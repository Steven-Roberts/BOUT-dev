#include "bout/build_defines.hxx"

#if BOUT_HAS_ADIOS2

#include "bout/adios_object.hxx"
#include "bout/array.hxx"
#include "bout/assert.hxx"
#include "bout/bout_types.hxx"
#include "bout/boutcomm.hxx"
#include "bout/boutexception.hxx"
#include "bout/field2d.hxx"
#include "bout/field3d.hxx"
#include "bout/fieldperp.hxx"
#include "bout/mesh.hxx"
#include "bout/utils.hxx"

#include <adios2.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bout {

static ADIOSPtr adios = nullptr;
static std::unordered_map<std::string, ADIOSStream> adiosStreams;

namespace {
bool isReadMode(adios2::Mode mode) {
  return mode == adios2::Mode::Read or mode == adios2::Mode::ReadRandomAccess;
}

auto streamKey(const std::string& fname, adios2::Mode mode) -> std::string {
  if (isReadMode(mode)) {
    return fname + ":read";
  }
  return fname + ":write";
}

using bout::utils::tuple_index_sequence;

template <class Tuple, std::size_t... I>
auto make_shape_impl(std::size_t first, const Tuple& t,
                     std::index_sequence<I...> /* index */) {
  return adios2::Dims{first, static_cast<std::size_t>(std::get<I>(t))...};
}

// Return an `adios2::Dims` with value ``{first, value.shape()[0]...}``
template <class T>
auto make_shape(std::size_t first, const T& value) {
  const auto shape = value.shape();
  return make_shape_impl(first, shape, tuple_index_sequence<decltype(shape)>{});
}

template <class Tuple, std::size_t... I>
auto make_start_impl(const Tuple& t, std::index_sequence<I...> /* index */) {
  // Hey look, a legitimate use of the comma operator to get a bunch
  // of zeros the length of the index_sequence!
  return adios2::Dims{static_cast<std::size_t>(BoutComm::rank()),
                      (std::get<I>(t), std::size_t{0})...};
}

// Return an `adios2::Dims` with value ``{rank, 0...}``, with as many zeros as
// the dimension of ``T``
template <class T>
auto make_start(const T& value) {
  const auto shape = value.shape();
  return make_start_impl(shape, tuple_index_sequence<decltype(shape)>{});
}

template <std::size_t... I>
auto make_dims_impl(std::index_sequence<I...> /*index*/) {
  return std::vector<std::string>{"rank", ("dim_" + std::to_string(I))...};
}

// Return vector of dimension names: `{"rank", "dim_0", ...}`
template <class T>
auto make_dims(const T& value) {
  return make_dims_impl(tuple_index_sequence<decltype(value.shape())>{});
}

template <template <class> class T, class U>
void adiosPutArray(ADIOSStream& stream, const std::string& name, const T<U>& value) {
  const auto shape = make_shape(static_cast<std::size_t>(BoutComm::size()), value);
  auto var = stream.GetArrayVariable<U>(name, shape, make_dims(value), BoutComm::rank());
  var.SetSelection({make_start(value), make_shape(1, value)});
  stream.engine().Put<U>(var, value.begin());
}
} // namespace

ADIOSSelection::ADIOSSelection(const std::vector<std::string>& dim_names,
                               const std::vector<int>& dim_sizes, const Mesh& mesh) {
  const auto ndims = dim_names.size();
  const bool dim0_is_rank = ndims > 0 ? (dim_names[0] == "rank") : false;
  const bool dim0_is_x = ndims > 0 ? (dim_names[0] == "x") : false;
  const bool dim1_is_y = ndims > 1 ? (dim_names[1] == "y") : false;
  const bool dim1_is_z = ndims > 1 ? (dim_names[1] == "z") : false;
  const bool dim2_is_z = ndims > 2 ? (dim_names[2] == "z") : false;

  should_set_selection = dim0_is_rank or (ndims == 2 and dim0_is_x and dim1_is_y)
                         or (ndims == 2 and dim0_is_x and dim1_is_z)
                         or (ndims == 3 and dim0_is_x and dim1_is_y and dim2_is_z);

  if (dim0_is_rank) {
    const auto ndim_sizes = dim_sizes.size();
    ASSERT3(ndim_sizes > 1);

    // This is a distributed array, so the local variable is going
    // to be shape (dim_sizes[1]...) (that is, drop the rank)
    dims.push_back(dim_sizes[1]);
    // but we tell ADIOS to read our rank's bit with the full ndims
    start = {static_cast<std::size_t>(BoutComm::rank()), 0};
    count = {std::size_t{1}, static_cast<std::size_t>(dim_sizes[0])};

    if (ndim_sizes > 2) {
      dims.push_back(dim_sizes[2]);
      start.push_back(0);
      count.push_back(dim_sizes[1]);
    }
    if (ndim_sizes > 3) {
      dims.push_back(dim_sizes[3]);
      start.push_back(0);
      count.push_back(dim_sizes[2]);
    }
    mem_count = count;
    mem_start = start;
    return;
  }

  if (ndims > 0) {
    dims.push_back(dim0_is_x ? mesh.LocalNx : dim_sizes[0]);
  }
  if (ndims > 1) {
    if (dim1_is_y) {
      dims.push_back(mesh.LocalNy);
    } else if (dim1_is_z) {
      dims.push_back(mesh.LocalNz);
    } else {
      dims.push_back(dim_sizes[1]);
    }
  }
  if (ndims > 2) {
    dims.push_back(dim2_is_z ? mesh.LocalNz : dim_sizes[2]);
  }

  shape.push_back(static_cast<std::size_t>(mesh.GlobalNx));
  start.push_back(static_cast<std::size_t>(mesh.MapGlobalX));
  count.push_back(static_cast<std::size_t>(mesh.MapCountX));
  mem_start.push_back(static_cast<std::size_t>(mesh.MapLocalX));
  mem_count.push_back(static_cast<std::size_t>(mesh.LocalNx));

  if (dim1_is_y) {
    shape.push_back(static_cast<std::size_t>(mesh.GlobalNy));
    start.push_back(static_cast<std::size_t>(mesh.MapGlobalY));
    count.push_back(static_cast<std::size_t>(mesh.MapCountY));
    mem_start.push_back(static_cast<std::size_t>(mesh.MapLocalY));
    mem_count.push_back(static_cast<std::size_t>(mesh.LocalNy));
  } else if (dim1_is_z) {
    shape.push_back(static_cast<std::size_t>(mesh.GlobalNz));
    start.push_back(static_cast<std::size_t>(mesh.MapGlobalZ));
    count.push_back(static_cast<std::size_t>(mesh.MapCountZ));
    mem_start.push_back(static_cast<std::size_t>(mesh.MapLocalZ));
    mem_count.push_back(static_cast<std::size_t>(mesh.LocalNz));
  }

  if (dim2_is_z) {
    shape.push_back(static_cast<std::size_t>(mesh.GlobalNz));
    start.push_back(static_cast<std::size_t>(mesh.MapGlobalZ));
    count.push_back(static_cast<std::size_t>(mesh.MapCountZ));
    mem_start.push_back(static_cast<std::size_t>(mesh.MapLocalZ));
    mem_count.push_back(static_cast<std::size_t>(mesh.LocalNz));
  }
}

void ADIOSInit(MPI_Comm comm) { adios = std::make_shared<adios2::ADIOS>(comm); }

void ADIOSInit(const std::string configFile, MPI_Comm comm) {
  adios = std::make_shared<adios2::ADIOS>(configFile, comm);
}

void ADIOSFinalize() {
  if (adios == nullptr) {
    throw BoutException(
        "ADIOS needs to be initialized first before calling ADIOSFinalize()");
  }
  adiosStreams.clear();
  adios.reset();
}

ADIOSPtr GetADIOSPtr() {
  if (adios == nullptr) {
    throw BoutException(
        "ADIOS needs to be initialized first before calling GetADIOSPtr()");
  }
  return adios;
}

IOPtr GetIOPtr(const std::string IOName) {
  auto adios = GetADIOSPtr();
  IOPtr io = nullptr;
  try {
    io = std::make_shared<adios2::IO>(adios->AtIO(IOName));
  } catch (std::invalid_argument& e) {
  }
  return io;
}

ADIOSStream::~ADIOSStream() { close(); }

ADIOSStream& ADIOSStream::ADIOSGetStream(const std::string& fname, adios2::Mode mode,
                                         const std::string& engineType) {
  const auto key = streamKey(fname, mode);
  auto it = adiosStreams.find(key);
  if (it == adiosStreams.end()) {
    it = adiosStreams.emplace(key, ADIOSStream(fname, mode, engineType)).first;
  }
  return it->second;
}

ADIOSStream::ADIOSStream(const std::string& fname, adios2::Mode mode,
                         const std::string& engineType)
    : fname(fname), file_mode(mode) {

  ADIOSPtr adiosp = GetADIOSPtr();
  try {
    io = adiosp->AtIO(fname);
  } catch (const std::invalid_argument& e) {
    io = adiosp->DeclareIO(fname);
    if (not isReadMode(mode)) {
      io.SetEngine(engineType);
    }
  }
}

void ADIOSStream::close() {
  if (engine_) {
    if (isInStep) {
      engine_.EndStep();
      isInStep = false;
    }
    engine_.Close();
    engine_ = adios2::Engine();
  }
}

void ADIOSSetParameters(const std::string& input, char delimKeyValue, char delimItem,
                        adios2::IO& io) {
  auto lf_Trim = [](std::string& input) {
    input.erase(0, input.find_first_not_of(" \n\r\t")); // prefixing spaces
    input.erase(input.find_last_not_of(" \n\r\t") + 1); // suffixing spaces
  };

  std::istringstream inputSS(input);
  std::string parameter;
  while (std::getline(inputSS, parameter, delimItem)) {
    const size_t position = parameter.find(delimKeyValue);
    if (position == std::string::npos) {
      throw BoutException("ADIOSSetParameters(): wrong format for IO parameter "
                          + parameter + ", format must be key" + delimKeyValue
                          + "value for each entry");
    }

    std::string key = parameter.substr(0, position);
    lf_Trim(key);
    std::string value = parameter.substr(position + 1);
    lf_Trim(value);
    if (value.length() == 0) {
      throw BoutException("ADIOS2SetParameters: empty value in IO parameter " + parameter
                          + ", format must be key" + delimKeyValue + "value");
    }
    io.SetParameter(key, value);
  }
}

void adiosPut(ADIOSStream& stream, const std::string& name, int value) {
  // Scalars are only written from processor 0
  if (BoutComm::rank() != 0) {
    return;
  }
  stream.engine().Put(stream.GetValueVariable<int>(name), value);
}

void adiosPut(ADIOSStream& stream, const std::string& name, BoutReal value) {
  // Scalars are only written from processor 0
  if (BoutComm::rank() != 0) {
    return;
  }
  stream.engine().Put(stream.GetValueVariable<BoutReal>(name), value);
}

void adiosPut(ADIOSStream& stream, const std::string& name, const std::string& value) {
  // Scalars are only written from processor 0
  if (BoutComm::rank() != 0) {
    return;
  }
  stream.engine().Put<std::string>(stream.GetValueVariable<std::string>(name), value,
                                   adios2::Mode::Sync);
}

void adiosPut(ADIOSStream& stream, const std::string& name, const Array<int>& value) {
  adiosPutArray(stream, name, value);
}

void adiosPut(ADIOSStream& stream, const std::string& name,
              const Array<BoutReal>& value) {
  adiosPutArray(stream, name, value);
}

void adiosPut(ADIOSStream& stream, const std::string& name, const Matrix<int>& value) {
  adiosPutArray(stream, name, value);
}

void adiosPut(ADIOSStream& stream, const std::string& name,
              const Matrix<BoutReal>& value) {
  adiosPutArray(stream, name, value);
}

void adiosPut(ADIOSStream& stream, const std::string& name, const Tensor<int>& value) {
  adiosPutArray(stream, name, value);
}

void adiosPut(ADIOSStream& stream, const std::string& name,
              const Tensor<BoutReal>& value) {
  adiosPutArray(stream, name, value);
}

void adiosPut(ADIOSStream& stream, const std::string& name, const Field2D& value) {
  ADIOSSelection selection(ADIOS_DIMS_XY, {}, *value.getMesh());
  auto var = stream.GetArrayVariable<BoutReal>(name, selection.shape, ADIOS_DIMS_XY,
                                               BoutComm::rank());
  var.SetSelection(selection.selection());
  var.SetMemorySelection(selection.memorySelection());
  stream.engine().Put(var, &value(0, 0));
}

void adiosPut(ADIOSStream& stream, const std::string& name, const Field3D& value) {
  ADIOSSelection selection(ADIOS_DIMS_XYZ, {}, *value.getMesh());
  auto var = stream.GetArrayVariable<BoutReal>(name, selection.shape, ADIOS_DIMS_XYZ,
                                               BoutComm::rank());
  var.SetSelection(selection.selection());
  var.SetMemorySelection(selection.memorySelection());
  stream.engine().Put(var, &value(0, 0, 0));
}

void adiosPut(ADIOSStream& stream, const std::string& name, const FieldPerp& value) {
  ADIOSSelection selection(ADIOS_DIMS_XZ, {}, *value.getMesh());
  auto var = stream.GetArrayVariable<BoutReal>(name, selection.shape, ADIOS_DIMS_XZ,
                                               BoutComm::rank());
  var.SetSelection(selection.selection());
  var.SetMemorySelection(selection.memorySelection());
  stream.engine().Put<BoutReal>(var, &value(0, 0));
}

void adiosGet(adios2::IO& io, adios2::Engine& reader, const std::string& name,
              Field2D& value) {
  auto var = io.InquireVariable<BoutReal>(name);
  if (!var) {
    throw BoutException("Could not find ADIOS variable '{:s}' in file '{:s}'", name,
                        reader.Name());
  }
  value.allocate();
  ADIOSSelection selection(ADIOS_DIMS_XY, {}, *value.getMesh());
  var.SetSelection(selection.selection());
  var.SetMemorySelection(selection.memorySelection());
  reader.Get<BoutReal>(var, &value(0, 0), adios2::Mode::Sync);
}

void adiosGet(adios2::IO& io, adios2::Engine& reader, const std::string& name,
              Field3D& value) {
  auto var = io.InquireVariable<BoutReal>(name);
  if (!var) {
    throw BoutException("Could not find ADIOS variable '{:s}' in file '{:s}'", name,
                        reader.Name());
  }
  value.allocate();
  ADIOSSelection selection(ADIOS_DIMS_XYZ, {}, *value.getMesh());
  var.SetSelection(selection.selection());
  var.SetMemorySelection(selection.memorySelection());
  reader.Get<BoutReal>(var, &value(0, 0, 0), adios2::Mode::Sync);
}

void adiosGet(adios2::IO& io, adios2::Engine& reader, const std::string& name,
              FieldPerp& value) {
  auto var = io.InquireVariable<BoutReal>(name);
  if (!var) {
    throw BoutException("Could not find ADIOS variable '{:s}' in file '{:s}'", name,
                        reader.Name());
  }
  value.allocate();
  ADIOSSelection selection(ADIOS_DIMS_XZ, {}, *value.getMesh());
  var.SetSelection(selection.selection());
  var.SetMemorySelection(selection.memorySelection());
  reader.Get<BoutReal>(var, &value(0, 0), adios2::Mode::Sync);
}

void adiosGet(ADIOSStream& stream, const std::string& name, Field2D& value) {
  adiosGet(stream.io, stream.engine(), name, value);
}

void adiosGet(ADIOSStream& stream, const std::string& name, Field3D& value) {
  adiosGet(stream.io, stream.engine(), name, value);
}

void adiosGet(ADIOSStream& stream, const std::string& name, FieldPerp& value) {
  adiosGet(stream.io, stream.engine(), name, value);
}

} // namespace bout
#endif //BOUT_HAS_ADIOS2
