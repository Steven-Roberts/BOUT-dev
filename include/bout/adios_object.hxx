/*!\file*******************************************************************
 * Provides access to the ADIOS library, handling initialisation and
 * finalisation.
 *
 * Usage
 * -----
 *
 * #include <bout/adios_object.hxx>
 *
 **************************************************************************/

#ifndef ADIOS_OBJECT_HXX
#define ADIOS_OBJECT_HXX

#include "bout/build_defines.hxx"

#if BOUT_HAS_ADIOS2

#include "bout/array.hxx"
#include "bout/bout_types.hxx"
#include "bout/boutexception.hxx"
#include "bout/utils.hxx"

#include <adios2.h>
#include <memory>
#include <mpi.h>
#include <string>
#include <vector>

class Field2D;
class Field3D;
class FieldPerp;
class Mesh;

namespace bout {

void ADIOSInit(MPI_Comm comm);
void ADIOSInit(const std::string configFile, MPI_Comm comm);
void ADIOSFinalize();

using ADIOSPtr = std::shared_ptr<adios2::ADIOS>;
using EnginePtr = std::shared_ptr<adios2::Engine>;
using IOPtr = std::shared_ptr<adios2::IO>;

ADIOSPtr GetADIOSPtr();
IOPtr GetIOPtr(const std::string IOName);

inline const std::vector<std::string> ADIOS_DIMS_X = {"x"};
inline const std::vector<std::string> ADIOS_DIMS_XY = {"x", "y"};
inline const std::vector<std::string> ADIOS_DIMS_XZ = {"x", "z"};
inline const std::vector<std::string> ADIOS_DIMS_XYZ = {"x", "y", "z"};

// Helper class to construct ADIOS hyperslices for BOUT++ distributed data
struct ADIOSSelection {
  // Offset of this processor's data into the global array
  adios2::Dims start;
  // The size of the mapped region
  adios2::Dims count;
  // Where the actual data starts in data pointer (to exclude ghost cells)
  adios2::Dims mem_start;
  // The actual size of data pointer in memory (including ghost cells)
  adios2::Dims mem_count;
  // Global shape, including boundaries but not guard cells
  adios2::Dims shape;
  // Shape of the local variable to read into
  std::vector<int> dims;

  // Distributed Field/Array/Matrix/Tensor
  bool should_set_selection{false};

  ADIOSSelection(const std::vector<std::string>& dim_names,
                 const std::vector<int>& dim_sizes, const Mesh& mesh);

  auto selection() const { return adios2::Box<adios2::Dims>{start, count}; }
  auto memorySelection() const { return adios2::Box<adios2::Dims>{mem_start, mem_count}; }
};

class ADIOSStream {
public:
  adios2::IO io;
  adios2::Variable<double> vTime;
  adios2::Variable<int> vStep;
  int adiosStep = 0;

  /** create or return the ADIOSStream based on the target file name */
  static ADIOSStream& ADIOSGetStream(const std::string& fname, adios2::Mode mode,
                                     const std::string& engineType = "BP5");

  ~ADIOSStream();

  template <class T>
  adios2::Variable<T> GetValueVariable(const std::string& varname) {
    auto v = io.InquireVariable<T>(varname);
    if (!v) {
      v = io.DefineVariable<T>(varname);
    }
    return v;
  }

  template <class T>
  adios2::Variable<T>
  GetArrayVariable(const std::string& varname, const adios2::Dims& shape,
                   const std::vector<std::string>& dimNames, int rank) {
    adios2::Variable<T> v = io.InquireVariable<T>(varname);
    if (!v) {
      adios2::Dims start(shape.size());
      v = io.DefineVariable<T>(varname, shape, start, shape);
      if (!rank && dimNames.size()) {
        io.DefineAttribute<std::string>("__xarray_dimensions__", dimNames.data(),
                                        dimNames.size(), varname, "/", true);
      }
    } else {
      v.SetShape(shape);
    }
    return v;
  }

  auto engine() -> adios2::Engine& {
    if (not engine_) {
      engine_ = io.Open(fname, file_mode);
      if (not engine_) {
        throw BoutException("Could not open ADIOS file '{:s}'", fname);
      }
    }
    return engine_;
  }

  void beginStep() {
    if (not isInStep) {
      engine().BeginStep();
      isInStep = true;
      adiosStep = static_cast<int>(engine().CurrentStep());
    }
  }

  void endStep() {
    if (isInStep) {
      engine().EndStep();
      isInStep = false;
    }
  }

  void finish() {
    if (engine_) {
      endStep();
      close();
    }
  }

  void close();

private:
  ADIOSStream(const std::string& fname, adios2::Mode mode, const std::string& engineType);

  std::string fname;
  adios2::Mode file_mode;
  adios2::Engine engine_;

  /// true if BeginStep was called and EndStep was not yet called
  bool isInStep = false;
};

/** Set user parameters for an IO group */
void ADIOSSetParameters(const std::string& input, char delimKeyValue, char delimItem,
                        adios2::IO& io);

void adiosPut(ADIOSStream& stream, const std::string& name, int value);
void adiosPut(ADIOSStream& stream, const std::string& name, BoutReal value);
void adiosPut(ADIOSStream& stream, const std::string& name, const std::string& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const Array<int>& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const Array<BoutReal>& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const Matrix<int>& value);
void adiosPut(ADIOSStream& stream, const std::string& name,
              const Matrix<BoutReal>& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const Tensor<int>& value);
void adiosPut(ADIOSStream& stream, const std::string& name,
              const Tensor<BoutReal>& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const Field2D& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const Field3D& value);
void adiosPut(ADIOSStream& stream, const std::string& name, const FieldPerp& value);

void adiosGet(adios2::IO& io, adios2::Engine& reader, const std::string& name,
              Field2D& value);
void adiosGet(adios2::IO& io, adios2::Engine& reader, const std::string& name,
              Field3D& value);
void adiosGet(adios2::IO& io, adios2::Engine& reader, const std::string& name,
              FieldPerp& value);
void adiosGet(ADIOSStream& stream, const std::string& name, Field2D& value);
void adiosGet(ADIOSStream& stream, const std::string& name, Field3D& value);
void adiosGet(ADIOSStream& stream, const std::string& name, FieldPerp& value);

} // namespace bout

#endif //BOUT_HAS_ADIOS2
#endif //ADIOS_OBJECT_HXX
