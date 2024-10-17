/**************************************************************************
 * Copyright 2010 B.D.Dudson, S.Farley, M.V.Umansky, X.Q.Xu
 *
 * Contact: Ben Dudson, bd512@york.ac.uk
 * 
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#ifndef BOUT_SUNDIAL_SOLVER_H
#define BOUT_SUNDIAL_SOLVER_H

#include <bout/field.hxx>
#include <bout/field2d.hxx>
#include <bout/field3d.hxx>
#include <bout/vector2d.hxx>
#include <bout/vector3d.hxx>
#include <nvector/nvector_manyvector.h>
#include <sundials/sundials_nvector.h>

#include "bout/sundials_backports.hxx"

namespace {
template <typename T>
T allReduce(const T local, const MPI_Op op = MPI_SUM) {
  T global;
  constexpr auto type = []() {
    if constexpr (std::is_same_v<T, double>) {
      return MPI_DOUBLE;
    } else if constexpr (std::is_same_v<T, unsigned int>) {
      return MPI_UNSIGNED;
    }
  }();
  MPI_Allreduce(&local, &global, 1, type, op, BoutComm::get());
  return global;
}

template <typename T>
struct Content {
  T& field; // A Field2D or Field3D
  const bool own;
  const bool evolve_bndry;
  const sunindextype length;

  Content(T& field, const bool own, const bool evolve_bndry)
      : field(field), own(own), evolve_bndry(evolve_bndry),
        length(allReduce(getRegion().size())) {}

  auto getRegion() const { return field.getRegion(evolve_bndry ? RGN_ALL : RGN_NOBNDRY); }

  ~Content() {
    if (own) {
      delete &field;
    }
  }
};

template <typename R, typename C, typename... D>
BoutReal fieldReduce(const R reduce, const bool allpe, const C& c1, const D&... c2) {
  BoutReal result = 0;
  const auto region = c1.getRegion();
  BOUT_FOR_OMP(i, region, parallel for reduction(+:result)) {
    result += reduce(c1.field[i], c2.field[i]...);
  }

  return allpe ? allReduce(result) : result;
}

template <typename T>
BoutReal dotProd(const Content<T>& c1, const Content<T>& c2, const bool allpe) {
  return fieldReduce(std::multiplies<BoutReal>(), allpe, c1, c2);
}

template <typename T>
BoutReal weightedL2Norm(const Content<T>& c1, const Content<T>& c2, const bool allpe) {
  return std::sqrt(fieldReduce(
      [](const auto v1, const auto v2) { return std::pow(v1 * v2, 2); }, allpe,
      c1, c2));
}

template <typename T>
BoutReal l1Norm(const Content<T>& c, const bool allpe) {
  return fieldReduce(std::abs<BoutReal>, allpe, c);
}

template <typename T>
Content<T>& N_VContent_Bout(const N_Vector v) {
  return *static_cast<Content<T>*>(v->content);
}
} // namespace

template <typename T>
T& N_VField_Bout(const N_Vector v) {
  return N_VContent_Bout<T>(v).field;
}

template <typename T, typename = bout::utils::EnableIfField<T>>
N_Vector N_VNew_Bout(const SUNContext ctx, T& field, const bool evolve_bndry,
                     const bool own = false) {N_VNew_Bout
  N_Vector v = callWithSUNContext(N_VNewEmpty, ctx);
  if (v == nullptr) {
    throw BoutException("N_VNewEmpty failed\n");
  }

  v->content = static_cast<void*>(new Content<T>(field, own, evolve_bndry));

  v->ops->nvgetvectorid = [](N_Vector) {
    return SUNDIALS_NVEC_CUSTOM;
  };

  v->ops->nvclone = [](N_Vector x) {
    T* const fieldClone = new T(N_VField_Bout<T>(x));
    fieldClone->allocate();
    return N_VNew_Bout(x->sunctx, *fieldClone, true);
  };

  v->ops->nvdestroy = [](N_Vector x) {
    if (x == nullptr) {
      return;
    }
    delete &N_VContent_Bout<T>(x);
    N_VFreeEmpty(x);
  };

  v->ops->nvgetlength = [](N_Vector x) { return N_VContent_Bout<T>(x).length; };

  v->ops->nvlinearsum = [](sunrealtype a, N_Vector x, sunrealtype b, N_Vector y,
                           N_Vector z) {
    N_VField_Bout<T>(z) = a * N_VField_Bout<T>(x) + b * N_VField_Bout<T>(y);
  };

  v->ops->nvconst = [](sunrealtype c, N_Vector x) { N_VField_Bout<T>(x) = c; };

  v->ops->nvprod = [](N_Vector x, N_Vector y, N_Vector z) {
    N_VField_Bout<T>(z) = N_VField_Bout<T>(x) * N_VField_Bout<T>(y);
  };

  v->ops->nvdiv = [](N_Vector x, N_Vector y, N_Vector z) {
    N_VField_Bout<T>(z) = N_VField_Bout<T>(x) / N_VField_Bout<T>(y);
  };

  v->ops->nvscale = [](sunrealtype a, N_Vector x, N_Vector y) {
    N_VField_Bout<T>(y) = a * N_VField_Bout<T>(x);
  };

  v->ops->nvabs = [](N_Vector x, N_Vector y) {
    N_VField_Bout<T>(y) = abs(N_VField_Bout<T>(x));
  };

  v->ops->nvinv = [](N_Vector x, N_Vector y) {
    N_VField_Bout<T>(y) = 1 / N_VField_Bout<T>(x);
  };

  v->ops->nvaddconst = [](N_Vector x, sunrealtype a, N_Vector y) {
    N_VField_Bout<T>(y) = a + N_VField_Bout<T>(x);
  };

  v->ops->nvdotprod = [](N_Vector x, N_Vector y) {
    return dotProd(N_VContent_Bout<T>(x), N_VContent_Bout<T>(y), true);
  };

  v->ops->nvmaxnorm = [](N_Vector x) { return max(abs(N_VField_Bout<T>(x)), true); };

  v->ops->nvwrmsnorm = [](N_Vector x, N_Vector y) {
    const auto sqrtLen = std::sqrt(static_cast<BoutReal>(N_VGetLength(x)));
    return weightedL2Norm(N_VContent_Bout<T>(x), N_VContent_Bout<T>(y), true) / sqrtLen;
  };

  v->ops->nvmin = [](N_Vector x) { return min(N_VField_Bout<T>(x), true); };

  v->ops->nvwl2norm = [](N_Vector x, N_Vector y) {
    return weightedL2Norm(N_VContent_Bout<T>(x), N_VContent_Bout<T>(y), true);
  };

  v->ops->nvl1norm = [](N_Vector x) { return l1Norm(N_VContent_Bout<T>(x), true); };

  /* Other functions to implement (most optional)
  N_Vector (*nvcloneempty)(N_Vector); // Probably not needed
   void (*nvdestroy)(N_Vector); // Implemented
   void (*nvspace)(N_Vector, sunindextype*, sunindextype*); // Probably not needed
   sunrealtype* (*nvgetarraypointer)(N_Vector); // Probably not needed
   sunrealtype* (*nvgetdevicearraypointer)(N_Vector); // Probably not needed
   void (*nvsetarraypointer)(sunrealtype*, N_Vector); // Probably not needed
   SUNComm (*nvgetcommunicator)(N_Vector); // Probably not needed
   sunindextype (*nvgetlength)(N_Vector); // Implemented
   sunindextype (*nvgetlocallength)(N_Vector);  // Probably not needed
   void (*nvlinearsum)(sunrealtype, N_Vector, sunrealtype, N_Vector, N_Vector); // Implemented
   void (*nvconst)(sunrealtype, N_Vector); // Implemented
   void (*nvprod)(N_Vector, N_Vector, N_Vector); // Implemented
   void (*nvdiv)(N_Vector, N_Vector, N_Vector); // Implemented
   void (*nvscale)(sunrealtype, N_Vector, N_Vector); // Implemented
   void (*nvabs)(N_Vector, N_Vector); // Implemented
   void (*nvinv)(N_Vector, N_Vector); // Implemented
   void (*nvaddconst)(N_Vector, sunrealtype, N_Vector); // Implemented
   sunrealtype (*nvdotprod)(N_Vector, N_Vector); // Implemented
   sunrealtype (*nvmaxnorm)(N_Vector); // Implemented
   sunrealtype (*nvwrmsnorm)(N_Vector, N_Vector); // Implemented
   sunrealtype (*nvwrmsnormmask)(N_Vector, N_Vector, N_Vector); // TODO
   sunrealtype (*nvmin)(N_Vector); // Implemented
   sunrealtype (*nvwl2norm)(N_Vector, N_Vector); // Implemented
   sunrealtype (*nvl1norm)(N_Vector); // Implemented
   void (*nvcompare)(sunrealtype, N_Vector, N_Vector); // TODO
   sunbooleantype (*nvinvtest)(N_Vector, N_Vector); // TODO
   sunbooleantype (*nvconstrmask)(N_Vector, N_Vector, N_Vector); // TODO
   sunrealtype (*nvminquotient)(N_Vector, N_Vector); // TODO
   SUNErrCode (*nvlinearcombination)(int, sunrealtype*, N_Vector*, N_Vector); // TODO
   SUNErrCode (*nvscaleaddmulti)(int, sunrealtype*, N_Vector, N_Vector*,
                                 N_Vector*); // Probably not needed
   SUNErrCode (*nvdotprodmulti)(int, N_Vector, N_Vector*, sunrealtype*); // Probably not needed
   SUNErrCode (*nvlinearsumvectorarray)(int, sunrealtype, N_Vector*, sunrealtype,
                                          N_Vector*, N_Vector*); // Probably not needed
   SUNErrCode (*nvscalevectorarray)(int, sunrealtype*, N_Vector*, N_Vector*); // Probably not needed
   SUNErrCode (*nvconstvectorarray)(int, sunrealtype, N_Vector*); // Probably not needed
   SUNErrCode (*nvwrmsnormvectorarray)(int, N_Vector*, N_Vector*, sunrealtype*); // Probably not needed
   SUNErrCode (*nvwrmsnormmaskvectorarray)(int, N_Vector*, N_Vector*, N_Vector,
                                             sunrealtype*); // Probably not needed
   SUNErrCode (*nvscaleaddmultivectorarray)(int, int, sunrealtype*, N_Vector*,
                                             N_Vector**, N_Vector**); // Probably not needed
   SUNErrCode (*nvlinearcombinationvectorarray)(int, int, sunrealtype*,
                                                N_Vector**, N_Vector*); // Probably not needed
   sunrealtype (*nvdotprodlocal)(N_Vector, N_Vector); // Probably not needed
   sunrealtype (*nvmaxnormlocal)(N_Vector); // Probably not needed
   sunrealtype (*nvminlocal)(N_Vector); // Probably not needed
   sunrealtype (*nvl1normlocal)(N_Vector); // Probably not needed
   sunbooleantype (*nvinvtestlocal)(N_Vector, N_Vector); // Probably not needed
   sunbooleantype (*nvconstrmasklocal)(N_Vector, N_Vector, N_Vector); // Probably not needed
   sunrealtype (*nvminquotientlocal)(N_Vector, N_Vector); // Probably not needed
   sunrealtype (*nvwsqrsumlocal)(N_Vector, N_Vector); // Probably not needed
   sunrealtype (*nvwsqrsummasklocal)(N_Vector, N_Vector, N_Vector); // Probably not needed
   SUNErrCode (*nvdotprodmultilocal)(int, N_Vector, N_Vector*, sunrealtype*); // Probably not needed
   SUNErrCode (*nvdotprodmultiallreduce)(int, N_Vector, sunrealtype*); // Probably not needed
   SUNErrCode (*nvbufsize)(N_Vector, sunindextype*); // Probably not needed
   SUNErrCode (*nvbufpack)(N_Vector, void*); // Probably not needed
   SUNErrCode (*nvbufunpack)(N_Vector, void*); // Probably not needed
   void (*nvprint)(N_Vector); // Nice, but optional
   void (*nvprintfile)(N_Vector, FILE*); // Probably not needed
   */

  return v;
}

N_Vector N_VNew_Bout(const SUNContext ctx, Vector2D& vec, const bool own = false) {
  N_Vector vecs[] = {N_VNew_Bout(ctx, vec.x, own), N_VNew_Bout(ctx, vec.y, own)};
  return N_VNew_ManyVector(2, vecs, ctx);
}

N_Vector N_VNew_Bout(const SUNContext ctx, Vector3D& vec, const bool own = false) {
  N_Vector vecs[] = {N_VNew_Bout(ctx, vec.x, own), N_VNew_Bout(ctx, vec.y, own),
                     N_VNew_Bout(ctx, vec.z, own)};
  return N_VNew_ManyVector(3, vecs, ctx);
}

template <typename... T>
N_Vector N_VNew_Bout(const SUNContext ctx, std::vector<T>&... args) {
  std::vector<N_Vector> vecs((args.size() + ...));
  auto vecsBegin = vecs.begin();
  ((std::transform(args.begin(), args.end(), vecsBegin,
                   [ctx](T& field) { return N_VNew_Bout(ctx, field); }),
    vecsBegin += args.size()),
   ...);
  return N_VNew_ManyVector(vecs.size(), vecs.data(), ctx);
}

auto test() {
  Vector3D* f = nullptr;
  sundials::Context ctx;
  std::vector v{*f, *f};
  return N_VNew_Bout(ctx, v, v, v);
}

#endif
