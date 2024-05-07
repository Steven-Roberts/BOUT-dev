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
struct Content {
  T& data; // A vector or field
  const bool own;

  ~Content() {
    if (own) {
      delete &data;
    }
  }
};
} // namespace

template <typename T>
T& N_VData_Bout(const N_Vector v) {
  return static_cast<Content<T>*>(v->content)->data;
}

namespace {
template <typename T, typename F, typename... Args>
std::enable_if_t<bout::utils::is_Field_v<T>, void> mapFields(F f, Args... vectors) {
  f(N_VData_Bout<T>(vectors)...);
}

template <typename T, typename F, typename... Args>
std::enable_if_t<std::is_base_of_v<Vector2D, T>, void> mapFields(F f, Args... vectors) {
  f(N_VData_Bout<T>(vectors).x...);
  f(N_VData_Bout<T>(vectors).y...);
}

template <typename T, typename F, typename... Args>
std::enable_if_t<std::is_base_of_v<Vector3D, T>, void> mapFields(F f, Args... vectors) {
  f(N_VData_Bout<T>(vectors).x...);
  f(N_VData_Bout<T>(vectors).y...);
  f(N_VData_Bout<T>(vectors).z...);
}
} // namespace

template <typename T>
N_Vector N_VNew_Bout(const SUNContext ctx, T& data, const bool own = false) {
  N_Vector v = callWithSUNContext(N_VNewEmpty, ctx);
  if (v == nullptr) {
    throw BoutException("N_VNewEmpty failed\n");
  }

  v->content = static_cast<void*>(new Content<T>({data, own}));

  v->ops->nvgetvectorid = []([[maybe_unused]] N_Vector x) {
    return SUNDIALS_NVEC_CUSTOM;
  };

  v->ops->nvclone = [](N_Vector x) {
    T* data = new T(N_VData_Bout<T>(x));
    return N_VNew_Bout<T>(x->sunctx, *data, true);
  };

  v->ops->nvdestroy = [](N_Vector x) {
    if (x == nullptr) {
      return;
    }
    delete static_cast<Content<T>*>(x->content);
    N_VFreeEmpty(x);
  };

  v->ops->nvgetlength = [](N_Vector x) {
    sunindextype len = 0;
    mapFields<T>(
        [&len](auto& fx) {
          len += fx.size(); // TODO: check this isn't the MPI local size
        },
        x);
    return len;
  };

  v->ops->nvconst = [](sunrealtype c, N_Vector x) {
    mapFields<T>([c](auto& f) { f = c; }, x);
  };

  v->ops->nvprod = [](N_Vector x, N_Vector y, N_Vector z) {
    mapFields<T>([](auto& fx, auto& fy, auto& fz) { fz = fx * fy; }, x, y, z);
  };

  v->ops->nvabs = [](N_Vector x, N_Vector y) {
    mapFields<T>([](auto& fx, auto& fy) { fy = abs(fx); }, x, y);
  };

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
   void (*nvlinearsum)(sunrealtype, N_Vector, sunrealtype, N_Vector, N_Vector); // TODO
   void (*nvconst)(sunrealtype, N_Vector); // Implemented
   void (*nvprod)(N_Vector, N_Vector, N_Vector); // Implemented
   void (*nvdiv)(N_Vector, N_Vector, N_Vector); // TODO
   void (*nvscale)(sunrealtype, N_Vector, N_Vector); // TODO
   void (*nvabs)(N_Vector, N_Vector); // Implemented
   void (*nvinv)(N_Vector, N_Vector); // TODO
   void (*nvaddconst)(N_Vector, sunrealtype, N_Vector); // TODO
   sunrealtype (*nvdotprod)(N_Vector, N_Vector); // TODO
   sunrealtype (*nvmaxnorm)(N_Vector); // TODO
   sunrealtype (*nvwrmsnorm)(N_Vector, N_Vector); // TODO
   sunrealtype (*nvwrmsnormmask)(N_Vector, N_Vector, N_Vector); // TODO
   sunrealtype (*nvmin)(N_Vector); // TODO
   sunrealtype (*nvwl2norm)(N_Vector, N_Vector); // TODO
   sunrealtype (*nvl1norm)(N_Vector); // TODO
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

template <typename... Args>
N_Vector N_VNew_Bout(SUNContext ctx, Args&... args) {
  N_Vector vecs[] = {N_VNew_Bout(ctx, args)...};
  return N_VNew_ManyVector(sizeof...(Args), vecs, ctx);
}

#endif
