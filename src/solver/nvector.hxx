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
    T& field;
    const bool own;
  };

  template <typename T>
  Content<T>& N_VContent_Bout(const N_Vector v) {
    return *static_cast<Content<T>*>(v->content);
  }
}

template <typename T, typename = bout::utils::EnableIfField<T>>
T& N_VField_Bout(const N_Vector v) {
  return N_VContent_Bout<T>(v).field;
}

template <typename T, typename = bout::utils::EnableIfField<T>>
N_Vector N_VNew_Bout(const SUNContext ctx, T& field, const bool own = false) {
  N_Vector v = callWithSUNContext(N_VNewEmpty, ctx);
  if (v == nullptr) {
    throw BoutException("N_VNewEmpty failed\n");
  }

  v->content = static_cast<void*>(new Content<T>({field, own}));

  v->ops->nvgetvectorid = []([[maybe_unused]] N_Vector x) {
    return SUNDIALS_NVEC_CUSTOM;
  };

  v->ops->nvclone = [](N_Vector x) {
    T* field = new T(N_VField_Bout<T>(x));
    return N_VNew_Bout<T>(x->sunctx, *field, true);
  };

  v->ops->nvdestroy = [](N_Vector x) {
    if (x == nullptr) {
      return;
    }
    if (x->content != nullptr && N_VContent_Bout<T>(x).own) {
      delete &N_VField_Bout<T>(x);
    }
    delete &N_VContent_Bout<T>(x);
    N_VFreeEmpty(x);
  };

  v->ops->nvgetlength = [](N_Vector x) -> sunindextype {
    // Is the the full domain size or just the subset fo the MPI process?
    return N_VField_Bout<T>(x).size();
  };

  v->ops->nvconst = [](sunrealtype c, N_Vector x) {
    N_VField_Bout<T>(x) = c;
  };

  v->ops->nvprod = [](N_Vector x, N_Vector y, N_Vector z) {
    // TODO: make sure this is the most efficient way
    N_VField_Bout<T>(z) = N_VField_Bout<T>(x) * N_VField_Bout<T>(y);
  };

  v->ops->nvabs = [](N_Vector x, N_Vector y) {
    N_VField_Bout<T>(y) = abs(N_VField_Bout<T>(x));
  };

  /* Other functions to implement (most optional)
  N_Vector (*nvcloneempty)(N_Vector);
   void (*nvdestroy)(N_Vector);
   void (*nvspace)(N_Vector, sunindextype*, sunindextype*);
   sunrealtype* (*nvgetarraypointer)(N_Vector);
   sunrealtype* (*nvgetdevicearraypointer)(N_Vector);
   void (*nvsetarraypointer)(sunrealtype*, N_Vector);
   SUNComm (*nvgetcommunicator)(N_Vector);
   sunindextype (*nvgetlength)(N_Vector);
   sunindextype (*nvgetlocallength)(N_Vector);
   void (*nvlinearsum)(sunrealtype, N_Vector, sunrealtype, N_Vector, N_Vector);
   void (*nvconst)(sunrealtype, N_Vector);
   void (*nvprod)(N_Vector, N_Vector, N_Vector);
   void (*nvdiv)(N_Vector, N_Vector, N_Vector);
   void (*nvscale)(sunrealtype, N_Vector, N_Vector);
   void (*nvabs)(N_Vector, N_Vector);
   void (*nvinv)(N_Vector, N_Vector);
   void (*nvaddconst)(N_Vector, sunrealtype, N_Vector);
   sunrealtype (*nvdotprod)(N_Vector, N_Vector);
   sunrealtype (*nvmaxnorm)(N_Vector);
   sunrealtype (*nvwrmsnorm)(N_Vector, N_Vector);
   sunrealtype (*nvwrmsnormmask)(N_Vector, N_Vector, N_Vector);
   sunrealtype (*nvmin)(N_Vector);
   sunrealtype (*nvwl2norm)(N_Vector, N_Vector);
   sunrealtype (*nvl1norm)(N_Vector);
   void (*nvcompare)(sunrealtype, N_Vector, N_Vector);
   sunbooleantype (*nvinvtest)(N_Vector, N_Vector);
   sunbooleantype (*nvconstrmask)(N_Vector, N_Vector, N_Vector);
   sunrealtype (*nvminquotient)(N_Vector, N_Vector);
   SUNErrCode (*nvlinearcombination)(int, sunrealtype*, N_Vector*, N_Vector);
   SUNErrCode (*nvscaleaddmulti)(int, sunrealtype*, N_Vector, N_Vector*,
                                 N_Vector*);
   SUNErrCode (*nvdotprodmulti)(int, N_Vector, N_Vector*, sunrealtype*);
   SUNErrCode (*nvlinearsumvectorarray)(int, sunrealtype, N_Vector*, sunrealtype,
                                          N_Vector*, N_Vector*);
   SUNErrCode (*nvscalevectorarray)(int, sunrealtype*, N_Vector*, N_Vector*);
   SUNErrCode (*nvconstvectorarray)(int, sunrealtype, N_Vector*);
   SUNErrCode (*nvwrmsnormvectorarray)(int, N_Vector*, N_Vector*, sunrealtype*);
   SUNErrCode (*nvwrmsnormmaskvectorarray)(int, N_Vector*, N_Vector*, N_Vector,
                                             sunrealtype*);
   SUNErrCode (*nvscaleaddmultivectorarray)(int, int, sunrealtype*, N_Vector*,
                                             N_Vector**, N_Vector**);
   SUNErrCode (*nvlinearcombinationvectorarray)(int, int, sunrealtype*,
                                                N_Vector**, N_Vector*);
   sunrealtype (*nvdotprodlocal)(N_Vector, N_Vector);
   sunrealtype (*nvmaxnormlocal)(N_Vector);
   sunrealtype (*nvminlocal)(N_Vector);
   sunrealtype (*nvl1normlocal)(N_Vector);
   sunbooleantype (*nvinvtestlocal)(N_Vector, N_Vector);
   sunbooleantype (*nvconstrmasklocal)(N_Vector, N_Vector, N_Vector);
   sunrealtype (*nvminquotientlocal)(N_Vector, N_Vector);
   sunrealtype (*nvwsqrsumlocal)(N_Vector, N_Vector);
   sunrealtype (*nvwsqrsummasklocal)(N_Vector, N_Vector, N_Vector);
   SUNErrCode (*nvdotprodmultilocal)(int, N_Vector, N_Vector*, sunrealtype*);
   SUNErrCode (*nvdotprodmultiallreduce)(int, N_Vector, sunrealtype*);
   SUNErrCode (*nvbufsize)(N_Vector, sunindextype*);
   SUNErrCode (*nvbufpack)(N_Vector, void*);
   SUNErrCode (*nvbufunpack)(N_Vector, void*);
   void (*nvprint)(N_Vector);
   void (*nvprintfile)(N_Vector, FILE*);
   */

  return v;
}

template <typename... Args>
N_Vector N_VNew_Bout(const SUNContext ctx, Args&... args) {
  N_Vector vecs[] = {N_VNew_Bout(ctx, args)...};
  return N_VNew_ManyVector(sizeof...(Args), vecs, ctx);
}

N_Vector N_VNew_Bout(const SUNContext ctx, Vector2D& vec) {
  return N_VNew_Bout(ctx, vec.x, vec.y);
}

N_Vector N_VNew_Bout(const SUNContext ctx, Vector3D& vec) {
  return N_VNew_Bout(ctx, vec.x, vec.y, vec.z);
}

#endif
