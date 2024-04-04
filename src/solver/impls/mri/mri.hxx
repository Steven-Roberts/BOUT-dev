/**************************************************************************
 * Interface to ARKODE's multirate infinitesimal (MRI) solver
 * NOTE: This requires SUNDIALS version 5 or newer.
 *
 **************************************************************************
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

#ifndef BOUT_MRI_SOLVER_H
#define BOUT_MRI_SOLVER_H

#include "bout/build_config.hxx"
#include "bout/solver.hxx"

#if not BOUT_HAS_ARKODE

namespace {
RegisterUnavailableSolver
    registerunavailablemri("mri", "BOUT++ was not configured with ARKODE/SUNDIALS");
}

#else

#include <sundials/sundials_config.h>

#if SUNDIALS_VERSION_MAJOR >= 5

#include "bout/bout_types.hxx"
#include "bout/sundials_backports.hxx"

#include <nvector/nvector_parallel.h>
#include <sundials/sundials_config.h>

#include <vector>

class MriSolver;
class Options;

namespace {
RegisterSolver<MriSolver> registersolvermri("mri");
}

class MriSolver : public Solver {
public:
  explicit MriSolver(Options* opts = nullptr);
  ~MriSolver();

  BoutReal getCurrentTimestep() override { return hcur; }

  int init() override;

  int run() override;
  BoutReal run(BoutReal tout);

  // These functions used internally (but need to be public)
  void rhs_f(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void rhs_e(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void rhs_i(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void rhs(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void pre(BoutReal t, BoutReal gamma, BoutReal delta, BoutReal* udata, BoutReal* rvec,
           BoutReal* zvec);
  void jac(BoutReal t, BoutReal* ydata, BoutReal* vdata, BoutReal* Jvdata);

private:
  BoutReal hcur; //< Current internal timestep

  bool diagnose{false}; //< Output additional diagnostics

  N_Vector uvec{nullptr};    //< Values
  void* arkode_mem{nullptr}; //< ARKODE internal memory block

  BoutReal pre_Wtime{0.0}; //< Time in preconditioner
  int pre_ncalls{0};       //< Number of calls to preconditioner

  /// Maximum number of steps to take between outputs
  int mxsteps;
  /// Use linear implicit solver (only evaluates jacobian inversion once)
  bool set_linear;
  /// Solve explicit portion in fixed timestep mode. NOTE: This is not recommended except
  /// for code comparison
  BoutReal timestep;
  /// Order of internal step
  int order;
  /// Use accelerated fixed point solver instead of Newton iterative
  bool fixed_point;
  /// Use user-supplied preconditioner function
  bool use_precon;
  /// Number of Krylov basis vectors to use
  int maxl;
  /// Use right preconditioning instead of left preconditioning
  bool rightprec;
  /// Use user-supplied Jacobian function
  bool use_jacobian;

  // Diagnostics from ARKODE
  int nsteps{0};
  int nfe_evals{0};
  int nfi_evals{0};
  int nniters{0};
  int npevals{0};
  int nliters{0};

  /// SPGMR solver structure
  SUNLinearSolver sun_solver{nullptr};
  /// Solver for implicit stages
  SUNNonlinearSolver nonlinear_solver{nullptr};

  /// Context for SUNDIALS memory allocations
  sundials::Context suncontext;
};

#endif // SUNDIALS_VERSION_MAJOR
#endif // BOUT_HAS_MRI
#endif // BOUT_MRI_SOLVER_H
