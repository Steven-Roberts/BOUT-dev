/**************************************************************************
 * Experimental interface to SUNDIALS ARKode IMEX solver
 *
 * NOTE: ARKode is still in beta testing so use with cautious optimism
 *
 **************************************************************************
 * Copyright 2010 B.D.Dudson, S.Farley, M.V.Umansky, X.Q.Xu
 *
 * Contact: Nick Walkden, nick.walkden@ccfe.ac.uk
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

#include "bout/build_config.hxx"

#include "mri.hxx"

#if BOUT_HAS_ARKODE and (SUNDIALS_VERSION_MAJOR >= 5)

#include "bout/boutcomm.hxx"
#include "bout/boutexception.hxx"
#include "bout/field3d.hxx"
#include "bout/mesh.hxx"
#include "bout/msg_stack.hxx"
#include "bout/options.hxx"
#include "bout/output.hxx"
#include "bout/unused.hxx"
#include "bout/utils.hxx"

#include <arkode/arkode_mristep.h>
#include <arkode/arkode_bbdpre.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_types.h>

#include <algorithm>
#include <numeric>

class Field2D;

static int arkode_rhs_fast(BoutReal t, N_Vector u, N_Vector du, void* user_data);
static int arkode_rhs_explicit(BoutReal t, N_Vector u, N_Vector du, void* user_data);
static int arkode_rhs_implicit(BoutReal t, N_Vector u, N_Vector du, void* user_data);

static int arkode_bbd_rhs(sunindextype Nlocal, BoutReal t, N_Vector u, N_Vector du,
                          void* user_data);
static int arkode_pre(BoutReal t, N_Vector yy, N_Vector yp, N_Vector rvec, N_Vector zvec,
                      BoutReal gamma, BoutReal delta, int lr, void* user_data);

static int arkode_jac(N_Vector v, N_Vector Jv, BoutReal t, N_Vector y, N_Vector fy,
                      void* user_data, N_Vector tmp);

MriSolver::MriSolver(Options* opts)
    : Solver(opts), diagnose((*options)["diagnose"]
                                 .doc("Print some additional diagnostics")
                                 .withDefault(false)),
      mxsteps((*options)["mxstep"]
                  .doc("Maximum number of steps to take between outputs")
                  .withDefault(500)),
      set_linear(
          (*options)["set_linear"]
              .doc("Use linear implicit solver (only evaluates jacobian inversion once)")
              .withDefault(false)),
      timestep((*options)["timestep"]
                     .doc("The slow timestep size")
                     .withDefault(0)),
      order((*options)["order"].doc("Order of internal step").withDefault(4)),
      fixed_point(
          (*options)["fixed_point"]
              .doc("Use accelerated fixed point solver instead of Newton iterative")
              .withDefault(false)),
      use_precon((*options)["use_precon"]
                     .doc("Use user-supplied preconditioner function")
                     .withDefault(false)),
      maxl(
          (*options)["maxl"].doc("Number of Krylov basis vectors to use").withDefault(0)),
      rightprec((*options)["rightprec"]
                    .doc("Use right preconditioning instead of left preconditioning")
                    .withDefault(false)),
      use_jacobian((*options)["use_jacobian"]
                       .doc("Use user-supplied Jacobian function")
                       .withDefault(false)),
      suncontext(createSUNContext(BoutComm::get())) {
  has_constraints = false; // This solver doesn't have constraints

  // Add diagnostics to output
  add_int_diagnostic(nsteps, "arkode_nsteps", "Cumulative number of internal steps");
  add_int_diagnostic(nfe_evals, "arkode_nfe_evals",
                     "No. of calls to fe (explicit portion of the right-hand-side "
                     "function) function");
  add_int_diagnostic(nfi_evals, "arkode_nfi_evals",
                     "No. of calls to fi (implicit portion of the right-hand-side "
                     "function) function");
  add_int_diagnostic(nniters, "arkode_nniters", "No. of nonlinear solver iterations");
  add_int_diagnostic(npevals, "arkode_npevals", "No. of preconditioner evaluations");
  add_int_diagnostic(nliters, "arkode_nliters", "No. of linear iterations");
}

MriSolver::~MriSolver() {
  N_VDestroy(uvec);
  MRIStepFree(&arkode_mem);
  SUNLinSolFree(sun_solver);
  SUNNonlinSolFree(nonlinear_solver);
}

/**************************************************************************
 * Initialise
 **************************************************************************/

int MriSolver::init() {
  TRACE("Initialising MRI solver");

  Solver::init();

  output.write("Initialising SUNDIALS' MRI solver\n");

  // Calculate number of variables (in generic_solver)
  const int local_N = getLocalN();

  // Get total problem size
  int neq;
  if (bout::globals::mpi->MPI_Allreduce(&local_N, &neq, 1, MPI_INT, MPI_SUM,
                                        BoutComm::get())) {
    throw BoutException("Allreduce localN -> GlobalN failed!\n");
  }

  output.write("\t3d fields = {:d}, 2d fields = {:d} neq={:d}, local_N={:d}\n", n3Dvars(),
               n2Dvars(), neq, local_N);

  // Allocate memory
  uvec = callWithSUNContext(N_VNew_Parallel, suncontext, BoutComm::get(), local_N, neq);
  if (uvec == nullptr) {
    throw BoutException("SUNDIALS memory allocation failed\n");
  }

  // Put the variables into uvec
  save_vars(N_VGetArrayPointer(uvec));

  arkode_mem = callWithSUNContext(MRIStepCreate, suncontext, arkode_rhs_explicit, arkode_rhs_implicit,
                                  simtime, uvec, nullptr);
  if (arkode_mem == nullptr) {
    throw BoutException("MRIStepCreate failed\n");
  }

  // For callbacks, need pointer to solver object
  if (MRIStepSetUserData(arkode_mem, this) != ARK_SUCCESS) {
    throw BoutException("MRIStepSetUserData failed\n");
  }

  if (MRIStepSetLinear(arkode_mem, set_linear) != ARK_SUCCESS) {
    throw BoutException("MRIStepSetLinear failed\n");
  }

  if (MRIStepSetFixedStep(arkode_mem, timestep) != ARK_SUCCESS) {
    throw BoutException("MRIStepSetFixedStep failed\n");
  }

  if (MRIStepSetOrder(arkode_mem, order) != ARK_SUCCESS) {
    throw BoutException("MRIStepSetOrder failed\n");
  }

  if (MRIStepSetMaxNumSteps(arkode_mem, mxsteps) != ARK_SUCCESS) {
    throw BoutException("MRIStepSetMaxNumSteps failed\n");
  }

  if (fixed_point) {
    output.write("\tUsing accelerated fixed point solver\n");
    nonlinear_solver = callWithSUNContext(SUNNonlinSol_FixedPoint, suncontext, uvec, 3);
    if (nonlinear_solver == nullptr) {
      throw BoutException("Creating SUNDIALS fixed point nonlinear solver failed\n");
    }
    if (MRIStepSetNonlinearSolver(arkode_mem, nonlinear_solver) != ARK_SUCCESS) {
      throw BoutException("MRIStepSetNonlinearSolver failed\n");
    }
  } else {
    output.write("\tUsing Newton iteration\n");

    const auto prectype =
        use_precon ? (rightprec ? SUN_PREC_RIGHT : SUN_PREC_LEFT) : SUN_PREC_NONE;
    sun_solver = callWithSUNContext(SUNLinSol_SPGMR, suncontext, uvec, prectype, maxl);
    if (sun_solver == nullptr) {
      throw BoutException("Creating SUNDIALS linear solver failed\n");
    }
    if (MRIStepSetLinearSolver(arkode_mem, sun_solver, nullptr) != ARKLS_SUCCESS) {
      throw BoutException("MRIStepSetLinearSolver failed\n");
    }

    /// Set Preconditioner
    if (use_precon) {
      if (hasPreconditioner()) {
        output.write("\tUsing user-supplied preconditioner\n");

        if (MRIStepSetPreconditioner(arkode_mem, nullptr, arkode_pre) != ARKLS_SUCCESS) {
          throw BoutException("MRIStepSetPreconditioner failed\n");
        }
      } else {
        output.write("\tUsing BBD preconditioner\n");

        /// Get options
        // Compute band_width_default from actually added fields, to allow for multiple
        // Mesh objects
        //
        // Previous implementation was equivalent to:
        //   int MXSUB = mesh->xend - mesh->xstart + 1;
        //   int band_width_default = n3Dvars()*(MXSUB+2);
        const int band_width_default = std::accumulate(
            begin(f3d), end(f3d), 0, [](int a, const VarStr<Field3D>& fvar) {
              Mesh* localmesh = fvar.var->getMesh();
              return a + localmesh->xend - localmesh->xstart + 3;
            });

        const auto mudq = (*options)["mudq"]
                              .doc("Upper half-bandwidth to be used in the difference "
                                   "quotient Jacobian approximation")
                              .withDefault(band_width_default);
        const auto mldq = (*options)["mldq"]
                              .doc("Lower half-bandwidth to be used in the difference "
                                   "quotient Jacobian approximation")
                              .withDefault(band_width_default);
        const auto mukeep = (*options)["mukeep"]
                                .doc("Upper half-bandwidth of the retained banded "
                                     "approximate Jacobian block")
                                .withDefault(n3Dvars() + n2Dvars());
        const auto mlkeep = (*options)["mlkeep"]
                                .doc("Lower half-bandwidth of the retained banded "
                                     "approximate Jacobian block")
                                .withDefault(n3Dvars() + n2Dvars());

        if (ARKBBDPrecInit(arkode_mem, local_N, mudq, mldq, mukeep, mlkeep, 0,
                           arkode_bbd_rhs, nullptr)
            != ARKLS_SUCCESS) {
          throw BoutException("ARKBBDPrecInit failed\n");
        }
      }
    } else {
      // Not using preconditioning
      output.write("\tNo preconditioning\n");
    }
  }

  /// Set Jacobian-vector multiplication function

  if (use_jacobian and hasJacobian()) {
    output.write("\tUsing user-supplied Jacobian function\n");

    if (MRIStepSetJacTimes(arkode_mem, nullptr, arkode_jac) != ARKLS_SUCCESS) {
      throw BoutException("MRIStepSetJacTimes failed\n");
    }
  } else {
    output.write("\tUsing difference quotient approximation for Jacobian\n");
  }

  return 0;
}

/**************************************************************************
 * Run - Advance time
 **************************************************************************/

int MriSolver::run() {
  TRACE("MriSolver::run()");

  if (!initialised) {
    throw BoutException("MriSolver not initialised\n");
  }

  for (int i = 0; i < getNumberOutputSteps(); i++) {

    /// Run the solver for one output timestep
    simtime = run(simtime + getOutputTimestep());

    /// Check if the run succeeded
    if (simtime < 0.0) {
      // Step failed
      output.write("Timestep failed. Aborting\n");

      throw BoutException("ARKode timestep failed\n");
    }

    // Get additional diagnostics
    long int temp_long_int, temp_long_int2;
    MRIStepGetNumSteps(arkode_mem, &temp_long_int);
    nsteps = int(temp_long_int);
    MRIStepGetNumRhsEvals(arkode_mem, &temp_long_int, &temp_long_int2);
    nfe_evals = int(temp_long_int);
    nfi_evals = int(temp_long_int2);
    MRIStepGetNumNonlinSolvIters(arkode_mem, &temp_long_int);
    nniters = int(temp_long_int);
    MRIStepGetNumPrecEvals(arkode_mem, &temp_long_int);
    npevals = int(temp_long_int);
    MRIStepGetNumLinIters(arkode_mem, &temp_long_int);
    nliters = int(temp_long_int);

    if (diagnose) {
      output.write("\nARKODE: nsteps {:d}, nfe_evals {:d}, nfi_evals {:d}, nniters {:d}, "
                   "npevals {:d}, nliters {:d}\n",
                   nsteps, nfe_evals, nfi_evals, nniters, npevals, nliters);

      output.write("    -> Newton iterations per step: {:e}\n",
                   static_cast<BoutReal>(nniters) / static_cast<BoutReal>(nsteps));
      output.write("    -> Linear iterations per Newton iteration: {:e}\n",
                   static_cast<BoutReal>(nliters) / static_cast<BoutReal>(nniters));
      output.write("    -> Preconditioner evaluations per Newton: {:e}\n",
                   static_cast<BoutReal>(npevals) / static_cast<BoutReal>(nniters));
    }

    if (call_monitors(simtime, i, getNumberOutputSteps())) {
      // User signalled to quit
      break;
    }
  }

  return 0;
}

BoutReal MriSolver::run(BoutReal tout) {
  TRACE("Running solver: solver::run({:e})", tout);

  bout::globals::mpi->MPI_Barrier(BoutComm::get());

  pre_Wtime = 0.0;
  pre_ncalls = 0;

  int flag;
  if (!monitor_timestep) {
    // Run in normal mode
    flag = MRIStepEvolve(arkode_mem, tout, uvec, &simtime, ARK_NORMAL);
  } else {
    // Run in single step mode, to call timestep monitors
    BoutReal internal_time;
    MRIStepGetCurrentTime(arkode_mem, &internal_time);
    while (internal_time < tout) {
      // Run another step
      const BoutReal last_time = internal_time;
      flag = MRIStepEvolve(arkode_mem, tout, uvec, &internal_time, ARK_ONE_STEP);

      if (flag != ARK_SUCCESS) {
        output_error.write("ERROR ARKODE solve failed at t = {:e}, flag = {:d}\n",
                           internal_time, flag);
        return -1.0;
      }

      // Call timestep monitor
      call_timestep_monitors(internal_time, internal_time - last_time);
    }
    // Get output at the desired time
    flag = MRIStepGetDky(arkode_mem, tout, 0, uvec);
    simtime = tout;
  }

  // Copy variables
  load_vars(N_VGetArrayPointer(uvec));
  // Call rhs function to get extra variables at this time
  run_rhs(simtime);
  // run_diffusive(simtime);
  if (flag != ARK_SUCCESS) {
    output_error.write("ERROR ARKODE solve failed at t = {:e}, flag = {:d}\n", simtime,
                       flag);
    return -1.0;
  }

  return simtime;
}

/**************************************************************************
 * Explicit RHS function du = F_E(t, u)
 **************************************************************************/

void MriSolver::rhs_e(BoutReal t, BoutReal* udata, BoutReal* dudata) {
  TRACE("Running RHS: MriSolver::rhs_e({:e})", t);

  // Load state from udata
  load_vars(udata);
  MRIStepGetLastStep(arkode_mem, &hcur);

  // Call RHS function
  run_convective(t);

  // Save derivatives to dudata
  save_derivs(dudata);
}

/**************************************************************************
 *   Implicit RHS function du = F_I(t, u)
 **************************************************************************/

void MriSolver::rhs_i(BoutReal t, BoutReal* udata, BoutReal* dudata) {
  TRACE("Running RHS: MriSolver::rhs_i({:e})", t);

  load_vars(udata);
  MRIStepGetLastStep(arkode_mem, &hcur);
  // Call Implicit RHS function
  run_diffusive(t);
  save_derivs(dudata);
}

/**************************************************************************
 * Preconditioner function
 **************************************************************************/

void MriSolver::pre(BoutReal t, BoutReal gamma, BoutReal delta, BoutReal* udata,
                       BoutReal* rvec, BoutReal* zvec) {
  TRACE("Running preconditioner: MriSolver::pre({:e})", t);

  const BoutReal tstart = bout::globals::mpi->MPI_Wtime();

  if (!hasPreconditioner()) {
    // Identity (but should never happen)
    const int N = N_VGetLocalLength_Parallel(uvec);
    std::copy(rvec, rvec + N, zvec);
    return;
  }

  // Load state from udata (as with res function)
  load_vars(udata);

  // Load vector to be inverted into F_vars
  load_derivs(rvec);

  runPreconditioner(t, gamma, delta);

  // Save the solution from F_vars
  save_derivs(zvec);

  pre_Wtime += bout::globals::mpi->MPI_Wtime() - tstart;
  pre_ncalls++;
}

/**************************************************************************
 * Jacobian-vector multiplication function
 **************************************************************************/

void MriSolver::jac(BoutReal t, BoutReal* ydata, BoutReal* vdata, BoutReal* Jvdata) {
  TRACE("Running Jacobian: MriSolver::jac({:e})", t);

  if (not hasJacobian()) {
    throw BoutException("No jacobian function supplied!\n");
  }

  // Load state from ydate
  load_vars(ydata);

  // Load vector to be multiplied into F_vars
  load_derivs(vdata);

  // Call function
  runJacobian(t);

  // Save Jv from vars
  save_derivs(Jvdata);
}

/**************************************************************************
 * ARKODE explicit RHS functions
 **************************************************************************/

static int arkode_rhs_explicit(BoutReal t, N_Vector u, N_Vector du, void* user_data) {

  BoutReal* udata = N_VGetArrayPointer(u);
  BoutReal* dudata = N_VGetArrayPointer(du);

  auto* s = static_cast<MriSolver*>(user_data);

  // Calculate RHS function
  try {
    s->rhs_e(t, udata, dudata);
  } catch (BoutRhsFail& error) {
    return 1;
  }
  return 0;
}

static int arkode_rhs_implicit(BoutReal t, N_Vector u, N_Vector du, void* user_data) {

  BoutReal* udata = N_VGetArrayPointer(u);
  BoutReal* dudata = N_VGetArrayPointer(du);

  auto* s = static_cast<MriSolver*>(user_data);

  // Calculate RHS function
  try {
    s->rhs_i(t, udata, dudata);
  } catch (BoutRhsFail& error) {
    return 1;
  }
  return 0;
}

/// RHS function for BBD preconditioner
static int arkode_bbd_rhs(sunindextype UNUSED(Nlocal), BoutReal t, N_Vector u,
                          N_Vector du, void* user_data) {
  return arkode_rhs_implicit(t, u, du, user_data);
}

/// Preconditioner function
static int arkode_pre(BoutReal t, N_Vector yy, N_Vector UNUSED(yp), N_Vector rvec,
                      N_Vector zvec, BoutReal gamma, BoutReal delta, int UNUSED(lr),
                      void* user_data) {
  BoutReal* udata = N_VGetArrayPointer(yy);
  BoutReal* rdata = N_VGetArrayPointer(rvec);
  BoutReal* zdata = N_VGetArrayPointer(zvec);

  auto* s = static_cast<MriSolver*>(user_data);

  // Calculate residuals
  s->pre(t, gamma, delta, udata, rdata, zdata);

  return 0;
}

/// Jacobian-vector multiplication function
static int arkode_jac(N_Vector v, N_Vector Jv, BoutReal t, N_Vector y,
                      N_Vector UNUSED(fy), void* user_data, N_Vector UNUSED(tmp)) {
  BoutReal* ydata = N_VGetArrayPointer(y);   ///< System state
  BoutReal* vdata = N_VGetArrayPointer(v);   ///< Input vector
  BoutReal* Jvdata = N_VGetArrayPointer(Jv); ///< Jacobian*vector output

  auto* s = static_cast<MriSolver*>(user_data);

  s->jac(t, ydata, vdata, Jvdata);

  return 0;
}

#endif
