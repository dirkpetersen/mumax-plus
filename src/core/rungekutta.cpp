#include "rungekutta.hpp"

#include <cmath>
#include <vector>

#include "butchertableau.hpp"
#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "timesolver.hpp"
#include "variable.hpp"

RungeKuttaStepper::RungeKuttaStepper(TimeSolver* solver, RKmethod method)
    : butcher_(ButcherTableau::get(method)) {
  setParentTimeSolver(solver);
}

int RungeKuttaStepper::nStages() const {
  return butcher_.nodes.size();
}

void RungeKuttaStepper::step() {
  // If there are no equations to solve, advance time and return early
  if (solver_->equations().empty()) {
    solver_->setTime(solver_->time() + solver_->timestep());
    return;
  }
  // construct a Runge Kutta stage executor for every equation
  std::vector<RungeKuttaStepper::RungeKuttaStageExecutor> equations;
  for (auto eq : solver_->equations())
    equations.emplace_back(eq, *this);

  real t0 = solver_->time();

  bool success = false;
  while (!success) {
    real dt = solver_->timestep();

    // apply the stages
    for (int stage = 0; stage < nStages(); stage++) {
      solver_->setTime(t0 + dt * butcher_.nodes[stage]);
      for (auto& eq : equations)
        eq.setStageX(stage);
      for (auto& eq : equations)
        eq.setStageK(stage);
    }

    // make the actual step
    solver_->setTime(t0 + dt);
    for (auto& eq : equations)
      eq.setFinalX();

    // nothing more to do if time step is fixed
    if (!solver_->hasAdaptiveTimeStep())
      break;

    // loop over equations and get the largest error
    real error = 0.0;
    for (auto& eq : equations)
      if (real e = eq.getError(); e > error)
        error = e;

    success = error < solver_->maxError();

    // update the timestep
    real corrFactor;
    if (success) {
      corrFactor = std::pow(solver_->maxError() / error, 1. / (butcher_.order2 + 1));
    } else {
      corrFactor = std::pow(solver_->maxError() / error, 1. / (butcher_.order1 + 1));
    }
    solver_->adaptTimeStep(corrFactor);

    // undo step if not successful
    if (!success) {
      for (auto& eq : equations)
        eq.resetX();
      solver_->setTime(t0);
    }
  }
}

RungeKuttaStepper::RungeKuttaStageExecutor::RungeKuttaStageExecutor(
    DynamicEquation eq,
    const RungeKuttaStepper& stepper)
    : x0(eq.x->eval()),
      butcher(stepper.butcher_),
      stepper(stepper),
      x(*eq.x),
      k(stepper.nStages()),
      eq_(eq) {
  // Noise term evaluated only here, it remains constant throughout all stages
  if (eq_.noiseTerm && !eq_.noiseTerm->assuredZero())
    noise = eq_.noiseTerm->eval();
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setStageK(int stage) {
  k[stage] = eq_.rhs->eval();

  auto dt = stepper.solver_->timestep();
  if (noise)
    addTo(k[stage], 1 / sqrt(dt), noise.value());
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setStageX(int stage) {
  if (stage == 0)
    return;

  Field xstage = x0;
  auto dt = stepper.solver_->timestep();
  for (int i = 0; i < stage; i++)
    addTo(xstage, dt * butcher.rkMatrix[stage][i], k[i]);

  // If x is a normalized variable, then xstage will be normalized during the
  // assignment. For this reason, we use a temporary xstage field instead of
  // working directly on x
  x = xstage;
}

void RungeKuttaStepper::RungeKuttaStageExecutor::setFinalX() {
  Field xstage = x0;
  auto dt = stepper.solver_->timestep();

  for (int i = 0; i < stepper.nStages(); i++)
    addTo(xstage, dt * butcher.weights1[i], k[i]);

  // If x is a normalized variable, then xstage will be normalized during the
  // assignment. For this reason, we use a temporary xstage field instead of
  // working directly on x
  x = xstage;
}

void RungeKuttaStepper::RungeKuttaStageExecutor::resetX() {
  x = x0;
}

real RungeKuttaStepper::RungeKuttaStageExecutor::getError() const {
  Field err(x.system(), x.ncomp());

  err.makeZero();

  auto dt = stepper.solver_->timestep();

  for (int i = 0; i < stepper.nStages(); i++)
    addTo(err, dt * (butcher.weights1[i] - butcher.weights2[i]), k[i]);

  return maxVecNorm(err);
}
