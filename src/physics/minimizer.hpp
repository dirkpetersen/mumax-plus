#pragma once

#include <deque>

#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;

// Minimize follows the steepest descent method as per Exl et al., JAP 115,
// 17D118 (2014).

class Minimizer {
 public:
  Minimizer(const Ferromagnet*, real stopMaxMagDiff, int nMagDiffSamples);
  Minimizer(const Antiferromagnet*, real stopMaxMagDiff, int nMagDiffSamples);

  void exec();

 private:
  void step();
  std::vector<const Ferromagnet*> magnet_;
  std::vector<real> stepsize_;
  int nsteps_;

  std::vector<FM_FieldQuantity> torque_;
  std::vector<Field> t0, t1, m0, m1;

  real stopMaxMagDiff_;
  int nMagDiffSamples_;
  bool converged() const;
  void addMagDiff(real);
  std::deque<real> lastMagDiffs_;
};
