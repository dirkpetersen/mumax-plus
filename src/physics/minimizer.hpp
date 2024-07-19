#pragma once

#include <deque>

#include "ferromagnetquantity.hpp"

class Antiferromagnet;
class Ferromagnet;
class MumaxWorld;

// Minimize follows the steepest descent method as per Exl et al., JAP 115,
// 17D118 (2014).

class Minimizer {
 public:
  Minimizer(const Ferromagnet*, real stopMaxMagDiff, int nMagDiffSamples);
  Minimizer(const Antiferromagnet*, real stopMaxMagDiff, int nMagDiffSamples);
  Minimizer(const MumaxWorld*, real stopMaxMagDiff, int nMagDiffSamples);

  void exec();

 private:
  void step();
  std::vector<const Ferromagnet*> magnets_;
  std::vector<real> stepsizes_;
  int nsteps_;

  std::vector<FM_FieldQuantity> torques_;
  std::vector<Field> t0, t1, m0, m1;

  size_t N; // Number of magnets in world.
  real stopMaxMagDiff_;
  int nMagDiffSamples_;
  bool converged() const;
  void addMagDiff(real);
  std::deque<real> lastMagDiffs_;
};
