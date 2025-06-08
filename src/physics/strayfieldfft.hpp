#pragma once

#include <cufft.h>
#include <cufftXt.h>

#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "strayfield.hpp"
#include "strayfieldkernel.hpp"
#include "../cudautil/deviceutils.hpp"

class Field;
class System;
class Magnet;

/** A StraFieldFFTExecutor uses the FFT method to compute stray fields. */
class StrayFieldFFTExecutor : public StrayFieldExecutor {
 public:
  /**
   * Construct a StrayFieldFFTExecutor.
   *
   * @param magnet the source of the stray field
   * @param system the system in which to compute the stray field
   * @param use_multi_gpu whether to use multi-GPU mode if available (default: true)
   */
  StrayFieldFFTExecutor(const Magnet* magnet,
                        std::shared_ptr<const System> system,
                        bool use_multi_gpu = true);

  /** Destruct the executor. */
  ~StrayFieldFFTExecutor();

  /** Compute and return the stray field. */
  Field exec() const;

  /** Return the computation method which is METHOD_FFT. */
  Method method() const { return StrayFieldExecutor::METHOD_FFT; }

 private:
  StrayFieldKernel kernel_;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
  bool use_multi_gpu_;
  int gpu_count_;
  std::vector<int> gpu_ids_;
  bool is_multi_gpu_enabled_;
};
