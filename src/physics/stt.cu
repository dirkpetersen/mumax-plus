#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "stt.hpp"
#include "world.hpp"

bool spinTransferTorqueAssuredZero(const Ferromagnet* magnet) {
  return magnet->msat.assuredZero() || magnet->jcur.assuredZero() ||
         magnet->pol.assuredZero();
}

bool ZhangLiSTTAssuredZero(const Ferromagnet* magnet) {
  return spinTransferTorqueAssuredZero(magnet) || (!magnet->Lambda.assuredZero());
}

bool SlonczewskiSTTAssuredZero(const Ferromagnet* magnet) {
  return ZhangLiSTTAssuredZero(magnet) || magnet->Lambda.assuredZero();
}
__global__ void k_ZhangLi(CuField torque,
                                     const CuField mField,
                                     const CuParameter msatParam,
                                     const CuParameter polParam,
                                     const CuParameter xiParam,
                                     const CuParameter alphaParam,
                                     const CuVectorParameter jcurParam) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  const Grid grid = torque.system.grid;
  const real3 cellsize = torque.system.cellsize;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.vectorAt(idx);

  const real3 j = jcurParam.vectorAt(idx);
  const real msat = msatParam.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real xi = xiParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);

  if (msat == 0 || pol == 0 || j == real3{0, 0, 0}) {
    torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const int3 coo = grid.index2coord(idx);

  const real3 u = MUB * pol / (QE * msat * (1 + xi * xi)) * j;

  real3 hspin{0, 0, 0};

  // x derivative
  for (int sign : {-1, 1}) {  // left and right neighbor
    const int3 coo_ = coo + int3{sign, 0, 0};
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.x * m_ / (2 * cellsize.x);  // central finite difference
    }
  }
  // y derivative
  for (int sign : {-1, 1}) {
    const int3 coo_ = coo + int3{0, sign, 0};
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.y * m_ / (2 * cellsize.y);
    }
  }
  // z derivative
  for (int sign : {-1, 1}) {
    const int3 coo_ = coo + int3{0, 0, sign};
    if (grid.cellInGrid(coo_) && msatParam.valueAt(coo_) != 0) {
      real3 m_ = mField.vectorAt(coo_);
      hspin += sign * u.z * m_ / (2 * cellsize.z);  // central finite difference
    }
  }

  const real3 mxh = cross(m, hspin);
  const real3 mxmxh = cross(m, mxh);
  const real3 t = (-1 / (1 + alpha * alpha)) *
                  ((1 + xi * alpha) * mxmxh + (xi - alpha) * mxh); // In 1/s
  
  torque.setVectorInCell(idx, t);
}

__global__ void k_Slonczewski(CuField torque,
                                     const CuField mField,
                                     const CuParameter msatParam,
                                     const CuParameter polParam,
                                     const CuParameter lambdaParam,
                                     const CuParameter alphaParam,
                                     const CuVectorParameter jcurParam,
                                     const CuParameter eps_prime,
                                     const CuVectorParameter FixedLayer,
                                     const CuParameter FreeLayerThickness) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // When outside the geometry, set to zero and return early
  if (!torque.cellInGeometry(idx)) {
    if (torque.cellInGrid(idx))
      torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  real3 m = mField.vectorAt(idx);
 
  const real3 jj = jcurParam.vectorAt(idx);
  const real jz = jj.z;  // consistent with mumax3 TODO: make more general?

  const real msat = msatParam.valueAt(idx);
  const real pol = polParam.valueAt(idx);
  const real alpha = alphaParam.valueAt(idx);

  const real3 p = FixedLayer.vectorAt(idx);
  const real lambda = lambdaParam.valueAt(idx);
  const real eps_p = eps_prime.valueAt(idx);
  const real d = FreeLayerThickness.valueAt(idx);

  if (msat == 0 || pol == 0 || jj == real3{0, 0, 0}) {
    torque.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

  const real B = (HBAR / QE) * jz / (msat * d); // in Tesla
  const real lambda2 = lambda * lambda;
  const real eps = pol * lambda2 / ((lambda2 + 1) + (lambda2 - 1) * dot(m, p));

 
  const real3 pxm = cross(p, m);
  const real3 mxpxm = cross(m, pxm);
  const real3 t = ((eps  + eps_p * alpha) * mxpxm 
                   + (eps_p - eps  * alpha) * pxm) * (B / (1 + alpha * alpha)) * GAMMALL;

  torque.setVectorInCell(idx, t);
}

Field evalSpinTransferTorque(const Ferromagnet* magnet) {
  Field torque(magnet->system(), 3);

  if (spinTransferTorqueAssuredZero(magnet)) {
    torque.makeZero();
    return torque;
  }

  int ncells = magnet->grid().ncells();
  auto m = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto pol = magnet->pol.cu();
  auto xi = magnet->xi.cu();
  auto alpha = magnet->alpha.cu();
  auto jcur = magnet->jcur.cu();
  auto lambda = magnet->Lambda.cu();
  auto eps_prime = magnet->eps_prime.cu();
  auto FixedLayer = magnet->FixedLayer.cu();
  auto FreeLayerThickness = magnet->FreeLayerThickness.cu();

  auto cellsize = magnet->world()->cellsize();

  if (magnet->Lambda.assuredZero() && magnet->eps_prime.assuredZero())
    cudaLaunch(ncells, k_ZhangLi, torque.cu(), m, msat, pol, xi, alpha, jcur);
  else
    cudaLaunch(ncells, k_Slonczewski, torque.cu(), m, msat, pol, lambda, alpha,
             jcur, eps_prime, FixedLayer, FreeLayerThickness);
  return torque;
}

FM_FieldQuantity spinTransferTorqueQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalSpinTransferTorque, 3, "spintransfer_torque", "1/s");
}
