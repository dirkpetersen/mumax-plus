#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "dmi.hpp"
#include "dmitensor.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "world.hpp"

bool dmiAssuredZero(const Ferromagnet* magnet) {
  return (magnet->dmiTensor.assuredZero() || magnet->msat.assuredZero());
}

__global__ void k_dmiField(CuField hField,
                           const CuField mField,
                           const CuDmiTensor dmiTensor,
                           const CuParameter msat,
                           const Grid mastergrid,
                           const CuParameter aex,
                           bool openBC) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

	real3 m = mField.vectorAt(idx);
  const int3 coo = system.grid.index2coord(idx);

  // Accumulate DMI field of cell at idx in h. Divide by msat at the end.
  real3 h{0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx;
    if (!system.inGeometry(neighbor_coo)) { neighbor_idx = idx; }
    else { neighbor_idx = system.grid.coord2index(neighbor_coo); }

    // If there is no FM-exchange at the boundary, open BC are assumed
    real a = aex.valueAt(idx);
    openBC = (a == 0) ? true : openBC;

    // If we assume open boundary conditions and if there is no neighbor, 
    // then simply continue without adding to the effective field.    
    if (openBC && (!system.inGeometry(neighbor_coo)
               || msat.valueAt(neighbor_idx) == 0))
      continue;
    
    // Get the dmi strengths between the center cell and the neighbor, which are
    // the harmonic means of the dmi strengths of both cells.
    real Dxz, Dxy, Dyz, Dzx, Dyx, Dzy;
    
    if (relative_coo.x) {  // derivative along x
      Dxz = dmiTensor.xxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.xxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.xyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.y) {  // derivative along y
      Dxz = dmiTensor.yxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.yxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.yyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.z) {  // derivative along z
      Dxz = dmiTensor.zxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.zxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.zyz.harmonicMean(idx, neighbor_idx);
    }

    Dzx = -Dxz;  // dmi tensor is assymetric
    Dyx = -Dxy;
    Dzy = -Dyz;

    // Distance between neighbors (the finite difference)
    real delta = relative_coo.x * system.cellsize.x +
                 relative_coo.y * system.cellsize.y +
                 relative_coo.z * system.cellsize.z;

    real3 m_;
    if (!system.inGeometry(neighbor_coo) && !openBC) { // Neumann BC
      int3 n = relative_coo * relative_coo;
      real3 Gamma = getGamma(dmiTensor, idx, n, m);
      m_ = m + (Gamma / (2*a)) * delta;
    }
    else {
      m_ = mField.vectorAt(neighbor_idx);
    }

    // Compute the effective field contribution of the DMI with the neighbor
    h.x += (Dxy * m_.y + Dxz * m_.z) / delta;
    h.y += (Dyx * m_.x + Dyz * m_.z) / delta;
    h.z += (Dzx * m_.x + Dzy * m_.y) / delta;

  }  // end loop over neighbors

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

__global__ void k_dmiField(CuField hField,
                           const CuField m1Field,
                           const CuField m2Field,
                           const CuDmiTensor dmiTensor,
                           const CuParameter msat,
                           Grid mastergrid,
                           const CuParameter aex,
                           const CuParameter afmex_nn) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto system = hField.system;
 
  if (!system.grid.cellInGrid(idx))
    return;

  // When outside the geometry or msat=0, set to zero and return early
  if (!system.inGeometry(idx) || (msat.valueAt(idx) == 0)) {
    hField.setVectorInCell(idx, real3{0, 0, 0});
    return;
  }

	real3 m = m1Field.vectorAt(idx);
  const int3 coo = system.grid.index2coord(idx);

  // Accumulate DMI field of cell at idx in h. Divide by msat at the end.
  real3 h{0, 0, 0};

// Loop over the 6 nearest neighbors using the neighbor's relative coordinate.
// Compute for each neighbor the DMI effective field term.
#pragma unroll
  for (int3 relative_coo : {int3{-1, 0, 0}, int3{1, 0, 0}, int3{0, -1, 0},
                            int3{0, 1, 0}, int3{0, 0, -1}, int3{0, 0, 1}}) {
    int3 neighbor_coo = mastergrid.wrap(coo + relative_coo);
    int neighbor_idx;
    if (!system.inGeometry(neighbor_coo)) { neighbor_idx = idx; }
    else { neighbor_idx = system.grid.coord2index(neighbor_coo); }
    
    // Get the dmi strengths between the center cell and the neighbor, which are
    // the harmonic means of the dmi strengths of both cells.
    real Dxz, Dxy, Dyz, Dzx, Dyx, Dzy;
    
    if (relative_coo.x) {  // derivative along x
      Dxz = dmiTensor.xxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.xxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.xyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.y) {  // derivative along y
      Dxz = dmiTensor.yxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.yxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.yyz.harmonicMean(idx, neighbor_idx);
    } else if (relative_coo.z) {  // derivative along z
      Dxz = dmiTensor.zxz.harmonicMean(idx, neighbor_idx);
      Dxy = dmiTensor.zxy.harmonicMean(idx, neighbor_idx);
      Dyz = dmiTensor.zyz.harmonicMean(idx, neighbor_idx);
    }

    Dzx = -Dxz;  // dmi tensor is assymetric
    Dyx = -Dxy;
    Dzy = -Dyz;

    // Distance between neighbors (the finite difference)
    real delta = relative_coo.x * system.cellsize.x +
                 relative_coo.y * system.cellsize.y +
                 relative_coo.z * system.cellsize.z;

    real3 m_;
    if (!system.inGeometry(neighbor_coo)) { // Neumann BC
      real a = aex.valueAt(idx);
      if(a == 0) // Assume open BC
          continue;
      int3 n = relative_coo * relative_coo;
      real3 Gamma1 = getGamma(dmiTensor, idx, n, m);

      real an = afmex_nn.valueAt(idx);
      real a2 = 2 * a;
      real an_a2 = an / a2;
      if (abs(an_a2) == 1)
        m_ = m + Gamma1 / (4*a) * delta;
      else {
        real3 Gamma2 = getGamma(dmiTensor, idx, n, m2Field.vectorAt(idx));
        m_ = m + delta / (a2 * (1 - an_a2*an_a2)) * (Gamma1 - an_a2 * Gamma2);
      }
    }
    else {
      m_ = m1Field.vectorAt(neighbor_idx);
    }

    // Compute the effective field contribution of the DMI with the neighbor
    h.x += (Dxy * m_.y + Dxz * m_.z) / delta;
    h.y += (Dyx * m_.x + Dyz * m_.z) / delta;
    h.z += (Dzx * m_.x + Dzy * m_.y) / delta;

  }  // end loop over neighbors

  /* TO DO: implement DMI-contribution to afm-exchange at a single site
  if (comp == 6) {
    // Compute effective field contribution of the DMI between sublattices at a single site
    real Dyz = dmiTensor.xyz.valueAt(idx);
    real Dxz = dmiTensor.yxz.valueAt(idx);
    real Dxy = dmiTensor.zxy.valueAt(idx);
    real Dzx = -Dxz;
    real Dyx = -Dxy;
    real Dzy = -Dyz;
    real l = latcon.valueAt(idx);
    h.x1 += (Dxy * m.y2 + Dxz * m.z2) / l;
    h.y1 += (Dyx * m.x2 + Dyz * m.z2) / l;
    h.z1 += (Dzx * m.x2 + Dzy * m.y2) / l;
    h.x2 += (Dxy * m.y1 + Dxz * m.z1) / l;
    h.y2 += (Dyx * m.x1 + Dyz * m.z1) / l;
    h.z2 += (Dzx * m.x1 + Dzy * m.y1) / l;
  }*/

  h /= msat.valueAt(idx);
  hField.setVectorInCell(idx, h);
}

Field evalDmiField(const Ferromagnet* magnet) {
  Field hField(magnet->system(), 3);
  if (dmiAssuredZero(magnet)) {
    hField.makeZero();
    return hField;
  }

  int ncells = hField.grid().ncells();
  auto grid = magnet->world()->mastergrid();
  auto mag = magnet->magnetization()->field().cu();
  auto msat = magnet->msat.cu();
  auto aex = magnet->aex.cu();
  auto dmiTensor = magnet->dmiTensor.cu();
  auto BC = magnet->enableOpenBC;
  
  if (!magnet->isSublattice() || magnet->enableOpenBC)
    cudaLaunch(ncells, k_dmiField, hField.cu(),
              mag, dmiTensor, msat, grid, aex, BC);
  else {
    // In case `magnet` is a sublattice, it's sister sublattice affects
    // the Neumann BC. There are no open boundaries when in this scope.
    auto mag2 = magnet->hostMagnet()->getOtherSublattice(magnet)->magnetization()->field().cu();
    auto afmex_nn = magnet->hostMagnet()->afmex_nn.cu();
    cudaLaunch(ncells, k_dmiField, hField.cu(), mag, mag2,
              dmiTensor, msat, grid, aex, afmex_nn);
  }
  return hField;
}

Field evalDmiEnergyDensity(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return Field(magnet->system(), 1, 0.0);

  return evalEnergyDensity(magnet, evalDmiField(magnet), 0.5);
}

real evalDmiEnergy(const Ferromagnet* magnet) {
  if (dmiAssuredZero(magnet))
    return 0;

  real edens = dmiEnergyDensityQuantity(magnet).average()[0];

  int ncells = magnet->grid().ncells();
  real cellVolume = magnet->world()->cellVolume();
  return ncells * edens * cellVolume;
}

FM_FieldQuantity dmiFieldQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDmiField, 3, "dmi_field", "T");
}

FM_FieldQuantity dmiEnergyDensityQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalDmiEnergyDensity, 1, "dmi_emergy_density",
                          "J/m3");
}

FM_ScalarQuantity dmiEnergyQuantity(const Ferromagnet* magnet) {
  return FM_ScalarQuantity(magnet, evalDmiEnergy, "dmi_energy", "J");
}
