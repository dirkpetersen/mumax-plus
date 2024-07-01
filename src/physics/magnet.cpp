#include "magnet.hpp"

#include <curand.h>

#include <memory>
#include <random>
#include <math.h>
#include <cfloat>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "mumaxworld.hpp"
#include "strayfield.hpp"

Magnet::Magnet(std::shared_ptr<System> system_ptr,
               std::string name)
    : system_(system_ptr),
      name_(name) {}

Magnet::Magnet(MumaxWorld* world,
               Grid grid,
               std::string name,
               GpuBuffer<bool> geometry)
    : Magnet(std::make_shared<System>(world, grid, geometry), name) {}

std::string Magnet::name() const {
  return name_;
}

std::shared_ptr<const System> Magnet::system() const {
  return system_;
}

const World* Magnet::world() const {
  return system()->world();
}

const MumaxWorld* Magnet::mumaxWorld() const {
  // static_cast: no check needed, world() is always a MumaxWorld
  return static_cast<const MumaxWorld*>(this->world());
}

Grid Magnet::grid() const {
  return system()->grid();
}

real3 Magnet::cellsize() const {
  return world()->cellsize();
}

const GpuBuffer<bool>& Magnet::getGeometry() const {
  return system_->geometry();
}

const StrayField* Magnet::getStrayField(const Magnet* magnet) const {
  auto it = strayFields_.find(magnet);
  if (it == strayFields_.end())
    return nullptr;
  return it->second;
}

std::vector<const StrayField*> Magnet::getStrayFields() const {
  std::vector<const StrayField*> strayFields;
  strayFields.reserve(strayFields_.size());
  for (const auto& entry : strayFields_) {
    strayFields.push_back(entry.second);
  }
  return strayFields;
}

void Magnet::addStrayField(const Magnet* magnet,
                                StrayFieldExecutor::Method method) {
  if (world() != magnet->world()) {
    throw std::runtime_error(
        "Can not define the field of the magnet on this magnet because it is "
        "not in the same world.");
  }

  auto it = strayFields_.find(magnet);
  if (it != strayFields_.end()) {
    // StrayField is already registered, just need to update the method
    it->second->setMethod(method);
    return;
  }
  // Stray field of magnet (parameter) on this magnet (the object)
  strayFields_[magnet] = new StrayField(magnet, system(), method);
}

void Magnet::removeStrayField(const Magnet* magnet) {
  auto it = strayFields_.find(magnet);
  if (it != strayFields_.end()) {
    delete it->second;
    strayFields_.erase(it);
  }
}