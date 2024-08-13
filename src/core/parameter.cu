#include <memory>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value)
    : system_(system), staticField_(nullptr), uniformValue_(value) {}

Parameter::~Parameter() {
  if (staticField_)
    delete staticField_;
}

void Parameter::set(real value) {
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

void Parameter::set(const Field& values) {
  staticField_ = new Field(values);
}

bool Parameter::isUniform() const {
  return !staticField_ && DynamicParameter<real>::isUniform();
}

bool Parameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

std::shared_ptr<const System> Parameter::system() const {
  return system_;
}

Field Parameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

CuParameter Parameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return CuParameter(this);
}

VectorParameter::VectorParameter(std::shared_ptr<const System> system,
                                 real3 value)
    : system_(system), staticField_(nullptr), uniformValue_(value) {}

VectorParameter::~VectorParameter() {
  if (staticField_)
    delete staticField_;
}

void VectorParameter::set(real3 value) {
  uniformValue_ = value;
  if (staticField_)
    delete staticField_;
}

void VectorParameter::set(const Field& values) {
  staticField_ = new Field(values);
}

bool VectorParameter::isUniform() const {
  return !staticField_ && DynamicParameter<real3>::isUniform();
}

bool VectorParameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

std::shared_ptr<const System> VectorParameter::system() const {
  return system_;
}

Field VectorParameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

real Parameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument("Cannot get uniform value of non-uniform Parameter.");
  }
  return uniformValue_;
}

CuVectorParameter VectorParameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return CuVectorParameter(this);
}
