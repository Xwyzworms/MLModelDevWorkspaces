#include "LinearRegression.h"

LinearRegression::LinearRegression() {

	weights = register_parameter("weights", torch::randn(1), true);
	bias = register_parameter("bias", torch::randn(1), true);

}

torch::Tensor LinearRegression::forward(torch::Tensor x) {
	return weights * x + bias;

}