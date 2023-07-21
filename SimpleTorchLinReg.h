#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H
#include <torch/torch.h>

class LinearRegressionModel : public torch::nn::Module {
public:
	LinearRegressionModel(int inputSize, int outputSize);

	torch::Tensor forward(torch::Tensor x);
	
	void train(torch::Tensor x_train, torch::Tensor y_train,
		int num_epochs, double learning_rate);

	torch::Tensor predict(torch::Tensor x_test);

private :
	torch::nn::Linear linear ;
};

#endif // !LINEAR_REGRESSION_H

