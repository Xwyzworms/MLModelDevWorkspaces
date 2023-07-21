#include "SimpleTorchLinReg.h"

LinearRegressionModel::LinearRegressionModel(int inputSize, int outputSize) : linear(register_module("linear", torch::nn::Linear(inputSize, outputSize))) {};

torch::Tensor LinearRegressionModel::forward(torch::Tensor x)
{
	return linear->forward(x);
};

void LinearRegressionModel::train(torch::Tensor x_train, torch::Tensor y_train,
								 int num_epochs, double learning_rate)  
{
	torch::optim::SGD optimizer(parameters(), torch::optim::SGDOptions(learning_rate));
	torch::nn::MSELoss loss_fn;

	for (int epoch = 1; epoch < num_epochs; epoch++) {

		torch::Tensor predictions = forward(x_train);

		torch::Tensor loss = loss_fn(predictions, y_train);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
		
		if ((epoch + 1) % 50 == 0) {
		  std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Loss: " << loss.item<float>() << std::endl;
		}
		
		this->eval();

		if ((epoch + 1) % validation_interval == 0) {
		
		}

			
		
	}

}


torch::Tensor LinearRegressionModel::predict(torch::Tensor x_test) 
{	
	return forward(x_test);
}

	
