#include <torch/torch.h>
#include <iostream>
#include "linreg.h"

void mainLinregDriver() {

	torch::manual_seed(2);
	int input_size = 1;
	int output_size = 1;
	int epochs = 1000;
	double learningRate = 0.01;

	torch::Tensor x_train = torch::randn({ 100, input_size });
	torch::Tensor y_train = 3 * x_train + torch::randn({ 100 , input_size });
	torch::Tensor x_val = torch::randn({ 100 , input_size });
	torch::Tensor y_val = 3 * x_val + torch::randn({ 100, input_size });

	LinearRegression model = LinearRegression();

	if (torch::cuda::is_available()) {
		std::cout << "CUDA is available! Training on GPU" << std::endl;

		model.trainData(
			x_train, y_train,
			x_val, y_val
		);

		model.getModelPerformance();

		//  model.savemodel();
		{
			// Loading Scope
			torch::NoGradGuard dr;
			torch::InferenceMode infrencemodel;
			torch::load(model.parameters(), "01_pytorch_workflow_model_0.pth");
			model.predict(x_val);
		}

	}
	else
	{
		std::cout << "CUDA is not available! Training on CPU" << std::endl;
		// std::cout << tensor << std::endl;	  
	}
}

void mainClassification() 
{
	
}
int main() {
	std::cin.get();
}