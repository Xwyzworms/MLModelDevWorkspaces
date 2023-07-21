#include <torch/torch.h>
#include <iostream>
#include "SimpleTorchLinReg.h"
int main() {

	int input_size = 1;
	int output_size = 1;
	int epochs = 1000;
	double learningRate = 0.01;

	torch::Tensor x_train = torch::randn({100, input_size });
	torch::Tensor y_train = 3 * x_train;

	LinearRegressionModel model(input_size, output_size);

  if (torch::cuda::is_available()) {
	  std::cout << "CUDA is available! Training on GPU" << std::endl;
	  model.train(x_train, y_train, 1000, learningRate);
	  model.eval();
  }
  else
  {
	  std::cout << "CUDA is not available! Training on CPU" << std::endl;
	 // std::cout << tensor << std::endl;	  
  }

  std::cin.get();
}