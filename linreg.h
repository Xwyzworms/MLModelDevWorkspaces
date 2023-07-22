#ifndef LINREG
#define LINREG

#include<torch/torch.h>
#include<vector>
#include<fstream>

class LinearRegression : public torch::nn::Module {
public:
	LinearRegression();

	torch::Tensor forward(torch::Tensor x);

	void trainData(torch::Tensor xTrain, torch::Tensor yTrain,
		torch::Tensor xVal, torch::Tensor yVal);

	void getModelPerformance();

	void predict(torch::Tensor xtest) {
		this->eval();
		torch::Tensor pred = this->forward(xtest);
		torch::print(pred);
	}

	void savemodel();
private:
	torch::Tensor weights;
	torch::Tensor bias;
	std::vector <float> trainLosses;
	std::vector <float> validationLosses;
	std::vector<int> epochs;
	float calculateMean(std::vector<float> vect);
};

#endif
