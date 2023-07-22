#include "linreg.h"

LinearRegression::LinearRegression() {

	weights = register_parameter("weights", torch::randn(1), true);
	bias = register_parameter("bias", torch::randn(1), true);

}

torch::Tensor LinearRegression::forward(torch::Tensor x) {
	return weights * x + bias;
}

void LinearRegression::trainData(torch::Tensor xTrain, torch::Tensor yTrain,
	torch::Tensor xVal, torch::Tensor yVal) {
	int EPOCHS = 1000;
	double learningRate = 0.001;

	torch::nn::MSELoss mse = torch::nn::MSELoss();

	torch::optim::AdamOptions adamOptions = torch::optim::AdamOptions(learningRate);
	torch::optim::Adam adam = torch::optim::Adam(this->parameters(), adamOptions);

	for (int epoch = 0; epoch < EPOCHS; epoch++) {

		this->train();
		torch::Tensor predictions = round(this->forward(xTrain));

		torch::Tensor currentLoss = mse(predictions, yTrain);

		adam.zero_grad();

		currentLoss.backward();

		adam.step();

		// Model Evaluation
		this->eval();

		torch::Tensor validationPred = round(this->forward(xVal));
		torch::Tensor validationLoss = mse(validationPred, yVal);

		this->epochs.push_back(epoch + 1);
		this->trainLosses.push_back(currentLoss.item<float>());
		this->validationLosses.push_back(validationLoss.item<float>());


		if ((epoch + 1) % 50 == 0)
		{
			std::cout << "Epoch :" << (epoch + 1) << " / " << EPOCHS << " MSE : " << currentLoss.item<float>() <<
				"\nValidation Loss : " << validationLoss.item<float>() << std::endl;
		}

	}

}
void LinearRegression::getModelPerformance() {
	float trainMean = this->calculateMean(this->trainLosses);
	float validationMean = this->calculateMean(this->validationLosses);


	std::cout << "Training Mean MSE : " << trainMean << std::endl;
	std::cout << "Validation Mean MSE : " << validationMean << std::endl;


}

float LinearRegression::calculateMean(std::vector<float> vect)
{
	if (vect.empty()) return 0.0f;
	float totalSum = 0.0f;
	for (auto value : vect) {
		totalSum += value;
	}

	return totalSum / vect.size();

}

void LinearRegression::savemodel() {
	std::string model_path = "01_pytorch_workflow_model_0.pth";
	try {
		torch::save(this->parameters(), model_path);
		std::cout << "Model Saved Successfully" << std::endl;
	}
	catch (const std::exception e) {
		std::cerr << "Error Saving model " << e.what() << std::endl;
	}
}
