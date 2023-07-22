#include "genericClassification.h"

BinaryClassification::BinaryClassification() {
	device = (torch::cuda::is_available() ) ? "cuda" : "cpu";
	inputLayer = register_module("input_layer", torch::nn::Linear(2,5));
	hiddenLayer1 = register_module("hidden_layer_1", torch::nn::Linear(5, 10));
	outputLayer = register_module("output_layer", torch::nn::Linear(10, 1));

	binaryLogits = torch::nn::BCEWithLogitsLoss();
}

torch::Tensor BinaryClassification::forward(torch::Tensor x) {
	torch::Tensor data = inputLayer->forward(x);
	torch::Tensor data = hiddenLayer1->forward(data);
	return outputLayer->forward(data);
}

void BinaryClassification::trainingData(
	torch::Tensor xTrain, torch::Tensor yTrain,
	torch::Tensor xVal, torch::Tensor yVal) {
	
	torch::optim::Adam optimizer = torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(this->lr));

	for (int epoch = 0; epoch < epochs; epoch++) {
	
		this->train();

		torch::Tensor currentPredictions = this->forward(xTrain);
		
		torch::Tensor currentLoss = this->binaryLogits(currentPredictions, yTrain);
		float trainAccuracy = this->accuracy(yTrain, currentPredictions);
		optimizer.zero_grad();
		currentLoss.backward();
		optimizer.step();

		this->eval();
		
		torch::Tensor currentPredValidation = this->forward(xVal);
		torch::Tensor predLoss = this->binaryLogits(currentPredValidation, yVal);
		float validationAccuracy = this->accuracy(yVal, currentPredValidation);

		this->trainAcc.push_back(trainAccuracy);
		this->validationAcc.push_back(validationAccuracy);
		this->trainLosses.push_back( currentLoss.item<float>());
		this->validationLosses.push_back( predLoss.item<float>());
		this->totalEpoch.push_back(epoch);

		if ((epoch + 1) % 100 == 0) 
		{
			std::cout << "Epoch : " 
				<< epoch + 1 <<"\nAcc : " 
				<< trainAccuracy << " Val Acc : " << validationAccuracy 
				<< "\nLoss : " << currentLoss.item<float>()
				<< "Validation Loss : " << predLoss << "\n\n";
		}

	}
}

float accuracy(torch::Tensor y_true, torch::Tensor y_pred) {
	float totalCorrect = torch::eq(y_true, y_pred).sum().item<float>();
	return (totalCorrect / y_true.size(0)) * 100.0f;
}

void BinaryClassification::predict(torch::Tensor xVal) {
	torch::print(this->forward(xVal));
}

torch::Tensor round(torch::Tensor x) 
{
	return torch::round(torch::sigmoid(x));
}
