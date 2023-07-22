#include<torch/torch.h>
#include<vector>

class BinaryClassification : torch::nn::Module {
	
public :
	BinaryClassification();

	torch::Tensor forward(torch::Tensor x);

	void trainingData(
		torch::Tensor xTrain,
		torch::Tensor yTrain,
		torch::Tensor xVal,
		torch::Tensor yVal
	);

	void predict(torch::Tensor xVal);
	
	float accuracy(torch::Tensor y_true, torch::Tensor y_pred);

private :
	torch::nn::Linear inputLayer;
	torch::nn::Linear hiddenLayer1;
	torch::nn::Linear outputLayer;
	std::string device;
	
	int epochs = 1000;
	float lr = 0.01;

	std::vector<float> trainLosses;
	std::vector<float> validationLosses;
	std::vector<float> trainAcc;
	std::vector<float> validationAcc;
	std::vector<int> totalEpoch;

	torch::Tensor round(torch::Tensor x);
	torch::nn::BCEWithLogitsLoss binaryLogits;
	
};
