#include<torch/torch.h>

class LinearRegression : public torch::nn::Module {
public:
	LinearRegression();

	torch::Tensor forward(torch::Tensor x);
private:
	torch::Tensor weights;
	torch::Tensor bias;
};
