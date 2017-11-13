#include "CombatRL.h"

CombatRL::CombatRL(CombatNN* p)
{
	ptr_net = p;
	weights_layer_1_ip_diff_et.assign(ptr_net->dim_hidden, std::vector<double>(ptr_net->dim_input, 0.0)); 
	bias_layer_1_ip_diff_et.assign(ptr_net->dim_hidden, 0.0);
	weights_layer_3_ip_diff_et.assign(ptr_net->dim_output, std::vector<double>(ptr_net->dim_hidden, 0.0));
	bias_layer_3_ip_diff_et.assign(ptr_net->dim_output, 0.0);
}

std::vector<double> CombatRL::GetQValues(const std::vector<double>& s)
{
	ptr_net->Forward(s);
	return ptr_net->GetOutput();
}

double CombatRL::GetQValue(const std::vector<double>& s, size_t a)
{
	return GetQValues(s)[a];
}

size_t CombatRL::GetAction(const std::vector<double>& s)
{
	std::vector<double> Qvalues = GetQValues(s);
	double Qmax = Qvalues[0];
	size_t Qmaxind = 0;
	for (size_t i = 1; i < Qvalues.size(); i++)
	{
		if (Qvalues[i] > Qmax)
		{
			Qmax = Qvalues[i];
			Qmaxind = i;
		}
	}
	return Qmaxind;
}

size_t CombatRL::GetAction(const std::vector<double>& s, const double& rate)
{
	if ((rand() % 100) < (100 * rate))
		return rand() % ptr_net->dim_output; 
	return GetAction(s);
}

void CombatRL::UpdateEligibility(const size_t i, 
	const double& lambda,
	const double& gamma)
{
	ptr_net->Backward(i); 
	//
	weights_layer_1_ip_diff_et *= gamma*lambda;
	bias_layer_1_ip_diff_et *= gamma*lambda;
	weights_layer_3_ip_diff_et *= gamma*lambda;
	bias_layer_3_ip_diff_et *= gamma*lambda;
	//
	weights_layer_1_ip_diff_et += ptr_net->weights_layer_1_ip_diff;
	bias_layer_1_ip_diff_et += ptr_net->bias_layer_1_ip_diff;
	weights_layer_3_ip_diff_et += ptr_net->weights_layer_3_ip_diff;
	bias_layer_3_ip_diff_et += ptr_net->bias_layer_3_ip_diff;
}

void CombatRL::TrainCombatNN(const double& temperal_diff,
	const double& learning_rate)
{
	ptr_net->weights_layer_1_ip += learning_rate*temperal_diff*weights_layer_1_ip_diff_et;
	ptr_net->bias_layer_1_ip += 2.0*learning_rate*temperal_diff*bias_layer_1_ip_diff_et; 
	ptr_net->weights_layer_3_ip += learning_rate*temperal_diff*weights_layer_3_ip_diff_et; 
	ptr_net->bias_layer_3_ip += 2.0*learning_rate*temperal_diff*bias_layer_3_ip_diff_et;
}
