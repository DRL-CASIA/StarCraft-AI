#pragma once

#include "CombatNN.h"
#include <vector>

class CombatNN; 

/** CombatRL is a SARSA with eligibility trace RL algorithm
  * it is built based on a network 
  * it trains the network weights using the eligibility trace */
class CombatRL
{
public:
	// construct RL algorithm 
	CombatRL() :ptr_net(nullptr) {}; 
	CombatRL(CombatNN* p); 
	// calculate Q value(s)
	std::vector<double> GetQValues(const std::vector<double>& s);
	double GetQValue(const std::vector<double>& s, size_t a);
	// calculate action
	size_t GetAction(const std::vector<double>& s);
	size_t GetAction(const std::vector<double>& s, const double& rate); 
	// update eligibility trace
	void UpdateEligibility(const size_t i, 
		const double& lambda,
		const double& gamma);
	// train net weights 
	void TrainCombatNN(const double& temperal_diff, 
		const double& learning_rate); 
private:
	// store a net pointer
	CombatNN* ptr_net; 
	// net layer 1 weights and bias eligibility trace
	std::vector<std::vector<double> > weights_layer_1_ip_diff_et;
	std::vector<double> bias_layer_1_ip_diff_et;
	// net layer 3 weights and bias eligibility trace
	std::vector<std::vector<double> > weights_layer_3_ip_diff_et;
	std::vector<double> bias_layer_3_ip_diff_et;
};