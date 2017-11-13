/** design a NN for StarCraft combat environment 
  * take d_emenys, d_owns, d_terrain as input, and output Q value */

#pragma once

#include <vector>
#include <string>
#include "CombatRL.h"

class CombatRL; 

/** CombatNNWeights stores NN data, weights, and derivatives
  * input - layer1(ip) - layer2(relu) - layer3(ip) - output
  * dim_input - dim_hidden - dim_hidden - dim_output */
class CombatNN
{
	friend class CombatRL; 
public:
	// construct function, define layers dimensions
	CombatNN(size_t dimi, size_t dimh, size_t dimo);
	// randomize net parameters around [0,0.01]
	void Reset();
	// load net parameters from file
	bool Load(const std::string& file_str);
	// save net parameters to file
	bool Save(const std::string& file_str);
	// get net outputs
	std::vector<double> GetOutput() { return data_output_layer_3; };
	// get net output
	double GetOutput(size_t i) { return data_output_layer_3[i]; };
private:
	// vector and matrix inner product
	std::vector<double> InnerProduct(const std::vector<double>& data, const std::vector<std::vector<double> >& weights, const std::vector<double>& bias);
	// vector and vector inner product
	double InnerProduct(const std::vector<double>& data, const std::vector<double>& weights, const double& bias);
	// relu activition layer
	std::vector<double> ReLU(const std::vector<double>& data); 
	// calculate output with input data
	bool Forward(const std::vector<double>& input);
	// differentiate vector and matrix inner product
	void InnerProductBackward(const std::vector<double>& data,
		const std::vector<std::vector<double> >& weights,
		const std::vector<double>& bias,
		const std::vector<double>& output_diff,
		std::vector<double>& data_diff, 
		std::vector<std::vector<double> >& weights_diff, 
		std::vector<double>& bias_diff);
	// differentiate vector and vector inner product
	std::vector<double> InnerProductBackward(const std::vector<double>& data,
		const std::vector<double>& weights, 
		const double& bias, 
		const double& output_diff, 
		std::vector<double>& weights_diff, 
		double& bias_diff); 
	// differentiate relu activition layer
	void ReLUBackward(const std::vector<double>& data,
		const std::vector<double>& output_diff,
		std::vector<double>& data_diff);
	// calculate derivatives with net output
	void Backward(const std::vector<double>& output_diff);
	void Backward(const size_t& i); 
private:
	// net layers dimensions
	size_t dim_input, dim_hidden, dim_output;
	// layers data
	std::vector<double> data_input, data_output_layer_1, data_output_layer_2, data_output_layer_3;
	// layers data derivative from output
	std::vector<double> data_input_diff, data_output_layer_1_diff, data_output_layer_2_diff, data_output_layer_3_diff;
	// layer 1 weights and weights derivatives
	std::vector<std::vector<double> > weights_layer_1_ip, weights_layer_1_ip_diff;
	// layer 1 bias and bias derivatives
	std::vector<double> bias_layer_1_ip, bias_layer_1_ip_diff;
	// layer 3 weights and weights derivatives
	std::vector<std::vector<double> > weights_layer_3_ip, weights_layer_3_ip_diff;
	// layer 3 bias and bias derivatives
	std::vector<double> bias_layer_3_ip, bias_layer_3_ip_diff;
};

// vector and vector equal function
bool operator==(const std::vector<double>& a, const std::vector<double>& b); 
// vector and vector add function
std::vector<double> operator+ (const std::vector<double>& a, const std::vector<double>& b);
// vector and vector add function
void operator += (std::vector<double>& a, const std::vector<double>& b);
// vector and scale times function
void operator *= (std::vector<double>& a, const double& b);
// vector and scale times function
std::vector<double> operator* (const std::vector<double>& a, const double& b);
// vector and scale times function
std::vector<double> operator* (const double& a, const std::vector<double>& b);

// matrix and matrix add function
void operator += (std::vector<std::vector<double> >&a, const std::vector<std::vector<double> >& b); 
// matrix and matrix add function
std::vector<std::vector<double> > operator + (const std::vector<std::vector<double> >&a, const std::vector<std::vector<double> >& b);
// matrix and scale times function
void operator *= (std::vector<std::vector<double> >& a, const double& b);
// matrix and scale times function
std::vector<std::vector<double> > operator* (const std::vector<std::vector<double> >& a, const double& b);
// matrix and scale times function
std::vector<std::vector<double> > operator* (const double& a, const std::vector<std::vector<double> >& b);
