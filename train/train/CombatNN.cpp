#include "CombatNN.h"

#include <algorithm>
#include <fstream>

CombatNN::CombatNN(size_t dimi, size_t dimh, size_t dimo) :dim_input(dimi), dim_hidden(dimh), dim_output(dimo)
{
	// layers data dimensions
	data_input.assign(dim_input, 0.0);
	data_output_layer_1.assign(dim_hidden, 0.0);
	data_output_layer_2.assign(dim_hidden, 0.0);
	data_output_layer_3.assign(dim_output, 0.0);
	// layers data derivatives dimensions
	data_input_diff.assign(dim_input, 0.0);
	data_output_layer_1_diff.assign(dim_hidden, 0.0);
	data_output_layer_2_diff.assign(dim_hidden, 0.0);
	data_output_layer_3_diff.assign(dim_output, 0.0);
	// layer 1 weights and bias (derivatives) dimension
	weights_layer_1_ip.assign(dim_hidden, std::vector<double>(dim_input, 0.0));
	weights_layer_1_ip_diff.assign(dim_hidden, std::vector<double>(dim_input, 0.0));
	bias_layer_1_ip.assign(dim_hidden, 0.0);
	bias_layer_1_ip_diff.assign(dim_hidden, 0.0);
	// layer 3 weights and bias (derivatives) dimension
	weights_layer_3_ip.assign(dim_output, std::vector<double>(dim_hidden, 0.0));
	weights_layer_3_ip_diff.assign(dim_output, std::vector<double>(dim_hidden, 0.0));
	bias_layer_3_ip.assign(dim_output, 0.0);
	bias_layer_3_ip_diff.assign(dim_output, 0.0);
}

void CombatNN::Reset()
{
	// layer 1 weights and bias reset
	for (size_t i = 0; i < weights_layer_1_ip.size(); i++)
	{
		for (size_t j = 0; j < weights_layer_1_ip[i].size(); j++)
		{
			weights_layer_1_ip[i][j] = 0.01*double(rand() % 100) / 100; 
		}
	}
	bias_layer_1_ip.assign(dim_hidden, 1.0);
	// layer 3 weights and bias reset
	for (size_t i = 0; i < weights_layer_3_ip.size(); i++)
	{
		for (size_t j = 0; j < weights_layer_3_ip[i].size(); j++)
		{
			weights_layer_3_ip[i][j] = 0.01*double(rand() % 100) / 100;
		}
	}
	bias_layer_3_ip.assign(dim_output, 1.0);
}

bool CombatNN::Load(const std::string& file_str) 
{ 
	std::ifstream is(file_str.c_str(), std::ios::binary);
	if (!is.is_open())
	{
		is.close(); 
		return false;
	}
	// dim
	is.read((char*)&dim_input, sizeof(size_t)); 
	is.read((char*)&dim_hidden, sizeof(size_t)); 
	is.read((char*)&dim_output, sizeof(size_t)); 
	// layer 1
	for (size_t i = 0; i < dim_hidden; i++)
	{
		weights_layer_1_ip[i].resize(dim_input);
		is.read((char*)&weights_layer_1_ip[i][0], dim_input*sizeof(double));
	}
	bias_layer_1_ip.resize(dim_hidden); 
	is.read((char*)&bias_layer_1_ip[0], dim_hidden*sizeof(double)); 
	// layer 3
	for (size_t i = 0; i < dim_output; i++)
	{
		weights_layer_3_ip[i].resize(dim_hidden);
		is.read((char*)&weights_layer_3_ip[i][0], dim_hidden*sizeof(double));
	}
	bias_layer_3_ip.resize(dim_output);
	is.read((char*)&bias_layer_3_ip[0], dim_output*sizeof(double));
	is.close();
	return true;
}

bool CombatNN::Save(const std::string& file_str)
{
	std::ofstream os(file_str.c_str(), std::ios::binary);
	// dim
	os.write((const char*)&dim_input, sizeof(size_t)); 
	os.write((const char*)&dim_hidden, sizeof(size_t)); 
	os.write((const char*)&dim_output, sizeof(size_t)); 
	// layer 1
	for (size_t i = 0; i < dim_hidden; i++)
		os.write((const char*)&weights_layer_1_ip[i][0], dim_input*sizeof(double)); 
	os.write((const char*)&bias_layer_1_ip[0], dim_hidden*sizeof(double)); 
	// layer 3
	for (size_t i = 0; i < dim_output; i++)
		os.write((const char*)&weights_layer_3_ip[i][0], dim_hidden*sizeof(double));
	os.write((const char*)&bias_layer_3_ip[0], dim_output*sizeof(double));
	os.close();
	return true;
}

std::vector<double> CombatNN::InnerProduct(const std::vector<double>& data, const std::vector<std::vector<double> >& weights, const std::vector<double>& bias)
{
	std::vector<double> res(weights.size());
	for (size_t i = 0; i < weights.size(); i++)
		res[i] = InnerProduct(data, weights[i], bias[i]);
	return res;
}

double CombatNN::InnerProduct(const std::vector<double>& data, const std::vector<double>& weights, const double& bias)
{
	double res = bias;
	for (size_t i = 0; i < weights.size(); i++)
		res += data[i] * weights[i];
	return res;
}

std::vector<double> CombatNN::ReLU(const std::vector<double>& data)
{
	std::vector<double> res(data.size());
	for (size_t i = 0; i < data.size(); i++)
	{
		res[i] = std::max(0.0, data[i]); 
	}
	return res;
}

bool CombatNN::Forward(const std::vector<double>& input)
{
	if (dim_input != input.size())
		return false;
	data_input = input;
	data_output_layer_1 = InnerProduct(data_input, weights_layer_1_ip, bias_layer_1_ip);
	data_output_layer_2 = ReLU(data_output_layer_1);
	data_output_layer_3 = InnerProduct(data_output_layer_2, weights_layer_3_ip, bias_layer_3_ip);
	return true;
}

void CombatNN::InnerProductBackward(const std::vector<double>& data,
	const std::vector<std::vector<double> >& weights,
	const std::vector<double>& bias,
	const std::vector<double>& output_diff,
	std::vector<double>& data_diff,
	std::vector<std::vector<double> >& weights_diff,
	std::vector<double>& bias_diff)
{
	data_diff.assign(data.size(), 0.0);
	for (size_t i = 0; i < output_diff.size(); i++)
	{
		data_diff += InnerProductBackward(data, 
			weights[i], 
			bias[i], 
			output_diff[i], 
			weights_diff[i], 
			bias_diff[i]);
	}
}

std::vector<double> CombatNN::InnerProductBackward(const std::vector<double>& data,
	const std::vector<double>& weights,
	const double& bias,
	const double& output_diff,
	std::vector<double>& weights_diff,
	double& bias_diff)
{
	std::vector<double> data_diff;
	data_diff = output_diff*weights;
	weights_diff = output_diff*data;
	bias_diff = output_diff; 
	return data_diff; 
}

void CombatNN::ReLUBackward(const std::vector<double>& data,
	const std::vector<double>& output_diff,
	std::vector<double>& data_diff)
{
	data_diff.assign(data.size(), 0.0);
	for (size_t i = 0; i < output_diff.size(); i++)
		data_diff[i] = (data[i]>0) ? output_diff[i] : 0; 
}

void CombatNN::Backward(const std::vector<double>& output_diff)
{
	data_output_layer_3_diff = output_diff;
	// layer 3 (ip) backward
	InnerProductBackward(data_output_layer_2,
		weights_layer_3_ip, 
		bias_layer_3_ip,
		data_output_layer_3_diff,
		data_output_layer_2_diff, 
		weights_layer_3_ip_diff,
		bias_layer_3_ip_diff); 
	// layer 2 (relu) backward
	ReLUBackward(data_output_layer_1,
		data_output_layer_2_diff,
		data_output_layer_1_diff); 
	// layer 1 (ip) backward
	InnerProductBackward(data_input,
		weights_layer_1_ip,
		bias_layer_1_ip,
		data_output_layer_1_diff,
		data_input_diff,
		weights_layer_1_ip_diff,
		bias_layer_1_ip_diff); 
}

void CombatNN::Backward(const size_t& i)
{
	std::vector<double> output_diff(dim_output, 0.0); 
	output_diff[i] = 1.0;
	Backward(output_diff); 
}

bool operator==(const std::vector<double>& a, const std::vector<double>& b)
{
	for (size_t i = 0; i < a.size(); i++)
	{
		if (a[i] != b[i])
			return false;
	}
	return true;
}

std::vector<double> operator+ (const std::vector<double>& a, const std::vector<double>& b)
{
	std::vector<double> res(a);
	res += b;
	return res;
}

void operator += (std::vector<double>& a, const std::vector<double>& b)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] += b[i];
}

void operator *= (std::vector<double>& a, const double& b)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] *= b;
}

std::vector<double> operator* (const std::vector<double>& a, const double& b)
{
	std::vector<double> res(a);
	res *= b;
	return res;
}

std::vector<double> operator* (const double& a, const std::vector<double>& b)
{
	return b*a;
}

void operator += (std::vector<std::vector<double> >&a, const std::vector<std::vector<double> >& b)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] += b[i];
}

std::vector<std::vector<double> > operator + (const std::vector<std::vector<double> >&a, const std::vector<std::vector<double> >& b)
{
	std::vector<std::vector<double> > res(a);
	res += b;
	return res;
}

void operator *= (std::vector<std::vector<double> >& a, const double& b)
{
	for (size_t i = 0; i < a.size(); i++)
		a[i] *= b;
}

std::vector<std::vector<double> > operator* (const std::vector<std::vector<double> >& a, const double& b)
{
	std::vector<std::vector<double> > res(a);
	res *= b; 
	return res;
}
std::vector<std::vector<double> > operator* (const double& a, const std::vector<std::vector<double> >& b)
{
	return b*a;
}