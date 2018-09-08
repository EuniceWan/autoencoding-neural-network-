#include "stdafx.h"
#include "cstdlib"
#include "cmath"
#include "iostream"
#include "fstream"
#include "limits"

#include "Neuron.h"
#include "Layer.h"

using namespace std;

CNeuron::CNeuron()
{
	m_nWeightNum = 0;
	m_fThreshold = 0;
	m_fOutput = 0;
	m_fDeltaWeights = 0;
	m_fSumDeltaWeights = 0;
	m_fLearnRate = 0;
	m_nLearnSampleNum = 0;
	m_pWeights = NULL;
}


CNeuron::~CNeuron()
{
	if (m_pWeights)
	{
		delete[] m_pWeights;
	}
	m_pWeights = NULL;

}

// 初始化阈值
void CNeuron::Initalize(int weight_num)
{
	m_nWeightNum = weight_num;
	m_pWeights = new float[weight_num];

	float sigma = 1 / sqrt(m_nWeightNum);
	for (int i = 0; i < m_nWeightNum; i++)
	{
		// 权值为（0，sigma2）高斯分布
		m_pWeights[i] = GussianRand(0, sigma);
	}

	// 阈值为（0,1）高斯分布
	m_fThreshold = GussianRand(0, 1);
}

// 计算神经元的输出
void CNeuron::Compute(CLayer *previousLayer)
{
	float sum = 0.0;
	for (int i = 0; i < m_nWeightNum; i++)
	{
		sum += previousLayer->GetNeuron(i)->m_fOutput * m_pWeights[i];
	}
	// 加阈值，为了方便计算梯度
	m_fOutput = ActivationFun(sum + m_fThreshold);
}

// 计算输入层神经元的输出
void CNeuron::ComputeFirst(float input)
{
	m_fOutput = input;
}

// 激励函数
float CNeuron::ActivationFun(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

// 调整该层所有神经元的权值
void CNeuron::AdjustWeights(CLayer *previousLayer)
{
	m_fSumDeltaWeights = m_fSumDeltaWeights / m_nLearnSampleNum;
	for (int i = 0; i < m_nWeightNum; i++)
	{
		m_pWeights[i] = m_pWeights[i] + m_fLearnRate*m_fSumDeltaWeights*previousLayer->GetNeuron(i)->m_fOutput;
	}
	m_fThreshold = m_fThreshold + m_fLearnRate*m_fSumDeltaWeights;
	m_fSumDeltaWeights = 0.0; // 调整完之后，归零，为下次累积误差做准备
}

// 误差逆向传播
// neuron_id指当前节点是该层第几个节点
void CNeuron::BackPropagate(int neuron_id, CLayer *backLayer)
{
	m_fDeltaWeights = 0.0;
	// 误差逆向传播，加权累积
	for (int i = 0; i < backLayer->GetNeuronNum(); i++)
	{
		m_fDeltaWeights += backLayer->GetNeuron(i)->m_fDeltaWeights * backLayer->GetNeuron(i)->m_pWeights[neuron_id];
	}
	m_fDeltaWeights = m_fDeltaWeights*m_fOutput*(1 - m_fOutput);

	// 加到sumDelta
	m_fSumDeltaWeights += m_fDeltaWeights;
}

// 保存权值
void CNeuron::Save(ofstream &fout)
{
	// 写入权值
	for (int i = 0; i < m_nWeightNum; i++)
	{
		fout << m_pWeights[i] << " ";
	}
	// 写入阈值
	fout << m_fThreshold << endl;
}

// 读取权值
void CNeuron::Load(ifstream &fin)
{
	// 读出权值
	for (int i = 0; i < m_nWeightNum; i++)
	{
		fin >> m_pWeights[i];
	}
	// 读出阈值
	fin >> m_fThreshold;
}

// 产生正态分布随机数
float  CNeuron::GussianRand(float mu, float sigma)
{
	const float epsilon = numeric_limits<float>::min();
	const float tau = 2.0f*3.1415926f;

	static float z0, z1;
	static bool generate;
	generate = !generate;
	if (!generate)
	{
		return z1*sigma + mu;
	}

	float u1, u2;
	do
	{
		u1 = (float)rand() / RAND_MAX;
		u2 = (float)rand() / RAND_MAX;
	} while (u1 <= epsilon);
	
	z0 = sqrt(-2.0f * log(u1))*cos(tau*u2);
	z1 = sqrt(-2.0f * log(u1))*sin(tau*u2);
	return z0*sigma + mu;
}