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

// ��ʼ����ֵ
void CNeuron::Initalize(int weight_num)
{
	m_nWeightNum = weight_num;
	m_pWeights = new float[weight_num];

	float sigma = 1 / sqrt(m_nWeightNum);
	for (int i = 0; i < m_nWeightNum; i++)
	{
		// ȨֵΪ��0��sigma2����˹�ֲ�
		m_pWeights[i] = GussianRand(0, sigma);
	}

	// ��ֵΪ��0,1����˹�ֲ�
	m_fThreshold = GussianRand(0, 1);
}

// ������Ԫ�����
void CNeuron::Compute(CLayer *previousLayer)
{
	float sum = 0.0;
	for (int i = 0; i < m_nWeightNum; i++)
	{
		sum += previousLayer->GetNeuron(i)->m_fOutput * m_pWeights[i];
	}
	// ����ֵ��Ϊ�˷�������ݶ�
	m_fOutput = ActivationFun(sum + m_fThreshold);
}

// �����������Ԫ�����
void CNeuron::ComputeFirst(float input)
{
	m_fOutput = input;
}

// ��������
float CNeuron::ActivationFun(float x)
{
	return 1.0f / (1.0f + exp(-x));
}

// �����ò�������Ԫ��Ȩֵ
void CNeuron::AdjustWeights(CLayer *previousLayer)
{
	m_fSumDeltaWeights = m_fSumDeltaWeights / m_nLearnSampleNum;
	for (int i = 0; i < m_nWeightNum; i++)
	{
		m_pWeights[i] = m_pWeights[i] + m_fLearnRate*m_fSumDeltaWeights*previousLayer->GetNeuron(i)->m_fOutput;
	}
	m_fThreshold = m_fThreshold + m_fLearnRate*m_fSumDeltaWeights;
	m_fSumDeltaWeights = 0.0; // ������֮�󣬹��㣬Ϊ�´��ۻ������׼��
}

// ������򴫲�
// neuron_idָ��ǰ�ڵ��Ǹò�ڼ����ڵ�
void CNeuron::BackPropagate(int neuron_id, CLayer *backLayer)
{
	m_fDeltaWeights = 0.0;
	// ������򴫲�����Ȩ�ۻ�
	for (int i = 0; i < backLayer->GetNeuronNum(); i++)
	{
		m_fDeltaWeights += backLayer->GetNeuron(i)->m_fDeltaWeights * backLayer->GetNeuron(i)->m_pWeights[neuron_id];
	}
	m_fDeltaWeights = m_fDeltaWeights*m_fOutput*(1 - m_fOutput);

	// �ӵ�sumDelta
	m_fSumDeltaWeights += m_fDeltaWeights;
}

// ����Ȩֵ
void CNeuron::Save(ofstream &fout)
{
	// д��Ȩֵ
	for (int i = 0; i < m_nWeightNum; i++)
	{
		fout << m_pWeights[i] << " ";
	}
	// д����ֵ
	fout << m_fThreshold << endl;
}

// ��ȡȨֵ
void CNeuron::Load(ifstream &fin)
{
	// ����Ȩֵ
	for (int i = 0; i < m_nWeightNum; i++)
	{
		fin >> m_pWeights[i];
	}
	// ������ֵ
	fin >> m_fThreshold;
}

// ������̬�ֲ������
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