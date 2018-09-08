#include "stdafx.h"
#include "Layer.h"

using namespace std;

CLayer::CLayer()
{
	m_nNeuronNum = 0;
	m_pNeurons = NULL;
}

CLayer::~CLayer()
{
	if (m_pNeurons)
		delete[] m_pNeurons;
	m_pNeurons = NULL;
}

// 初始化该层
void CLayer::Initalize(int neuron_num, int weight_num)
{
	m_nNeuronNum = neuron_num;
	m_pNeurons = new CNeuron[m_nNeuronNum];
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Initalize(weight_num);
	}
}

 // 计算输入层神经元输出
void CLayer::ComputeFirstLayer(float *input)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].ComputeFirst(input[i]);
	}
}

// 计算该层所有神经元的输出
void CLayer::Compute(CLayer *previousLayer)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Compute(previousLayer);
	}
}

// 调整该层所有神经元的权值
void CLayer::AdjustWeights(CLayer *previousLayer)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].AdjustWeights(previousLayer);
	}
}

// 误差逆向传播
void CLayer::BackPropagate(CLayer *backLayer)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].BackPropagate(i, backLayer);
	}
}

// 设定学习率
void CLayer::SetLearnRate(float lr)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].SetLearnRate(lr);
	}
}

// 设定一次学习的样本数目
void CLayer::SetLearnSampleNum(int num)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].SetLearnSampleNum(num);
	}
}

// 保存权值
void CLayer::Save(ofstream &fout)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Save(fout);
	}
}

// 读取权值
void CLayer::Load(ifstream &fin)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Load(fin);
	}
}