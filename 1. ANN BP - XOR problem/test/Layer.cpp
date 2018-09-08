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

// ��ʼ���ò�
void CLayer::Initalize(int neuron_num, int weight_num)
{
	m_nNeuronNum = neuron_num;
	m_pNeurons = new CNeuron[m_nNeuronNum];
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Initalize(weight_num);
	}
}

 // �����������Ԫ���
void CLayer::ComputeFirstLayer(float *input)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].ComputeFirst(input[i]);
	}
}

// ����ò�������Ԫ�����
void CLayer::Compute(CLayer *previousLayer)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Compute(previousLayer);
	}
}

// �����ò�������Ԫ��Ȩֵ
void CLayer::AdjustWeights(CLayer *previousLayer)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].AdjustWeights(previousLayer);
	}
}

// ������򴫲�
void CLayer::BackPropagate(CLayer *backLayer)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].BackPropagate(i, backLayer);
	}
}

// �趨ѧϰ��
void CLayer::SetLearnRate(float lr)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].SetLearnRate(lr);
	}
}

// �趨һ��ѧϰ��������Ŀ
void CLayer::SetLearnSampleNum(int num)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].SetLearnSampleNum(num);
	}
}

// ����Ȩֵ
void CLayer::Save(ofstream &fout)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Save(fout);
	}
}

// ��ȡȨֵ
void CLayer::Load(ifstream &fin)
{
	for (int i = 0; i < m_nNeuronNum; i++)
	{
		m_pNeurons[i].Load(fin);
	}
}