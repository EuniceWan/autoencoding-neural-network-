#pragma once

#include "Neuron.h"

class CLayer
{
public:
	CLayer();
	~CLayer();

	void Initalize(int neuron_num, int weight_num); // ��ʼ���ò�
	void Compute(CLayer *previousLayer);              // ����ò�������Ԫ�����
	void ComputeFirstLayer(float *input);                // �����������Ԫ���
	void AdjustWeights(CLayer *previousLayer);      // �����ò�������Ԫ��Ȩֵ
	void BackPropagate(CLayer *backLayer);           // ������򴫲�
	void SetLearnRate(float lr);                                // �趨ѧϰ��
	void SetLearnSampleNum(int num);                  // �趨һ��ѧϰ��������Ŀ
	void Save(std::ofstream &fout);                         // ����Ȩֵ
	void Load(std::ifstream &fin);                            // ��ȡȨֵ

	inline CNeuron *GetNeuron(int i); // ��ȡ�ò��i����Ԫ
	inline int GetNeuronNum();          // ��ȡ�ò���Ԫ����Ŀ

private:
	int m_nNeuronNum;        // ��Ԫ����Ŀ
	CNeuron *m_pNeurons;   // ��Ԫ����ָ��
};

// ��ȡ�ò��i����Ԫ
inline CNeuron *CLayer::GetNeuron(int i)
{
	return m_pNeurons + i;
}

// ��ȡ�ò��i����Ԫ
inline int CLayer::GetNeuronNum()
{
	return m_nNeuronNum;
}
