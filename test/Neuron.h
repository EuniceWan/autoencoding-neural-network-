#pragma once
#include "fstream"

class CLayer;
class CNeuron
{
public:
	CNeuron();  // Ĭ�Ϲ��캯��
	~CNeuron();

	void Initalize(int weight_num);                                                  // ��ʼ����ֵ
	void Compute(CLayer *previousLayer);                                     // ������Ԫ�����
	void ComputeFirst(float input);                                                // �����������Ԫ�����
	float ActivationFun(float x);                                                     // ��������
	void AdjustWeights(CLayer *previousLayer);                            // �����ò�������Ԫ��Ȩֵ
	void BackPropagate(int neuron_id, CLayer *backLayer);           // ������򴫲�
	void Save(std::ofstream &fout);                                               // ����Ȩֵ
	void Load(std::ifstream &fin);                                                   // ��ȡȨֵ
	float  GussianRand(float mu, float sigma);                                // ������̬�ֲ������

	inline void SetLearnRate(float lr);                                              // �趨ѧϰ��
	inline void SetLearnSampleNum(int num);                               // �趨һ��ѧϰ��������Ŀ

	float m_fOutput;                     // �洢���
	float m_fDeltaWeights;            // �洢Ȩֵ�����delta
	float m_fSumDeltaWeights;     // �洢Ȩֵ�Ķ�����delta�ĺͣ�����һ��ѧϰ��

private:
	float *m_pWeights;             // �洢Ȩֵָ��
	float m_fThreshold;             // �洢��ֵָ��
	int m_nWeightNum;           // �洢Ȩֵ����Ŀ
	int m_nLearnSampleNum;  // һ��ѧϰ��������Ŀ
	float m_fLearnRate;             // �洢ѧϰ��
};

// �趨ѧϰ��
inline void CNeuron::SetLearnRate(float lr)
{
	m_fLearnRate = lr;
}

// �趨һ��ѧϰ��������Ŀ
inline void CNeuron::SetLearnSampleNum(int num)
{
	m_nLearnSampleNum = num;
}

