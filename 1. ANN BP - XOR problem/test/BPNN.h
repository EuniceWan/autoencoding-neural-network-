#pragma once

#include "Data.h"
#include "Layer.h"

class CBPNN
{
public:
	CBPNN(int layer_num, int *layer_neuron_num, float learn_rate, int learn_sample_num);   // layer_neuron_num����ÿһ���Ӧ�Ľڵ���Ŀ
	~CBPNN();

	void Compute(float *input);                                // ��������������
	void ComputeMSE(CData *pData); 	                  // ����MSE
	void ComputeCrossE(CData *pData);                  // ���㽻����cross entropy
	void ComputeDelta(CData *pData, int i); 	          // �����������ڵ�i�����������������
	void BackPropagate();                                        //  �������򴫲����
	void AdjustWeights();                                         //  ����Ȩֵ����ֵ
	void Train(int iterateNum, CData *pData);          //  ѵ��������
	void Test(CData *pData);                                    //  ����������

	void Save(char *filename);                                  // ����Ȩֵ
	void Load(char *filename);                                 // ��ȡȨֵ

private:
	int m_nInputNum;                // �������Ԫ����Ŀ
	int m_nOutputNum;             // �������Ԫ����Ŀ
	int m_nLayerNum;                // ������Ĳ���
	float m_fMSE;                      // �������
	float m_fCrossE;                   // ������
	float m_fLearnRate;              // �洢ѧϰ��
	int m_nLearnSampleNum;    // һ��ѧϰ��������Ŀ
	CLayer *m_pLayers;              // ������������ָ��
};

