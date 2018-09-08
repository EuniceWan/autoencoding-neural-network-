#include "stdafx.h"
#include "BPNN.h"
#include "iostream"

using namespace std;

// ���캯��
CBPNN::CBPNN(int layer_num, int *layer_neuron_num, float learn_rate, int learn_sample_num)
{
	m_nLayerNum = layer_num;
	m_pLayers = new CLayer[m_nLayerNum];
	m_pLayers[0].Initalize(layer_neuron_num[0], 1); // ������Ȩֵ��Ŀ����1
	for (int i = 1; i < m_nLayerNum; i++) // ����㲻���г�ʼ��
	{
		m_pLayers[i].Initalize(layer_neuron_num[i], layer_neuron_num[i - 1]);
	}
	m_nInputNum = m_pLayers[0].GetNeuronNum();
	m_nOutputNum = m_pLayers[m_nLayerNum - 1].GetNeuronNum();

	// �趨ѧϰ��
	m_fLearnRate = learn_rate;
	m_nLearnSampleNum = learn_sample_num;
	for (int i = 0; i < m_nLayerNum; i++)
	{
		m_pLayers[i].SetLearnRate(m_fLearnRate);
		m_pLayers[i].SetLearnSampleNum(m_nLearnSampleNum);
	}

	m_fMSE = 0.0;
	m_fCrossE = 0.0;
}

CBPNN::~CBPNN()
{
	if (m_pLayers)
		delete[] m_pLayers;
	m_pLayers = NULL;
}

// ��������������
void CBPNN::Compute(float *input)
{
	m_pLayers[0].ComputeFirstLayer(input);  // ���������

	// ����������
	for (int i = 1; i < m_nLayerNum; i++)
	{
		m_pLayers[i].Compute(&m_pLayers[i-1]);
	}
}

// ����MSE
void CBPNN::ComputeMSE(CData *pData)
{
	m_fMSE = 0.0;
	// ���ÿ������ڵ����ƫ��
	for (int k = 0; k < m_nOutputNum; k++)
	{
		// ���ÿ����������ƫ��
		for (int i = 0; i < pData->m_nSampleNum; i++)
		{
			// �����i��������Ӧ����Ԫʵ�����
			Compute(pData->m_pX[i]);

			float output_k = m_pLayers[m_nLayerNum - 1].GetNeuron(k)->m_fOutput;
			m_fMSE += (pData->m_pY[i][k] - output_k)*(pData->m_pY[i][k] - output_k);
		}
	}
	m_fMSE = m_fMSE / pData->m_nSampleNum / 2.0f; 		// ����������MSE
}

// ���㽻����cross entropy
void CBPNN::ComputeCrossE(CData *pData)
{
	m_fCrossE = 0.0;
	// ���ÿ������ڵ����ƫ��
	for (int k = 0; k < m_nOutputNum; k++)
	{
		// ���ÿ����������ƫ��
		for (int i = 0; i < pData->m_nSampleNum; i++)
		{
			// �����i��������Ӧ����Ԫʵ�����
			Compute(pData->m_pX[i]);

			float output_k = m_pLayers[m_nLayerNum - 1].GetNeuron(k)->m_fOutput;
			m_fCrossE += - pData->m_pY[i][k] * log(output_k) - (1 - pData->m_pY[i][k])*log(1 - output_k);
		}
	}
	m_fCrossE = m_fCrossE / pData->m_nSampleNum;
}

// �����������ڵ�i�����������������
void CBPNN::ComputeDelta(CData *pData, int sample_id)
{
	Compute(pData->m_pX[sample_id]);

	// ����ÿһ������ڵ㣬�������
	for (int j = 0; j < m_nOutputNum; j++)
	{
		float &delta = m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fDeltaWeights;
		float output = m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fOutput;
		delta = (pData->m_pY[sample_id][j] - output);
		m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fSumDeltaWeights += delta;
	}
}

//  �������򴫲����
void CBPNN::BackPropagate()
{
	// ��������������
	for (int k = m_nLayerNum - 2; k > 0 ; k--)
	{
		m_pLayers[k].BackPropagate(&m_pLayers[k + 1]);  // ����k����Ԫ�����
	}
}

// ����������Ԫ��Ȩֵ����ֵ
void CBPNN::AdjustWeights()
{
	// �������Ȩֵ����ֵ
	for (int k = m_nLayerNum - 1; k > 0; k--)
	{
		m_pLayers[k].AdjustWeights(&m_pLayers[k - 1]);
	}
}

//  ѵ��������
void CBPNN::Train(int iterateNum, CData *pData)
{
	ofstream of("Data\\output MSE 128 5ά 0.05 ����.txt");
		//ofstream of2("Data\\code.txt");
		if (!of.is_open())
		{
			cout << "Can't open output.txt" << endl;
			exit(0);
		}
		for (int k = 0; k < iterateNum; k++)
	{
		
		// �������MSE
		ComputeMSE(pData);
		if (m_fMSE < 0.000001)
		{
			break;
		}
		if (k % 100 == 0)
		{
			cout << k + 1 << " MSE: " << m_fMSE << endl;
			of << m_fMSE << " ";
		}
		// ������ۻ��������������
		for (int i = 0; i < m_nOutputNum; i++)
		{
			m_pLayers[m_nLayerNum - 1].GetNeuron(i)->m_fSumDeltaWeights = 0.0;
		}

		// ����ÿһ���ڵ���ۻ����������
		for (int i = 0; i < m_nLearnSampleNum; i++)
		{
			// ���ѡ������
			int sample_id = rand() % pData->m_nSampleNum;
		//	int sample_id = k%pData->m_nSampleNum;

			// ������������������
			ComputeDelta(pData, sample_id);

			// ������򴫲�������������Ԫ�����
			BackPropagate();
		}

		// ����Ȩֵ����ֵ
		AdjustWeights();
		
	}of << endl;
	of.close();
}

//  ����������
void CBPNN::Test(CData *pData)
{
	cout << "Test Results: " << endl;
	// ���ÿһ���������������㲢���
	for (int i = 0; i < pData->m_nSampleNum; i++)
	{
		Compute(pData->m_pX[i]);
		for (int j = 0; j < m_nOutputNum; j++)
		{
			cout << "yp=" << pData->m_pY[i][j] << " ";
			cout << m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fOutput << " ";
			if (fabs(pData->m_pY[i][j] - m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fOutput) > 0.5)
				cout << "****error****";
		}
		cout << endl;
	}
}

// ����Ȩֵ
void CBPNN::Save(char *filename)
{
	ofstream fout(filename);
	if (!fout.is_open())
	{
		cout << "Can't Save Weights" << endl;
		exit(0);
	}
	for (int i = 1; i < m_nLayerNum; i++) // �����Ȩֵ������
	{
		m_pLayers[i].Save(fout);
	}
	fout.close();
}

// ��ȡȨֵ
void CBPNN::Load(char *filename)
{
	ifstream fin(filename);
	if (!fin.is_open())
	{
		cout << "Can't Load Weights" << endl;
		exit(0);
	}
	for (int i = 1; i < m_nLayerNum; i++) // �����Ȩֵ����ȡ
	{
		m_pLayers[i].Load(fin);
	}
	fin.close();
}