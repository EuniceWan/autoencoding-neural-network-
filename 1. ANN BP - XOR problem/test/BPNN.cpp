#include "stdafx.h"
#include "BPNN.h"
#include "iostream"

using namespace std;

// 构造函数
CBPNN::CBPNN(int layer_num, int *layer_neuron_num, float learn_rate, int learn_sample_num)
{
	m_nLayerNum = layer_num;
	m_pLayers = new CLayer[m_nLayerNum];
	m_pLayers[0].Initalize(layer_neuron_num[0], 1); // 输入层的权值数目等于1
	for (int i = 1; i < m_nLayerNum; i++) // 输入层不进行初始化
	{
		m_pLayers[i].Initalize(layer_neuron_num[i], layer_neuron_num[i - 1]);
	}
	m_nInputNum = m_pLayers[0].GetNeuronNum();
	m_nOutputNum = m_pLayers[m_nLayerNum - 1].GetNeuronNum();

	// 设定学习率
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

// 计算神经网络的输出
void CBPNN::Compute(float *input)
{
	m_pLayers[0].ComputeFirstLayer(input);  // 计算输入层

	// 计算其他层
	for (int i = 1; i < m_nLayerNum; i++)
	{
		m_pLayers[i].Compute(&m_pLayers[i-1]);
	}
}

// 计算MSE
void CBPNN::ComputeMSE(CData *pData)
{
	m_fMSE = 0.0;
	// 针对每个输出节点计算偏差
	for (int k = 0; k < m_nOutputNum; k++)
	{
		// 针对每个样本计算偏差
		for (int i = 0; i < pData->m_nSampleNum; i++)
		{
			// 计算第i个样本对应的神经元实际输出
			Compute(pData->m_pX[i]);

			float output_k = m_pLayers[m_nLayerNum - 1].GetNeuron(k)->m_fOutput;
			m_fMSE += (pData->m_pY[i][k] - output_k)*(pData->m_pY[i][k] - output_k);
		}
	}
	m_fMSE = m_fMSE / pData->m_nSampleNum / 2.0f; 		// 计算均方误差MSE
}

// 计算交叉熵cross entropy
void CBPNN::ComputeCrossE(CData *pData)
{
	m_fCrossE = 0.0;
	// 针对每个输出节点计算偏差
	for (int k = 0; k < m_nOutputNum; k++)
	{
		// 针对每个样本计算偏差
		for (int i = 0; i < pData->m_nSampleNum; i++)
		{
			// 计算第i个样本对应的神经元实际输出
			Compute(pData->m_pX[i]);

			float output_k = m_pLayers[m_nLayerNum - 1].GetNeuron(k)->m_fOutput;
			m_fCrossE += - pData->m_pY[i][k] * log(output_k) - (1 - pData->m_pY[i][k])*log(1 - output_k);
		}
	}
	m_fCrossE = m_fCrossE / pData->m_nSampleNum;
}

// 计算输出层对于第i个样本的误差修正量
void CBPNN::ComputeDelta(CData *pData, int sample_id)
{
	Compute(pData->m_pX[sample_id]);

	// 对于每一个输出节点，计算误差
	for (int j = 0; j < m_nOutputNum; j++)
	{
		float &delta = m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fDeltaWeights;
		float output = m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fOutput;
		delta = (pData->m_pY[sample_id][j] - output);
		m_pLayers[m_nLayerNum - 1].GetNeuron(j)->m_fSumDeltaWeights += delta;
	}
}

//  计算逆向传播误差
void CBPNN::BackPropagate()
{
	// 计算其他层的误差
	for (int k = m_nLayerNum - 2; k > 0 ; k--)
	{
		m_pLayers[k].BackPropagate(&m_pLayers[k + 1]);  // 计算k层神经元的误差
	}
}

// 调整所有神经元的权值、阈值
void CBPNN::AdjustWeights()
{
	// 逐层修正权值、阈值
	for (int k = m_nLayerNum - 1; k > 0; k--)
	{
		m_pLayers[k].AdjustWeights(&m_pLayers[k - 1]);
	}
}

//  训练神经网络
void CBPNN::Train(int iterateNum, CData *pData)
{
	ofstream of("Data\\output MSE 128 5维 0.05 五万.txt");
		//ofstream of2("Data\\code.txt");
		if (!of.is_open())
		{
			cout << "Can't open output.txt" << endl;
			exit(0);
		}
		for (int k = 0; k < iterateNum; k++)
	{
		
		// 计算误差MSE
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
		// 输出层累积误差修正量清零
		for (int i = 0; i < m_nOutputNum; i++)
		{
			m_pLayers[m_nLayerNum - 1].GetNeuron(i)->m_fSumDeltaWeights = 0.0;
		}

		// 计算每一个节点的累积误差修正量
		for (int i = 0; i < m_nLearnSampleNum; i++)
		{
			// 随机选择样本
			int sample_id = rand() % pData->m_nSampleNum;
		//	int sample_id = k%pData->m_nSampleNum;

			// 计算输出层误差修正量
			ComputeDelta(pData, sample_id);

			// 误差逆向传播，计算其他神经元的误差
			BackPropagate();
		}

		// 调整权值、阈值
		AdjustWeights();
		
	}of << endl;
	of.close();
}

//  测试神经网络
void CBPNN::Test(CData *pData)
{
	cout << "Test Results: " << endl;
	// 针对每一个测试样本，计算并输出
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

// 保存权值
void CBPNN::Save(char *filename)
{
	ofstream fout(filename);
	if (!fout.is_open())
	{
		cout << "Can't Save Weights" << endl;
		exit(0);
	}
	for (int i = 1; i < m_nLayerNum; i++) // 输入层权值不保存
	{
		m_pLayers[i].Save(fout);
	}
	fout.close();
}

// 读取权值
void CBPNN::Load(char *filename)
{
	ifstream fin(filename);
	if (!fin.is_open())
	{
		cout << "Can't Load Weights" << endl;
		exit(0);
	}
	for (int i = 1; i < m_nLayerNum; i++) // 输入层权值不读取
	{
		m_pLayers[i].Load(fin);
	}
	fin.close();
}