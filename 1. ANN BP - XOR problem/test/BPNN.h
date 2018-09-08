#pragma once

#include "Data.h"
#include "Layer.h"

class CBPNN
{
public:
	CBPNN(int layer_num, int *layer_neuron_num, float learn_rate, int learn_sample_num);   // layer_neuron_num代表每一层对应的节点数目
	~CBPNN();

	void Compute(float *input);                                // 计算神经网络的输出
	void ComputeMSE(CData *pData); 	                  // 计算MSE
	void ComputeCrossE(CData *pData);                  // 计算交叉熵cross entropy
	void ComputeDelta(CData *pData, int i); 	          // 计算输出层对于第i个样本的误差修正量
	void BackPropagate();                                        //  计算逆向传播误差
	void AdjustWeights();                                         //  修正权值、阈值
	void Train(int iterateNum, CData *pData);          //  训练神经网络
	void Test(CData *pData);                                    //  测试神经网络

	void Save(char *filename);                                  // 保存权值
	void Load(char *filename);                                 // 读取权值

private:
	int m_nInputNum;                // 输入层神经元的数目
	int m_nOutputNum;             // 输出层神经元的数目
	int m_nLayerNum;                // 神经网络的层数
	float m_fMSE;                      // 均方误差
	float m_fCrossE;                   // 交叉熵
	float m_fLearnRate;              // 存储学习率
	int m_nLearnSampleNum;    // 一次学习样本的数目
	CLayer *m_pLayers;              // 神经网络层数组的指针
};

