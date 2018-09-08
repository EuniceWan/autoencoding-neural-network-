#pragma once

#include "Neuron.h"

class CLayer
{
public:
	CLayer();
	~CLayer();

	void Initalize(int neuron_num, int weight_num); // 初始化该层
	void Compute(CLayer *previousLayer);              // 计算该层所有神经元的输出
	void ComputeFirstLayer(float *input);                // 计算输入层神经元输出
	void AdjustWeights(CLayer *previousLayer);      // 调整该层所有神经元的权值
	void BackPropagate(CLayer *backLayer);           // 误差逆向传播
	void SetLearnRate(float lr);                                // 设定学习率
	void SetLearnSampleNum(int num);                  // 设定一次学习的样本数目
	void Save(std::ofstream &fout);                         // 保存权值
	void Load(std::ifstream &fin);                            // 读取权值

	inline CNeuron *GetNeuron(int i); // 获取该层第i个神经元
	inline int GetNeuronNum();          // 获取该层神经元的数目

private:
	int m_nNeuronNum;        // 神经元的数目
	CNeuron *m_pNeurons;   // 神经元数组指针
};

// 获取该层第i个神经元
inline CNeuron *CLayer::GetNeuron(int i)
{
	return m_pNeurons + i;
}

// 获取该层第i个神经元
inline int CLayer::GetNeuronNum()
{
	return m_nNeuronNum;
}
