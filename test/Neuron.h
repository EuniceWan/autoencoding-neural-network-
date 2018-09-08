#pragma once
#include "fstream"

class CLayer;
class CNeuron
{
public:
	CNeuron();  // 默认构造函数
	~CNeuron();

	void Initalize(int weight_num);                                                  // 初始化阈值
	void Compute(CLayer *previousLayer);                                     // 计算神经元的输出
	void ComputeFirst(float input);                                                // 计算输入层神经元的输出
	float ActivationFun(float x);                                                     // 激励函数
	void AdjustWeights(CLayer *previousLayer);                            // 调整该层所有神经元的权值
	void BackPropagate(int neuron_id, CLayer *backLayer);           // 误差逆向传播
	void Save(std::ofstream &fout);                                               // 保存权值
	void Load(std::ifstream &fin);                                                   // 读取权值
	float  GussianRand(float mu, float sigma);                                // 产生正态分布随机数

	inline void SetLearnRate(float lr);                                              // 设定学习率
	inline void SetLearnSampleNum(int num);                               // 设定一次学习的样本数目

	float m_fOutput;                     // 存储输出
	float m_fDeltaWeights;            // 存储权值的误差delta
	float m_fSumDeltaWeights;     // 存储权值的多个误差delta的和（用于一括学习）

private:
	float *m_pWeights;             // 存储权值指针
	float m_fThreshold;             // 存储阈值指针
	int m_nWeightNum;           // 存储权值的数目
	int m_nLearnSampleNum;  // 一次学习的样本数目
	float m_fLearnRate;             // 存储学习率
};

// 设定学习率
inline void CNeuron::SetLearnRate(float lr)
{
	m_fLearnRate = lr;
}

// 设定一次学习的样本数目
inline void CNeuron::SetLearnSampleNum(int num)
{
	m_nLearnSampleNum = num;
}

