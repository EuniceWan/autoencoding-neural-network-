#pragma once

class CData
{
public:
	CData(char* filename                                 // 文件名
		, int sample_num                                    // 样本数目
		, int input_dim                                        // 输入数据的维数
		, int ouput_dim);                                     // 输出数据的维数
	~CData(void);

public:
	int m_nSampleNum;                            // 样本数目
	float **m_pX;                                       // 实际样本输入x
	float **m_pY;                                       // 期望输出y
	int m_nInputDim;                                // 输入样本维数
	int m_nOutputDim;                             // 输出样本维数
};
