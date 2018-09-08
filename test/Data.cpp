#include "stdafx.h"
#include "Data.h"

#include "cstdlib"
#include "fstream"
#include "iostream"

using namespace std;

CData::CData(char* filename                       // 文件名
	, int sample_num                                    // 样本数目
	, int input_dim                                        // 输入数据的维数
	, int ouput_dim)                                     // 输出数据的维数
{
	// 初始化
	m_nSampleNum = sample_num;
	m_nInputDim = input_dim;
	m_nOutputDim = ouput_dim;

	// 读文件
	int i, j;
	ifstream fin(filename);
	if (!fin.is_open())
	{
		cout << "无法读取文件：" << filename;
		exit(0);
	}

	// 申请空间
	m_pX = new float*[m_nSampleNum];
	m_pY = new float*[m_nSampleNum];
	for (i = 0; i < m_nSampleNum; i++)
	{
		m_pX[i] = new float[m_nInputDim];
		m_pY[i] = new float[m_nOutputDim];
	}

	// 读数据
	for (i = 0; i < m_nSampleNum; i++)
	{
		for (j = 0; j < m_nInputDim; j++)
		{
			fin >> m_pX[i][j];
		}
		for (j = 0; j < m_nOutputDim; j++)
		{
			fin >> m_pY[i][j];
		}
	}
	fin.close();  // 关闭流
}

CData::~CData(void)
{
	// 释放空间
	if (m_pX)
	{
		for (int i = 0; i < m_nSampleNum; i++)
		{
			delete[] m_pX[i];
		}
		delete[] m_pX;
	}
	if (m_pY)
	{
		for (int i = 0; i < m_nSampleNum; i++)
		{
			delete[] m_pY[i];
		}
		delete[] m_pY;
	}
}
