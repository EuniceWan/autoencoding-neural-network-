#include "stdafx.h"
#include "Data.h"

#include "cstdlib"
#include "fstream"
#include "iostream"

using namespace std;

CData::CData(char* filename                       // �ļ���
	, int sample_num                                    // ������Ŀ
	, int input_dim                                        // �������ݵ�ά��
	, int ouput_dim)                                     // ������ݵ�ά��
{
	// ��ʼ��
	m_nSampleNum = sample_num;
	m_nInputDim = input_dim;
	m_nOutputDim = ouput_dim;

	// ���ļ�
	int i, j;
	ifstream fin(filename);
	if (!fin.is_open())
	{
		cout << "�޷���ȡ�ļ���" << filename;
		exit(0);
	}

	// ����ռ�
	m_pX = new float*[m_nSampleNum];
	m_pY = new float*[m_nSampleNum];
	for (i = 0; i < m_nSampleNum; i++)
	{
		m_pX[i] = new float[m_nInputDim];
		m_pY[i] = new float[m_nOutputDim];
	}

	// ������
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
	fin.close();  // �ر���
}

CData::~CData(void)
{
	// �ͷſռ�
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
