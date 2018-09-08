#pragma once

class CData
{
public:
	CData(char* filename                                 // �ļ���
		, int sample_num                                    // ������Ŀ
		, int input_dim                                        // �������ݵ�ά��
		, int ouput_dim);                                     // ������ݵ�ά��
	~CData(void);

public:
	int m_nSampleNum;                            // ������Ŀ
	float **m_pX;                                       // ʵ����������x
	float **m_pY;                                       // �������y
	int m_nInputDim;                                // ��������ά��
	int m_nOutputDim;                             // �������ά��
};
