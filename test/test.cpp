// test.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "define.h"
#include "bpnn.h"
#include "data.h"
#include "ctime"
#include "cstdlib"

using namespace std;

void main()
{
	srand((unsigned int)time(NULL));

	CData data("Data\\single0to9.txt", 10, 784, 784);

	int layerNeuronNum[] = { 784, 100, 784 };
	
	CBPNN bpnn(3, layerNeuronNum, 0.01f, 10);

	//bpnn.Train(500, &data);
	//bpnn.Save("Data\\weights 50000 single0to9  .txt");
	bpnn.Load("Data\\weights 50000 single0to9  .txt");
	bpnn.Test(&data);
}

