// test.cpp : 定义控制台应用程序的入口点。
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

	CData data("Data\\sample 128.txt", 128, 6, 6);

	int layerNeuronNum[] = { 6, 5, 6 };

	CBPNN bpnn(3, layerNeuronNum, 0.05f, 1);

	bpnn.Train(50000, &data);
	bpnn.Save("Data\\weights 128.txt");
	//bpnn.Load("Data\\weights 128.txt");
	bpnn.Test(&data);
}

