#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include "RNN.h"

double **X;
double *matrix_out_FL;
double out_net;

double *matrix_syn_transform_FL;
double matrix_syn_transform_SL;

double *old_matrix_out_FL;


double **matrix_first_layer;
double  *matrix_sec_layer;

double **matrix_context;


double  *T1;
double   T2;

double  *G1;
double   G2;

double  *D;

double  alpha;
double e = 0.002;

int p;
int HL_size;
int lern_size;

void init_waight_matrix() {

	matrix_first_layer = (double **)malloc(HL_size*sizeof(double*));
	for (int i = 0; i < HL_size; i++)
		matrix_first_layer[i] = (double *)malloc(p * sizeof(double));

	matrix_sec_layer = (double *)malloc(HL_size * sizeof(double));

	matrix_context = (double **)malloc(HL_size*sizeof(double*));
	for (int i = 0; i < HL_size; i++)
		matrix_context[i] = (double *)malloc(HL_size * sizeof(double));
	

	srand(time(NULL));
	for (int i = 0; i<HL_size; i++)
		for (int j = 0; j<p; j++)
			matrix_first_layer[i][j] = ((((double)rand() / (double)(RAND_MAX)) * 2) - 1);

	for (int j = 0; j<HL_size; j++)
		matrix_sec_layer[j] = ((((double)rand() / (double)(RAND_MAX)) * 2) - 1);

	for (int i = 0; i<HL_size; i++)
		for (int j = 0; j<HL_size; j++)
			matrix_context[i][j] = ((((double)rand() / (double)(RAND_MAX)) * 2) - 1);



}

void choose_sequence(int window_size, int hidden_layer_size, double err, double A) {

	int size = 16;
	double sequence[16];
	char number;

	e = err;
	alpha = A;
	p = window_size;
	HL_size = hidden_layer_size;

	cout << "Выберите последовательность\n";
	cout << "1) Fibonacci number\n";
	cout << "2) 2^x\n";
	cout << "3) 1,2,3,4,5...\n";
	cout << "4) x^2\n";
	cout << "5) х+2\n";

	cin >> number;

	switch (number) {
	case '1': {
		double tmp[] = { 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610 };
		size = 12;
		//  e=0.05;
		//  A =0.001;
		//  p=2;
		//  m=1;

		memcpy(sequence, tmp, sizeof(tmp));
		break;
	}
	case '2': {
		double tmp[] = { 1,4,8,16,32,64,128,256 };
		size = 7;

		//  A =0.001;
		// p=6;
		//  m=1*p;
		memcpy(sequence, tmp, sizeof(tmp));
		break;
	}
	case '3': {
		double tmp[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
		size = 15;
		// e=0.1;
		//  A =0.001;
		//  p=2;
		// m=1*p;
		memcpy(sequence, tmp, sizeof(tmp));
		break;
	}
	case '4': {
		double tmp[] = { 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225, 256 };
		size = 8;

		//   e=0.05;
		//   A =0.001;
		// p=size-6;
		//  m=5*p;
		memcpy(sequence, tmp, sizeof(tmp));
		break;
	}
	case '5': {
		double tmp[] = { 0, 2, 4, 6, 8, 10, 12};
		size = 7;

		//   e=0.05;
		//   A =0.001;
		// p=3;
		//  m=4;
		memcpy(sequence, tmp, sizeof(tmp));  //копируем куда, откуда, сколько
		break;
	}
	default:
		cout << "Input error\n"; 
		exit(0);

	}

	lern_size = size - p;

	X = (double **)malloc(lern_size*sizeof(double*));
	for (int i = 0; i < lern_size; i++)
		X[i] = (double *)malloc(p * sizeof(double));

	matrix_out_FL = (double *)malloc(HL_size*sizeof(double));

	matrix_syn_transform_FL = (double *)malloc(HL_size*sizeof(double));

	old_matrix_out_FL = (double *)malloc(HL_size*sizeof(double));

	D = (double *)malloc(lern_size*sizeof(double));

	T1 = (double *)malloc(HL_size*sizeof(double));

	G1 = (double *)malloc(HL_size*sizeof(double));

	for (int i = 0;i<lern_size;i++)
		memset(X[i], 0, p * sizeof(double)); //заполнение нулями

	memset(matrix_out_FL, 0, HL_size * sizeof(double));
	memset(old_matrix_out_FL, 0, HL_size * sizeof(double));
	memset(T1, 0, HL_size * sizeof(double));
	memset(G1, 0, HL_size * sizeof(double));
	out_net = 0;
	T2 = 0;
	G2 = 0;

	int offset = 0;
	for (int index = 0; index < lern_size; index++) {

		for (int i = 0; i < p; i++) {
			X[index][i] = sequence[offset + i];

		}
		D[index] = sequence[offset + p];
		offset++;
	}
}

void count_matrix_out_FL(int index) {
	for (int i = 0;i<HL_size;i++) {
		matrix_syn_transform_FL[i] = 0.0;
		for (int j = 0;j<p;j++) {
			matrix_syn_transform_FL[i] += (X[index][j] * matrix_first_layer[i][j] - T1[i]);
		}
	}
	for (int i = 0;i<HL_size;i++) {
		for (int j = 0;j<HL_size;j++) {
			matrix_syn_transform_FL[i] += ((old_matrix_out_FL[j] * matrix_context[i][j]) - T1[i]);
		}
	}
	for (int j = 0;j<HL_size;j++) {
		matrix_syn_transform_FL[j] += (T1[j]);
	}

	for (int j = 0;j<HL_size;j++) {
		matrix_out_FL[j] = soft_plus(matrix_syn_transform_FL[j]);
	}
}

void count_out_net() {
	matrix_syn_transform_SL = 0.0;
	for (int j = 0;j<HL_size;j++) {
		matrix_syn_transform_SL += ((matrix_out_FL[j] * matrix_sec_layer[j]) - T2);
	}
	out_net = soft_plus(matrix_syn_transform_SL);

}

double soft_plus(double x) {
	double return_value;

	return_value = log(1 + (double)exp(x));
	return (double)return_value;
}

double d_soft_plus(double x)
{
	double return_value;
	return_value = (double)exp(x) / (double)(1 + exp(x));


	return return_value;
}


void learn(int it) {
	int k = 0;
	double E;
	int l = lern_size - 2;
	do
	{
		k++;
		if (k >= it)
			break;
		E = 0;

		for (int i = 0; i<l; i++)
		{
			count_matrix_out_FL(i);
			count_out_net();
			countment_increment_matrix_sec_layer_T2(i);
			countment_increment_matrix_first_layer_T1(i);
			set_old();
		}


		for (int j = 0;j<HL_size;j++) {
			old_matrix_out_FL[j] = 0;
		}

		for (int i = 0; i<l; i++)
		{
			count_matrix_out_FL(i);
			count_out_net();
			E += (out_net - D[i])*(out_net - D[i]) / 2;
			set_old();
		}


		for (int j = 0;j<HL_size;j++) {
			old_matrix_out_FL[j] = 0;
		}

		cout  << "Итерация: " << k << setw(15) << "ошибка = " << E << endl;
	} while (E>e);

	cout << endl;
	cout << endl;
	print_all();
	
	for (int i = 0; i<lern_size; i++)
	{
		count_matrix_out_FL(i);
		count_out_net();
		cout << setw(10) << out_net << setw(10) << (int)D[i]<<endl;
		set_old();


	}
}

void countment_increment_matrix_sec_layer_T2(int index) {
	G2 = out_net - D[index];
	double temp = alpha*G2*d_soft_plus(matrix_syn_transform_SL);

	for (int j = 0; j<HL_size; j++)
		G1[j] = G2*d_soft_plus(matrix_syn_transform_SL)*matrix_sec_layer[j];

	for (int j = 0; j<HL_size; j++)
		matrix_sec_layer[j] -= temp*matrix_out_FL[j];

	T2 += temp;
}

void countment_increment_matrix_first_layer_T1(int index) {
	for (int j = 0; j<HL_size; j++)
		T1[j] += alpha*G1[j] * d_soft_plus(matrix_syn_transform_FL[j]);

	for (int i = 0;i<HL_size;i++) {
		for (int j = 0;j<p;j++) {
			matrix_first_layer[i][j] -= alpha*(G1[i])*d_soft_plus(matrix_syn_transform_FL[i])*X[index][j];
		}
	}
	for (int i = 0;i<HL_size;i++) {
		for (int j = 0;j<HL_size;j++) {
			matrix_context[i][j] -= alpha*G1[i] * d_soft_plus(matrix_syn_transform_FL[i])*old_matrix_out_FL[j];
		}
	}
	
}

void set_old() {
	for (int j = 0;j<HL_size;j++) {
		old_matrix_out_FL[j] = matrix_out_FL[j];
	}

}

void print_all() {
	cout << endl;
	cout << "Матрица весов на 1-м слое: \n";
	for (int i = 0; i<HL_size; i++) {
		for (int j = 0; j < p; j++)
			cout << setw(13)<< matrix_first_layer[i][j];
		cout << endl;
	}
	cout << endl;

	cout << "Матрица весов на 2-м слое: \n";
	for (int j = 0; j < HL_size; j++)
		cout << setw(13) << matrix_sec_layer[j];
	cout << endl;
	cout << endl;

	cout << "Матрица весов для контекстных нейронов со скрытого слоя:\n";
	for (int i = 0; i<HL_size; i++) {
		for (int j = 0; j < HL_size; j++)
			cout << setw(13) << matrix_context[i][j];
		cout << endl;
	}
	cout << endl;
	/*
	cout << "Матрица выхода первого слоя Y1:\n";
	for (int j = 0; j < m; j++)
		cout << Y1[j];
	cout << endl;
	cout << endl;

	cout << "выход самой сети out_net: ";
	cout << out_net;
	cout << endl;
	cout << endl;
	*/
	cout << "Результат синаптического преобразования на 1-ом слое:\n";
	for (int j = 0; j < HL_size; j++)
		cout << setw(13) << matrix_syn_transform_FL[j];
	cout << endl;
	cout << endl;

	cout << "Результат синаптического преобразования на 2-ом слое: ";
	cout << matrix_syn_transform_SL;
	cout << endl;
	cout << endl;
	/*
	cout << "матрица порогов на первом слое T1:\n";
	for (int j = 0; j < m; j++)
		cout << T1[j];
	cout << endl;
	cout << endl;
	cout << "порог на 2-ом слое T2: ";
	cout << T2;
	cout << endl;
	cout << endl;
	*/
}