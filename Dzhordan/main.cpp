#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "RNN.h"

int main()
{
	setlocale(LC_ALL, "Russian");

	int p = 8;
	int m = 8;
	double alpha = 0.001;
	double e = 0.0001;
	int it = 8;
	cout << "������� ������ ����: ";
	cin >> p;
	cout << endl;

	cout << "������� ������ �������� ����: ";
	cin >> m;
	cout << endl;

	cout << "������� ������������ ���������� ����� ��������: ";
	cin >> it;
	cout << endl;

	cout << "������� ����������� ���������� �������������������� ������: ";
	cin >> e;
	cout << endl;

	cout << "������� ��� ��������: ";
	cin >> alpha;
	cout << endl;

	choose_sequence(p, m, e, alpha);
	init_waight_matrix();
	learn(it);
	system("pause");

	return 0;
}