#ifndef RNN_H_INCLUDED
#define RNN_H_INCLUDED
#include <iostream>
using namespace std;

void count_matrix_out_FL(int index);
void choose_sequence(int x, int y, double _e, double _A);
void count_out_net();
void init_waight_matrix();
void set_old();
void print_all();
double soft_plus(double x);
double d_soft_plus(double x);
void learn(int it);
void countment_increment_matrix_sec_layer_T2(int index);
void countment_increment_matrix_first_layer_T1(int index);

#endif // RNN_H_INCLUDED
