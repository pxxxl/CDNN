#ifndef OPTIM_H
#define OPTIM_H

#include "Tensor.h"

typedef struct Optimizer{
    int type;
    double lr;
    int data_len;
    void *data;
    int n_params;
    Tensor** params;
    Tensor** param_grads;
    void (*step)(struct Optimizer*);
}Optimizer;

unsigned char* serialize_optimizer(Optimizer *o, int *size);

Optimizer* deserialize_optimizer(unsigned char *data, int size);

Optimizer* SGD(double lr);

#endif