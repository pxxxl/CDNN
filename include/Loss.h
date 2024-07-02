#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"

typedef struct Loss{
    int type;
    double (*loss)(Tensor *output, Tensor *target);
    Tensor* (*grad_loss)(Tensor *output, Tensor *target);
}Loss;

Loss* MSELoss();

#endif