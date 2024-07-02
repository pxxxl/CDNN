#ifndef F_H
#define F_H

#include "Tensor.h"
#include "Layer.h"

Layer* ReLU(int input_size);
Tensor* relu_forward(struct Layer* self, Tensor *t);
Tensor* relu_backward(struct Layer* self, Tensor *t);

#endif