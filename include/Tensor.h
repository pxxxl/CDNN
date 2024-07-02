#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdio.h>

typedef struct Tensor{
    int ndim;
    int *shape;
    int *strides;
    int size;
    double *data;
}Tensor;

// create a tensor with shape ndim and shape, not initialize the data
Tensor* create_tensor(int ndim, int *shape);

// create a tensor with shape ndim and shape, and initialize the data with the given data
Tensor* create_tensor_from_data(int ndim, int *shape, double *data);

// set a uni value to the tensor
void tensor_set_uni_value(Tensor *t, double value);

// free the tensor
void free_tensor(Tensor *t);

// serialize the tensor to a unsigned char array
unsigned char* serialize_tensor(Tensor *t, int *size);

// deserialize the unsigned char array to a tensor
Tensor* deserialize_tensor(unsigned char *data, int size);

// reshape the tensor
void reshape(Tensor *t, int ndim, int *shape);

// matrix multiplication, will return a new tensor. return NULL if shape mismatch
Tensor* matmul(Tensor *a, Tensor *b);

// matrix addition, will return a new tensor. return NULL if shape mismatch
Tensor* tensor_add(Tensor *a, Tensor *b);

// matrix subtraction, will return a new tensor. return NULL if shape mismatch
Tensor* tensor_minus(Tensor *a, Tensor *b);

// tensor copy
Tensor* tensor_copy(Tensor *a);

// tensor implace copy
void tensor_implace_copy(Tensor *a, Tensor *b);

// matrix scalar multiplication, will return a new tensor.
Tensor* tensor_mul_scalar(Tensor *a, double b);

// matrix scalar division, will return a new tensor.
Tensor* tensor_div_scalar(Tensor *a, double b);

// matrix transpose, will return a new tensor.
Tensor* transpose(Tensor *a);

// matrix negation, will return a new tensor.
Tensor* tensor_negate(Tensor *a);

// matrix sum
double tensor_sum(Tensor *a);

// matrix mean
double tensor_mean(Tensor *a);

// matrix max
double tensor_max(Tensor *a);

// matrix min
double tensor_min(Tensor *a);

// slice the tensor, return a new tensor, [start, end)
Tensor* slice(Tensor* a, int axis, int start, int end);

// merge the tensor, return a new tensor
Tensor* merge(Tensor* a, Tensor* b, int axis);

// -----------------------IO-----------------------
void print_tensor(Tensor *t);

Tensor* from_binary(const char* filename);

void to_binary(Tensor *t, const char* filename);

// -----------------------ERROR_CODES-----------------------

extern int tensor_error_code;

#define SHAPE_MISMATCH 1
#define DIMENSION_MISMATCH 2
#define SIZE_MISMATCH 3
#define FILE_NOT_FOUND 4

#endif