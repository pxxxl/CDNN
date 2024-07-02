#ifndef DATASET_H
#define DATASET_H

#include "../include/Tensor.h"

typedef struct Dataset{
    void *data;
    int (*len)(struct Dataset *);
    void (*get)(struct Dataset *, int, Tensor **, Tensor **);
    void (*free)(struct Dataset *);
} Dataset;

Dataset* ExampleLinearDataset();

#endif