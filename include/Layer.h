#ifndef LAYER_H
#define LAYER_H

/*
 * The layer will store the params and the grad of the loss to the weights
 * each time call forward, it will receive the input, and return the output of the layer
 * each time call backward, it will receive the grad of the loss to the output, and return the grad of the loss to the input, and save the grad to the weights
 */

#include "Tensor.h"



#define FORWARD_FUNC 0
#define BACKWARD_FUNC 1

extern Tensor *(*flow_funcs[][2])(Layer*, Tensor*);

typedef struct Layer{
    int type;
    int input_size;
    int output_size;
    Tensor *params;
    Tensor *input;
    Tensor *output;
    Tensor *grads;
    Tensor *(*forward)(struct Layer*, Tensor*);
    Tensor *(*backward)(struct Layer*, Tensor*);
} Layer;

Layer* Linear(int input_size, int output_size);

unsigned char* serialize_layer(Layer *l, int *size);

Layer* deserialize_layer(unsigned char *data, int size);

void free_layer(Layer *l);

#endif