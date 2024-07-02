// implement the functions in F/F.h

#include "../include/Layer.h"
#include "../include/Types.h"
#include <math.h>
#include <stdlib.h>

Tensor* relu_forward(struct Layer* self, Tensor *t){
    Tensor *out = create_tensor(t->ndim, t->shape);
    for(int i=0; i<t->size; i++){
        out->data[i] = t->data[i]>0 ? t->data[i] : 0;
    }
    return out;
}

Tensor* relu_backward(struct Layer* self, Tensor *t){
    Tensor *out = create_tensor(t->ndim, t->shape);
    for(int i=0; i<t->size; i++){
        out->data[i] = t->data[i]>0?1:0;
    }
    return out;
}

Layer* ReLU(int input_size){
    Layer *l = (Layer*)malloc(sizeof(Layer));
    l->type = RELU;
    l->input_size = input_size;
    l->output_size = input_size;
    l->input = NULL;
    l->output = NULL;
    l->params = NULL;
    l->grads = NULL;
    l->forward = flow_funcs[RELU][FORWARD_FUNC];
    l->backward = flow_funcs[RELU][BACKWARD_FUNC];
    return l;
}