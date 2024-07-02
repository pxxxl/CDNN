#include "../include/Optim.h"
#include "../include/Types.h"
#include <memory.h>

void sgd_step(Optimizer* self){
    for(int i = 0; i < self->n_params; i++){
        Tensor *param = self->params[i];
        Tensor *grad = self->param_grads[i];
        if (param == NULL || grad == NULL){
            continue;
        }
        Tensor *mul_grad = tensor_mul_scalar(grad, self->lr);
        Tensor *new_param = tensor_minus(param, mul_grad);
        tensor_implace_copy(param, new_param);
        free_tensor(mul_grad);
        free_tensor(new_param);
    }
}

Optimizer* SGD(double lr){
    Optimizer *opt = (Optimizer*)malloc(sizeof(Optimizer));
    opt->type = SGD_OPTIM;
    opt->lr = lr;
    opt->data_len = 0;
    opt->data = NULL;
    opt->n_params = 0;
    opt->params = NULL;
    opt->param_grads = NULL;
    opt->step = &sgd_step;
    return opt;
}


unsigned char* serialize_optimizer(Optimizer *o, int *size){
    int data_len = 2 * sizeof(int) + sizeof(double) + o->data_len;
    unsigned char *data = (unsigned char*)malloc(data_len);
    unsigned char *cursor = data;
    memcpy(cursor, &o->type, sizeof(int));
    cursor += sizeof(int);
    memcpy(cursor, &o->lr, sizeof(double));
    cursor += sizeof(double);
    memcpy(cursor, &o->data_len, sizeof(int));
    cursor += sizeof(int);
    memcpy(cursor, o->data, o->data_len);
    cursor += o->data_len;
    *size = data_len;
    return data;
}

Optimizer* deserialize_optimizer(unsigned char *data, int size){
    Optimizer *opt = (Optimizer*)malloc(sizeof(Optimizer));
    unsigned char *cursor = data;
    memcpy(&opt->type, cursor, sizeof(int));
    cursor += sizeof(int);
    memcpy(&opt->lr, cursor, sizeof(double));
    cursor += sizeof(double);
    memcpy(&opt->data_len, cursor, sizeof(int));
    cursor += sizeof(int);
    opt->data = (void*)malloc(opt->data_len);
    memcpy(opt->data, cursor, opt->data_len);
    cursor += opt->data_len;
    opt->n_params = 0;
    opt->params = NULL;
    opt->param_grads = NULL;
    opt->step = NULL;
    switch(opt->type){
        case SGD_OPTIM:
            opt->step = &sgd_step;
            break;
    }
    return opt;
}




