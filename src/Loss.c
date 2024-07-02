#include "../include/Loss.h"
#include "../include/Types.h"

double mse_loss(Tensor *output, Tensor *target){
    double loss = 0;
    for(int i=0; i<output->size; i++){
        loss += (output->data[i]-target->data[i])*(output->data[i]-target->data[i]);
    }
    return loss/output->size;
}

Tensor* grad_mse_loss(Tensor *output, Tensor *target){
    Tensor *grad = create_tensor(output->ndim, output->shape);
    for(int i=0; i<output->size; i++){
        grad->data[i] = 2*(output->data[i]-target->data[i]);
    }
    return grad;
}

Loss* MSELoss(){
    Loss *l = (Loss*)malloc(sizeof(Loss));
    l->type = MSE_LOSS;
    l->loss = &mse_loss;
    l->grad_loss = &grad_mse_loss;
    return l;
}