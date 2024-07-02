#include "../include/Module.h"
#include "../include/Layer.h"
#include "../include/Loss.h"
#include "../include/Optim.h"
#include "../include/F.h"
#include <stdio.h>
#include <string.h>

typedef struct IO{
    Tensor* input[1000];
    Tensor* target[1000];
} IO;

IO* read_io(){
    char* dir = "/home/whisper/Programs/C/DeepLearning/datasets/linear/";
    IO* io = (IO*)malloc(sizeof(IO));
    for (int i = 0; i < 1000; i++){
        char* input_dir = (char*)malloc(100);
        char* target_dir = (char*)malloc(100);
        sprintf(input_dir, "%s%d.tsr", dir, i);
        sprintf(target_dir, "%s%d_y.tsr", dir, i);
        io->input[i] = from_binary(input_dir);
        io->target[i] = from_binary(target_dir);
        free(input_dir);
        free(target_dir);
    }
    return io;
}

int main(){
    Module* M = create_module();
    Layer* L1 = Linear(2, 4);
    Layer* R = ReLU(4);
    Layer* L2 = Linear(4, 1);
    Loss* l = MSELoss();
    Optimizer* o = SGD(0.01);

    add_node(M, L1, -1);
    add_node(M, R, 0);
    add_node(M, L2, 1);
    set_loss(M, l);
    set_optimizer(M, o);
    attach(M);

    IO* io = read_io();
    Tensor* input = NULL;
    Tensor* target = NULL;

    for(int i=0; i<1000; i++){
        input = io->input[i];
        target = io->target[i];
        Tensor* out = forward(M, input);
        backward(M, target);
        step(M);
        double loss = M->loss->loss(out, target);
        printf("loss: %f\n", loss);
        free_tensor(out);
    }
    free_tensor(target);
    free_module(M);
    return 0;
}
