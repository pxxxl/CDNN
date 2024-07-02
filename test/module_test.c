#include "../include/Module.h"
#include "../include/Layer.h"
#include "../include/Loss.h"
#include "../include/Optim.h"
#include "../include/F.h"

int main(){
    Module* M = create_module();
    Layer* L1 = Linear(1, 2);
    Layer* R = ReLU(2);
    Layer* L2 = Linear(2, 1);
    Loss* l = MSELoss();
    Optimizer* o = SGD(0.01);

    add_node(M, L1, -1);
    add_node(M, R, 0);
    add_node(M, L2, 1);
    set_loss(M, l);
    set_optimizer(M, o);
    attach(M);

    Tensor* input = create_tensor(2, (int[]){1, 1});
    input->data[0] = 0;
    Tensor* target = create_tensor(2, (int[]){1, 1});
    target->data[0] = 1;

    for(int i=0; i<100; i++){
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
