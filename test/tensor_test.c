#include <stdio.h>
#include "../include/Tensor.h"

void t1(){
    printf("\nt1\n\n");
    Tensor *t = create_tensor_from_data(2, (int[]){2, 2}, (double[]){1, 2, 3, 4});
    Tensor *t2 = tensor_negate(t);
    print_tensor(t2);
    free_tensor(t);
    free_tensor(t2);
}

void t2(){
    printf("\nt2\n\n");
    Tensor *t = create_tensor_from_data(2, (int[]){2, 2}, (double[]){-1, -2, -3, -4});
    Tensor *t2 = tensor_mul_scalar(t, 2);
    printf("raw\n");
    print_tensor(t);
    printf("mul 2\n");
    print_tensor(t2);
    printf("t1 * t2\n");
    Tensor *t3 = matmul(t, t2);
    print_tensor(t3);
    free_tensor(t);
    free_tensor(t2);
    free_tensor(t3);
}

void t3(){
    printf("\nt3\n\n");
    Tensor *t = create_tensor_from_data(2, (int[]){2, 2}, (double[]){-1, -2, -3, -4});
    Tensor *t2 = slice(t, 1, 0, 1);
    printf("raw\n");
    print_tensor(t);
    printf("slice 0, 0, 1\n");
    print_tensor(t2);
    Tensor* t3 = merge(t, t2, 1);
    printf("merge 0\n");
    print_tensor(t3);
    free_tensor(t);
    free_tensor(t2);
    free_tensor(t3);
}

void t4(){
    printf("\nt4\n\n");
    Tensor *t = create_tensor_from_data(2, (int[]){2, 2}, (double[]){-1, -2, -3, -4});
    to_binary(t, "/home/whisper/Programs/C/DeepLearning/cache/tensor1");
    Tensor *t2 = from_binary("/home/whisper/Programs/C/DeepLearning/cache/tensor1");
    printf("raw\n");
    print_tensor(t);
    printf("from binary\n");
    print_tensor(t2);
    free_tensor(t);
    free_tensor(t2);
}

void t5(){
    Tensor* t = create_tensor(2, (int[]){2, 2});
    t->data[0] = 1;
    t->data[1] = 2;
    t->data[2] = 3;
    t->data[3] = 4;
    Tensor *b = slice(t, 0, 1, 2);
    print_tensor(b);
    free_tensor(t);
    free_tensor(b);
}

int main(){
    printf("Hello World\n");
    //t1();
    //t2();
    //t3();
    //t4();
    t5();
    return 0;
}