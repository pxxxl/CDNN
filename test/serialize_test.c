#include <stdio.h>
#include "../include/Tensor.h"

void t1(){
    printf("\nt1\n\n");
    Tensor *t = create_tensor_from_data(2, (int[]){2, 2}, (double[]){1, 2, 3, 4});
    int size = 0;
    unsigned char* data = serialize_tensor(t, &size);
    Tensor *t2 = deserialize_tensor(data, size);
    print_tensor(t);
    print_tensor(t2);
    free_tensor(t);
    free_tensor(t2);
    free(data);
}

int main(){
    printf("Hello World\n");
    t1();
    return 0;
}