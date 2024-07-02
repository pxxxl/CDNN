#include "../include/Dataset.h"

int len_ExampleLinearDataset(Dataset *self){
    return 1000;
}

void get_ExampleLinearDataset(Dataset *self, int idx, Tensor **x, Tensor **y){
    char* x_path = "../datasets/linear/";
    char* y_path = "../datasets/linear/";
    char x_filename[100];
    char y_filename[100];
    sprintf(x_filename, "%s%d.tsr", x_path, idx);
    sprintf(y_filename, "%s%d_y.tsr", y_path, idx);
    // open the x_file, and read the first 8 bytes as x1
    FILE *x_file = fopen(x_filename, "rb");
    if (x_file == NULL){
        printf("Error opening file %s\n", x_filename);
        exit(1);
    }
    double x1;
    fread(&x1, sizeof(double), 1, x_file);
    //read the second 8 bytes as x2
    double x2;
    fread(&x2, sizeof(double), 1, x_file);
    fclose(x_file);
    // open the y_file, and read the first 8 bytes as y
    FILE *y_file = fopen(y_filename, "rb");
    if (y_file == NULL){
        printf("Error opening file %s\n", y_filename);
        exit(1);
    }
    double y_1;
    fread(&y_1, sizeof(double), 1, y_file);
    fclose(y_file);
    // create the x tensor
    int x_shape[2] = {2};
    Tensor *x_tensor = create_tensor(1, x_shape);
    x_tensor->data[0] = x1;
    x_tensor->data[1] = x2;
    // create the y tensor
    int y_shape[1] = {1};
    Tensor *y_tensor = create_tensor(1, y_shape);
    y_tensor->data[0] = y_1;
    *x = x_tensor;
    *y = y_tensor;
}

void free_ExampleLinearDataset(Dataset *self){
    free(self);
}

Dataset* ExampleLinearDataset(){
    Dataset *dataset = (Dataset *)malloc(sizeof(Dataset));
    dataset->data = NULL;
    dataset->len = len_ExampleLinearDataset;
    dataset->get = get_ExampleLinearDataset;
    dataset->free = free_ExampleLinearDataset;
    return dataset;
}