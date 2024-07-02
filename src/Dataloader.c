#include "../include/Dataloader.h"

DataLoader* create_dataloader(Dataset *dataset, int batch_size, int shuffle){
    DataLoader *dataloader = (DataLoader *)malloc(sizeof(DataLoader));
    dataloader->dataset = dataset;
    if (batch_size <= 0){
        // Batch size must be greater than 0
        return NULL;
    }
    dataloader->batch_size = batch_size;
    dataloader->shuffle = shuffle;
    dataloader->cursor = 0;
    dataloader->indices = (int *)malloc(dataset->len(dataset) * sizeof(int));
    for (int i = 0; i < dataset->len(dataset); i++){
        dataloader->indices[i] = i;
    }
    if (shuffle){
        for (int i = 0; i < dataset->len(dataset); i++){
            int j = rand() % dataset->len(dataset);
            int temp = dataloader->indices[i];
            dataloader->indices[i] = dataloader->indices[j];
            dataloader->indices[j] = temp;
        }
    }
    return dataloader;
}

Tensor* dl_get(DataLoader *self){
    int iter_num = 0;
    int dataset_len = self->dataset->len(self->dataset);
    int batch_size = self->batch_size;
    if (dataset_len - self->cursor < batch_size){
        iter_num = dataset_len - self->cursor;
    } else {
        iter_num = batch_size;
    }
    Tensor* base_x;
    Tensor* base_y;
    self->dataset->get(self->dataset, self->indices[self->cursor], &base_x, &base_y);
    int desired_x_shape[2] = {1, base_x->size};
    int desired_y_shape[2] = {1, base_y->size};
    reshape(base_x, 2, desired_x_shape);
    reshape(base_y, 2, desired_y_shape);
    for (int i = 1; i < iter_num; i++){
        Tensor* x;
        Tensor* y;
        self->dataset->get(self->dataset, self->indices[self->cursor + i], &x, &y);
        reshape(x, 2, desired_x_shape);
        reshape(y, 2, desired_y_shape);
        base_x = merge(base_x, x, 1);
        base_y = merge(base_y, y, 1);
    }
    self->cursor += iter_num;
    return base_x;
}

void dl_rewind(DataLoader *self){
    self->cursor = 0;
    if (self->shuffle){
        for (int i = 0; i < self->dataset->len(self->dataset); i++){
            int j = rand() % self->dataset->len(self->dataset);
            int temp = self->indices[i];
            self->indices[i] = self->indices[j];
            self->indices[j] = temp;
        }
    }
}

void dl_free(DataLoader *self){
    free(self->indices);
    free(self);
}
