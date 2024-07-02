#ifndef DATALOADER_H
#define DATALOADER_H

#include "Dataset.h"

typedef struct DataLoader{
    Dataset *dataset;
    int batch_size;
    int shuffle;
    int cursor;
    int *indices;
    Tensor (*get)(DataLoader *self);
    Tensor (*rewind)(DataLoader *self);
    void (*free)(DataLoader *self);
} DataLoader;

DataLoader* create_dataloader(Dataset *dataset, int batch_size, int shuffle);

Tensor* dl_get(DataLoader *self);

void dl_rewind(DataLoader *self);

void dl_free(DataLoader *self);

#endif
