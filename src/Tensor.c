/*
 * implement tensor in C
 */
#include "../include/Tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
int tensor_error_code;

// ----------------------CREATE TENSOR----------------------

Tensor* create_tensor(int ndim, int *shape){
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = (int*)malloc(sizeof(int)*ndim);
    t->strides = (int*)malloc(sizeof(int)*ndim);
    t->size = 1;
    for(int i=0; i<ndim; i++){
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }
    t->data = (double*)malloc(sizeof(double)*t->size);
    memset(t->data, 0, sizeof(double)*t->size);
    t->strides[ndim-1] = 1;
    for(int i=ndim-2; i>=0; i--){
        t->strides[i] = t->strides[i+1]*t->shape[i+1];
    }
    return t;
}

void free_tensor(Tensor *t){
    if (t == NULL){
        return;
    }
    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t);
}

Tensor* create_tensor_from_data(int ndim, int *shape, double *data){
    Tensor *t = create_tensor(ndim, shape);
    for(int i=0; i<t->size; i++){
        t->data[i] = data[i];
    }
    return t;
}

unsigned char* serialize_tensor(Tensor *t, int *size){
    *size = sizeof(int) + sizeof(int)*t->ndim + 2 * sizeof(int)*t->size + sizeof(double)*t->size;
    unsigned char *data = (unsigned char*)malloc(*size);
    unsigned char *cursor = data;
    memcpy(cursor, &t->ndim, sizeof(int));
    cursor += sizeof(int);
    memcpy(cursor, t->shape, sizeof(int)*t->ndim);
    cursor += sizeof(int)*t->ndim;
    memcpy(cursor, t->strides, sizeof(int)*t->ndim);
    cursor += sizeof(int)*t->ndim;
    memcpy(cursor, &t->size, sizeof(int));
    cursor += sizeof(int);
    memcpy(cursor, t->data, sizeof(double)*t->size);
    return data;
}

Tensor* deserialize_tensor(unsigned char *data, int size){
    unsigned char *cursor = data;
    int ndim;
    memcpy(&ndim, cursor, sizeof(int));
    cursor += sizeof(int);
    int *shape = (int*)malloc(sizeof(int)*ndim);
    memcpy(shape, cursor, sizeof(int)*ndim);
    cursor += sizeof(int)*ndim;
    int *strides = (int*)malloc(sizeof(int)*ndim);
    memcpy(strides, cursor, sizeof(int)*ndim);
    cursor += sizeof(int)*ndim;
    int tensor_size;
    memcpy(&tensor_size, cursor, sizeof(int));
    cursor += sizeof(int);
    double *tensor_data = (double*)malloc(sizeof(double)*tensor_size);
    memcpy(tensor_data, cursor, sizeof(double)*tensor_size);
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    t->ndim = ndim;
    t->shape = shape;
    t->strides = strides;
    t->size = tensor_size;
    t->data = tensor_data;
    return t;
}

void reshape(Tensor *t, int ndim, int *shape){
    int size = 1;
    for(int i=0; i<ndim; i++){
        size *= shape[i];
    }
    if(size != t->size){
        tensor_error_code = SIZE_MISMATCH;
        return NULL;
    }
    t->ndim = ndim;
    free(t->shape);
    free(t->strides);
    t->shape = (int*)malloc(sizeof(int)*ndim);
    t->strides = (int*)malloc(sizeof(int)*ndim);
    for(int i=0; i<ndim; i++){
        t->shape[i] = shape[i];
    }
    t->strides[ndim-1] = 1;
    for(int i=ndim-2; i>=0; i--){
        t->strides[i] = t->strides[i+1]*t->shape[i+1];
    }
}

// ----------------------CREATE TENSOR----------------------

// ----------------------INDEX----------------------

double tensor_at(Tensor *t, int *index){
    int i = 0;
    for(int j=0; j<t->ndim; j++){
        i += index[j]*t->strides[j];
    }
    return t->data[i];
}

void tensor_set(Tensor *t, int *index, double value){
    int i = 0;
    for(int j=0; j<t->ndim; j++){
        i += index[j]*t->strides[j];
    }
    t->data[i] = value;
}

void tensor_set_uni_value(Tensor *t, double value){
    for(int i=0; i<t->size; i++){
        t->data[i] = value;
    }
}

// ----------------------INDEX----------------------


// ----------------------DOUBLE MATRIX OPERATIONS----------------------

Tensor* matmul(Tensor *a, Tensor *b){
    if(a->shape[1] != b->shape[0]){
        tensor_error_code = SHAPE_MISMATCH;
        return NULL;
    }
    int shape[2] = {a->shape[0], b->shape[1]};
    Tensor *c = create_tensor(2, shape);
    for(int i=0; i<a->shape[0]; i++){
        for(int j=0; j<b->shape[1]; j++){
            c->data[i*c->strides[0]+j] = 0;
            for(int k=0; k<a->shape[1]; k++){
                c->data[i*c->strides[0]+j] += a->data[i*a->strides[0]+k]*b->data[k*b->strides[0]+j];
            }
        }
    }
    return c;
}

Tensor* tensor_add(Tensor *a, Tensor *b){
    if (a->ndim != b->ndim){
        tensor_error_code = DIMENSION_MISMATCH;
        return NULL;
    }
    
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            tensor_error_code = SHAPE_MISMATCH;
            return NULL;
        }
    }
    Tensor *c = create_tensor(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] + b->data[i];
    }
    return c;
}

Tensor* tensor_minus(Tensor *a, Tensor *b){
    if (a->ndim != b->ndim){
        tensor_error_code = DIMENSION_MISMATCH;
        return NULL;
    }
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            tensor_error_code = SHAPE_MISMATCH;
            return NULL;
        }
    }
    Tensor *c = create_tensor(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] - b->data[i];
    }
    return c;
}

Tensor* tensor_mul_scalar(Tensor *a, double b){
    Tensor *c = create_tensor(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] * b;
    }
    return c;
}

Tensor* tensor_div_scalar(Tensor *a, double b){
    Tensor *c = create_tensor(a->ndim, a->shape);
    for (int i = 0; i < a->size; i++){
        c->data[i] = a->data[i] / b;
    }
    return c;
}

// ----------------------DOUBLE MATRIX OPERATIONS----------------------

// -----------------------SINGLE MATRIX OPERATIONS-----------------------

Tensor* transpose(Tensor *a){
    int *shape = (int*)malloc(sizeof(int)*a->ndim);
    shape[0] = a->shape[1];
    shape[1] = a->shape[0];
    Tensor *b = create_tensor(2, shape);
    for(int i=0; i<a->shape[0]; i++){
        for(int j=0; j<a->shape[1]; j++){
            b->data[j*b->strides[0]+i] = a->data[i*a->strides[0]+j];
        }
    }
    free(shape);
    return b;
}

Tensor *tensor_copy(Tensor *a){
    Tensor *b = create_tensor(a->ndim, a->shape);
    for(int i=0; i<a->size; i++){
        b->data[i] = a->data[i];
    }
    return b;
}

void tensor_implace_copy(Tensor *a, Tensor *b){
    if (a->size != b->size){
        tensor_error_code = SIZE_MISMATCH;
        return;
    }
    for(int i=0; i<a->size; i++){
        a->data[i] = b->data[i];
    }
}

Tensor* tensor_negate(Tensor *a){
    Tensor *b = create_tensor(2, a->shape);
    for(int i=0; i<a->shape[0]; i++){
        for(int j=0; j<a->shape[1]; j++){
            b->data[i*b->strides[0]+j] = -a->data[i*a->strides[0]+j];
        }
    }
    return b;
}

// -----------------------SINGLE MATRIX OPERATIONS-----------------------

// -----------------------MATRIX REDUCTION-----------------------

double tensor_sum(Tensor *a){
    double sum = 0;
    for(int i=0; i<a->size; i++){
        sum += a->data[i];
    }
    return sum;
}

double tensor_mean(Tensor *a){
    return tensor_sum(a) / a->size;
}

double tensor_max(Tensor *a){
    double max = a->data[0];
    for(int i=1; i<a->size; i++){
        if(a->data[i] > max){
            max = a->data[i];
        }
    }
    return max;
}

double tensor_min(Tensor *a){
    double min = a->data[0];
    for(int i=1; i<a->size; i++){
        if(a->data[i] < min){
            min = a->data[i];
        }
    }
    return min;
}

// -----------------------MATRIX REDUCTION-----------------------

// -----------------------MATRIX SLICING AND MERGE-----------------------
void recursive_slice_value_set(Tensor *t, Tensor *s, int *t_index, int *s_index, int depth, int axis, int start, int end){
    if (depth == t->ndim - 1){
        if (axis == depth){
            for (int i = start; i < end; i++){
                t_index[depth] = i - start;
                s_index[depth] = i;
                tensor_set(t, t_index, tensor_at(s, s_index));
            }
        } else {
            for (int i = 0; i < t->shape[depth]; i++){
                t_index[depth] = i;
                s_index[depth] = i;
                tensor_set(t, t_index, tensor_at(s, s_index));
            }
        }
    } else {
        if (axis == depth){
            for (int i = start; i < end; i++){
                t_index[depth] = i - start;
                s_index[depth] = i;
                recursive_slice_value_set(t, s, t_index, s_index, depth + 1, axis, start, end);
            }
        } else {
            for (int i = 0; i < t->shape[depth]; i++){
                t_index[depth] = i;
                s_index[depth] = i;
                recursive_slice_value_set(t, s, t_index, s_index, depth + 1, axis, start, end);
            }
        }
    }
}

Tensor *slice(Tensor* a, int axis, int start, int end){
    int* shape = (int*)malloc(sizeof(int)*a->ndim);
    for(int i=0; i<a->ndim; i++){
        shape[i] = a->shape[i];
    }
    shape[axis] = end - start;
    Tensor *s = create_tensor(a->ndim, shape);

    int *target_index = (int*)malloc(sizeof(int) * a->ndim);
    int *source_index = (int*)malloc(sizeof(int) * a->ndim);
    recursive_slice_value_set(s, a, target_index, source_index, 0, axis, start, end);
    free(target_index);
    free(source_index);
    free(shape);
    return s;
}

Tensor *recursive_merge_value_set(Tensor *t, Tensor *s, int *t_index, int *s_index, int depth, int axis, int padding){
    if (depth == t->ndim - 1){
        if (axis == depth){
            for (int i = 0; i < s->shape[depth]; i++){
                t_index[depth] = i + padding;
                s_index[depth] = i;
                tensor_set(t, t_index, tensor_at(s, s_index));
            }
        } else {
            for (int i = 0; i < s->shape[depth]; i++){
                t_index[depth] = i;
                s_index[depth] = i;
                tensor_set(t, t_index, tensor_at(s, s_index));
            }
        }
    } else {
        if (axis == depth){
            for (int i = 0; i < s->shape[depth]; i++){
                t_index[depth] = i + padding;
                s_index[depth] = i;
                recursive_merge_value_set(t, s, t_index, s_index, depth + 1, axis, padding);
            }
        } else {
            for (int i = 0; i < s->shape[depth]; i++){
                t_index[depth] = i;
                s_index[depth] = i;
                recursive_merge_value_set(t, s, t_index, s_index, depth + 1, axis, padding);
            }
        }
    }
    return t;
}

Tensor* merge(Tensor* a, Tensor* b, int axis){
    if(a->ndim != b->ndim){
        tensor_error_code = SHAPE_MISMATCH;
        return NULL;
    }
    for(int i=0; i<a->ndim; i++){
        if(i != axis && a->shape[i] != b->shape[i]){
            tensor_error_code = SHAPE_MISMATCH;
            return NULL;
        }
    }
    int* shape = (int*)malloc(sizeof(int)*a->ndim);
    for(int i=0; i<a->ndim; i++){
        shape[i] = a->shape[i];
    }
    shape[axis] = a->shape[axis] + b->shape[axis];
    Tensor *c = create_tensor(a->ndim, shape);
    int *target_index = (int*)malloc(sizeof(int) * a->ndim);
    int *source_index = (int*)malloc(sizeof(int) * a->ndim);
    recursive_merge_value_set(c, a, target_index, source_index, 0, axis, 0);
    recursive_merge_value_set(c, b, target_index, source_index, 0, axis, a->shape[axis]);
    free(target_index);
    free(source_index);
    free(shape);
    return c;
}

void recursive_print_tensor(double* data, int* shape, int ndim, double* cursor, int depth){
    if (depth == ndim - 1){
        printf("[");
        for (int i = 0; i < shape[ndim - 1]; i++){
            printf("%.2lf ", *cursor);
            cursor++;
        }
        printf("]\n");
    } else {
        printf("[\n");
        for (int i = 0; i < shape[depth]; i++){
            recursive_print_tensor(data, shape, ndim, cursor + i * shape[depth + 1], depth + 1);
        }
        printf("]\n");
    }
}

void print_tensor(Tensor *t){
    double* cursor = t->data;
    recursive_print_tensor(t->data, t->shape, t->ndim, cursor, 0);
}

Tensor* from_binary(const char* filename){
    FILE *fp;
    fp = fopen(filename, "rb");
    if (fp == NULL){
        tensor_error_code = FILE_NOT_FOUND;
        return NULL;
    }
    int ndim;
    fread(&ndim, sizeof(int), 1, fp);
    int* shape = (int*)malloc(sizeof(int) * ndim);
    fread(shape, sizeof(int), ndim, fp);
    Tensor* t = create_tensor(ndim, shape);
    fread(t->data, sizeof(double), t->size, fp);
    return t;
}

void to_binary(Tensor *t, const char* filename){
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL){
        tensor_error_code = FILE_NOT_FOUND;
        return;
    }
    fwrite(&t->ndim, sizeof(int), 1, fp);
    fwrite(t->shape, sizeof(int), t->ndim, fp);
    fwrite(t->data, sizeof(double), t->size, fp);
    fclose(fp);
}