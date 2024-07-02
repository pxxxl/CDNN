#include "../include/Layer.h"
#include "../include/F.h"

Tensor *(*flow_funcs[][2])(Layer*, Tensor*) = {
    {&linear_forward, &linear_backward},
    {&relu_forward, &relu_backward}
};

void free_layer(Layer *l){
    free_tensor(l->params);
    free_tensor(l->input);
    free_tensor(l->output);
    free_tensor(l->grads);
    free(l);
}

unsigned char* serialize_layer(Layer *l, int *size){
    int type = l->type;
    int input_size = l->input_size;
    int output_size = l->output_size;
    int params_size = 0;
    unsigned char *params_data = serialize_tensor(l->params, &params_size);
    int input_size = 0;
    unsigned char *input_data = serialize_tensor(l->input, &input_size);
    int output_size = 0;
    unsigned char *output_data = serialize_tensor(l->output, &output_size);
    int grads_size = 0;
    unsigned char *grads_data = serialize_tensor(l->grads, &grads_size);
    *size = sizeof(int) * 7 + params_size + input_size + output_size + grads_size;
    unsigned char *data = (unsigned char*)malloc(*size);
    unsigned char *p = data;
    memcpy(p, &type, sizeof(int));
    p += sizeof(int);
    memcpy(p, &input_size, sizeof(int));
    p += sizeof(int);
    memcpy(p, &output_size, sizeof(int));
    p += sizeof(int);
    memcpy(p, &params_size, sizeof(int));
    p += sizeof(int);
    memcpy(p, params_data, params_size);
    p += params_size;
    memcpy(p, &input_size, sizeof(int));
    p += sizeof(int);
    memcpy(p, input_data, input_size);
    p += input_size;
    memcpy(p, &output_size, sizeof(int));
    p += sizeof(int);
    memcpy(p, output_data, output_size);
    p += output_size;
    memcpy(p, &grads_size, sizeof(int));
    p += sizeof(int);
    memcpy(p, grads_data, grads_size);
    p += grads_size;
    free(params_data);
    free(input_data);
    free(output_data);
    free(grads_data);
    return data;
}

Layer* deserialize_layer(unsigned char *data, int size){
    Layer *l = (Layer*)malloc(sizeof(Layer));
    unsigned char *p = data;
    memcpy(&l->type, p, sizeof(int));
    p += sizeof(int);
    memcpy(&l->input_size, p, sizeof(int));
    p += sizeof(int);
    memcpy(&l->output_size, p, sizeof(int));
    p += sizeof(int);
    int params_size = 0;
    memcpy(&params_size, p, sizeof(int));
    p += sizeof(int);
    l->params = deserialize_tensor(p, params_size);
    p += params_size;
    int input_size = 0;
    memcpy(&input_size, p, sizeof(int));
    p += sizeof(int);
    l->input = deserialize_tensor(p, input_size);
    p += input_size;
    int output_size = 0;
    memcpy(&output_size, p, sizeof(int));
    p += sizeof(int);
    l->output = deserialize_tensor(p, output_size);
    p += output_size;
    int grads_size = 0;
    memcpy(&grads_size, p, sizeof(int));
    p += sizeof(int);
    l->grads = deserialize_tensor(p, grads_size);
    p += grads_size;
    l->forward = flow_funcs[l->type][FORWARD_FUNC];
    l->backward = flow_funcs[l->type][BACKWARD_FUNC];
    return l;
}

Tensor *linear_forward(struct Layer *self, Tensor *t){
    int axis_0_len = self->params->shape[0] - 1;
    Tensor *w = slice(self->params, 0, 0, axis_0_len);
    Tensor *b = slice(self->params, 0, axis_0_len, axis_0_len + 1);
    Tensor *out = matmul(t, w);
    Tensor *out_save = tensor_copy(out);
    for(int i=0; i<out->size; i++){
        out->data[i] += b->data[0];
    }
    free_tensor(w);
    free_tensor(b);
    self->input = t;
    self->output = out_save;
    return out;
}

Tensor *linear_backward(struct Layer *self, Tensor *t){
    Tensor *w = slice(self->params, 0, 0, self->input_size);
    Tensor *b = slice(self->params, 0, self->input_size, self->input_size + 1);
    Tensor *wT = transpose(w);
    Tensor *grads = matmul(t, wT);
    Tensor *grads_w = matmul(transpose(self->input), t);
    Tensor *grads_b = tensor_copy(t);
    Tensor *grads_params = merge(grads_w, grads_b, 0);
    tensor_implace_copy(self->grads, grads_params);
    free_tensor(w);
    free_tensor(b);
    free_tensor(wT);
    free_tensor(grads_w);
    free_tensor(grads_b);
    free_tensor(grads_params);
    return grads;
}

Layer *Linear(int input_size, int output_size){
    Layer *l = (Layer*)malloc(sizeof(Layer));
    l->type = LINEAR;
    l->input_size = input_size;
    l->output_size = output_size;
    l->input = NULL;
    l->output = NULL;
    int shape[2] = {input_size + 1, output_size};
    l->grads = create_tensor(2, shape);
    l->params = create_tensor(2, shape);
    tensor_set_uni_value(l->params, 0.0);
    l->forward = flow_funcs[LINEAR][FORWARD_FUNC];
    l->backward = flow_funcs[LINEAR][BACKWARD_FUNC];
    return l;
}
