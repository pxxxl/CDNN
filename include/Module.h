#ifndef MODULE_H
#define MODULE_H
/*
 * For every DNN, we will have a graph structure that will store the layers and 
 the connections between them.
 * 1. add node for the network(currently only supports sequential networks)
 * 2. set loss function and optimizer
 * 3. begin training, for each round, first will forward the input to get the loss
 * 4. then use backward to get the gradients(the grad is saved in Layer, refering to dL/dw)
 * 5. then use the optimizer to update the weights(step)
 */

#include "Layer.h"
#include "Loss.h"
#include "Optim.h"

#define MAX_LAYER_NUM 100
#define MAX_CHILDREN_NUM 1
#define MAX_PARENTS_NUM 1

typedef struct Node{
    int id;
    int n_children;
    int children[MAX_CHILDREN_NUM];
    int n_parents;
    int parents[MAX_PARENTS_NUM];
    Layer *layer;
}Node;

typedef struct Module{
    int n_nodes;
    int loss_type;
    int optim_type;
    Node *nodes[MAX_LAYER_NUM];
    Tensor *output;
    Loss *loss;
    Optimizer* optim;
}Module;

Module *create_module();

// add a node to the graph, the parent_id is the id of the parent node. The first node is automately the head node(regardless of the input parent id). 
// The function will return the id of the new-added node.
int add_node(Module *g, Layer *layer, int parent_id);

// set the loss function for the graph
void set_loss(Module *g, Loss *loss);

// set the optimizer for the graph, while do not attach the optimizer to the layers
void set_optimizer(Module *g, Optimizer *optim);

// attach the optimizer to the layers
void attach(Module *g);

// forward the input through the graph, return the loss
Tensor *forward(Module *g, Tensor *input);

// backward the loss through the graph
void backward(Module *g, Tensor *target);

// update the weights of the graph
void step(Module *g);

// free the graph
void free_module(Module *g);

int dump_module(Module *g, const char* path);

Module *load_module(const char* path);

#endif