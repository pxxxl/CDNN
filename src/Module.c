#include "../include/Module.h"
#include "../include/sqlite3.h"
#include "../include/Sql.h"
#include "../include/Types.h"
Module *create_module(){
    Module *g = (Module*)malloc(sizeof(Module));
    g->n_nodes = 0;
    g->loss = NULL;
    g->optim = NULL;
    return g;
}

int add_node(Module *g, Layer *layer, int parent_id){
    if(g->n_nodes >= MAX_LAYER_NUM){
        return -1;
    }
    if(parent_id >= g->n_nodes){
        return -1;
    }

    Node *node = (Node*)malloc(sizeof(Node));
    node->id = g->n_nodes;
    node->layer = layer;
    node->n_children = 0;
    node->n_parents = 0;
    if(g->n_nodes == 0){
        g->nodes[g->n_nodes] = node;
        g->n_nodes++;
        return 0;
    }
    if(g->nodes[parent_id]->n_children >= MAX_CHILDREN_NUM){
        free(node);
        return -1;
    }
    g->nodes[parent_id]->children[g->nodes[parent_id]->n_children] = node->id;
    g->nodes[parent_id]->n_children++;
    node->parents[0] = parent_id;
    node->n_parents = 1;
    g->nodes[g->n_nodes] = node;
    g->n_nodes++;
    return node->id;
}

void set_loss(Module *g, Loss *loss){
    g->loss = loss;
    g->loss_type = loss->type;
}

void set_optimizer(Module *g, Optimizer *optim){
    g->optim = optim;
    g->optim_type = optim->type;
}

void attach(Module *g){
    g->optim->params = (Tensor**)malloc(sizeof(Tensor*)*g->n_nodes);
    g->optim->param_grads = (Tensor**)malloc(sizeof(Tensor*)*g->n_nodes);
    g->optim->n_params = g->n_nodes;
    for(int i=0; i<g->n_nodes; i++){
        g->optim->params[i] = g->nodes[i]->layer->params;
        g->optim->param_grads[i] = g->nodes[i]->layer->grads;
    }
}

Tensor *forward(Module *g, Tensor *input){
    Tensor *out = input;
    for(int i=0; i<g->n_nodes; i++){
        out = g->nodes[i]->layer->forward(g->nodes[i]->layer, out);
    }
    Tensor* out_save = tensor_copy(out);
    g->output = out_save;
    return out;
}

void backward(Module *g, Tensor *target){
    Tensor *grads = g->loss->grad_loss(g->output, target);
    Tensor *legacy_grads = grads;
    for(int i=g->n_nodes-1; i>=0; i--){
        grads = g->nodes[i]->layer->backward(g->nodes[i]->layer, grads);
        free_tensor(legacy_grads);
        legacy_grads = grads;
    }
    free_tensor(grads);
    free_tensor(g->output);
    g->output = NULL;
}

void step(Module *g){
    g->optim->step(g->optim);
}

void free_module(Module *g){
    for(int i=0; i<g->n_nodes; i++){
        free_layer(g->nodes[i]->layer);
        free(g->nodes[i]);
    }
    free(g->optim->params);
    free(g->optim->param_grads);
    free(g->optim);
    free(g->loss);
    free(g->output);   
    free(g);
}

int dump_module(Module *g, const char* path){
    // use sqlite3 to serialize the graph
    char* err_msg = NULL;
    int rc;
    sqlite3 *db;
    rc = sqlite3_open(path, &db);
    if(rc){
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }

    // create tables
    const char* create_table_sql[] = {CLEAR_TABLES, CREATE_NODE_TABLE, CREATE_LINK_TABLE, CREATE_ETC_TABLE};
    int create_table_sql_len = 4;
    for (int i = 0; i < create_table_sql_len; i++){
        rc = sqlite3_exec(db, create_table_sql[i], 0, 0, &err_msg);
        if(rc != SQLITE_OK){
            fprintf(stderr, "SQL error: %s\n", err_msg);
            sqlite3_free(err_msg);
            sqlite3_close(db);
            return -1;
        }
    }

    // insert nodes
    for(int i=0; i<g->n_nodes; i++){
        unsigned char* layer_data;
        int layer_data_size;
        layer_data = serialize_layer(g->nodes[i]->layer, &layer_data_size);
        sqlite3_stmt *stmt;
        rc = sqlite3_prepare_v2(db, INSERT_NODE, -1, &stmt, 0);
        if(rc != SQLITE_OK){
            fprintf(stderr, "SQL error: %s\n", err_msg);
            sqlite3_free(err_msg);
            sqlite3_close(db);
            return -1;
        }
        sqlite3_bind_int(stmt, 1, i);
        sqlite3_bind_blob(stmt, 2, layer_data, layer_data_size, SQLITE_STATIC);
        rc = sqlite3_step(stmt);
        if(rc != SQLITE_DONE){
            fprintf(stderr, "SQL error: %s\n", err_msg);
            sqlite3_free(err_msg);
            sqlite3_close(db);
            return -1;
        }
        sqlite3_finalize(stmt);
        free(layer_data);
    }

    // insert links
    for(int i=0; i<g->n_nodes; i++){
        for(int j=0; j<g->nodes[i]->n_children; j++){
            sqlite3_stmt *stmt;
            rc = sqlite3_prepare_v2(db, INSERT_LINK, -1, &stmt, 0);
            if(rc != SQLITE_OK){
                fprintf(stderr, "SQL error: %s\n", err_msg);
                sqlite3_free(err_msg);
                sqlite3_close(db);
                return -1;
            }
            sqlite3_bind_int(stmt, 1, i);
            sqlite3_bind_int(stmt, 2, g->nodes[i]->children[j]);
            rc = sqlite3_step(stmt);
            if(rc != SQLITE_DONE){
                fprintf(stderr, "SQL error: %s\n", err_msg);
                sqlite3_free(err_msg);
                sqlite3_close(db);
                return -1;
            }
            sqlite3_finalize(stmt);
        }
    }

    // insert etc
    sqlite3_stmt *stmt;
    rc = sqlite3_prepare_v2(db, INSERT_ETC, -1, &stmt, 0);
    if(rc != SQLITE_OK){
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return -1;
    }
    int optim_data_size = 0;
    unsigned char* optim_data = serialize_optimizer(g->optim, &optim_data_size);
    sqlite3_bind_blob(stmt, 1, g->output->data, g->output->size, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, g->loss->loss);
    sqlite3_bind_blob(stmt, 3, optim_data, optim_data_size, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    if(rc != SQLITE_DONE){
        fprintf(stderr, "SQL error: %s\n", err_msg);
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return -1;
    }
    sqlite3_finalize(stmt);

    sqlite3_close(db);
    return 0;
}

Module *load_module(const char* path){
    sqlite3 *db;
    sqlite3_stmt *stmt;
    int rc;
    const char *tail;
    rc = sqlite3_open(path, &db);
    if(rc){
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }

    // load nodes
    rc = sqlite3_prepare_v2(db, "SELECT * FROM node;", -1, &stmt, &tail);
    if(rc != SQLITE_OK){
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }
    Module *g = create_module();
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
        int id = sqlite3_column_int(stmt, 0);
        const void* data = sqlite3_column_blob(stmt, 1);
        int data_size = sqlite3_column_bytes(stmt, 1);
        Layer *layer = deserialize_layer((unsigned char*)data, data_size);
        g->nodes[id]->layer = layer;
        g->nodes[id]->id = id;
        g->nodes[id]->n_children = 0;
        g->nodes[id]->n_parents = 0;
        g->n_nodes++;
    }
    sqlite3_finalize(stmt);

    // load links
    rc = sqlite3_prepare_v2(db, "SELECT * FROM link;", -1, &stmt, &tail);
    if(rc != SQLITE_OK){
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }
    while((rc = sqlite3_step(stmt)) == SQLITE_ROW){
        int id = sqlite3_column_int(stmt, 0);
        int child_id = sqlite3_column_int(stmt, 1);
        g->nodes[id]->children[g->nodes[id]->n_children] = child_id;
        g->nodes[id]->n_children++;
        g->nodes[child_id]->parents[g->nodes[child_id]->n_parents] = id;
        g->nodes[child_id]->n_parents++;
    }
    sqlite3_finalize(stmt);

    // load etc
    rc = sqlite3_prepare_v2(db, "SELECT * FROM etc;", -1, &stmt, &tail);
    if(rc != SQLITE_OK){
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }
    rc = sqlite3_step(stmt);
    if(rc != SQLITE_ROW){
        fprintf(stderr, "SQL error: %s\n", sqlite3_errmsg(db));
        sqlite3_close(db);
        return NULL;
    }
    const void* output_data = sqlite3_column_blob(stmt, 0);
    int output_size = sqlite3_column_bytes(stmt, 0);
    g->output = deserialize_tensor((unsigned char*)output_data, output_size);
    int loss_type = (sqlite3_column_int(stmt, 1));
    g->loss = NULL;
    switch(loss_type){
        case MSE_LOSS:
            g->loss = MSE();
            break;
    }
    const void* optim_data = sqlite3_column_blob(stmt, 2);
    int optim_data_size = sqlite3_column_bytes(stmt, 2);
    g->optim = deserialize_optimizer((unsigned char*)optim_data, optim_data_size);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return g;
}