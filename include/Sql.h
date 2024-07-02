#ifndef SQL_H
#define SQL_H

const char* CLEAR_TABLES = "DROP TABLE IF EXISTS link;"
                           "DROP TABLE IF EXISTS node;"
                           "DROP TABLE IF EXISTS etc;";

const char* CREATE_LINK_TABLE = "CREATE TABLE IF NOT EXISTS link("
                                "   link_id INTEGER PRIMARY KEY AUTOINCREMENT,"
                                "   node_id INTEGER NOT NULL,"
                                "   child_id INTEGER NOT NULL,"
                                "   FOREIGN KEY(node_id) REFERENCES node(node_id),"
                                ");";

const char* CREATE_NODE_TABLE = "CREATE TABLE IF NOT EXISTS node("
                                "   node_id INTEGER PRIMARY KEY,"
                                "   layer BLOB NOT NULL,"
                                ");";

const char* CREATE_ETC_TABLE = "CREATE TABLE IF NOT EXISTS etc("
                                  "   output BLOB,"
                                  "   loss INTEGER,"
                                  "   optim BLOB"
                                  ");";

const char* INSERT_LINK = "INSERT INTO link(node_id, child_id) VALUES(?, ?);";

const char* INSERT_NODE = "INSERT INTO node(node_id, layer) VALUES(?, ?);";

const char *INSERT_ETC = "INSERT INTO etc(output, loss, optim) VALUES(?, ?, ?);";

#endif