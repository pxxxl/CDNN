cmake_minimum_required(VERSION 3.0.0)
project(DeepLearning VERSION 0.1.0 LANGUAGES C)

set(SOURCES
    ./src/F.c
    ./src/Layer.c
    ./src/Loss.c
    ./src/Module.c
    ./src/Optim.c
    ./src/Tensor.c
    ./src/sqlite3.c
)

include_directories(./include)

add_executable(TensorTest ${SOURCES} ./test/tensor_test.c)
add_executable(ModuleTest ${SOURCES} ./test/module_test.c)
add_executable(LinearTest ${SOURCES} ./test/linear_test.c)
add_executable(SerializeTest ${SOURCES} ./test/serialize_test.c)