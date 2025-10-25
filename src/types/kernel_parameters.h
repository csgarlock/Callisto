#ifndef KERNEL_PARAMETERS_H_INCLUDED
#define KERNEL_PARAMETERS_H_INCLUDED

struct MatrixMultShape {
    int input;
    int output;
    int batch_size = 1;
};

#endif