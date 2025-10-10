#ifndef HOST_LOSS_H_INCLUDED
#define HOST_LOSS_H_INCLUDED

template <typename T>
float mse_cpu(const T *__restrict__ predicted, const T *__restrict__ actual, int n) {
    T acc = 0;
    for (int i = 0; i < n; i++) {
        T diff = actual[i] - predicted[i];
        acc += diff * diff;
    }
    return acc / n;
}

#endif