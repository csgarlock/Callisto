#include "sequential.h"

void Sequential::forward() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i].forward();
    }
}

void Sequential::propagate_error() {
    for (int i = layers.size() - 1; i >= 0; i--) {
        layers[i].propagate_error();
    }
}

void Sequential::find_gradients() {
    for (int i = layers.size() - 1; i >= 0; i--) {
        layers[i].find_gradients();
    }
}

void Sequential::update_parameters() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i].update_parameters();
    }
}

void Sequential::zero_grads() {
    for (int i = 0; i < layers.size(); i++) {
        layers[i].zero_grads();
    }
}