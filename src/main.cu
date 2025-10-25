#include "tests/loss_test.h"
#include "tests/linear_test.h"
#include "types/activation_types.h"
#include "util.h"

#include <array>

int main() {
    linear_forward_test_batch<ReLU>(256, 256, 64);
    linear_forward_test_batch<ReLU>(256, 256, 80);
    linear_forward_test_batch<ReLU>(1, 128, 1024);
    linear_forward_test_batch<ReLU>(634, 422, 87);
    return 0;
}