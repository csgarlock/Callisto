#include "tests/inference_test.h"
#include "tests/loss_test.h"
#include "tests/linear_test.h"

#include "types.h"
#include "util.h"
#include <array>

int main() {
    linear_forward_test_batch(256, 256, 64);
    linear_forward_test_batch(1, 256, 1074);
    linear_forward_test_batch(1, 256, 1074);
    linear_forward_test_batch(1, 256, 1074);
    linear_forward_test_batch(1, 256, 1074);
    linear_forward_test_batch(1, 256, 325);
    linear_forward_test_batch(1, 256, 325);
    linear_forward_test_batch(1, 256, 325);
    linear_forward_test_batch(1, 256, 325);
    return 0;
}