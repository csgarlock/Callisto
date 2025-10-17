#include "tests/inference_test.h"
#include "tests/loss_test.h"
#include "tests/linear_test.h"
#include "types.h"
#include "util.h"

#include <array>

int main() {
    linear_forward_test_batch(256, 256, 80, true);
    linear_forward_test_batch(4, 4, 2);
    linear_forward_test_batch(4, 4, 2);
    linear_forward_test_batch(4, 4, 2);
    return 0;
}