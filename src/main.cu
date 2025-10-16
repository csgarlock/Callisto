#include "tests/inference_test.h"
#include "tests/loss_test.h"

#include "types.h"
#include "util.h"
#include <array>

int main() {
    test_mse(32);
    test_mse(1024);
    test_mse(524288);
    return 0;
}