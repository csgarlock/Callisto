#include "tests/inference_test.h"
#include "tests/loss_test.h"

#include "types.h"
#include "util.h"
#include <array>

int main() {
    test_mse(6233);
    test_mse(131072);
    return 0;
}