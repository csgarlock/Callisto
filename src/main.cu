#include "tests/inference_test.h"
#include "tests/loss_test.h"

#include "types.h"
#include "util.h"
#include <array>

int main() {
    test_feed_forward(32, 32);
    test_feed_forward(8192, 32768);
    test_feed_forward(32768, 8192);
    test_feed_forward(32768, 32768);
    return 0;
}