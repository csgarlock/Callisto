#include "tests/inference_test.h"
#include "tests/loss_test.h"

#include "types.h"
#include "util.h"
#include <array>

int main() {
    test_feed_forward_batch(32, 32, 32);
    test_feed_forward_batch(256, 256, 256);
    test_feed_forward_batch(1024, 1024, 1024);
    return 0;
}