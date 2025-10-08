#include "tests/inference_test.h"


int main() {
    // Example: 4096×4096 matvec (big enough for stable timing)
    test_feed_forward(32, 32);
    test_feed_forward(4096, 4096);
    test_feed_forward(4096, 32768);
    test_feed_forward(32768, 4096);
    test_feed_forward(32768, 32768);
    return 0;
}