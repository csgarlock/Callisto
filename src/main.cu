
#include "tests/inference_test.h"



int main() {
    // Example: 4096×4096 matvec (big enough for stable timing)
    test_feed_forward(4096, 4096);
    test_feed_forward(8192, 32768);
    test_feed_forward(32768, 8192);
    return 0;
}