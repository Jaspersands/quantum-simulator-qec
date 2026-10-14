#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdint>
// Keys from a small set (many ties); indices sorted by key, descending, as ldpc sorts.
int main() {
    uint64_t x = 88172645463325252ull;
    auto next = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (int n : {3, 5, 7, 20, 23, 24, 25, 26, 29, 40, 64, 100, 129, 200, 1000, 5000}) {
        for (int distinct : {2, 5, 1000}) {
            std::vector<double> key(n); for (int i = 0; i < n; i++) key[i] = double(next() % distinct);
            std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = (int)(next() % n);  // with repeats, as an order may have
            for (int i = 0; i < n; i++) idx[i] = i;
            // a shuffled start
            for (int i = n - 1; i > 0; i--) { int j = next() % (i + 1); std::swap(idx[i], idx[j]); }
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return key[a] > key[b]; });
            long s = 0; for (int i = 0; i < n; i++) s = (s * 31 + idx[i] + 1) % 1000000007;
            printf("%d %d %ld\n", n, distinct, s);
        }
    }
}
