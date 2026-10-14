#include <random>
#include <algorithm>
#include <vector>
#include <cstdio>
int main() {
    for (int n : {1, 2, 5, 17, 64, 1000}) {
        std::vector<int> v(n); for (int i = 0; i < n; i++) v[i] = i;
        std::mt19937 g(12345);
        std::shuffle(v.begin(), v.end(), g);
        std::shuffle(v.begin(), v.end(), g);
        long s = 0; for (int i = 0; i < n; i++) s = s * 31 + v[i] + 1, s %= 1000000007;
        printf("mt %d %ld %d\n", n, s, n > 3 ? v[3] : -1);
        std::vector<int> u(n); for (int i = 0; i < n; i++) u[i] = i;
        std::default_random_engine e(777);
        std::shuffle(u.begin(), u.end(), e);
        s = 0; for (int i = 0; i < n; i++) s = s * 31 + u[i] + 1, s %= 1000000007;
        printf("dre %d %ld\n", n, s);
    }
}
