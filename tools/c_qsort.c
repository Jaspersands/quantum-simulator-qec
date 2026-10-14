#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
struct str { double value; int index; };
static int cmp(const void *a, const void *b) {
    const struct str *x = a, *y = b;
    if (x->value > y->value) return 1;
    if (x->value < y->value) return -1;
    return 0;
}
int main(void) {
    uint64_t s = 88172645463325252ull;
    int sizes[] = {3, 6, 7, 8, 12, 20, 40, 41, 64, 100, 500, 3000};
    int distinct[] = {2, 5, 1000};
    for (unsigned a = 0; a < sizeof sizes / sizeof *sizes; a++) for (unsigned b = 0; b < 3; b++) {
        int n = sizes[a];
        struct str *o = malloc(n * sizeof *o);
        for (int i = 0; i < n; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; o[i].value = (double)(s % distinct[b]); o[i].index = i; }
        qsort(o, n, sizeof *o, cmp);
        long d = 0; for (int i = 0; i < n; i++) d = (d * 31 + o[i].index + 1) % 1000000007;
        printf("%d %d %ld\n", n, distinct[b], d);
        free(o);
    }
}
