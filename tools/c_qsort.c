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
static uint64_t s = 88172645463325252ull;
static uint64_t next(void) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
static long sorted_digest(struct str *o, int n) {
    qsort(o, n, sizeof *o, cmp);
    long d = 0; for (int i = 0; i < n; i++) d = (d * 31 + o[i].index + 1) % 1000000007;
    return d;
}
int main(void) {
    int sizes[] = {3, 6, 7, 8, 12, 20, 40, 41, 64, 100, 500, 3000};
    int distinct[] = {2, 5, 1000};
    for (unsigned a = 0; a < sizeof sizes / sizeof *sizes; a++) for (unsigned b = 0; b < 3; b++) {
        int n = sizes[a];
        struct str *o = malloc(n * sizeof *o);
        for (int i = 0; i < n; i++) { o[i].value = (double)(next() % distinct[b]); o[i].index = i; }
        printf("%d %d %ld\n", n, distinct[b], sorted_digest(o, n));
        free(o);
    }
    /* Already ordered input with ties, where Apple's qsort tries its bounded insertion sort and
       can run into its depth limit: shape 1 ascending, 2 descending, 3 up then down (organ pipe),
       a new value after each run of about `run` equal ones. */
    int runs[] = {2, 5, 20};
    for (int shape = 1; shape <= 3; shape++)
        for (unsigned a = 0; a < sizeof sizes / sizeof *sizes; a++) for (unsigned b = 0; b < 3; b++) {
            int n = sizes[a];
            struct str *o = malloc(n * sizeof *o);
            double x = 0;
            for (int i = 0; i < n; i++) {
                if (next() % runs[b] == 0) x += 1;
                double v = shape == 2 ? -x : x;
                if (shape == 3 && n - 1 - i < i) v = o[n - 1 - i].value;
                o[i].value = v; o[i].index = i;
            }
            printf("%d %d %d %ld\n", shape, n, runs[b], sorted_digest(o, n));
            free(o);
        }
}
