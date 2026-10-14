// The reference for tests/data/mbp_reference.json: ldpc's mbp.hpp (on ldpc's main branch; ldpc
// 2.4.1 ships its Python but not this C++), run on cases read from stdin.
//
// mbp.hpp calls pcm->iterate_column_ptr(i) and pcm->iterate_row_ptr(i), which ldpc's sparse
// matrix no longer has; the adapters below give the same entries in the same order. Build with
// a copy of mbp.hpp in which those calls are spelled col_ptrs(*pcm, i) and row_ptrs(*pcm, i):
//
//   sed -E 's/pcm->iterate_column_ptr\(([a-z_]+)\)/col_ptrs(*pcm, \1)/g; s/pcm->iterate_row_ptr\(([a-z_]+)\)/row_ptrs(*pcm, \1)/g' ldpc/src_cpp/mbp.hpp > inc/mbp.hpp
//   clang++ -std=c++20 -O2 -ffp-contract=off -Iinc -Ildpc/src_cpp tools/mbp_reference.cpp -o mbp_reference
//   python tools/mbp_reference.py ./mbp_reference tests/data/mbp_reference.json
//
// (-ffp-contract=off: no fused multiply-adds, which Rust does not make either.)
#include <cstdio>
#include <vector>
#include "sparse_matrix.hpp"
// The entries of a column or row, in iteration order, as pointers (the API mbp.hpp was written for).
template <class M> auto col_ptrs(M &m, int i) { std::vector<std::remove_reference_t<decltype(*m.column_heads[0])>*> v; for (auto &e : m.iterate_column(i)) v.push_back(&e); return v; }
template <class M> auto row_ptrs(M &m, int i) { std::vector<std::remove_reference_t<decltype(*m.row_heads[0])>*> v; for (auto &e : m.iterate_row(i)) v.push_back(&e); return v; }
#include "mbp.hpp"
int main() {
    int m, n, nnz;
    if (scanf("%d %d %d", &m, &n, &nnz) != 3) return 1;
    auto *pcm = new mbp_sparse(m, n);
    for (int k = 0; k < nnz; k++) { int i, j, v; scanf("%d %d %d", &i, &j, &v); pcm->insert_entry(i, j, (uint8_t) v); }
    std::vector<std::vector<double>> ch(3, std::vector<double>(n)), al(3, std::vector<double>(n));
    for (int w = 0; w < 3; w++) for (int i = 0; i < n; i++) scanf("%lf", &ch[w][i]);
    for (int w = 0; w < 3; w++) for (int i = 0; i < n; i++) scanf("%lf", &al[w][i]);
    double beta, gamma; int method, max_iter, shots;
    scanf("%lf %lf %d %d %d", &beta, &gamma, &method, &max_iter, &shots);
    mbp_decoder dec(pcm, ch, max_iter, al, beta, method, gamma);
    std::vector<uint8_t> s(m);
    for (int t = 0; t < shots; t++) {
        for (int i = 0; i < m; i++) { int x; scanf("%d", &x); s[i] = (uint8_t) x; }
        auto &d = dec.decode(s);
        for (int i = 0; i < n; i++) printf("%d ", (int) d[i]);
        printf("\n%d %d\n", (int) dec.converge, dec.iterations);
        for (int w = 0; w < 3; w++) { for (int i = 0; i < n; i++) printf("%.17g ", dec.log_prob_ratios[w][i]); printf("\n"); }
    }
    return 0;
}
