// Microsoft's std::sort, transcribed from microsoft/STL (stl/inc/algorithm and
// stl/inc/__msvc_heap_algorithms.hpp; Copyright (c) Microsoft Corporation, Apache License 2.0 WITH
// LLVM-exception) into portable C++, so that its results can be computed off Windows: the
// expected digests of `msvc_sort_is_microsofts` in src/cpp_random.rs. Same driver as cpp_sort.cpp.
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

using std::swap;
constexpr long ISORT_MAX = 32;

template <class T, class P> void insertion_sort(T *first, T *last, P pred) {
    if (first != last) {
        for (T *mid = first; ++mid != last;) {
            T *hole = mid;
            T val = *mid;
            if (pred(val, *first)) {
                for (T *p = mid; p != first; --p) *p = *(p - 1);
                *first = val;
            } else {
                for (T *prev = hole; pred(val, *--prev); hole = prev) *hole = *prev;
                *hole = val;
            }
        }
    }
}

template <class T, class P> void med3(T *first, T *mid, T *last, P pred) {
    if (pred(*mid, *first)) swap(*mid, *first);
    if (pred(*last, *mid)) {
        swap(*last, *mid);
        if (pred(*mid, *first)) swap(*mid, *first);
    }
}

template <class T, class P> void guess_median(T *first, T *mid, T *last, P pred) {
    long count = last - first;
    if (40 < count) {
        long step = (count + 1) >> 3, two_step = step << 1;
        med3(first, first + step, first + two_step, pred);
        med3(mid - step, mid, mid + step, pred);
        med3(last - two_step, last - step, last, pred);
        med3(first + step, mid, last - step, pred);
    } else {
        med3(first, mid, last, pred);
    }
}

template <class T, class P> std::pair<T *, T *> partition_by_pivot(T *first, T *pfirst, T *last, P pred) {
    T *plast = pfirst + 1;
    while (first < pfirst && !pred(*(pfirst - 1), *pfirst) && !pred(*pfirst, *(pfirst - 1))) --pfirst;
    while (plast < last && !pred(*plast, *pfirst) && !pred(*pfirst, *plast)) ++plast;
    T *gfirst = plast, *glast = pfirst;
    for (;;) {
        for (; gfirst < last; ++gfirst) {
            if (pred(*pfirst, *gfirst)) continue;
            else if (pred(*gfirst, *pfirst)) break;
            else if (plast != gfirst) { swap(*plast, *gfirst); ++plast; }
            else ++plast;
        }
        for (; first < glast; --glast) {
            T *gp = glast - 1;
            if (pred(*gp, *pfirst)) continue;
            else if (pred(*pfirst, *gp)) break;
            else if (--pfirst != gp) swap(*pfirst, *gp);
        }
        if (glast == first && gfirst == last) return {pfirst, plast};
        if (glast == first) {
            if (plast != gfirst) swap(*pfirst, *plast);
            ++plast;
            swap(*pfirst, *gfirst);
            ++pfirst;
            ++gfirst;
        } else if (gfirst == last) {
            if (--glast != --pfirst) swap(*glast, *pfirst);
            swap(*pfirst, *--plast);
        } else {
            swap(*gfirst, *--glast);
            ++gfirst;
        }
    }
}

template <class T, class P> void push_heap_by_index(T *first, long hole, long top, T val, P pred) {
    for (long idx = (hole - 1) >> 1; top < hole && pred(*(first + idx), val); idx = (hole - 1) >> 1) {
        *(first + hole) = *(first + idx);
        hole = idx;
    }
    *(first + hole) = val;
}

template <class T, class P> void pop_heap_hole_by_index(T *first, long hole, long bottom, T val, P pred) {
    long top = hole, idx = hole;
    long max_non_leaf = (bottom - 1) >> 1;
    while (idx < max_non_leaf) {
        idx = 2 * idx + 2;
        if (pred(*(first + idx), *(first + (idx - 1)))) --idx;
        *(first + hole) = *(first + idx);
        hole = idx;
    }
    if (idx == max_non_leaf && bottom % 2 == 0) {
        *(first + hole) = *(first + (bottom - 1));
        hole = bottom - 1;
    }
    push_heap_by_index(first, hole, top, val, pred);
}

template <class T, class P> void make_heap(T *first, T *last, P pred) {
    long bottom = last - first;
    for (long hole = bottom >> 1; hole > 0;) {
        --hole;
        T val = *(first + hole);
        pop_heap_hole_by_index(first, hole, bottom, val, pred);
    }
}

template <class T, class P> void sort_heap(T *first, T *last, P pred) {
    for (; last - first >= 2; --last) {
        T *l = last - 1;
        T val = *l;
        *l = *first;
        pop_heap_hole_by_index(first, 0L, (long) (l - first), val, pred);
    }
}

template <class T, class P> void sort_unchecked(T *first, T *last, long ideal, P pred) {
    for (;;) {
        if (last - first <= ISORT_MAX) { insertion_sort(first, last, pred); return; }
        if (ideal <= 0) { make_heap(first, last, pred); sort_heap(first, last, pred); return; }
        T *mid0 = first + ((last - first) >> 1);
        guess_median(first, mid0, last - 1, pred);
        auto mid = partition_by_pivot(first, mid0, last, pred);
        ideal = (ideal >> 1) + (ideal >> 2);
        if (mid.first - first < last - mid.second) { sort_unchecked(first, mid.first, ideal, pred); first = mid.second; }
        else { sort_unchecked(mid.second, last, ideal, pred); last = mid.first; }
    }
}

int main() {
    uint64_t x = 88172645463325252ull;
    auto next = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
    for (int n : {3, 5, 7, 20, 23, 24, 25, 26, 29, 32, 33, 40, 41, 42, 64, 100, 129, 200, 1000, 5000}) {
        for (int distinct : {2, 5, 1000}) {
            std::vector<double> key(n); for (int i = 0; i < n; i++) key[i] = double(next() % distinct);
            std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = (int)(next() % n);
            for (int i = 0; i < n; i++) idx[i] = i;
            for (int i = n - 1; i > 0; i--) { int j = next() % (i + 1); std::swap(idx[i], idx[j]); }
            auto pred = [&](int a, int b) { return key[a] > key[b]; };
            sort_unchecked(idx.data(), idx.data() + n, (long) n, pred);
            long s = 0; for (int i = 0; i < n; i++) s = (s * 31 + idx[i] + 1) % 1000000007;
            printf("(%d, %d, %ld), ", n, distinct, s);
        }
    }
    printf("\n");
    // The heap sort it falls back on, alone (no divisions allowed).
    for (int n : {33, 100, 1000}) {
        for (int distinct : {2, 1000}) {
            std::vector<double> key(n); for (int i = 0; i < n; i++) key[i] = double(next() % distinct);
            std::vector<int> idx(n); for (int i = 0; i < n; i++) idx[i] = i;
            for (int i = n - 1; i > 0; i--) { int j = next() % (i + 1); std::swap(idx[i], idx[j]); }
            auto pred = [&](int a, int b) { return key[a] > key[b]; };
            sort_unchecked(idx.data(), idx.data() + n, 0L, pred);
            long s = 0; for (int i = 0; i < n; i++) s = (s * 31 + idx[i] + 1) % 1000000007;
            printf("(%d, %d, %ld), ", n, distinct, s);
        }
    }
    printf("\n");
}
