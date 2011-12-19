__kernel void local_test(__local char *p, int local_size) {
    for (int i=0; i<local_size; i++) {
        p[i] = i;               /* ローカルメモリに値を設定(結果は捨てられる) */
    }
}
