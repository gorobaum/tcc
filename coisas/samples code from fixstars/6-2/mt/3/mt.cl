typedef struct mt_struct_s {
    uint aaa;
    int mm,nn,rr,ww;
    uint wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint maskB, maskC;
} mt_struct ;

/* Initialize state using a seed */
static void sgenrand_mt(uint seed, __global const mt_struct *mts, uint *state)  {
    int i;
    for (i=0; i<mts->nn; i++) {
	state[i] = seed;
        seed = (1812433253 * (seed  ^ (seed >> 30))) + i + 1;
    }
    for (i=0; i<mts->nn; i++)
	state[i] &= mts->wmask;
}

/* Update state */
static void update_state(__global const mt_struct *mts, uint *st) {
    int n = mts->nn, m = mts->mm;
    uint aa = mts->aaa, x;
    uint uuu = mts->umask, lll = mts->lmask;
    int k,lim;
    lim = n - m;
    for (k=0; k<lim; k++) {
        x = (st[k]&uuu)|(st[k+1]&lll);
        st[k] = st[k+m] ^ (x>>1) ^ (x&1U ? aa : 0U);
    }
    lim = n - 1;
    for (; k<lim; k++) {
        x = (st[k]&uuu)|(st[k+1]&lll);
        st[k] = st[k+m-n] ^ (x>>1) ^ (x&1U ? aa : 0U);
    }
    x = (st[n-1]&uuu)|(st[0]&lll);
    st[n-1] = st[m-1] ^ (x>>1) ^ (x&1U ? aa : 0U);
}

static inline void gen(__global uint *out, const __global mt_struct *mts, uint *state, int num_rand) {
    int i, j, n, nn = mts->nn;
    n = (num_rand+(nn-1)) / nn;
    for (i=0; i<n; i++) {
        int m = nn;
        if (i == n-1) m = num_rand%nn;
        update_state(mts, state);
        for (j=0; j<m; j++) {
            /* Generate random numbers */
            uint x = state[j];
            x ^= x >> mts->shift0;
            x ^= (x << mts->shiftB) & mts->maskB;
            x ^= (x << mts->shiftC) & mts->maskC;
            x ^= x >> mts->shift1;
            out[i*nn + j] = x;
        }
    }
}

__kernel void genrand(__global uint *out,__global mt_struct *mts,int num_rand, int num_generator){
    int gid = get_global_id(0);
    uint seed = gid*3;
    uint state[17];

    if (gid >= num_generator) return;

    mts += gid;                 /* mts for this item */
    out += gid * num_rand;      /* Output buffer for this item */
    sgenrand_mt(0x33ff*gid, mts, (uint*)state); /* Initialize random numbers */
    gen(out, mts, (uint*)state, num_rand);      /* Generate random numbers */
}

/* Count the number of points within the circle */
__kernel void calc_pi(__global uint *out, __global uint *rand, int num_rand, int num_generator) {
    int gid = get_global_id(0);
    int count = 0;
    int i;

    if (gid >= num_generator) return;

    rand += gid*num_rand;

    for (i=0; i<num_rand; i++) {
        float x, y, len;
        x = ((float)(rand[i]>>16))/65535.0f;    /* x coordinate */
        y = ((float)(rand[i]&0xffff))/65535.0f; /* y coordinate */
        len = (x*x + y*y);      /* Distance from the origin */

        if (len < 1) {          /* sqrt(len) < 1 = len < 1 */
            count++;
        }
    }

    out[gid] = count;
}
