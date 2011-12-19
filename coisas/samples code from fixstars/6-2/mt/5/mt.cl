typedef struct mt_struct_s {
    uint aaa;
    int mm,nn,rr,ww;
    uint wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint maskB, maskC;
} mt_struct ;

/* Initialize state using a seed */
static void sgenrand_mt(uint seed, __local const mt_struct *mts, __local uint *state)  {
    int i;
    for (i=0; i<mts->nn; i++) {
	state[i] = seed;
        seed = (1812433253 * (seed  ^ (seed >> 30))) + i + 1;
    }
    for (i=0; i<mts->nn; i++)
	state[i] &= mts->wmask;
}

/* Update state */
static void update_state(__local const mt_struct *mts, __local uint *st, int wlid) {
    int n = 17, m = 8;
    uint aa = mts->aaa, x;
    uint uuu = mts->umask, lll = mts->lmask;
    int k,lim;

    if (wlid < 9) {
        k = wlid;
        uint stm = st[k+m];
        x = (st[k]&uuu)|(st[k+1]&lll);
        st[k] = st[k+m] ^ (x>>1) ^ (x&1U ? aa : 0U);
    }
    if (wlid<7) {
        k = wlid + 9;
        x = (st[k]&uuu)|(st[k+1]&lll);
        st[k] = st[k+m-n] ^ (x>>1) ^ (x&1U ? aa : 0U);
    }
    if (wlid == 0) {
        x = (st[n-1]&uuu)|(st[0]&lll);
        st[n-1] = st[m-1] ^ (x>>1) ^ (x&1U ? aa : 0U);
    }
}

extern int printf(const char*p, ...);

static inline void gen(__global uint *out, const __local mt_struct *mts, __local uint *state, int num_rand, int wlid) {
    int i, j, n, nn = mts->nn;
    n = (num_rand+(nn-1)) / nn;

    for (i=0; i<n; i++) {
        int m = nn;
        if (i == n-1) m = num_rand%nn;

        update_state(mts, state, wlid);

        barrier(CLK_LOCAL_MEM_FENCE);
        if (wlid < m) {
            int j = wlid;
            uint x = state[j];
            x ^= x >> mts->shift0;
            x ^= (x << mts->shiftB) & mts->maskB;
            x ^= (x << mts->shiftC) & mts->maskC;
            x ^= x >> mts->shift1;
            out[i*nn + j] = x;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void genrand(__global uint *out,__global mt_struct *mts_g,int num_rand, int num_generator,
                      uint num_param_per_warp, __local uint *state_mem, __local mt_struct *mts){
    int warp_per_compute_unit = 4;
    int workitem_per_warp = 32;
    int wid = get_group_id(0);
    int lid = get_local_id(0);
    int warp_id = wid * warp_per_compute_unit + lid / workitem_per_warp;
    int generator_id, end;
    int wlid = lid % workitem_per_warp;

    __local uint *state = state_mem + warp_id*17; /* Store state in local memory */

    end = num_param_per_warp*warp_id + num_param_per_warp;
    if (end > num_generator)
        end = num_generator;

    mts = mts + warp_id;


    for (generator_id = num_param_per_warp*warp_id;
         generator_id < end;
         generator_id ++)
    {
        if (wlid == 0) {
            /* Copy parameters to local memory */
            mts->aaa = mts_g[generator_id].aaa;
            mts->mm = mts_g[generator_id].mm;
            mts->nn = mts_g[generator_id].nn;
            mts->rr = mts_g[generator_id].rr;
            mts->ww = mts_g[generator_id].ww;
            mts->wmask = mts_g[generator_id].wmask;
            mts->umask = mts_g[generator_id].umask;
            mts->lmask = mts_g[generator_id].lmask;
            mts->shift0 = mts_g[generator_id].shift0;
            mts->shift1 = mts_g[generator_id].shift1;
            mts->shiftB = mts_g[generator_id].shiftB;
            mts->shiftC = mts_g[generator_id].shiftC;
            mts->maskB = mts_g[generator_id].maskB;
            mts->maskC = mts_g[generator_id].maskC;
            sgenrand_mt(0x33ff*generator_id, mts, (__local uint*)state); /* Initialize random numbers */
        }
        gen(out + generator_id*num_rand, mts, (__local uint*)state, num_rand, wlid);      /* Generate random numbers */
    }
}

/* Count the number of points within the circle */
__kernel void calc_pi(__global uint *out, __global uint *rand, int num_rand_per_compute_unit,
                      int num_compute_unit, int num_rand_all, __local uint *count_per_wi) {
    int gid = get_group_id(0);
    int lid = get_local_id(0);

    int count = 0;
    int i, end, begin;

    begin = gid * num_rand_per_compute_unit;
    end = begin + num_rand_per_compute_unit;
    if (end > num_rand_all)
        end = num_rand_all;

    rand += lid;

    for (i=begin; i<end-128; i+=128) {
        float x, y, len;
        x = ((float)(rand[i]>>16))/65535.0f;    /* x coordinate */
        y = ((float)(rand[i]&0xffff))/65535.0f; /* y coordinate */
        len = (x*x + y*y);      /* Distance from the origin */

        if (len < 1) {          /* sqrt(len) < 1 = len < 1 */
            count++;
        }
    }


    /* Process left-overs */
    if ((i + lid) < end) {
        float x, y, len;
        x = ((float)(rand[i]>>16))/65535.0f;    /* x coordinate */
        y = ((float)(rand[i]&0xffff))/65535.0f; /* y coordinate */
        len = (x*x + y*y);      /* Distance from the origin */

        if (len < 1) {          /* sqrt(len) < 1 = len < 1 */
            count++;
        }
    }

    count_per_wi[lid] = count;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        int count = 0;
        for (i=0; i<128; i++) {
            count += count_per_wi[i];
        }
        out[gid] = count;
    }
}
