#define PI       3.14159265358979323846
#define PI_2     1.57079632679489661923

__kernel void spinFact(__global double2* w, int n)
{
    unsigned int i = get_global_id(0);

    double2 angle = (double2)(2*i*PI/(double)n,(2*i*PI/(double)n)+PI_2);
    w[i] = cos(angle);
}

__kernel void bitReverse(__global double2 *dst, __global double2 *src, int m, int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int nid = get_global_id(1);
    
    unsigned int j = gid;
    j = (j & 0x55555555) <<  1 | (j & 0xAAAAAAAA) >>  1; 
    j = (j & 0x33333333) <<  2 | (j & 0xCCCCCCCC) >>  2; 
    j = (j & 0x0F0F0F0F) <<  4 | (j & 0xF0F0F0F0) >>  4; 
    j = (j & 0x00FF00FF) <<  8 | (j & 0xFF00FF00) >>  8; 
    j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16; 

    j >>= (32-m);
    
    dst[nid*n+j] = src[nid*n+gid];
}

__kernel void norm(__global double2 *x, int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int nid = get_global_id(1);
    
    x[nid*n+gid] = x[nid*n+gid] / (double2)((double)n, (double)n);
}

__kernel void butterfly(__global double2 *x, __global double2* w, int m, int n, int iter, ulong flag)
{
    unsigned int gid = get_global_id(0);
    unsigned int nid = get_global_id(1);
    
    int butterflySize      = 1 << (iter-1);
    int butterflyGrpDist   = 1 << iter;
    int butterflyGrpNum    = n >> iter;
    int butterflyGrpBase   = (gid >> (iter-1))*(butterflyGrpDist);
    int butterflyGrpOffset = gid & (butterflySize-1);
    
    int a = nid * n + butterflyGrpBase + butterflyGrpOffset;
    int b = a + butterflySize;
      
    int l = butterflyGrpNum * butterflyGrpOffset;

    double2 xa, xb, xbxx, xbyy, wab, wayx, wbyx, resa, resb;
  
    xa   = x[a];
    xb   = x[b];
    xbxx = xb.xx;
    xbyy = xb.yy;
    
    wab  = as_double2(as_ulong2(w[l])   ^ (ulong2)(0x0UL, flag));
    wayx = as_double2(as_ulong2(wab.yx) ^ (ulong2)(0x8000000000000000UL, 0x0UL));
    wbyx = as_double2(as_ulong2(wab.yx) ^ (ulong2)(0x0UL, 0x8000000000000000UL));

    resa = xa + xbxx*wab + xbyy*wayx;
    resb = xa - xbxx*wab + xbyy*wbyx;

    x[a] = resa;
    x[b] = resb;
}

__kernel void transpose(__global double2 *dst, __global double2* src, int n)
{
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);

    unsigned int iid = ygid * n + xgid;
    unsigned int oid = xgid * n + ygid;

    dst[oid] = src[iid];
}

__kernel void hiPassFilter(__global double2* image, int n, int radius)
{
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);

    int2 n_2  = (int2)(n>>1, n>>1);
    int2 mask = (int2)(n-1,  n-1);
    
    int2 gid = ((int2)(xgid, ygid) + n_2) & mask; 

    int2 diff  = n_2 - gid;
    int2 diff2 = diff * diff;
    int dist2 = diff2.x + diff2.y;
    
    long2 window;
          
    if (dist2 < radius*radius) {
        window = (long2)(0L, 0L);
    } else {
        window = (long2)(-1L, -1L);
    }
    
    image[ygid*n+xgid] = as_double2(as_long2(image[ygid*n+xgid]) & window);
}
