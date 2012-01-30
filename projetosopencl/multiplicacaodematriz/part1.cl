#define STRINGIFY(A) #A
std::string kernel_source = STRINGIFY(

__kernel void part1(__global int* a, __global int* b, __global int* c)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    c[i+j*3] = b[i]*a[j*3]+b[i+3]*a[j*3+1]+b[i+6]*a[j*3+2];
}
);
