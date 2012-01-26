#define STRINGIFY(A) #A
std::string kernel_source = STRINGIFY(

int function_example(int a, int b)
{
    return a + b;
}

__kernel void part1(__global int* a, __global int* b, __global int* c)
{
    unsigned int i = get_global_id(0);
    unsigned int j = i%3;
    if ( i < 3 ) c[i] = a[0]*b[j] + a[1]*b[j+1] + a[2]*b[j+2];
    else if ( i < 6 ) c[i] = a[3]*b[j] + a[4]*b[j+1] + a[5]*b[j+2];
    else c[i] = a[6]*b[j] + a[7]*b[j+1] + a[8]*b[j+2];

}
);
