#define STRINGIFY(A) #A
std::string kernel_source = STRINGIFY(

int function_example(int a, int b)
{
    return a + b;
}

__kernel void part1(__global int* a, __global int* b, __global int* c)
{
    unsigned int i = get_global_id(0);

    c[i] = function_example(a[i], b[i]);
}
);
