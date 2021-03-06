#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>

/* Read Stock data */
int stock_array1[] = {
    #include "stock_array1.txt"
};

/* Define width for the moving average */
#define WINDOW_SIZE (13)

#define MAX_SOURCE_SIZE (0x100000)

int main(void)
{
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobjIn = NULL;
    cl_mem memobjOut = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    size_t kernel_code_size;
    char *kernel_src_str;
    float *result;
    cl_int ret;
    FILE *fp;

    int data_num = sizeof(stock_array1) / sizeof(stock_array1[0]);
    int window_num = (int)WINDOW_SIZE;
    int i;

    /* Allocate space to read in kernel code */
    kernel_src_str = (char *)malloc(MAX_SOURCE_SIZE);

    /* Allocate space for the result on the host side */
    result = (float *)malloc(data_num*sizeof(float));

    /* Get Platform */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    /* Get Device */
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                         &ret_num_devices);

    /* Create Context */
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /* Read Kernel Code */
    fp = fopen("moving_average.cl", "r");
    kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create Program Object */
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str,
                                        (const size_t *)&kernel_code_size, &ret);
    /* Compile kernel */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Create Kernel */
    kernel = clCreateKernel(program, "moving_average", &ret);

    /* Create buffer for the input data on the device */
    memobjIn  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               data_num * sizeof(int), NULL, &ret);

    /* Create buffer for the result on the device */ 
    memobjOut = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               data_num * sizeof(float), NULL, &ret);

    /* Copy input data to the global memory on the device*/
    ret = clEnqueueWriteBuffer(command_queue, memobjIn, CL_TRUE, 0,
                               data_num * sizeof(int),
                               stock_array1, 0, NULL, NULL);

    /* Set kernel arguments */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobjIn);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memobjOut);
    ret = clSetKernelArg(kernel, 2, sizeof(int),    (void *)&data_num);
    ret = clSetKernelArg(kernel, 3, sizeof(int),    (void *)&window_num);

    /* Execute the kernel */ 
    ret = clEnqueueTask(command_queue, kernel, 0, NULL, NULL);

    /* Copy result from device to host */
    ret = clEnqueueReadBuffer(command_queue, memobjOut, CL_TRUE, 0,
                              data_num * sizeof(float),
                              result, 0, NULL, NULL);


    /* OpenCL Object Finalization */
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobjIn);
    ret = clReleaseMemObject(memobjOut);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /* Display Results */
    for(i=0; i<data_num; i++) {
        printf( "result[%d] = %f\n", i, result[i] );
    }

    /* Deallocate memory on the host */
    free(result);
    free(kernel_src_str);

    return 0;
}

