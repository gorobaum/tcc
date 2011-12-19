#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>

#define NAME_NUM (4)   /* Number of stocks */
#define DATA_NUM (100) /* Number of data to process for each stock */

/* Read Stock data */
int stock_array_4[NAME_NUM*DATA_NUM]= {
    #include "stock_array_4.txt"
};

/* Moving average width */
#define WINDOW_SIZE_13 (13)
#define WINDOW_SIZE_26 (26)


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
    cl_mem memobjOut13 = NULL;
    cl_mem memobjOut26 = NULL;
    cl_program program = NULL;
    cl_kernel kernel13 = NULL;
    cl_kernel kernel26 = NULL;
    cl_event event13, event26;
    size_t kernel_code_size;
    char *kernel_src_str;
    float *result13;
    float *result26;
    cl_int ret;
    FILE *fp;

    int window_num_13 = (int)WINDOW_SIZE_13;
    int window_num_26 = (int)WINDOW_SIZE_26;
    int point_num = NAME_NUM * DATA_NUM;
    int data_num = (int)DATA_NUM;
    int name_num = (int)NAME_NUM;

    int i,j;

    /* Allocate space to read in kernel code */
    kernel_src_str = (char *)malloc(MAX_SOURCE_SIZE);

    /* Allocate space for the result on the host side */
    result13 = (float *)malloc(point_num*sizeof(float)); /* average over 13 weeks */
    result26 = (float *)malloc(point_num*sizeof(float)); /* average over 26 weeks */

    /* Get Platform */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

    /* Get Device */
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id,
                         &ret_num_devices);

    /* Create Context */  
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command Queue */       
    command_queue = clCreateCommandQueue(context, device_id,
                                         CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);

    /* Read kernel source code */
    fp = fopen("moving_average_vec4.cl", "r");
    kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create Program Object */
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str,
                                        (const size_t *)&kernel_code_size, &ret);

    /* Compile kernel */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /* Create kernel */
    kernel13 = clCreateKernel(program, "moving_average_vec4", &ret); /* 13週用 */
    kernel26 = clCreateKernel(program, "moving_average_vec4", &ret); /* 26週用 */

    /* Create buffer for the input data on the device */
    memobjIn  = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               point_num * sizeof(int), NULL, &ret);

    /* Create buffer for the result on the device */  
    memobjOut13 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 point_num * sizeof(float), NULL, &ret); /* 13週用 */
    memobjOut26 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 point_num * sizeof(float), NULL, &ret); /* 26週用 */

    /* Copy input data to the global memory on the device*/
    ret = clEnqueueWriteBuffer(command_queue, memobjIn, CL_TRUE, 0,
                               point_num * sizeof(int),
                               stock_array_4, 0, NULL, NULL);

    /* Set Kernel Arguments (13 weeks) */
    ret = clSetKernelArg(kernel13, 0, sizeof(cl_mem), (void *)&memobjIn);
    ret = clSetKernelArg(kernel13, 1, sizeof(cl_mem), (void *)&memobjOut13);
    ret = clSetKernelArg(kernel13, 2, sizeof(int),    (void *)&data_num);
    ret = clSetKernelArg(kernel13, 3, sizeof(int),    (void *)&window_num_13);

    /* Submit task to compute the moving average over 13 weeks */  
    ret = clEnqueueTask(command_queue, kernel13, 0, NULL, &event13);

    /* Set Kernel Arguments (26 weeks) */
    ret = clSetKernelArg(kernel26, 0, sizeof(cl_mem), (void *)&memobjIn);
    ret = clSetKernelArg(kernel26, 1, sizeof(cl_mem), (void *)&memobjOut26);
    ret = clSetKernelArg(kernel26, 2, sizeof(int),    (void *)&data_num);
    ret = clSetKernelArg(kernel26, 3, sizeof(int),    (void *)&window_num_26);

    /* Submit task to compute the moving average over 26 weeks */ 
    ret = clEnqueueTask(command_queue, kernel26, 0, NULL, &event26);

    /* Copy result for the 13 weeks moving average from device to host */
    ret = clEnqueueReadBuffer(command_queue, memobjOut13, CL_TRUE, 0,
                              point_num * sizeof(float),
                              result13, 1, &event13, NULL);

    /* Copy result for the 26 weeks moving average from device to host */
    ret = clEnqueueReadBuffer(command_queue, memobjOut26, CL_TRUE, 0,
                              point_num * sizeof(float),
                              result26, 1, &event26, NULL);

    /* OpenCL Object Finalization */
    ret = clReleaseKernel(kernel13);
    ret = clReleaseKernel(kernel26);    
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobjIn);
    ret = clReleaseMemObject(memobjOut13);
    ret = clReleaseMemObject(memobjOut26);    
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /* Display results */
    for(i=window_num_26-1; i<data_num; i++) {
        printf( "result[%d]: ", i );
        for(j=0; j<name_num; j++ ) {
            /* Display whether the 13 week average is greater */
            printf( "[%d] ", (result13[i*NAME_NUM+j] >  result26[i*NAME_NUM+j]) );
        }
        printf( "\n" );
    }

    /* Deallocate memory on the host */ 
    free(result13);
    free(result26);
    free(kernel_src_str);

    return 0;
}

