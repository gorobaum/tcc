#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_BINARY_SIZE (0x100000)

int main()
{
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem memobj = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    float mem[MEM_SIZE];

    FILE *fp;
    char fileName[] = "./kernel.clbin";
    size_t binary_size;
    char *binary_buf;
    cl_int binary_status;
    cl_int i;

    /* Load kernel source code */
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    binary_buf = (char *)malloc(MAX_BINARY_SIZE);
    binary_size = fread( binary_buf, 1, MAX_BINARY_SIZE, fp );
    fclose( fp );

    /*Initialize Data */
    for( i = 0; i < MEM_SIZE; i++ ) {
        mem[i] = i;
    }

    /* Get platform/device information */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    /* Create OpenCL Context */
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
  
    /* Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  
    /* Create memory buffer*/
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE * sizeof(float), NULL, &ret);

    /* Transfer data to memory buffer */
    ret = clEnqueueWriteBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);

    /* Create Kernel program from the read in source */
    program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&binary_size, 
                                        (const unsigned char **)&binary_buf, &binary_status, &ret);
    
    /* Build Kernel Program */
    kernel = clCreateKernel(program, "vecAdd", &ret);
    printf("err:%d\n", ret);

    /* Create OpenCL Kernel */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);

    size_t global_work_size[3] = {MEM_SIZE, 0, 0};
    size_t local_work_size[3]  = {MEM_SIZE, 0, 0};

    /* Execute OpenCL kernel */
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    /* Transfer result from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, MEM_SIZE * sizeof(float), mem, 0, NULL, NULL);

    /* Display result */
    for(i=0; i<MEM_SIZE; i++) {
        printf("mem[%d] : %f\n", i, mem[i]);
    }
  
    /* Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    free(binary_buf);

    return 0;
}

