#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>

#define MAX_SOURCE_SIZE (0x100000)

int main() {
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL;
    size_t kernel_code_size, local_size_size;
    char *kernel_src_str;
    cl_int ret;
    FILE *fp;
    cl_ulong local_size;
    cl_int cl_local_size;
    cl_event ev;

    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, 
                   &ret_num_devices);
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    fp = fopen("local.cl", "r");
    kernel_src_str = (char*)malloc(MAX_SOURCE_SIZE);
    kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Get available local memory size */
    clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_size), &local_size, &local_size_size);
    printf("CL_DEVICE_LOCAL_MEM_SIZE = %d\n", (int)local_size);

    /* Build Program */
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str,
                                        (const size_t *)&kernel_code_size, &ret);
    clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    kernel = clCreateKernel(program, "local_test", &ret);

    cl_local_size = local_size / 2;

    /* Set kernel argument */
    clSetKernelArg(kernel, 0, cl_local_size, NULL);
    clSetKernelArg(kernel, 1, sizeof(cl_local_size), &cl_local_size);

    /* Execute kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL, &ev);
    if (ret == CL_OUT_OF_RESOURCES) {
        puts("too large local");
        return 1;
    }
    /* Wait for the kernel to finish */
    clWaitForEvents(1, &ev);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    free(kernel_src_str);

    return 0;
}
