#include <stdlib.h>
#include <CL/cl.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

typedef struct mt_struct_s {
    cl_uint aaa;
    cl_int mm,nn,rr,ww;
    cl_uint wmask,umask,lmask;
    cl_int shift0, shift1, shiftB, shiftC;
    cl_uint maskB, maskC;
} mt_struct ;

#include "mts.h"

#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char **argv)
{
    cl_int num_rand_all = 4096*256*32;        /* The number of random numbers generated using one generator */
    cl_int num_rand_per_compute_unit, num_rand_per_generator;
    int count_all, i;
    cl_uint num_generator;
    unsigned int num_work_item;
    double pi;
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_platforms;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel_mt = NULL, kernel_pi = NULL;
    size_t kernel_code_size;
    char *kernel_src_str;
    cl_uint *result;
    cl_int ret;
    FILE *fp;
    cl_mem rand, count;

    size_t global_item_size_mt[3], local_item_size_mt[3];
    size_t global_item_size_pi[3], local_item_size_pi[3];

    cl_mem dev_mts;
    cl_event ev_mt_end, ev_pi_end, ev_copy_end;
    cl_ulong prof_start, prof_mt_end, prof_pi_end, prof_copy_end;
    cl_uint num_compute_unit, warp_per_compute_unit, num_warp, num_param_per_warp;
    int mts_size;
    mt_struct *mts = NULL;

    if (argc >= 2) {
        int n = atoi(argv[1]);
        if (n == 128) {
            mts = mts128;
            mts_size = sizeof(mts128);
            num_generator = 128;
            num_rand_per_generator = num_rand_all / 128;
        }
        if (n == 256) {
            mts = mts256;
            mts_size = sizeof(mts256);
            num_generator = 256;
            num_rand_per_generator = num_rand_all / 256;
        }
    }

    if (mts == NULL) {
        mts = mts512;
        mts_size = sizeof(mts512);
        num_generator = 512;
        num_rand_per_generator = num_rand_all / 512;
    }

    clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, 
                   &ret_num_devices);
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    result = (cl_uint*)malloc(sizeof(cl_uint)*num_generator);

    command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    fp = fopen("mt.cl", "r");
    kernel_src_str = (char*)malloc(MAX_SOURCE_SIZE);
    kernel_code_size = fread(kernel_src_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /* Create output buffer */
    rand = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*num_rand_all, NULL, &ret);
    count = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint)*num_generator, NULL, &ret);

    /* Build Program*/
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_str,
                                        (const size_t *)&kernel_code_size, &ret);
    ret = clBuildProgram(program, 1, &device_id, "", NULL, NULL);

    if (ret != CL_SUCCESS) {
        char buffer[1024];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 1024, buffer, NULL);
        puts(buffer);
    }

    kernel_mt = clCreateKernel(program, "genrand", &ret);
    kernel_pi = clCreateKernel(program, "calc_pi", &ret);

    /* Create input parameter */
    dev_mts = clCreateBuffer(context, CL_MEM_READ_WRITE, mts_size, NULL, &ret);
    clEnqueueWriteBuffer(command_queue, dev_mts, CL_TRUE, 0, mts_size, mts, 0, NULL, NULL);

    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &num_compute_unit, NULL);

    /* Each compute unit processes 4 warps */
    warp_per_compute_unit = 4;
    /* #warps = #warps per compute units * computeunits  */
    num_warp = warp_per_compute_unit * num_compute_unit ;

    /* #parameters per warp (rounded up) */
    num_param_per_warp = (num_generator + (num_warp-1)) / num_warp;

    /* 32 work-items per warp */
    num_work_item = 32 * warp_per_compute_unit;

    /* #random numbers per group */
    num_rand_per_compute_unit = (num_rand_all + (num_compute_unit-1)) / num_compute_unit;

    /* Set Kernel Arguments */
    clSetKernelArg(kernel_mt, 0, sizeof(cl_mem), (void*)&rand); /* Random numbers (output of genrand) */
    clSetKernelArg(kernel_mt, 1, sizeof(cl_mem), (void*)&dev_mts); /* MT parameter (input to genrand) */
    clSetKernelArg(kernel_mt, 2, sizeof(num_rand_per_generator), &num_rand_per_generator); /* Number of random numbers to generate for each parameter*/
    clSetKernelArg(kernel_mt, 3, sizeof(num_generator), &num_generator); /* Number of parameters to generate */
    clSetKernelArg(kernel_mt, 4, sizeof(num_param_per_warp), &num_param_per_warp); /* Number of parameters per work-group */
    clSetKernelArg(kernel_mt, 5, sizeof(cl_uint)*17*num_work_item, NULL); /* Local memory (state)*/
    clSetKernelArg(kernel_mt, 6, sizeof(mt_struct)*num_work_item, NULL); /* Local memory (mts) */

    clSetKernelArg(kernel_pi, 0, sizeof(cl_mem), (void*)&count); /* Counter for points within circle (output of calc_pi) */
    clSetKernelArg(kernel_pi, 1, sizeof(cl_mem), (void*)&rand);  /* Random numbers (input to calc_pi) */
    clSetKernelArg(kernel_pi, 2, sizeof(num_rand_per_compute_unit), &num_rand_per_compute_unit);   /* Number of random numbers per group */
    clSetKernelArg(kernel_pi, 3, sizeof(num_compute_unit), &num_compute_unit); /* Number of parameters */
    clSetKernelArg(kernel_pi, 4, sizeof(num_rand_all), &num_rand_all); /* Number of random numbers */
    clSetKernelArg(kernel_pi, 5, sizeof(cl_uint)*128, NULL); /* Memory allocated for counter */


    /* Specify the number of work-groups and work-items */
    global_item_size_mt[0] = num_work_item * num_compute_unit; global_item_size_mt[1] = 1; global_item_size_mt[2] = 1;
    local_item_size_mt[0] = num_work_item;  local_item_size_mt[1] = 1;  local_item_size_mt[2] = 1;

    /* Create a random number array */
    clEnqueueNDRangeKernel(command_queue, kernel_mt, 1, NULL, global_item_size_mt, local_item_size_mt, 0, NULL, &ev_mt_end);

    global_item_size_pi[0] = num_compute_unit * 128; global_item_size_pi[1] = 1; global_item_size_pi[2] = 1;
    local_item_size_pi[0] = 128;  local_item_size_pi[1] = 1;  local_item_size_pi[2] = 1;

    /* Compute PI */
    ret = clEnqueueNDRangeKernel(command_queue, kernel_pi, 1, NULL, global_item_size_pi, local_item_size_pi, 0, NULL, &ev_pi_end);

    /* Get result */
    clEnqueueReadBuffer(command_queue, count, CL_TRUE, 0, sizeof(cl_uint)*num_generator, result, 0, NULL, &ev_copy_end);

    /* Average the values of PI */
    count_all = 0;
    for (i=0; i<(int)num_compute_unit; i++) {
        count_all += result[i];
    }

    pi = ((double)count_all)/(num_rand_all) * 4;
    printf("pi = %.20f\n", pi);

    /* Get execution time info */
    clGetEventProfilingInfo(ev_mt_end, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &prof_start, NULL);
    clGetEventProfilingInfo(ev_mt_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &prof_mt_end, NULL);
    clGetEventProfilingInfo(ev_pi_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &prof_pi_end, NULL);
    clGetEventProfilingInfo(ev_copy_end, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &prof_copy_end, NULL);

    printf("    mt: %f[ms]\n"
           "    pi: %f[ms]\n"
           "    copy: %f[ms]\n",
           (prof_mt_end - prof_start)/(1000000.0),
           (prof_pi_end - prof_mt_end)/(1000000.0),
           (prof_copy_end - prof_pi_end)/(1000000.0));

    clReleaseEvent(ev_mt_end);
    clReleaseEvent(ev_pi_end);
    clReleaseEvent(ev_copy_end);

    clReleaseMemObject(rand);
    clReleaseMemObject(count);
    clReleaseKernel(kernel_mt);
    clReleaseKernel(kernel_pi);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    free(kernel_src_str);
    free(result);
    return 0;
}
