#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "pgm.h"

#define MAX_SOURCE_SIZE (0x100000)

#define AMP(a, b) (sqrt((a)*(a)+(b)*(b)))

cl_context context = NULL;
cl_command_queue queue = NULL;
cl_program program = NULL;
 
cl_ulong prof_writ = 0;
cl_ulong prof_sfac = 0;
cl_ulong prof_brev = 0;
cl_ulong prof_bfly = 0;
cl_ulong prof_norm = 0;
cl_ulong prof_trns = 0;
cl_ulong prof_hpfl = 0;
cl_ulong prof_read = 0;

enum Mode {
    forward = 0,
    inverse = 1
};

cl_ulong checkTime(cl_event ev)
{
    cl_ulong start;
    cl_ulong end;
    
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &end,   NULL);

    return (end - start);
}

int setWorkSize(size_t* gws, size_t* lws, cl_int x, cl_int y)
{
    switch(y) {
        case 1:
            gws[0] = x;
            gws[1] = 1;
            lws[0] = 256;
            lws[1] = 1;
            break;
        default:
            gws[0] = x;
            gws[1] = y;
            lws[0] = 16;
            lws[1] = 32;
            break;
    }

    return 0;
}

int fftCore(cl_mem dst, cl_mem src, cl_mem spin, cl_int m, enum Mode direction)
{
    cl_int ret;

    cl_int iter;
    cl_uint flag;

    cl_int n = 1<<m;

    cl_event ev_brev;
    cl_event ev_bfly;
    cl_event ev_norm;

    cl_kernel brev = NULL;
    cl_kernel bfly = NULL;
    cl_kernel norm = NULL;

   
    brev = clCreateKernel(program, "bitReverse",   &ret);
    bfly = clCreateKernel(program, "butterfly",    &ret);
    norm = clCreateKernel(program, "norm",    &ret);

    size_t gws[2];
    size_t lws[2];

    switch (direction) {
        case forward: flag = 0x00000000; break;
        case inverse: flag = 0x80000000; break;
    }

    ret = clSetKernelArg(brev, 0, sizeof(cl_mem), (void *)&dst);
    ret = clSetKernelArg(brev, 1, sizeof(cl_mem), (void *)&src);
    ret = clSetKernelArg(brev, 2, sizeof(cl_int), (void *)&m);
    ret = clSetKernelArg(brev, 3, sizeof(cl_int), (void *)&n);

    ret = clSetKernelArg(bfly, 0, sizeof(cl_mem), (void *)&dst);
    ret = clSetKernelArg(bfly, 1, sizeof(cl_mem), (void *)&spin);
    ret = clSetKernelArg(bfly, 2, sizeof(cl_int), (void *)&m);
    ret = clSetKernelArg(bfly, 3, sizeof(cl_int), (void *)&n);
    ret = clSetKernelArg(bfly, 5, sizeof(cl_uint), (void *)&flag);

    ret = clSetKernelArg(norm, 0, sizeof(cl_mem), (void *)&dst);
    ret = clSetKernelArg(norm, 1, sizeof(cl_int), (void *)&n);

    /* Reverse bit ordering */
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, brev, 2, NULL, gws, lws, 0, NULL, &ev_brev);
    ret = clWaitForEvents(1, &ev_brev);
    prof_brev += checkTime(ev_brev);

    
    /* Perform Butterfly Operations*/
    setWorkSize(gws, lws, n/2, n);
    for(iter=1; iter<=m; iter++){
        ret = clSetKernelArg(bfly, 4, sizeof(cl_int), (void *)&iter);
        ret = clEnqueueNDRangeKernel(queue, bfly, 2, NULL, gws, lws, 0, NULL, &ev_bfly);
        ret = clWaitForEvents(1, &ev_bfly);
    
        prof_bfly += checkTime(ev_bfly);
    
    }
    
    if (direction == inverse) {
        setWorkSize(gws, lws, n, n);
        ret = clEnqueueNDRangeKernel(queue, norm, 2, NULL, gws, lws, 0, NULL, &ev_norm);
        ret = clWaitForEvents(1, &ev_norm);
        
        prof_norm += checkTime(ev_norm);
    }

    ret = clReleaseKernel(bfly);
    ret = clReleaseKernel(brev);
    ret = clReleaseKernel(norm);

    return 0;
}

int main()
{
    cl_mem xmobj = NULL;
    cl_mem rmobj = NULL;
    cl_mem wmobj = NULL;
    
    cl_event ev_sfac;
    cl_event ev_trns;
    cl_event ev_hpfl;
    cl_event ev_writ;
    cl_event ev_read;
    
    cl_kernel sfac = NULL;
    cl_kernel trns = NULL;
    cl_kernel hpfl = NULL;

    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;

    cl_uint ret_num_platforms;
    cl_uint ret_num_devices;

    cl_int ret;

    cl_float2 *xm;
    cl_float2 *rm;
    cl_float2 *wm;

    pgm_t ipgm;
    pgm_t opgm;

    FILE *fp;
    const char fileName[] = "./fft.cl";
    size_t source_size;
    char *source_str;
    cl_int i, j;
    cl_int n;
    cl_int m;

    size_t gws[2];
    size_t lws[2];

    /* Load kernel source code */
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp );
    fclose( fp );

    /* Read image */
    readPGM(&ipgm, "lena.pgm");

    n = ipgm.width; 
    m = (cl_int)(log((double)n)/log(2.0));

    xm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
    rm = (cl_float2 *)malloc(n * n * sizeof(cl_float2));
    wm = (cl_float2 *)malloc(n / 2 * sizeof(cl_float2));

    for( i = 0; i < n; i++ ) {
        for( j = 0; j < n; j++ ) {
            ((float*)xm)[(2*n*j)+2*i+0] = (float)ipgm.buf[n*j+i];
            ((float*)xm)[(2*n*j)+2*i+1] = (float)0;
        }
    }

    /* Get platform/device  */
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    /* Create OpenCL context */
    context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

    /* Create Command queue */
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &ret);

    /* Create Buffer Objects */
    xmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
    rmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, n*n*sizeof(cl_float2), NULL, &ret);
    wmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, (n/2)*sizeof(cl_float2), NULL, &ret);

    /* Transfer data to memory buffer */
    ret = clEnqueueWriteBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, &ev_writ);

    /* Create kernel program from source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

    /* Build kernel program */
    ret = clBuildProgram(program, 1, &device_id, NULL , NULL, NULL);

    /* Create OpenCL Kernel */
    sfac = clCreateKernel(program, "spinFact",     &ret);
    trns = clCreateKernel(program, "transpose",    &ret);
    hpfl = clCreateKernel(program, "hiPassFilter", &ret);

    /* Create spin factor */
    ret = clSetKernelArg(sfac, 0, sizeof(cl_mem), (void *)&wmobj);
    ret = clSetKernelArg(sfac, 1, sizeof(cl_int), (void *)&n);
    setWorkSize(gws, lws, n/2, 1);
    ret = clEnqueueNDRangeKernel(queue, sfac, 1, NULL, gws, lws, 0, NULL, &ev_sfac);
    
    /* Butterfly Operation */
    fftCore(rmobj, xmobj, wmobj, m, forward);

    /* Transpose matrix */
    ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&xmobj);
    ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&rmobj);
    ret = clSetKernelArg(trns, 2, sizeof(cl_int), (void *)&n);
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL, &ev_trns);
    clWaitForEvents(1, &ev_trns);

    prof_trns += checkTime(ev_trns);

    /* Butterfly Operation */
    fftCore(rmobj, xmobj, wmobj, m, forward);

    /* Apply high-pass filter */
    cl_int radius = n/8;
    ret = clSetKernelArg(hpfl, 0, sizeof(cl_mem), (void *)&rmobj);
    ret = clSetKernelArg(hpfl, 1, sizeof(cl_int), (void *)&n);
    ret = clSetKernelArg(hpfl, 2, sizeof(cl_int), (void *)&radius);
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, hpfl, 2, NULL, gws, lws, 0, NULL, &ev_hpfl);

    /* Inverse FFT */

    /* Butterfly Operation */
    fftCore(xmobj, rmobj, wmobj, m, inverse);

    /* Transpose matrix */
    ret = clSetKernelArg(trns, 0, sizeof(cl_mem), (void *)&rmobj);
    ret = clSetKernelArg(trns, 1, sizeof(cl_mem), (void *)&xmobj);
    setWorkSize(gws, lws, n, n);
    ret = clEnqueueNDRangeKernel(queue, trns, 2, NULL, gws, lws, 0, NULL, NULL);
    clWaitForEvents(1, &ev_trns);
    
    prof_trns += checkTime(ev_trns);

    /* Butterfly Operation */
    fftCore(xmobj, rmobj, wmobj, m, inverse);
    
    /* Read data from memory buffer */
    ret = clEnqueueReadBuffer(queue, xmobj, CL_TRUE, 0, n*n*sizeof(cl_float2), xm, 0, NULL, &ev_read);

    /* Calculate amplitude */
    float* ampd;
    ampd = (float*)malloc(n*n*sizeof(float));
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            ampd[n*((i))+((j))] = (AMP(((float*)xm)[(2*n*i)+2*j], ((float*)xm)[(2*n*i)+2*j+1]));
        }
    }
    opgm.width = n;
    opgm.height = n;
    normalizeF2PGM(&opgm, ampd);
    free(ampd);

    /* Write out image */
    writePGM(&opgm, "output.pgm");


    /* Time Measurement */
    prof_writ += checkTime(ev_writ);
    prof_sfac += checkTime(ev_sfac);
    prof_hpfl += checkTime(ev_hpfl);
    prof_read += checkTime(ev_read);
    
    printf(" membuf write: %10.5f [ms]\n", (prof_writ)/1000000.0);
    printf("   spinFactor: %10.5f [ms]\n", (prof_sfac)/1000000.0);
    printf("   bitReverse: %10.5f [ms]\n", (prof_brev)/1000000.0);
    printf("    butterfly: %10.5f [ms]\n", (prof_bfly)/1000000.0);
    printf("    normalize: %10.5f [ms]\n", (prof_norm)/1000000.0);
    printf("    transpose: %10.5f [ms]\n", (prof_trns)/1000000.0);
    printf(" hiPassFilter: %10.5f [ms]\n", (prof_hpfl)/1000000.0);
    printf("  membuf read: %10.5f [ms]\n", (prof_read)/1000000.0);

    /* Finalizations*/
    ret = clFlush(queue);
    ret = clFinish(queue);
    ret = clReleaseKernel(hpfl);
    ret = clReleaseKernel(trns);
    ret = clReleaseKernel(sfac);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(xmobj);
    ret = clReleaseMemObject(rmobj);
    ret = clReleaseMemObject(wmobj);
    ret = clReleaseCommandQueue(queue);
    ret = clReleaseContext(context);

    destroyPGM(&ipgm);
    destroyPGM(&opgm);

    free(source_str);
    free(wm);
    free(rm);
    free(xm);

    return 0;
}
