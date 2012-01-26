#include <stdio.h>

#include "cll.h"
#include "util.h"

#define MN 3 // Magic Number
void CL::popCorn()
{
    printf("in popCorn\n");

    //initialize our kernel from the program
    //kernel = clCreateKernel(program, "part1", &err);
    //printf("clCreateKernel: %s\n", oclErrorString(err));
    try{
        kernel = cl::Kernel(program, "part1", &err);
    }
    catch (cl::Error er) {
        printf("ERROR: %s(%d)\n", er.what(), er.err());
    }

       

    //initialize our CPU memory arrays, send them to the device and set the kernel arguements
    int a[MN][MN];
    int b[MN][MN];
    int c[MN][MN];
    for(int i=0; i < MN; i++)
    {
	for(int j = 0; j < MN; j++) {
	    a[i][j] = i+j;
            b[i][j] = i+j;
            c[i][j] = 0;
	}
    }

    printf("Creating OpenCL arrays\n");
    size_t array_size = sizeof(int) * MN * MN;
    //our input arrays
    ///cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num, NULL, &err);
    ///cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * num, NULL, &err);
    cl_a = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);
    cl_b = cl::Buffer(context, CL_MEM_READ_ONLY, array_size, NULL, &err);
    //our output array
    //cl_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num, NULL, &err);
    cl_c = cl::Buffer(context, CL_MEM_WRITE_ONLY, array_size, NULL, &err);
    printf("Pushing data to the GPU\n");
    //push our CPU arrays to the GPU
    ///err = clEnqueueWriteBuffer(command_queue, cl_a, CL_TRUE, 0, sizeof(float) * num, a, 0, NULL, &event);
    err = queue.enqueueWriteBuffer(cl_a, CL_TRUE, 0, array_size, a, NULL, &event);
    ///clReleaseEvent(event); //we need to release events in order to be completely clean (has to do with openclprof)
    ///err = clEnqueueWriteBuffer(command_queue, cl_b, CL_TRUE, 0, sizeof(float) * num, b, 0, NULL, &event);
    err = queue.enqueueWriteBuffer(cl_b, CL_TRUE, 0, array_size, b, NULL, &event);
    ///clReleaseEvent(event);
    ///err = clEnqueueWriteBuffer(command_queue, cl_c, CL_TRUE, 0, sizeof(float) * num, c, 0, NULL, &event);
    err = queue.enqueueWriteBuffer(cl_c, CL_TRUE, 0, array_size, c, NULL, &event);
    ///clReleaseEvent(event);
    queue.finish();
    int teste[MN][MN];
    err = queue.enqueueReadBuffer(cl_a, CL_TRUE, 0, sizeof(float) * MN * MN, &teste, NULL, &event);
    for( int i = 0; i < MN; i++ ){
	for( int j = 0; j < MN; j++ ) printf("%d  ", teste[i][j]);
	printf("\n");
    }
    //set the arguements of our kernel
    ///err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &cl_a);
    ///err  = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &cl_b);
    ///err  = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &cl_c);
    err = kernel.setArg(0, cl_a);
    err = kernel.setArg(1, cl_b);
    err = kernel.setArg(2, cl_c);
    //Wait for the command queue to finish these commands before proceeding
    ///clFinish(command_queue);
    queue.finish();
 
    //for now we make the workgroup size the same as the number of elements in our arrays
    //workGroupSize[0] = num;
    //delete a;
    //delete b;
    //delete c;
}



void CL::runKernel()
{
    printf("in runKernel\n");
    //execute the kernel
    ///err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &event);
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(MN*MN), cl::NullRange, NULL, &event); 
    ///clReleaseEvent(event);
    printf("clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
    ///clFinish(command_queue);
    queue.finish();

    //lets check our calculations by reading from the device memory and printing out the results
    int c_done[MN][MN];
    ///err = clEnqueueReadBuffer(command_queue, cl_c, CL_TRUE, 0, sizeof(float) * num, c_done, 0, NULL, &event);
    err = queue.enqueueReadBuffer(cl_c, CL_TRUE, 0, sizeof(float) * MN * MN, &c_done, NULL, &event);
    printf("clEnqueueReadBuffer: %s\n", oclErrorString(err));
    //clReleaseEvent(event);

    for(int i=0; i < MN; i++)
    {
	for(int j = 0; j < MN; j++) printf("%d  ", c_done[i][j]);
	printf("\n");
    }
}


