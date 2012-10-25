#!/bin/bash 
for i in {1..3000}; do /home/thiago/repositorios/testesopencl/testedememoria/memory >> /home/thiago/repositorios/tcc/resultados_opencl_memory.txt; done

for i in {1..3000}; do /home/thiago/repositorios/testesopencl/matrixmulti/matrix >> /home/thiago/repositorios/tcc/resultados_opencl_process.txt; done

for i in {1..3000}; do /home/thiago/repositorios/Testes-Cuda/memorybound/memory >> /home/thiago/repositorios/tcc/resultados_cuda_memory.txt; done

for i in {1..3000}; do /home/thiago/repositorios/Testes-Cuda/processbound/process >> /home/thiago/repositorios/tcc/resultados_cuda_process.txt; done