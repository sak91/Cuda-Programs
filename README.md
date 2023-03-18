# Cuda-Programs
This repository contains practice cuda programs done as part of my learning.



# Compute Unified Device Architecture:
CUDA C is an extension of C programming language with special constructs for supporting parallel computing. CUDA programmer perspective - CPU is a host : dispatches parallel jobs to GPU devices. 

# Concept of blocks, grids, SMs and SPs
In CUDA, blocks and grids are used to organize and schedule parallel computations on the GPU. Each block represents a group of threads that can be executed in parallel, while a grid is a collection of blocks.
The mapping of blocks and grids to the GPGPU architecture depends on the specific GPU being used, as well as the configuration of that GPU. Generally speaking, a CUDA-enabled GPU consists of several Streaming Multiprocessors (SMs), each of which contains multiple processing cores called Streaming Processors (SPs). When a CUDA kernel is launched, the blocks are distributed across the available SMs in the GPU. Each SM schedules the execution of the threads within its assigned blocks, and each thread is executed on one of the available SPs within that SM.
The number of blocks and threads in a CUDA kernel can be chosen based on the specific problem being solved and the capabilities of the GPU being used. The optimal configuration will depend on factors such as the amount of parallelism in the problem, the memory requirements of the kernel, and the number of available processing units on the GPU.


# Program 1 - The Hadamard Product Operation (pg1.cu)
This program computes the Hadamard product operation for two matrices of floating point numbers. 
Given input matrices A and B, both of size m*n, the Hadamard product is an elementwise product operation yielding an output matrix C such that every element 
C[i][j] = A[i][j] * B[i][j].
The program takes as input i)number of test cases ii)values of m and n depecting dimentions of input matrices A and B ii) The matrix elements of A iv) matrix elements of B, performs the Hadamard product operation and prints the resultant matrix C.
