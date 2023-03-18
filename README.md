# Cuda-Programs
This repository contains practice cuda programs done as part of my learning.

# Concept of block, grids, SMs and SPs
In CUDA, blocks and grids are used to organize and schedule parallel computations on the GPU. Each block represents a group of threads that can be executed in parallel, while a grid is a collection of blocks.
The mapping of blocks and grids to the GPGPU architecture depends on the specific GPU being used, as well as the configuration of that GPU. Generally speaking, a CUDA-enabled GPU consists of several Streaming Multiprocessors (SMs), each of which contains multiple processing cores called Streaming Processors (SPs). When a CUDA kernel is launched, the blocks are distributed across the available SMs in the GPU. Each SM schedules the execution of the threads within its assigned blocks, and each thread is executed on one of the available SPs within that SM.
The number of blocks and threads in a CUDA kernel can be chosen based on the specific problem being solved and the capabilities of the GPU being used. The optimal configuration will depend on factors such as the amount of parallelism in the problem, the memory requirements of the kernel, and the number of available processing units on the GPU.
