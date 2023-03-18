#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

// Kernal 
__global__ void hadamard(float *A, float *B, float *C, int M, int N)
{
	/*
		Threads are arranged in 2-D thread-blocks in a 2-D grid. 
		CUDA provides a simple indexing mechanism to obtain the
		thread-ID within a thread-block (threadIdx.x, threadIdx.y and threadIdx.z) 
		and block-ID within a grid (blockIdx.x, blockIdx.y and blockIdx.z) . 
		In our case rows are indexed in the y-dimension. 
		To compute the index of row r in terms of CUDA threadIdx and blockIdx,
		we can take blockIdx.y and multiply it with blockDim.y 
		to get the total number of threads up to blockIdx.y number of blocks. 
		Then we add threadIdx.y which is the thread-ID 
		along y-dimension within the block this thread belongs to.
		Column index can be computed similarly along x-dimension.
	*/
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	/*
	Because we take the ceiling of n/BLOCK_SIZE and m/BLOCK_SIZE 
	CUDA kernel launcher will launch more threads than we need. 
	Therefore we need values M and N (dimensions of C matrix) 
	to check if a given thread computes a valid element in the output matrix.	
	*/

	if(row > M || col > N) return;
	
	/*
		Since A and B are laid out in memory in row-major order
		we can access all elements in row "row" of A using A[row*width + col] (0≤col≤width)
		and row "row" of B using B[row*width + col] (0≤col≤width)
		here, width is the total number of threads in y-dimension of the grid and 
		width can be calculated as gridDim.y * blockDim.y
	*/
	
	int width = gridDim.y * blockDim.y;
	int ofs = row * width  + col;
	C[ofs] = A[ofs]*B[ofs];
}

/**
 * Host main routine
 */
void print_matrix(float *A,int m,int n)
{
	for(int i =0;i<m;i++)
	{
		for(int j=0;j<n;j++)
			printf("%.2f ",A[i*n+j]);
		printf("\n");
	}

}
int main(void)
{
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Print the vector length to be used, and compute its size
	
	int t; //number of test cases
	scanf("%d",&t);
	while(t--)
	{
		int m,n;
		scanf("%d %d",&m,&n);
		size_t size = m*n * sizeof(float);
		printf("[Hadamard product of two matrices ]\n");

		float *h_A = (float*)malloc(size); // Allocate the host input vector A
		
		float *h_B = (float*)malloc(size);// Allocate the host input vector B
		
		float *h_C = (float*)malloc(size);// Allocate the host output vector C
		

		// Verify that allocations succeeded
		if (h_A == NULL || h_B == NULL || h_C == NULL)
		{
			fprintf(stderr, "Failed to allocate host vectors!\n");
			exit(EXIT_FAILURE);
		}

		// Initialize the host input vectors
		
		for (int i = 0; i < n*m; ++i)
		{
			scanf("%f",&h_A[i]);
			

		}
		for (int i = 0; i < n*m; ++i)
		{
		   
			scanf("%f",&h_B[i]);

		}
		
		
		// Allocate the device input vector A
		float *d_A = NULL;
		err = cudaMalloc((void**)&d_A,size);
		if(err != cudaSuccess)
		{
			fprintf(stderr,"failed to allocate device vector A (error code %s)!",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Allocate the device input vector B
		float *d_B = NULL;
		err = cudaMalloc((void**)&d_B,size);
		if(err != cudaSuccess)
		{
			fprintf(stderr,"failed to allocate device vector B (error code %s)!",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Allocate the device output vector C
		float *d_C = NULL;
		err = cudaMalloc((void**)&d_C,size);
		if(err != cudaSuccess)
		{
			fprintf(stderr,"failed to allocate device vector C (error code %s)!",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Copy the host input vectors A and B in host memory to the device input vectors in
		// device memory
		
		err = cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
		{
			fprintf(stderr,"failed to copy vector A from host to device (error code %s)!",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		err = cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
		{
			fprintf(stderr,"failed to copy vector B from host to device (error code %s)!",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		/*
			initialize blocksPerGrid and threads Per Block
			Sine we are multiplying 2-D matrices 
			it only makes sense to arrange the thread-blocks and grid in 2-D.
			We are assuming a 32 x 32 2-D thread-block.
			Let’s assume that our thread-block size is BLOCK_SIZE x BLOCK_SIZE
		*/
		dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);
		
		/*
			Now how should we arrange our grid? Since the output matrix C is m × n,
			we need to have at least m/BLOCK_SIZE number of thread-blocks in y-dimension
			and n/BLOCK_SIZE number of thread-blocks in x-dimension
			So block and grid dimension can be specified as follows using CUDA.
			Here I assumed that columns in the matrix are indexed in x-dimension 
			and rows in y-dimension. So x-dimension of the grid will have 
			n/BLOCK_SIZE blocks and y-dimension of the grid will have 
			m/BLOCK_SIZE blocks
		*/
		dim3 dim_grid(ceilf(n/(float)BLOCK_SIZE), ceilf(m/(float)BLOCK_SIZE), 1);
		

		hadamard<<<dim_grid, dim_block>>>(d_A, d_B, d_C, m, n);
		err = cudaGetLastError();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Copy the device result vector in device memory to the host result vector
		// in host memory.
		err = cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
		if(err != cudaSuccess)
		{
			fprintf(stderr,"failed to copy vector C from device to host (error code %s)!\n ",cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		// Verify that the result vector is correct
		for (int i = 0; i < n*m; ++i)
		{
			if (fabs(h_A[i] * h_B[i] - h_C[i]) > 1e-5)
			{
				fprintf(stderr, "Result verification failed at element %d!\n", i);
				exit(EXIT_FAILURE);
			}
		}

		 printf("Test PASSED\n");

		// Free device global memory
		cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

		// Free host memory
		free(h_A); free(h_B); free(h_C);
		
		err = cudaDeviceReset();

		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		print_matrix(h_C,m,n);
		
		 printf("Done\n");
	}
	return 0;
}

