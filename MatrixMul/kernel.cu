
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int SIZE = 150;

cudaError_t cudaStatus;

__global__ void matrixMul(int* c, const int* a, const int* b)
{	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < SIZE && j < SIZE)
	{
		int temp = 0;
		for (int k = 0; k < SIZE; k++)
		{
			temp += a[i * SIZE + k] * b[k * SIZE + j];
		}
		c[i * SIZE + j] = temp;
	}
}

int main()
{	// Square Matrix Multiplication

	int a[SIZE][SIZE];
	int b[SIZE][SIZE];
	int c[SIZE][SIZE] = { {0} };

	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			a[i][j] = 1;
			b[i][j] = 1;
		}
	}
	

	// Device memory allocation
	int* dev_a;
	int* dev_b;
	int* dev_c;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	cudaStatus = cudaMalloc((void**)&dev_a, SIZE * SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMalloc((void**)&dev_b, SIZE * SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_c, SIZE * SIZE * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaMemset(dev_c, 0, SIZE * SIZE * sizeof(int));

	cudaStatus = cudaMemcpy(dev_a, a, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(dev_b, b, SIZE * SIZE * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks((SIZE + 31) / 32, (SIZE + 31) / 32); 

	// Kernel call
	matrixMul << <numBlocks, threadsPerBlock >> > (dev_c, dev_a, dev_b);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(cudaStatus));
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	cudaStatus = cudaMemcpy(c, dev_c, SIZE * SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	
	// Free device memory
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);



	// Displaying the result
	printf("Matrix C:\n");
	for (int i = 0; i < SIZE; i++)
	{
		for (int j = 0; j < SIZE; j++)
		{
			printf("%d ", c[i][j]);
		}
		printf("\n");
	}

}