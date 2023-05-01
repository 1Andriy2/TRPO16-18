#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>

using namespace std;

#define INF 999999
#define MAX_NODES 100
#define WIDTH 800
#define HEIGHT 800
#define MAX_ITERATIONS 10000

//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__device__ unsigned char computePixel(float x, float y, float a) {
	float suma = 0;
	float lastX = x;

	for (int i = 0; i < MAX_ITERATIONS; i++)
	{
		float newX = a * lastX * (1 - lastX);
		suma += logf(fabsf(a * (1 - 2 * lastX)));
		
		if (i > 100)
		{
			if (fabsf(newX - lastX) < 1e-6)
			{
				return (unsigned char)(suma * 255.0 / MAX_ITERATIONS);
			}
		}
		
		lastX = newX;
	}

	return 0;
}

__global__ void fractal(unsigned char* image, float aMin, float aMax, float bMin, float bMax, float dx, float dy) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float a = aMin + col * dx;
	float b = bMin + row * dy;
	float x = 0.5;
	float y = 0.5;

	unsigned char value = computePixel(x, y, a);
	image[row * WIDTH + col] = value;
}

__global__ void addKernel(int a, int b,int l, int *c)
{
    *c = a / b - l;
}

__global__ void addArrays(int* a, int* b, int* c) {
	int indx = threadIdx.x; 
	c[indx] = a[indx] - b[indx] + 2*a[indx]*b[indx];
}

__global__ void dijkstra(int* adjMatrix, int* dist, int* visited, int startNode, int numNodes) {
	int i, j, u, v, minDist;
	u = threadIdx.x;
	for (int i = 0; i < numNodes; i++)
	{
		dist[u * numNodes + i] = adjMatrix[u * numNodes + i];
		visited[i] = 0;
	}

	visited[startNode] = 1;

	for (int i = 0; i < numNodes - 1; i++)
	{
		minDist = INF;
		for (int j = 0; j < numNodes; j++)
		{
			if (!visited[j] && dist[u * numNodes + j] < minDist)
			{
				minDist = dist[u * numNodes + j];
				v = j;
			}
		}
		visited[v] = 1;
		for (int j = 0; j < numNodes; j++)
		{
			if (!visited[j] && dist[u * numNodes + v] + adjMatrix[v * numNodes + j] < dist[u * numNodes + j])
			{
				dist[u * numNodes + j] = dist[u * numNodes + v] + adjMatrix[v * numNodes + j];
			}
		}
	}
}

int main()
{
  /*  const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };*/

    // Add vectors in parallel.
    /*cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }*/

    /*printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);*/

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    /*cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

	//printf("Krysa Volodymyr: \n");
	//printf("19 laba - ");
	//unsigned char *image;
	//cudaMallocManaged(&image, WIDTH * HEIGHT * sizeof(unsigned char));
	//float aMin = 2.4, aMax = 4.0, bMin = 0.1, bMax = 0.9;
	//float dx = (aMax - aMin) / WIDTH;
	//float dy = (bMax - bMin) / HEIGHT;

	//dim3 blocks(WIDTH / 16, HEIGHT / 16);
	//dim3 threads(16,16);

	//fractal<<<blocks, threads>>>(image, aMin, aMax, bMin, bMax, dx, dy);
	//cudaDeviceSynchronize();

	//FILE *file = fopen("fractal.pgm", "wb");
	//fprintf(file, "P5\n%d %d\n255\n", WIDTH, HEIGHT);
	////fwrite(image,sizeof(unsigned char), WIDTH * HEIGHT, file);
	//fclose(file);
	//cudaFree(image);

	printf("18 laba: result = ");
	int numNodes = 5;
	int startNode = 0;

	int adjMatrix[MAX_NODES][MAX_NODES] = {
		{ 0, 3, INF, 1, INF },
		{ 3, 0, 3, 2, INF },
		{ INF, 3, 0, INF, 1 },
		{ 1, 2, INF, 0, 4 },
		{ INF, INF, 1, 4, 0 }
	};

	int dvsSizeDist = MAX_NODES * MAX_NODES * sizeof(int);
	int *deviceAdjMatrix, *deviceDist, *deviceVisited;
	cudaMalloc((void**)&deviceAdjMatrix, dvsSizeDist);
	cudaMalloc((void**)&deviceDist, dvsSizeDist);
	cudaMalloc((void**)&deviceVisited, dvsSizeDist);

	cudaMemcpy(deviceAdjMatrix, adjMatrix, dvsSizeDist, cudaMemcpyKind::cudaMemcpyHostToDevice);
	dijkstra<<<1, numNodes>>>(deviceAdjMatrix, deviceDist, deviceVisited, startNode, numNodes);
	
	int *dist = (int*)malloc(dvsSizeDist);
	cudaMemcpy(dist, deviceDist, dvsSizeDist, cudaMemcpyKind::cudaMemcpyDeviceToHost);

	printf("Shortest distances from node %d to node %d:\n", startNode, endNode);
	//printf("Distance: %d\n", dist[endNode]);
	for (int i = 0; i < numNodes; i++)
	{
		printf("Node %d: %d\n", i, dist[i]);
	}
	
	cudaFree(deviceAdjMatrix);
	cudaFree(deviceDist);
	cudaFree(deviceVisited);
	free(dist);

	
	printf("16 laba: ");
	int c; 
	int *dev_c;
	cudaMalloc((void**)&dev_c, sizeof(int));
	addKernel<<<1, 1>>>(456,2,5,dev_c);
	cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	printf("result = %d\n", c);
	cudaFree(dev_c);

	printf("17 laba: result = ");
	int ha[] = { 10, 20, 30, 40, 50 }; 
	int hb[] = { 1, 2, 3, 4, 5 };
	int hc[5];

	int *da, *db, *dc;
	int size = sizeof(int) * 5;
	cudaMalloc((void**)&da, size);
	cudaMalloc((void**)&db, size);
	cudaMalloc((void**)&dc, size);

	cudaMemcpy(da, ha, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

	addArrays<<<1,5>>>(da, db, dc);
	cudaMemcpy(hc, dc, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	for (int i = 0; i < 5; i++)
	{
		cout << hc[i] << "\t";
	}
	cout << endl;

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	
	getchar(); 
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
//cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
//{
//    int *dev_a = 0;
//    int *dev_b = 0;
//    int *dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate GPU buffers for three vectors (two input, one output)    .
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;Z\
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//    
//    return cudaStatus;
//}
