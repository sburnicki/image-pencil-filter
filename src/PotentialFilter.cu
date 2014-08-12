#include "PotentialFilter.h"

// TODO: remove/resolve duplicate code:
__device__ __host__ int PixelIndexOf5(int x, int y, int width) {
	return x + y * width;
}

__device__ __host__ bool IsInImage5(int x, int y, int width, int height) {
	return x >= 0 && x < width &&
			y >= 0 && y < height;
}

// TODO: remove the /255.0 operations and decide in which space we are

__global__ void PotentialKernel(
		float *img,
		float *beta,
		float *result,
		int image_width, int image_height) {
	// some neat index calculations:
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (IsInImage5(x, y, image_width, image_height)) {
		int pixel_index = PixelIndexOf5(x, y, image_width);
		result[pixel_index] = img[pixel_index] * beta[pixel_index];
	}
}

PotentialFilter::PotentialFilter(float *beta)
{
	this->beta = beta;
}

void PotentialFilter::Run()
{
	float * gpuBeta;

	int imagew = GetImageWidth();
	int imageh = GetImageHeight();
	int image_size = imagew * imageh;

	dim3 thread_block_size(32, 32, 1);
	dim3 block_grid_size(1 + imagew / thread_block_size.x,
			1 + imageh / thread_block_size.y,
			1);

	cudaMalloc((void**) &gpuBeta, image_size * sizeof(float));
	cudaMemcpy(gpuBeta, beta, image_size * sizeof(float), cudaMemcpyHostToDevice);

	PotentialKernel<<<block_grid_size, thread_block_size>>>(
			GetGpuImageData(),
			gpuBeta,
			GetGpuResultData(),
			imagew, imageh);

	cudaFree(gpuBeta);
}
