#include "PotentialFilter.h"

// TODO: remove/resolve duplicate code:
__device__ __host__ int PixelIndexOf5(int x, int y, int width) {
	return x + y * width;
}

__device__ __host__ bool IsInImage5(int x, int y, int width, int height) {
	return x >= 0 && x < width &&
			y >= 0 && y < height;
}

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
		result[pixel_index] = powf(img[pixel_index] , beta[pixel_index]) * 255.f;
	}
}

PotentialFilter::PotentialFilter(float *gpu_beta)
{
	gpu_beta_ = gpu_beta;
}

void PotentialFilter::Run()
{
	int imagew = GetImageWidth();
	int imageh = GetImageHeight();

	dim3 thread_block_size(32, 32, 1);
	dim3 block_grid_size(1 + imagew / thread_block_size.x,
			1 + imageh / thread_block_size.y,
			1);

	PotentialKernel<<<block_grid_size, thread_block_size>>>(
			GetGpuImageData(),
			gpu_beta_,
			GetGpuResultData(),
			imagew, imageh);
}
