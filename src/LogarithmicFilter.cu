#include "LogarithmicFilter.h"

// TODO: remove/resolve duplicate code:
__device__ __host__ int PixelIndexOf4(int x, int y, int width) {
	return x + y * width;
}

__device__ __host__ bool IsInImage4(int x, int y, int width, int height) {
	return x >= 0 && x < width &&
			y >= 0 && y < height;
}

// TODO: remove the /255.0 operations and decide in which space we are

__global__ void LogarithmicKernel(
		float *img,
		float *result,
		int image_width, int image_height) {
	// some neat index calculations:
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (IsInImage4(x, y, image_width, image_height)) {
		int pixel_index = PixelIndexOf4(x, y, image_width);
		result[pixel_index] = logf(img[pixel_index] / 255.f);
	}
}

LogarithmicFilter::LogarithmicFilter()
{
}

void LogarithmicFilter::Run()
{
	int imagew = GetImageWidth();
	int imageh = GetImageHeight();
	dim3 thread_block_size(32, 32, 1);
	dim3 block_grid_size(1 + imagew / thread_block_size.x,
			1 + imageh / thread_block_size.y,
			1);
	LogarithmicKernel<<<block_grid_size, thread_block_size>>>(
			GetGpuImageData(),
			GetGpuResultData(),
			imagew, imageh);
}
