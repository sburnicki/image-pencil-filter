#include "ToneMappingFilter.h"
#include <cmath>
#include <iostream>

#include "ImagePencilFilter.h"
#define EPSILON 0.0001

// TODO: remove/resolve duplicate code:
__device__ __host__ int PixelIndexOf2(int x, int y, int width) {
	return x + y * width;
}

__device__ __host__ bool IsInImage2(int x, int y, int width, int height) {
	return x >= 0 && x < width &&
			y >= 0 && y < height;
}

// search function
__device__ __host__ int binarySearch(float value, float* target, int minidx, int maxidx) {
    while(true)
    {
    	int pivot = (maxidx - minidx) / 2 + minidx;
    	if (maxidx <= minidx)
    	{
    		return minidx;
    	}
    	float mapval = target[pivot];
    	float diff = std::abs(value - mapval);
    	if (diff < EPSILON)
    	{
    		return pivot;
    	}
    	if (value < mapval)
    	{
    		maxidx = pivot - 1;
    	}
    	else if (value > mapval)
    	{
    		minidx = pivot + 1;
    	}
	}
}

/*
 * Important: This kernel does not only perform tone mapping,
 * but also applies the log2f function to it!
 */
__global__ void ToneMappingKernel(
		float *image,
		float *result,
		int image_width, int image_height,
		int num_tones, int *origHist, float *destHist) {
	// some neat index calculations:
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	float numpixels = image_width * image_height;

	if (IsInImage2(x, y, image_width, image_height)) {
		int pixel_index = PixelIndexOf2(x, y, image_width);
		float find = ((float) origHist[(int) image[pixel_index]]) / numpixels;
		int targetValue = binarySearch(find, destHist, 0, num_tones - 1);
		result[pixel_index] = log2f(targetValue / 255.0);
	}
}


ToneMappingFilter::ToneMappingFilter(ToneMap &destinationMap, int *gpuCumHistogram) : ImageFilter()
{
	gpu_histogram_ = gpuCumHistogram;
	const std::vector<float> &tonemap = destinationMap.getTonemap();
	cudaMalloc((void**) &gpu_tonemap_array_, COLOR_DEPTH * sizeof(float));
	cudaMemcpy(gpu_tonemap_array_, &tonemap[0], COLOR_DEPTH * sizeof(float), cudaMemcpyHostToDevice);
}

ToneMappingFilter::~ToneMappingFilter()
{
	cudaFree(gpu_tonemap_array_);
}

void ToneMappingFilter::Run() {
	int imagew = GetImageWidth();
	int imageh = GetImageHeight();
	dim3 thread_block_size(32, 32, 1);
	dim3 block_grid_size(1 + imagew / thread_block_size.x,
			1 + imageh / thread_block_size.y,
			1);
	ToneMappingKernel<<<block_grid_size, thread_block_size>>>(
			GetGpuImageData(),
			GetGpuResultData(),
			imagew, imageh, COLOR_DEPTH,
			gpu_histogram_, gpu_tonemap_array_);
}
