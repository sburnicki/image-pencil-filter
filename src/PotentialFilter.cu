#include "PotentialFilter.h"
#include "macros.h"

__global__ void PotentialKernel(
		float *img,
		float *beta,
		float *result,
		int image_width, int image_height) {
	// some neat index calculations:
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (IS_IN_IMAGE(x, y, image_width, image_height)) {
		int pixel_index = PIXEL_INDEX_OF(x, y, image_width);
		// * 2 because img[pixel_index] is the log2f of the real value
		result[pixel_index] = exp2f(img[pixel_index] * beta[pixel_index]) * 255.f;
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
