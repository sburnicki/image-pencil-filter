/*
 * TextureExpander.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: burnicki
 */

#include "TextureExpander.h"
#include "macros.h"

/*
 * Important: This kernel does not only perform tone mapping,
 * but also normalizes the value and applies the logarithmic function to it!
 */
__global__ void TextureExpansionKernel(
		float *result,
		unsigned char *colored_orig_image,
		int orig_width, int orig_height,
		int dest_width, int dest_height) {
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	// we know that we will get bank conflicts.
	if (IS_IN_IMAGE(x, y, orig_width, orig_height)) {
		// get target Y component from RGB values
		int srcIdx = PIXEL_INDEX_OF(x, y, orig_width) * RGB_COMPONENTS;
		float yValue = log2f((colored_orig_image[srcIdx] * 0.299 +
					    colored_orig_image[srcIdx+1] * 0.587 +
					    colored_orig_image[srcIdx+2] * 0.114) / 255.f);
		int destX = x;
		int destY = y;
		// replicate in x direction
		while(IS_IN_IMAGE(destX, destY, dest_width, dest_height))
		{
			// replicate in y direction
			while(IS_IN_IMAGE(destX, destY, dest_width, dest_height))
			{
				result[PIXEL_INDEX_OF(destX, destY, dest_width)] = yValue;
				destY += orig_height;
			}
			destX += orig_width;
			destY = y;
		}
	}
}

TextureExpander::TextureExpander(unsigned char *gpuColoredTexture, int origWidth, int origHeight)
{
	gpu_colored_texture_ = gpuColoredTexture;
	orig_width_ = origWidth;
	orig_height_ = origHeight;
}

void TextureExpander::ExpandDesaturateAndLogTo(float *gpuResultBuffer, int resultWidth, int resultHeight)
{
	dim3 thread_block_size(32, 32, 1);
	dim3 block_grid_size(1 + orig_width_ / thread_block_size.x,
			1 + orig_height_ / thread_block_size.y,
			1);
	TextureExpansionKernel<<<block_grid_size, thread_block_size>>>(
			gpuResultBuffer, gpu_colored_texture_,
			orig_width_, orig_height_, resultWidth, resultHeight);

}
