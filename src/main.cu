/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>

#include "../lib/jpge.h"
#include "../lib/jpgd.h"

#define MAX_BLOCKS 256
#define MAX_THREADS 256

__global__ void convertRGBToYUV(unsigned char* image, int image_size, int comps)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		int idx = pixel*comps;
		float r = (float) image[idx];
		float g = (float) image[idx+1];
		float b = (float) image[idx+2];
		float y = 0.299 * r + 0.587 * g + 0.114 * b;
		image[idx] = y;
		image[idx+1] = (b - y) * 0.493;
		image[idx+2] = (r - y) * 0.877;

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

__global__ void convertYUVToRGB(unsigned char* image, int image_size, int comps)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		int idx = pixel*comps;
		float y = (float) image[idx];
		float u = (float) image[idx+1];
		float v = (float) image[idx+2];
		float r = y+v/0.877;
		float b = y+u/0.493;
		image[idx] = r;
		image[idx+1] = 1.704 * y - 0.509 * r - 0.194*b;
		image[idx+2] = b;

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

__global__ void extractGrayscale(unsigned char* grayscale, unsigned char *image, int image_size, int comps)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		grayscale[pixel] = image[pixel*comps];

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

int main(int argc, char* argv[]) {
	int width, height, comps, image_size;

	if (argc < 3)
	{
		std::cout << "Please provide input and output filenames as arguments." << std::endl;
		return 1;
	}
	char *infilename = argv[1];
	char *outfilename = argv[2];

	// load image, allocate space on GPU
	unsigned char * image = jpgd::decompress_jpeg_image_from_file(infilename, &width, &height, &comps, 3);
	image_size = width * height;
	if (comps != 3)
	{
		std::cout << "Currently only images with 3 components are supported." << std::endl;
		free(image);
		return 1;
	}

	unsigned char * gpuImage;
	cudaMalloc((void**) &gpuImage, image_size * comps * sizeof(unsigned char));

	// upload to gpu
    cudaMemcpy(gpuImage, image, image_size * comps * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// convert to YUV
    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);
    convertRGBToYUV<<<blockGrid, threadBlock>>>(gpuImage, image_size, comps);

    // extract grayscale
    unsigned char * gpuGrayscale;
    cudaMalloc((void**) &gpuGrayscale, image_size * sizeof(unsigned char));
    extractGrayscale<<<blockGrid, threadBlock>>>(gpuGrayscale, gpuImage, image_size, comps);

    // TODO: use the grayscale, then modify gpuImage in the end

	// convert to RGB
    convertYUVToRGB<<<blockGrid, threadBlock>>>(gpuImage, image_size, comps);

	// download image
    cudaMemcpy(image, gpuImage, image_size * comps * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// write image
	if(!jpge::compress_image_to_jpeg_file(outfilename, width, height, comps, image))
	{
		std::cout << "Error writing the image." << std::endl;
	}

	free(image);
	cudaFree(gpuGrayscale);
	cudaFree(gpuImage);

	return 0;
}
