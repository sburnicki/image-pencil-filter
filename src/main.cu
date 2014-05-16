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

__global__ void convertRGBToYUV(float *outputImage, unsigned char* image, int image_size, int comps)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		int idx = pixel*comps;
		float r = (float) image[idx];
		float g = (float) image[idx+1];
		float b = (float) image[idx+2];
		float y = 0.299 * r + 0.587 * g + 0.114 * b;
		outputImage[idx] = y;
		outputImage[idx+1] = (b - y) * 0.493;
		outputImage[idx+2] = (r - y) * 0.877;

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

__global__ void convertYUVToRGB(unsigned char* outputImage, float *image, int image_size, int comps)
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
		outputImage[idx] = r;
		outputImage[idx+1] = 1.704 * y - 0.509 * r - 0.194*b;
		outputImage[idx+2] = b;

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

/**
 * \brief Kernel to transform the gradient image into a RGB image
 *
 *        This kernel is only for testing the gradient image output.
 *
 * \param kGradientImage  The input gradient image
 * \param kImageSize      The size of the image in pixel
 * \param rgb_image       RGB output image
 */
__global__ void ConvertGradienToRGB(
    const float *kGradientImage,
    const int kImageSize,
    const int kImageComponents,
    unsigned char *rgb_image) {
  // Calculate pixel position
  int pixel_pos_this = blockDim.x * blockIdx.x + threadIdx.x;

  // Calculate RGB value if pixel exists
  while (pixel_pos_this < kImageSize)
  {
    // Transform to YUV
    float y = kGradientImage[pixel_pos_this];
    float u = 0.0;
    float v = 0.0;

    // Calculate RGB
    float r = y + v / 0.877;
    float b = y + u / 0.493;

    // Save RGB
    int pixel_pos_output = pixel_pos_this * kImageComponents;
    rgb_image[pixel_pos_output]     = r;
    rgb_image[pixel_pos_output + 1] = 1.704 * y - 0.509 * r - 0.194 * b;
    rgb_image[pixel_pos_output + 2] = b;

    // Calculate next pixel position
    pixel_pos_this += MAX_BLOCKS * MAX_THREADS;
  }
}

__global__ void extractGrayscale(float* grayscale, float *image, int image_size, int comps)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		grayscale[pixel] = image[pixel*comps];

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

/**
 * \brief Kernel to calculate the forward gradient from a grayscale image
 *
 *        The bottom line and the very right line will be zero, as it is
 *        impossible to calculate the forward gradient for these points.
 *
 * \param kGrayscaleImage The input grayscale image
 * \param kImageSize      The size of the image in pixel
 * \param kImageWidth     The size of one line in the input image
 * \param gradient_image  Gradient output image
 *
 * TODO: Fix bank conflicts and do general optimization
 */
__global__ void CalculateGradientImage(
    const float *kGrayscaleImage,
    const int kImageSize,
    const int kImageWidth,
    float *gradient_image) {
  // Calculate pixel position
  int pixel_pos_this = blockDim.x * blockIdx.x + threadIdx.x;

  // Calculate gradient if pixel exists
  while (pixel_pos_this < kImageSize) {
    // Calculate forward pixels positions
    int pixel_pos_right = pixel_pos_this + 1;
    int pixel_pos_top   = pixel_pos_this + kImageWidth;

    // Set bottom and very right pixels to zero
    gradient_image[pixel_pos_this] = 0;

    // Calculate gradient if forward pixels exist
    if (pixel_pos_right < kImageSize && pixel_pos_top < kImageSize) {
      // Retrieve points value
      int pixel_this  = kGrayscaleImage[pixel_pos_this];
      int pixel_right = kGrayscaleImage[pixel_pos_right];
      int pixel_top   = kGrayscaleImage[pixel_pos_top];

      // Calculate difference between this and forward points
      int dx = pixel_right - pixel_this;
      int dy = pixel_top   - pixel_this;

      // Calculate the gradient for this point
      gradient_image[pixel_pos_this] =  sqrt(
          static_cast<float>( (dx * dx + dy * dy) )
      );
    }

    // Calculate next pixel position
    pixel_pos_this += MAX_BLOCKS * MAX_THREADS;
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
		if (comps == 0)
		{
			std::cout << "Loading the image failed! Wrong path?." << std::endl;
		}
		else
		{
			std::cout << "Currently only images with 3 components are supported." << std::endl;
		}
		free(image);
		return 1;
	}

	unsigned char * gpuCharImage;
	float * gpuFloatImage;
    float * gpuGrayscale;
	cudaMalloc((void**) &gpuCharImage, image_size * comps * sizeof(unsigned char));
	cudaMalloc((void**) &gpuFloatImage, image_size * comps * sizeof(float));
	cudaMalloc((void**) &gpuGrayscale, image_size * sizeof(float));

	// upload to gpu
    cudaMemcpy(gpuCharImage, image, image_size * comps * sizeof(unsigned char), cudaMemcpyHostToDevice);

	// convert to YUV
    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);
    convertRGBToYUV<<<blockGrid, threadBlock>>>(gpuFloatImage, gpuCharImage, image_size, comps);

    // extract grayscale
    extractGrayscale<<<blockGrid, threadBlock>>>(gpuGrayscale, gpuFloatImage, image_size, comps);

    // Calculate gradient image
    float *gpu_gradient_image;
    cudaMalloc((void**) &gpu_gradient_image, image_size * sizeof(float));
    CalculateGradientImage<<<blockGrid, threadBlock>>>(
        gpuGrayscale,
        image_size,
        width,
        gpu_gradient_image);

    // Output grayscale image
    ConvertGradienToRGB<<<blockGrid, threadBlock>>>(
        gpu_gradient_image,
        image_size,
        comps,
        gpuCharImage);

	// convert to RGB
    //convertYUVToRGB<<<blockGrid, threadBlock>>>(gpuCharImage, gpuFloatImage, image_size, comps);

	// download image
    cudaMemcpy(image, gpuCharImage, image_size * comps * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	// write image
	if(!jpge::compress_image_to_jpeg_file(outfilename, width, height, comps, image))
	{
		std::cout << "Error writing the image." << std::endl;
	}

	free(image);
	cudaFree(gpu_gradient_image);
	cudaFree(gpuGrayscale);
	cudaFree(gpuFloatImage);
	cudaFree(gpuCharImage);

	return 0;
}
