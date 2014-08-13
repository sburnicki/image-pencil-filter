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
#include <cmath>

#include "../lib/jpge.h"
#include "../lib/jpgd.h"

#include "ScetchFilter.h"
#include "ToneMappingFilter.h"
#include "ImageMultiplicationFilter.h"
#include "LogarithmicFilter.h"
#include "PotentialFilter.h"
#include "EquationSolver.h"


#define MAX_BLOCKS 256
#define MAX_THREADS 256
#define LAMBDA 2.f

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

/**
 * \brief Kernel to calculate the histogram from a grayscale image
 * \param kGrayscaleImage         The input grayscale image
 * \param kImageSize              The size of the image in pixel
 * \param histogram               Histogram output
 * \param accumulative_histogramm Accumulative histogram output
 *
 * TODO: Check this out - http://developer.download.nvidia.com/compute/cuda/\
 *                        1.1-Beta/x86_website/projects/histogram64/doc/\
 *                        histogram.pdf
 */
__global__ void CalculateHistogram(
    const float *kGrayscaleImage,
    const int kImageSize,
    int *histogram,
    int *accumulative_histogram) {
  __shared__ int shared_histogram[256];
  __shared__ int shared_accumulative_histogram[256];

  // Calculate ID and pixel position
  int tid       = threadIdx.x;
  int pixel_pos = blockDim.x * blockIdx.x + tid;

  // Clear histogram
  if (pixel_pos < 256) {
      histogram[pixel_pos] = 0;
      accumulative_histogram[pixel_pos] = 0;
  }
  if (tid < 256) {
    shared_histogram[tid] = 0;
    shared_accumulative_histogram[tid] = 0;
  }
  __syncthreads();

  // Calculate partial histogram if pixel exists
  while (pixel_pos < kImageSize) {
    int value = kGrayscaleImage[pixel_pos];

    // Increment position of value in histogram
    // TODO Remove sanity check if sure
    if (value < 256 && value >= 0) {
      atomicAdd(&shared_histogram[value], 1);
    }

    // Calculate next pixel position
    pixel_pos += MAX_BLOCKS * MAX_THREADS;
  }
  __syncthreads();

  // Calculate partial histogram and accumulate result to global memory
  if (tid < 256) {
    shared_accumulative_histogram[tid] = shared_histogram[tid];

    // TODO: Fix the commented code block and delete the uncommented slower one
    //       The result is too big in this faster solution
    //for (int i = 1; i <= tid; i *= 2) {
    //  __syncthreads();
    //  shared_accumulative_histogram[tid]
    //      += shared_accumulative_histogram[tid - i];
    //}
    __syncthreads();
    int sum = 0;
    for (int i = 0; i <= tid; i++) {
      sum += shared_accumulative_histogram[i];
    }
    __syncthreads();
    shared_accumulative_histogram[tid] = sum;

    // Copy result to global memory
    __syncthreads();
    atomicAdd(&histogram[tid], shared_histogram[tid]);
    atomicAdd(&accumulative_histogram[tid], shared_accumulative_histogram[tid]);
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

	int texture_width, texture_height, texture_comps;
	unsigned char * texture = jpgd::decompress_jpeg_image_from_file("resources/texture4.jpg", &texture_width, &texture_height, &texture_comps, 3);

	if (texture_comps != 3)
	{
		if (texture_comps == 0)
		{
			std::cout << "Loading the image failed! Wrong path?." << std::endl;
		}
		else
		{
			std::cout << "Currently only images with 3 components are supported." << std::endl;
		}
		free(texture);
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
    std::cout << "Converting RGB to YUV" << std::endl;
    convertRGBToYUV<<<blockGrid, threadBlock>>>(gpuFloatImage, gpuCharImage, image_size, comps);

    // extract grayscale
    std::cout << "Extracting grayscale Image" << std::endl;
    extractGrayscale<<<blockGrid, threadBlock>>>(gpuGrayscale, gpuFloatImage, image_size, comps);

    // Calculate gradient image
    float *gpu_gradient_image;
    cudaMalloc((void**) &gpu_gradient_image, image_size * sizeof(float));

    std::cout << "Calculating the Gradient" << std::endl;
    CalculateGradientImage<<<blockGrid, threadBlock>>>(
        gpuGrayscale,
        image_size,
        width,
        gpu_gradient_image);



    std::cout << "Calculating the scetch filter" << std::endl;
    // Apply Scetch Filter
    ScetchFilter scetch_filter;
    scetch_filter.SetImageFromGpu(gpu_gradient_image, width, height);
    scetch_filter.set_line_count(5);
    scetch_filter.set_line_length(20);
    scetch_filter.set_line_strength(1);
    scetch_filter.set_gamma(1.2);
    scetch_filter.Run();


    // std::cout << "Running scetch filter debug tests" << std::endl;
    // std::string debug_message, additional_message;
    // bool is_there_a_addditional_message = false;
    // if (!scetch_filter.TestGpuFunctions(&debug_message, 
    //       &additional_message, &is_there_a_addditional_message)) {
    // 	std::cerr << "scetch filter test failed with message:" << std::endl <<
    // 			debug_message << std::endl;
    // } else if (is_there_a_addditional_message) {
    //   std::cout << "scetch filter succeeded with additional message:" << std::endl <<
    //     additional_message << std::endl;
    // }

    std::cout << "Create rgb image from greyscale image" << std::endl;

    // Calculate histogram
    int * gpu_histogram;
    int * gpu_accumulative_histogram;
    cudaMalloc((void**) &gpu_histogram, 256 * sizeof(int));
    cudaMalloc((void**) &gpu_accumulative_histogram, 256 * sizeof(int));

    std::cout << "Calculating the histogram of the grayscale image"
              << std::endl;
    CalculateHistogram<<<blockGrid, threadBlock>>>(
        gpuGrayscale,
        image_size,
        gpu_histogram,
        gpu_accumulative_histogram);

    //cudaThreadSynchronize();

    /*
    // TODO: Only for testing purpose, remove for production
    int histogram[256];
    int accumulative_histogram[256];
    cudaMemcpy(
        &histogram,
        gpu_histogram,
        256 * sizeof(int),
        cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &accumulative_histogram,
        gpu_accumulative_histogram,
        256 * sizeof(int),
        cudaMemcpyDeviceToHost);
    std::cout << "Histogram: ";
    for (int i = 0; i <= 255; i++) {
      std::cout << " " << histogram[i];
    }
    std::cout << std::endl;
    std::cout << "Accumulative histogram: ";
    for (int i = 0; i <= 255; i++) {
      std::cout << " " << accumulative_histogram[i];
    }
    std::cout << std::endl;
    int sum = 0;
    for (int i = 0; i <= 255; i++) {
      sum += histogram[i];
    }
    std::cout << "Control Sum: " << sum << std::endl;
    */


    std::cout << "Calculating the tone mapping filter" << std::endl;
    // Apply Scetch Filter
    ToneMappingFilter tone_filter(256, gpu_accumulative_histogram);
    tone_filter.SetImageFromGpu(gpuGrayscale, width, height);
    tone_filter.Run();

    /*
    // DEBUG: show destination tonemap
    const std::vector<float> &tonemap = tone_filter.GetCpuTonemap();
    for (int i = 0; i < 256; i++)
    {
    	std::cout << "Value for " << i << ": " << tonemap[i] << std::endl;
    }
    //
    */

    // TODO: think about unifying with tone mapping
    std::cout << "Calculate the log of tonemapped image" << std::endl;
    LogarithmicFilter log_filter;
    log_filter.SetImageFromGpu(tone_filter.GetGpuResultData(), width, height);
    log_filter.Run();


    std::cout << "Expanding and apply log function to texture" << std::endl;
    float *logtext = (float *) malloc(sizeof(float) * image_size);
    float *expanded_text = (float *) malloc(sizeof(float) * image_size);
    for (int y = 0; y < height; y++) {
    	for (int x = 0; x < width; x++) {
    		int target_index = x + y * width;
    		int source_x = x % texture_width;
    		int source_y = y % texture_height;

    		int source_index = source_x + source_y * texture_width;
    		expanded_text[target_index] =  texture[3*source_index +0] * 0.299;
    		expanded_text[target_index] += texture[3*source_index +1] * 0.587;
    		expanded_text[target_index] += texture[3*source_index +2] * 0.114;
    		expanded_text[target_index] /= 255.f;
    		logtext[target_index] = logf(expanded_text[target_index]);
    	}
    }

    //
    float *gpuExpandedTexture;
    cudaMalloc((void**) &gpuExpandedTexture, image_size * sizeof(float));

    cudaMemcpy(gpuExpandedTexture, expanded_text, image_size * sizeof(float), cudaMemcpyHostToDevice);



    std::cout << "Solving equation for texture drawing" << std::endl;
    EquationSolver equation_solver(logtext, log_filter.GetCpuResultData(), width, height, LAMBDA);
    equation_solver.Run();
    float *beta_star = equation_solver.GetResult();

    std::cout << "Rendering computed texture" << std::endl;
    PotentialFilter potential_filter(beta_star);
    potential_filter.SetImageFromCpu(expanded_text, width, height);
    potential_filter.Run();

    std::cout << "Multiplicating both images" << std::endl;
    ImageMultiplicationFilter image_multiplication(scetch_filter.GetGpuResultData());
    image_multiplication.SetImageFromGpu(potential_filter.GetGpuResultData(), width, height);
    image_multiplication.Run();



    // Output grayscale image
    ConvertGradienToRGB<<<blockGrid, threadBlock>>>(
    	image_multiplication.GetGpuResultData(),
    	// gpuExpandedTexture,
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
	std::cout << "Done." << std::endl;

	free(image);
	free(logtext);
	free(expanded_text);
	cudaFree(gpu_histogram);
	cudaFree(gpu_accumulative_histogram);
	cudaFree(gpu_gradient_image);
	cudaFree(gpuGrayscale);
	cudaFree(gpuFloatImage);
	cudaFree(gpuCharImage);
}
