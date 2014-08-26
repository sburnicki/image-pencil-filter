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

#include "ImagePencilFilter.h"
#include "JpegImage.h"
#include "ExpandableTexture.h"
#include "GrayscaleHistogram.h"

#include "ScetchFilter.h"
#include "ToneMappingFilter.h"
#include "ImageMultiplicationFilter.h"
#include "LogarithmicFilter.h"
#include "PotentialFilter.h"
#include "EquationSolver.h"


__global__ void convertRGBToYUV(float *outputImage, unsigned char* image, int image_size)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		int idx = pixel*RGB_COMPONENTS;
		float r = image[idx];
		float g = image[idx+1];
		float b = image[idx+2];
		float y = RGB_TO_Y(r, g, b);
		outputImage[idx] = y;
		outputImage[idx+1] = (b - y) * 0.493;
		outputImage[idx+2] = (r - y) * 0.877;

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

__global__ void GrayscaleAndYUVToRGB(unsigned char* outputImage, float *grayscaleImage, float *yuvImage, bool useColors, int image_size)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		int idx = pixel*RGB_COMPONENTS;
		float y = (float) grayscaleImage[pixel];
		float u = useColors ? (float) yuvImage[idx+1] : 0.0f;
		float v = useColors ? (float) yuvImage[idx+2] : 0.0f;

		float r = y+v/0.877;
		float b = y+u/0.493;
		float g = 1.703 * y - 0.509 * r - 0.194*b;

		// make sure that our values fit in a byte
		r = r < 0 ? 0 : r;
		g = g < 0 ? 0 : g;
		b = b < 0 ? 0 : b;

		r = r > 255 ? 255 : r;
		g = g > 255 ? 255 : g;
		b = b > 255 ? 255 : b;

		outputImage[idx] = r;
		outputImage[idx+1] = g;
		outputImage[idx+2] = b;

		pixel += MAX_BLOCKS * MAX_THREADS;
	}
}

__global__ void extractGrayscale(float* grayscale, float *image, int image_size)
{
	int pixel = blockDim.x * blockIdx.x + threadIdx.x;
	while (pixel < image_size)
	{
		grayscale[pixel] = image[pixel*RGB_COMPONENTS];

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

void ExecutePipeline(const char *infilename, const char *outfilename, IPFConfiguration &config)
{
	/*
	 * CPU Preprocssing: Load image and texture from JPEG, set variables
	 */
	JpegImage cpuImage(infilename);
	ExpandableTexture pencilTexture(PENCIL_TEXTURE_PATH);

	int imageSize = cpuImage.PixelSize();
	int imageWidth = cpuImage.Width();
	int imageHeight = cpuImage.Height();

	/*
	 * GPU Setup: allocate buffers, set variables, upload image to GPU
	 */
	unsigned char * gpuCharImage;
	float * gpuFloatImage;
    float * gpuGrayscale;
	cudaMalloc((void**) &gpuCharImage, cpuImage.ByteSize());
	cudaMalloc((void**) &gpuFloatImage, imageSize * YUV_COMPONENTS * sizeof(float));
	cudaMalloc((void**) &gpuGrayscale, imageSize * sizeof(float));
    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);

    cudaMemcpy(gpuCharImage, cpuImage.Buffer(), cpuImage.ByteSize(), cudaMemcpyHostToDevice);

    /*
     * GPU Preprocessing: Convert to YUV and extract Grayscale
     */
    std::cout << "Converting RGB to YUV" << std::endl;
    convertRGBToYUV<<<blockGrid, threadBlock>>>(gpuFloatImage, gpuCharImage, imageSize);

    std::cout << "Extracting grayscale Image" << std::endl;
    extractGrayscale<<<blockGrid, threadBlock>>>(gpuGrayscale, gpuFloatImage, imageSize);



    /*
     * Image 1: Create the scetched gradient image from Grayscale
     */
    float *gpu_gradient_image;
    cudaMalloc((void**) &gpu_gradient_image, imageSize * sizeof(float));

    std::cout << "Calculating the Gradient" << std::endl;
    CalculateGradientImage<<<blockGrid, threadBlock>>>(
        gpuGrayscale,
        imageSize,
        imageWidth,
        gpu_gradient_image);

    std::cout << "Calculating the scetch filter" << std::endl;
    ScetchFilter scetch_filter(config);
    scetch_filter.SetImageFromGpu(gpu_gradient_image, imageWidth, imageHeight);
    scetch_filter.Run();


    /*
     * Image 2: Create the textured tone-mapped image from Grayscale
     */
    std::cout << "Calculating the target tone map on CPU" << std::endl;
    ToneMap targetToneMap(config);

    std::cout << "Calculating the histogram of the grayscale image" << std::endl;
    GrayscaleHistogram histogram(gpuGrayscale, imageSize);
    histogram.Run();

    std::cout << "Calculating the tone mapping filter" << std::endl;
    ToneMappingFilter tone_filter(targetToneMap, histogram.GpuCummulativeHistogram());
    tone_filter.SetImageFromGpu(gpuGrayscale, imageWidth, imageHeight);
    tone_filter.Run();

    std::cout << "Calculate the log of tonemapped image" << std::endl;
    LogarithmicFilter log_filter;
    log_filter.SetImageFromGpu(tone_filter.GetGpuResultData(), imageWidth, imageHeight);
    log_filter.Run();

    std::cout << "Expanding and apply log function to texture on CPU" << std::endl;
    pencilTexture.Expand(imageWidth, imageHeight);

    std::cout << "Solving equation for texture drawing" << std::endl;
    EquationSolver equation_solver(pencilTexture.LogBuffer(), log_filter.GetCpuResultData(),
    		imageWidth, imageHeight, config.TextureRenderingSmoothness);
    equation_solver.Run();
    float *beta_star = equation_solver.GetResult();

    std::cout << "Rendering computed texture" << std::endl;
    PotentialFilter potential_filter(beta_star);
    potential_filter.SetImageFromCpu(pencilTexture.ExpandedBuffer(), imageWidth, imageHeight);
    potential_filter.Run();



    /*
     * Combined Image: Multiplying texture tone-mapped image with scetched gradient image
     */
    std::cout << "Multiplicating both images" << std::endl;
    ImageMultiplicationFilter image_multiplication(scetch_filter.GetGpuResultData());
    image_multiplication.SetImageFromGpu(potential_filter.GetGpuResultData(), imageWidth, imageHeight);
    image_multiplication.Run();

    float *resultGrayscaleImage = image_multiplication.GetGpuResultData();


    /*
     * GPU Postprocessing: Convert to RGB, either with colors or without
     */
    GrayscaleAndYUVToRGB<<<blockGrid, threadBlock>>>(gpuCharImage, resultGrayscaleImage, gpuFloatImage, config.UseColors, imageSize);


    /*
     * CPU Postprocessing: Download image and save it as JPEG
     */


    cudaMemcpy(cpuImage.Buffer(), gpuCharImage, cpuImage.ByteSize(), cudaMemcpyDeviceToHost);
    cpuImage.Save(outfilename);
	std::cout << "Done." << std::endl;

	/*
	 * Cleanup
	 */
	cudaFree(gpu_gradient_image);
	cudaFree(gpuGrayscale);
	cudaFree(gpuFloatImage);
	cudaFree(gpuCharImage);
}





int main(int argc, char* argv[]) {
	if (argc < 3)
	{
		std::cout << "Please provide input and output filenames as arguments." << std::endl;
		return 1;
	}
	IPFConfiguration config;
	config.UseColors = !(argc > 3 && strcmp(argv[3], "-grayscale") == 0);

	try
	{
		ExecutePipeline(argv[1], argv[2], config);
	}
	catch (const char *msg)
	{
		std::cout << msg << std::endl;
		return 1;
	}

	return 0;
}
