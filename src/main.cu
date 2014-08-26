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

#include "macros.h"
#include "ImagePencilFilter.h"
#include "JpegImage.h"
#include "TextureExpander.h"
#include "GrayscaleHistogram.h"

#include "ScetchFilter.h"
#include "ToneMappingFilter.h"
#include "ImageMultiplicationFilter.h"
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
	PROF_RANGE_PUSH("Load Jpeg Images");
	JpegImage cpuImage(infilename);
	JpegImage cpuTextureImage(PENCIL_TEXTURE_PATH);

	int imageSize = cpuImage.PixelSize();
	int imageWidth = cpuImage.Width();
	int imageHeight = cpuImage.Height();
	PROF_RANGE_POP();

	/*
	 * GPU Setup: allocate buffers, set variables, upload image to GPU
	 */
	PROF_RANGE_PUSH("Buffer Allocation");
	unsigned char * gpuCharImage;
	unsigned char * gpuTextureImage;
	float * gpuYUVImage;
	float * gpuScetchImage;
    float * gpuBufferOne;
    float * gpuBufferTwo;
    float * cpuExpandedTexture = (float *) malloc(imageSize * sizeof(float));
    cudaMalloc((void**) &gpuCharImage, cpuImage.ByteSize());
    cudaMalloc((void**) &gpuTextureImage, cpuTextureImage.ByteSize());
	cudaMalloc((void**) &gpuYUVImage, imageSize * YUV_COMPONENTS * sizeof(float));
	cudaMalloc((void**) &gpuScetchImage, imageSize * sizeof(float));
	cudaMalloc((void**) &gpuBufferOne, imageSize * sizeof(float));
    cudaMalloc((void**) &gpuBufferTwo, imageSize * sizeof(float));
    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);

    cudaMemcpy(gpuCharImage, cpuImage.Buffer(), cpuImage.ByteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTextureImage, cpuTextureImage.Buffer(), cpuTextureImage.ByteSize(), cudaMemcpyHostToDevice);
    PROF_RANGE_POP();

    /*
     * GPU Preprocessing: Convert to YUV and extract Grayscale
     */
    PROF_RANGE_PUSH("GPU Preprocessing");
    std::cout << "Converting RGB to YUV" << std::endl;
    convertRGBToYUV<<<blockGrid, threadBlock>>>(gpuYUVImage, gpuCharImage, imageSize);

    std::cout << "Extracting grayscale Image into BufferOne" << std::endl;
    extractGrayscale<<<blockGrid, threadBlock>>>(gpuBufferOne, gpuYUVImage, imageSize);
    PROF_RANGE_POP();



    /*
     * Image 1: Create the scetched gradient image from Grayscale
     */
    PROF_RANGE_PUSH("Image 1");
    std::cout << "Calculating the Gradient from BufferOne (grayscale) in BufferTwo" << std::endl;
    CalculateGradientImage<<<blockGrid, threadBlock>>>(
        gpuBufferOne,
        imageSize,
        imageWidth,
        gpuBufferTwo);

    std::cout << "Calculating the scetch filter from BufferTwo(gradient) in gpuScetchImage" << std::endl;
    ScetchFilter scetch_filter(config);
    scetch_filter.SetImageFromGpu(gpuBufferTwo, imageWidth, imageHeight, gpuScetchImage);
    scetch_filter.Run();
    PROF_RANGE_POP();


    /*
     * Image 2: Create the textured tone-mapped image from Grayscale
     */
    PROF_RANGE_PUSH("Compute Target Tone Map");
    std::cout << "Calculating the target tone map on CPU" << std::endl;
    ToneMap targetToneMap(config);
    PROF_RANGE_POP();

    PROF_RANGE_PUSH("Compute Histogram");
    std::cout << "Calculating the histogram of BufferOne (grayscale)" << std::endl;
    GrayscaleHistogram histogram(gpuBufferOne, imageSize);
    histogram.Run();
    PROF_RANGE_POP();

    PROF_RANGE_PUSH("Use ToneMapping filter");
    std::cout << "Applying tone mapping fromn BufferOne (grayscale) to BufferTwo" << std::endl;
    ToneMappingFilter tone_filter(targetToneMap, histogram.GpuCummulativeHistogram());
    tone_filter.SetImageFromGpu(gpuBufferOne, imageWidth, imageHeight, gpuBufferTwo);
    tone_filter.Run();
    PROF_RANGE_POP();

    PROF_RANGE_PUSH("Expanding Texture");
    std::cout << "Expanding and apply log2f function to texture, copy it to BufferOne" << std::endl;
    TextureExpander textExpander(gpuTextureImage, cpuTextureImage.Width(), cpuTextureImage.Height());
    textExpander.ExpandDesaturateAndLogTo(gpuBufferOne , imageWidth, imageHeight);
    PROF_RANGE_POP();
    PROF_RANGE_PUSH("Download expanded log-Texture to CPU");
    cudaMemcpy(cpuExpandedTexture, gpuBufferOne, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
	PROF_RANGE_POP();

	PROF_RANGE_PUSH("Solve Equation");
    std::cout << "Solving equation for texture drawing, copy result to BufferTwo" << std::endl;
    EquationSolver equation_solver(cpuExpandedTexture, tone_filter.GetCpuResultData(),
    		imageWidth, imageHeight, config.TextureRenderingSmoothness);
    equation_solver.Run();
    PROF_RANGE_POP();
    PROF_RANGE_PUSH("Upload equation result");
    cudaMemcpy(gpuBufferTwo, equation_solver.GetResult(), imageSize * sizeof(float), cudaMemcpyHostToDevice);
    PROF_RANGE_POP();

    std::cout << "Rendering computed texture with BufferTwo (equation_solver result), BufferOne (expanded texture) to BufferOne" << std::endl;
    PROF_RANGE_PUSH("Rendering Texture");
    // This Filter can work inplace: Input Picture can be ouput picture
    PotentialFilter potential_filter(gpuBufferTwo);
    potential_filter.SetImageFromGpu(gpuBufferOne, imageWidth, imageHeight, gpuBufferOne);
    potential_filter.Run();
    PROF_RANGE_POP();



    /*
     * Combined Image: Multiplying texture tone-mapped image with scetched gradient image
     */
    PROF_RANGE_PUSH("Multiplicate Images");
    std::cout << "Multiplicating both images (gpuScetchImage and BufferTwo) into BufferOne" << std::endl;
    ImageMultiplicationFilter image_multiplication(scetch_filter.GetGpuResultData());
    image_multiplication.SetImageFromGpu(potential_filter.GetGpuResultData(), imageWidth, imageHeight, gpuBufferOne);
    image_multiplication.Run();
    PROF_RANGE_POP();

    /*
     * GPU Postprocessing: Convert to RGB, either with colors or without
     */
    PROF_RANGE_PUSH("GPU Postprocessing");
    GrayscaleAndYUVToRGB<<<blockGrid, threadBlock>>>(gpuCharImage, image_multiplication.GetGpuResultData(),
    		gpuYUVImage, config.UseColors, imageSize);
    PROF_RANGE_POP();


    /*
     * CPU Postprocessing: Download image and save it as JPEG
     */
    PROF_RANGE_PUSH("Download result");
    cudaMemcpy(cpuImage.Buffer(), gpuCharImage, cpuImage.ByteSize(), cudaMemcpyDeviceToHost);
    PROF_RANGE_POP();

    PROF_RANGE_PUSH("Save result");
	cpuImage.Save(outfilename);
	PROF_RANGE_POP();

	std::cout << "Done." << std::endl;

	/*
	 * Cleanup
	 */
	PROF_RANGE_PUSH("Cleanup");
	cudaFree(gpuBufferTwo);
	cudaFree(gpuBufferOne);
	cudaFree(gpuYUVImage);
	cudaFree(gpuCharImage);
	cudaFree(gpuTextureImage);
	free(cpuExpandedTexture);
	PROF_RANGE_POP();
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
