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

#define NUM_GPU_IMAGE_BUFFERS 3

//#define BENCHMARKING

#ifdef BENCHMARKING
	#include <time.h>
	#define BENCHMARK_REPETITIONS 50
	#define COUT(msg) DUMMYOP()
#else
	#define COUT(msg) std::cout << msg << std::endl;
#endif

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

void ExecutePipelineCore(unsigned char *gpuCharResult, unsigned char *gpuCharImage, float *gpuYUVBuffer,
						 unsigned char *gpuTextureImage, float *cpuExpandedBuffer,
						 std::vector<float *> gpuImageBuffers, int imageWidth, int imageHeight,
						 int textureWidth, int textureHeight, IPFConfiguration &config)
{
	int imageSize = imageWidth * imageHeight;
	float * gpuScetchImage = gpuImageBuffers[0];
    float * gpuBufferOne = gpuImageBuffers[1];
    float * gpuBufferTwo = gpuImageBuffers[2];

    dim3 blockGrid(MAX_BLOCKS);
    dim3 threadBlock(MAX_THREADS);

    /*
      * GPU Preprocessing: Convert to YUV and extract Grayscale
      */
     PROF_RANGE_PUSH("GPU Preprocessing");
     COUT("Converting RGB to YUV");
     convertRGBToYUV<<<blockGrid, threadBlock>>>(gpuYUVBuffer, gpuCharImage, imageSize);

     COUT("Extracting grayscale Image into BufferOne");
     extractGrayscale<<<blockGrid, threadBlock>>>(gpuBufferOne, gpuYUVBuffer, imageSize);
     PROF_RANGE_POP();



     /*
      * Image 1: Create the scetched gradient image from Grayscale
      */
     PROF_RANGE_PUSH("Image 1");
     COUT("Calculating the Gradient from BufferOne (grayscale) in BufferTwo");
     CalculateGradientImage<<<blockGrid, threadBlock>>>(
         gpuBufferOne,
         imageSize,
         imageWidth,
         gpuBufferTwo);

     COUT("Calculating the scetch filter from BufferTwo(gradient) in gpuScetchImage");
     ScetchFilter scetch_filter(config);
     scetch_filter.SetImageFromGpu(gpuBufferTwo, imageWidth, imageHeight, gpuScetchImage);
     scetch_filter.Run();
     PROF_RANGE_POP();


     /*
      * Image 2: Create the textured tone-mapped image from Grayscale
      */
     PROF_RANGE_PUSH("Compute Target Tone Map");
     COUT("Calculating the target tone map on CPU");
     ToneMap targetToneMap(config);
     PROF_RANGE_POP();

     PROF_RANGE_PUSH("Compute Histogram");
     COUT("Calculating the histogram of BufferOne (grayscale)");
     GrayscaleHistogram histogram(gpuBufferOne, imageSize);
     histogram.Run();
     PROF_RANGE_POP();

     PROF_RANGE_PUSH("Use ToneMapping filter");
     COUT("Applying tone mapping fromn BufferOne (grayscale) to BufferTwo");
     ToneMappingFilter tone_filter(targetToneMap, histogram.GpuCummulativeHistogram());
     tone_filter.SetImageFromGpu(gpuBufferOne, imageWidth, imageHeight, gpuBufferTwo);
     tone_filter.Run();
     PROF_RANGE_POP();

     PROF_RANGE_PUSH("Expanding Texture");
     COUT("Expanding and apply log2f function to texture, copy it to BufferOne");
     TextureExpander textExpander(gpuTextureImage, textureWidth, textureHeight);
     textExpander.ExpandDesaturateAndLogTo(gpuBufferOne , imageWidth, imageHeight);
     PROF_RANGE_POP();
     PROF_RANGE_PUSH("Download expanded log-Texture to CPU");
     cudaMemcpy(cpuExpandedBuffer, gpuBufferOne, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
 	 PROF_RANGE_POP();

 	 PROF_RANGE_PUSH("Solve Equation");
 	 COUT("Solving equation for texture drawing, copy result to BufferTwo");
     EquationSolver equation_solver(cpuExpandedBuffer, tone_filter.GetCpuResultData(),
     		imageWidth, imageHeight, config.TextureRenderingSmoothness);
     equation_solver.Run();
     PROF_RANGE_POP();
     PROF_RANGE_PUSH("Upload equation result");
     cudaMemcpy(gpuBufferTwo, equation_solver.GetResult(), imageSize * sizeof(float), cudaMemcpyHostToDevice);
     PROF_RANGE_POP();

     COUT("Rendering computed texture with BufferTwo (equation_solver result), BufferOne (expanded texture) to BufferOne");
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
     COUT("Multiplicating both images (gpuScetchImage and BufferTwo) into BufferOne");
     ImageMultiplicationFilter image_multiplication(scetch_filter.GetGpuResultData());
     image_multiplication.SetImageFromGpu(potential_filter.GetGpuResultData(), imageWidth, imageHeight, gpuBufferOne);
     image_multiplication.Run();
     PROF_RANGE_POP();

     /*
      * GPU Postprocessing: Convert to RGB, either with colors or without
      */
     PROF_RANGE_PUSH("GPU Postprocessing");
     GrayscaleAndYUVToRGB<<<blockGrid, threadBlock>>>(gpuCharResult, gpuBufferOne,
     		gpuYUVBuffer, config.UseColors, imageSize);
     PROF_RANGE_POP();
}

void ExecutePipeline(const char *infilename, const char *outfilename, IPFConfiguration &config)
{
	/*
	 * Setup: Load images, allocate buffers, set variables, upload image to GPU
	 */
	PROF_RANGE_PUSH("Load Jpeg Images");
	JpegImage cpuImage(infilename);
	JpegImage cpuTextureImage(PENCIL_TEXTURE_PATH);

	int imageSize = cpuImage.PixelSize();
	int imageWidth = cpuImage.Width();
	int imageHeight = cpuImage.Height();
	PROF_RANGE_POP();


	PROF_RANGE_PUSH("Buffer Allocation");
	unsigned char * gpuCharImage;
	unsigned char * gpuCharResult;
	unsigned char * gpuTextureImage;
	float * gpuYUVBuffer;
	float * cpuExpandedBuffer = (float *) malloc(imageSize * sizeof(float));
	cudaMalloc((void**) &gpuCharImage, cpuImage.ByteSize());
	cudaMalloc((void**) &gpuCharResult, cpuImage.ByteSize());
    cudaMalloc((void**) &gpuTextureImage, cpuTextureImage.ByteSize());
	cudaMalloc((void**) &gpuYUVBuffer, imageSize * YUV_COMPONENTS * sizeof(float));

    std::vector<float *> gpuImageBuffers;
    for (int i = 0; i < NUM_GPU_IMAGE_BUFFERS; i++)
    {
    	float *buff;
    	cudaMalloc((void**) &buff, imageSize * sizeof(float));
    	gpuImageBuffers.push_back(buff);
    }

    cudaMemcpy(gpuCharImage, cpuImage.Buffer(), cpuImage.ByteSize(), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuTextureImage, cpuTextureImage.Buffer(), cpuTextureImage.ByteSize(), cudaMemcpyHostToDevice);
    PROF_RANGE_POP();

#ifdef BENCHMARKING
    clock_t start = clock();
    for (int i = 0; i < BENCHMARK_REPETITIONS; i++)
    {
#endif
		ExecutePipelineCore(gpuCharResult, gpuCharImage, gpuYUVBuffer, gpuTextureImage, cpuExpandedBuffer,
							gpuImageBuffers, cpuImage.Width(), cpuImage.Height(),
							cpuTextureImage.Width(), cpuTextureImage.Height(), config);
#ifdef BENCHMARKING
    }
    float msecs = ((float) (clock() - start))/ CLOCKS_PER_SEC * 1000;
    std::cerr << "Needed " << (msecs/BENCHMARK_REPETITIONS) << "ms per run." <<  std::endl;
#endif


    /*
     * Postprocessing: Download image and save it as JPEG
     */
    PROF_RANGE_PUSH("Download result");
    cudaMemcpy(cpuImage.Buffer(), gpuCharResult, cpuImage.ByteSize(), cudaMemcpyDeviceToHost);
    PROF_RANGE_POP();

    PROF_RANGE_PUSH("Save result");
	cpuImage.Save(outfilename);
	PROF_RANGE_POP();

	std::cout << "Done." << std::endl;

	/*
	 * Cleanup
	 */
	PROF_RANGE_PUSH("Cleanup");
    for (int i = 0; i < NUM_GPU_IMAGE_BUFFERS; i++)
    {
    	float *buff;
    	cudaMalloc((void**) &buff, imageSize * sizeof(float));
    	gpuImageBuffers.push_back(buff);
    }
	cudaFree(gpuYUVBuffer);
	cudaFree(gpuCharImage);
	cudaFree(gpuCharResult);
	cudaFree(gpuTextureImage);
	free(cpuExpandedBuffer);
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
