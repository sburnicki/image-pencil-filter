/*
 * GrayscaleHistogram.cpp
 *
 *  Created on: Aug 21, 2014
 *      Author: burnicki
 */

#include "GrayscaleHistogram.h"
#include <stddef.h>

// TODO: fix this to make sense
#define MAX_BLOCKS 256
#define MAX_THREADS 256

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
  __shared__ int shared_histogram[COLOR_DEPTH];
  __shared__ int shared_accumulative_histogram[COLOR_DEPTH];

  // Calculate ID and pixel position
  int tid       = threadIdx.x;
  int pixel_pos = blockDim.x * blockIdx.x + tid;

  // Clear histogram
  if (pixel_pos < COLOR_DEPTH) {
      histogram[pixel_pos] = 0;
      accumulative_histogram[pixel_pos] = 0;
  }
  if (tid < COLOR_DEPTH) {
    shared_histogram[tid] = 0;
    shared_accumulative_histogram[tid] = 0;
  }
  __syncthreads();

  // Calculate partial histogram if pixel exists
  while (pixel_pos < kImageSize) {
    int value = kGrayscaleImage[pixel_pos];

    // Increment position of value in histogram
    // TODO Remove sanity check if sure
    if (value < COLOR_DEPTH && value >= 0) {
      atomicAdd(&shared_histogram[value], 1);
    }

    // Calculate next pixel position
    pixel_pos += MAX_BLOCKS * MAX_THREADS;
  }
  __syncthreads();

  // Calculate partial histogram and accumulate result to global memory
  if (tid < COLOR_DEPTH) {
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



GrayscaleHistogram::GrayscaleHistogram(float *gpuGrayscale, int imageSize)
{
	_gpuGrayscale = gpuGrayscale;
	_grayscaleSize = imageSize;
	_cummulativeHistogram = NULL;
	_histogram = NULL;
}

GrayscaleHistogram::~GrayscaleHistogram()
{
	if (_cummulativeHistogram != NULL)
	{
		cudaFree(_cummulativeHistogram);
	}
	if (_histogram != NULL)
	{
		cudaFree(_histogram);
	}
}

void GrayscaleHistogram::Run()
{
    cudaMalloc((void**) &_histogram, COLOR_DEPTH * sizeof(int));
    cudaMalloc((void**) &_cummulativeHistogram, COLOR_DEPTH * sizeof(int));
	// TODO: change threads/blocks
    CalculateHistogram<<<MAX_BLOCKS, MAX_THREADS>>>(
    	_gpuGrayscale,
        _grayscaleSize,
        _histogram,
        _cummulativeHistogram);
}

