/*
 * ConvolutionFilter.cpp
 *
 *  Created on: May 12, 2014
 *      Author: braunra
 */

#include "ScetchFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string>
#include <vector>

// shared Memory Size per Multiprocessor is 48KB with Cuda 2.0 to 4.x
// 48KB equals 48k/4 floats
// the side lenght of a shared memory block is the squareroot of 40k/4
// sqrt(48000 / 4) = 109.544...
#define SHARED_2D_BLOCK_DIMENSION 109

// Used Kernel functions
__device__ __host__ int clamp(int value, int mi, int ma) {
  return max(mi, min(ma, value));
}

__device__ __host__ int PixelIndexOf(int x, int y, int width) {
  return x + y * width;
}

__device__ __host__ bool IsInImage(int x, int y, int width, int height) {
  return x >= 0 && x < width &&
    y >= 0 && y < height;
}

__device__ __host__ bool IsInSharedMemoryBlock(int x, int y, int block_dim) {
  return x >= 0 && y >= 0 &&
    x < block_dim && y < block_dim;
}

__device__ __host__ void RotatedCoordinate(float *x, float *y, float angle) {
  float c = cos(angle);
  float s = sin(angle);
  float new_x = c * (*x) - s* (*y);
  float new_y = s * (*x) + c * (*y);
  (*x) = new_x;
  (*y) = new_y;
}


__device__ __host__ bool CalculateCoordinatesInSharedMemoryBlock(
    int x, int y,
    int thread_x, int thread_y,
    int image_x, int image_y,
    float rotation_angle,
    int length,
    int shared_width,
    int image_width, int image_height,
    int *shared_x, int *shared_y) {
  float start_x = 0;
  float start_y = 0;
  float current_x = start_x + x;
  float current_y = start_y + y;
  RotatedCoordinate(&current_x, &current_y, rotation_angle);
  int rotated_image_x = current_x + image_x;
  int rotated_image_y = current_y + image_y;
  current_x += length;  // in shared memory (0,0) is located at
  current_y += length;  // (length, length)
  *shared_x = current_x + thread_x;
  *shared_y = current_y + thread_y;
  return IsInSharedMemoryBlock(*shared_x, *shared_y, shared_width) &&
         IsInImage(rotated_image_x, rotated_image_y, image_width, image_height);
}

__device__ __host__ void ImageCoordinatesFromSharedAddress(
    int shared_address,
    int shared_width,
    int start_x,
    int start_y,
    int *image_x,
    int *image_y) {
  *image_x = shared_address % shared_width;
  *image_y = shared_address / shared_width;
  *image_x = (*image_x) + start_x;
  *image_y = (*image_y) + start_y;
}



// fast scetch kernel
__global__ void HighSpeedScetchKernel(
    float *image,
    float *result,
    int image_width,
    int image_height,
    int shared_width,
    int line_length,
    float line_strength,
    int line_count,
    float rotation_offset,
    float gamma) {
  // Create a shared memory block
  extern __shared__ float image_block[];

  int overhang = line_length;
  int x_image = threadIdx.x + blockDim.x * blockIdx.x;
  int y_image = threadIdx.y + blockDim.y * blockIdx.y;

  int thread_number = threadIdx.x + blockDim.x * threadIdx.y;
  int thread_count_in_block = blockDim.x * blockDim.y;
  int num_copy_iterations = ceil(static_cast<float>(shared_width * shared_width)
                            / thread_count_in_block);
  int start_x = blockDim.x * blockIdx.x - overhang;
  int start_y = blockDim.y * blockIdx.y - overhang;
  for (int i = 0; i < num_copy_iterations; i++) {
    int shared_address = thread_number + i * thread_count_in_block;
    if (shared_address < shared_width * shared_width) {
      int x, y;
      ImageCoordinatesFromSharedAddress(
          shared_address,
          shared_width,
          start_x,
          start_y,
          &x,
          &y);
      if (IsInImage(x, y, image_width, image_height))
        image_block[shared_address] = image[PixelIndexOf(x, y, image_width)];
      else // TODO(Raphael) debug! Necessary?
        image_block[shared_address] = 0.f;
    }
  }
  __syncthreads();

  if (IsInImage(x_image, y_image, image_width, image_height)) {
    // calculate line convolution for all directions
    float angle_step = 2.f * M_PI / line_count;
    float max_convolution_result = 0.f;
    for (int line_index = 0; line_index < line_count; line_index++) {
      float rotation_angle = angle_step * line_index + rotation_offset;
      int n_pixels = 0;
      float sum = 0.f;
      // move along the line from left to right and collect the pixel values
      for (int y = 0; y < line_strength; y++) {
        for (int x = 0; x < line_length; x++) {
          int shared_x, shared_y;
          bool is_inside_block = CalculateCoordinatesInSharedMemoryBlock(
              x, y,
              threadIdx.x, threadIdx.y,
              x_image, y_image,
              rotation_angle,
              overhang,
              shared_width,
              image_width, image_height,
              &shared_x, &shared_y);
          if (is_inside_block) {
            sum += image_block[PixelIndexOf(shared_x, shared_y, shared_width)];
            n_pixels += 1;
          }
        }
      }
      // do the convolution and take the line if its the best so far
      max_convolution_result = max(max_convolution_result, sum / n_pixels);
    }
    // calculate gamma
    result[PixelIndexOf(x_image, y_image, image_width)] =
      max(255.f - __powf(max_convolution_result, gamma), 50.f);
  }
}


ScetchFilter::ScetchFilter() : ImageFilter() {
  line_length_ = 20;
  line_strength_ = 1;
  line_count_  = 4;
  gamma_ = 1.f;
  rotation_offset_ = 0.f;
}


void ScetchFilter::set_line_strength(float line_strength) {
  line_strength_ = line_strength;
}

bool ScetchFilter::set_line_length(int line_length) {
  if (2*line_length + 32 <= SHARED_2D_BLOCK_DIMENSION) {
    line_length_ = line_length;
    return true;
  } else {
    // max line length exeeded
    line_length_ = (SHARED_2D_BLOCK_DIMENSION - 32) / 2;
    return false;
  }
}

void ScetchFilter::set_line_count(int line_count) {
  line_count_ = line_count;
}

void ScetchFilter::set_line_rotation_offset(float offset_angle) {
  rotation_offset_ = offset_angle;
}

void ScetchFilter::set_gamma(float gamma) {
  gamma_ = gamma;
}

void ScetchFilter::Run() {
  // Max threads per Block = 1024 ==> sqrt(1024) = 32
  int pixels_per_dimension = min(SHARED_2D_BLOCK_DIMENSION - (2 * line_length_), 32);
  dim3 high_speed_block_size(pixels_per_dimension, pixels_per_dimension, 1);
  dim3 high_speed_grid_size(GetImageWidth() / pixels_per_dimension + 1,
      GetImageHeight() / pixels_per_dimension + 1,
      1);
  int memory_per_dimension = pixels_per_dimension + 2*line_length_ + 1;
  int shared_memory_size = sizeof(float) * memory_per_dimension * memory_per_dimension;
  HighSpeedScetchKernel<<<high_speed_grid_size, high_speed_block_size, shared_memory_size>>>(
      GetGpuImageData(),
      GetGpuResultData(),
      GetImageWidth(),
      GetImageHeight(),
      memory_per_dimension,
      line_length_,
      line_strength_,
      line_count_,
      rotation_offset_,
      gamma_);
}
