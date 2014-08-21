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

// only for debugging!
#include "../lib/jpge.h"
#include "../lib/jpgd.h"
// -------------------

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


// calculates indices and corresponding weights of all pixels along a line
__device__ __host__ int LinePixels(int x, int y, float line_angle, int image_width, int image_height,
    int line_length, float line_strength,
    int *indices, float *weights) {
  int line_pixel_count = 0;
  float halve_length = static_cast<float>(line_length) / 2.f;
  float halve_strength = line_strength / 2.f;

  for (int j = ceil(y - halve_strength); j < ceil(y + halve_strength); j++) {
    for (int i = ceil(x - halve_length); i < ceil(x + halve_length); i++) {
      float rotated_x = i - x;
      float rotated_y = j - y;
      RotatedCoordinate(&rotated_x, &rotated_y, line_angle);
      rotated_x  += x;
      rotated_y  += y;
      if (IsInImage(rotated_x, rotated_y, image_width, image_height)) {
        indices[line_pixel_count] = PixelIndexOf(rotated_x, rotated_y, image_width);
        weights[line_pixel_count] = 1;
        line_pixel_count++;
      }
    }
  }
  return line_pixel_count;
}

// scetch kernel
__global__ void SimpleScetchKernel(
    float *image,
    float *result,
    int image_width, int image_height,
    int line_length, float line_strength, int line_count,
    float gamma) {
  // some neat index calculations:
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;

  if (IsInImage(x, y, image_width, image_height)) {
    int pixel_index = PixelIndexOf(x, y, image_width);

    // the number of pixels in a line equals the number of pixels in a rectangle
    // the true number of pixels might be smaller due to image boundaries
    int max_line_pixel_count = line_strength * line_length;

    // allocate some memory for the line pixel indices and the corresponding weights
    int* line_pixel_indices = new int[max_line_pixel_count];
    float* weights = new float[max_line_pixel_count];
    float max_value = 0.f;
    for (int line = 0; line < line_count; line++) {
      float line_angle = static_cast<float>(line) * (M_PI / line_count);
      int line_pixel_count = LinePixels(x, y, line_angle, image_width, image_height,
          line_length, line_strength,
          line_pixel_indices, weights);
      float convolution_result = 0;
      for  (int i = 0; i < line_pixel_count; i++) {
        float line_pixel_value = image[line_pixel_indices[i]];
        convolution_result += line_pixel_value * weights[i] / line_pixel_count;
      }
      max_value = max(max_value, convolution_result);
    }

    delete[] line_pixel_indices;
    delete[] weights;
    result[pixel_index] = max(255.f - __powf(max_value, gamma), 0.f);
  }
}

__device__ __host__ bool CalculateCoordinatesInSharedMemoryBlock(
    int x, int y,
    int thread_x, int thread_y,
    int image_x, int image_y,
    float rotation_angle,
    int half_length,
    int shared_width,
    int image_width, int image_height,
    int *shared_x, int *shared_y) {
  float start_x = -half_length;
  float start_y = 0;
  float current_x = start_x + x;
  float current_y = start_y + y;
  RotatedCoordinate(&current_x, &current_y, rotation_angle);
  int rotated_image_x = current_x + image_x;
  int rotated_image_y = current_y + image_y;
  current_x += half_length;  // in shared memory (0,0) is located at
  current_y += half_length;  // (half_length, half_length)
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

  int overhang = ceilf(static_cast<float>(line_length) / 2.f);
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
    float angle_step = M_PI / line_count;
    float max_convolution_result = 0.f;
    for (int line_index = 0; line_index < line_count; line_index++) {
      float rotation_angle = angle_step * line_index;
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

void ScetchFilter::set_line_length(int line_length) {
  line_length_ = line_length;
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
  int pixels_per_dimension = min(SHARED_2D_BLOCK_DIMENSION - (line_length_ + 1), 32);
  dim3 high_speed_block_size(pixels_per_dimension, pixels_per_dimension, 1);
  dim3 high_speed_grid_size(GetImageWidth() / pixels_per_dimension + 1,
      GetImageHeight() / pixels_per_dimension + 1,
      1);
  int memory_per_dimension = pixels_per_dimension + line_length_ + 1;
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
