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

__device__ __host__ void CalculateCoordinatesInSharedMemoryBlock(
    int x, int y,
    int thread_x, int thread_y,
    float rotation_angle,
    int &shared_x, int &shared_y) {
// TODO(Raphael) To be continued
}



// fast scetch kernel
__global__ void HighSpeedScetchKernel(
    float *image,
    float *result,
    int image_width,
    int image_height,
    int line_length,
    float line_strength,
    int line_count,
    float rotation_offset,
    float gamma) {
  // Create a shared memory block
  __shared__ float image_block[SHARED_2D_BLOCK_DIMENSION][SHARED_2D_BLOCK_DIMENSION];

  // fill the shared memory block
  int overhang = ceil(static_cast<float>(line_length) / 2.f);  //TODO(Raphael) was wenn line_length ungerade?
  // first copy in the pixels of the trivial mappings of the threads to pixels:
  int x_image = threadIdx.x + blockDim.x * blockIdx.x;
  int y_image = threadIdx.y + blockDim.y * blockIdx.y;
  image_block[overhang + threadIdx.x][overhang + threadIdx.y] = image[PixelIndexOf(x, y, image_width)];
  // then copy the 4 left overhanging regions clockwise starting left
  int thread_number = threadIdx.x + blockDim.x * threadIdx.y;
  // left overhang (complete)
  int start_x = blockDim.x * blockIdx.x - overhang;
  int start_y = blockDim.y * blockIdx.y - overhang;
  if ((start_x >= 0) &&
      (thread_number < 2 * overhang * overhang + overhang * SHARED_2D_BLOCK_DIMENSION)) {
    image_block[thread_number % overhang][thread_number / overhang] =
      image[PixelIndexOf(start_x + (thread_number % overhang),
          start_y + (thread_number / overhang),
          image_width)];
  }
  start_x = start_x + overhang;
  // top overhang (without left corner)
  if ((start_y >= 0) &&
      (thread_number < overhang * overhang + overhang * SHARED_2D_BLOCK_DIMENSION)) {
    image_block[overhang + thread_number % (blockDim.x + overhang)][thread_number / (blockDim.x + overhang)] =
      image[PixelIndexOf(start_x + (thread_number % (blockDim.x + overhang)),
          start_y + (thread_number % (blockDim.x + overhang)),
          image_width)];
  }
  start_x = start_x + blockDim.x;
  start_y = start_y + overhang;
  // right overhang (without top corner)
  if ((start_x + blockDim.x + 2 * overhang < image_width) &&
      (thread_number < overhang * overhang + overhang * SHARED_2D_BLOCK_DIMENSION)) {
    image_block[blockDim.x + (thread_number % overhang)][overhang + (thread_number / overhang)] =
      image[PixelIndexOf(start_x + (thread_number % overhang),
          start_y + (thread_number / overhang),
          image_width)];
  }
  start_x = blockDim.x * blockIdx.x;
  start_y = start_y - blockDim.y;
  // bottom overhang (without any corners)
  if ((start_y - overhang < image_height) &&
      (thread_number < overhang * blockDim.x)) {
    image_block[overhang + (thread_number % overhang)][overhang + blockDim.y + (thread_number / overhang)] =
      image[PixelIndexOf(start_x + (thread_number % blockDim.x),
          start_y + (thread_number / blockDim.x),
          image_width)];
  }

  // calculate line convolution for all directions
  angle_step = M_PI / line_count;
  for (int line_index = 0; direction_index < line_count; direction_index++) {
    float rotation_angle = angle_step * line_index;
    int n_pixels = 0;
    float sum = 0.f;
    // move along the line from left to right and collect the pixel values
    for (int y = 0; y < line_strength; y++) {
      for (int x = 0; x < line_length; x++) {
        int shared_x, shared_y;
        CalculateCoordinatesInSharedMemoryBlock(x, y,
            threadIdx.x, threadIdx.y,
            rotation_angle, &shared_x, &shared_y);
        sum = sum + image_block[shared_x][shared_y];
        n_pixels += 1;
      }
    }
    // do the convolution
    image[PixelIndexOf(x, y, image_width)] = sum / n_pixels;
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
  int imageh = GetImageHeight();
  int imagew = GetImageWidth();
  dim3 thread_block_size(32, 32, 1);
  dim3 block_grid_size(1 + imagew / thread_block_size.x,
      1 + imageh / thread_block_size.y,
      1);
  SimpleScetchKernel<<<block_grid_size, thread_block_size>>>(
      GetGpuImageData(),
      GetGpuResultData(),
      imagew, imageh,
      line_length_, line_strength_, line_count_,
      gamma_);


  dim3 high_speed_grid_size = block_grid_size;
  dim3 high_speed_block_size = thread_block_size;
  HighSpeedScetchKernel<<<high_speed_grid_size, high_speed_block_size>>>(
      GetGpuImageData(),
      GetGpuResultData(),
      GetImageWidth(),
      GetImageHeight(),
      line_length_,
      line_strength_,
      line_count_,
      rotation_offset_,
      gamma_);
}



bool ScetchFilter::TestGpuFunctions(std::string *message) {
  int** lines = new int*[line_count_];
  float** weights = new float*[line_count_];
  int max_line_pixel_count = line_strength_ * line_length_;
  int image_width = GetImageWidth();
  int image_height = GetImageHeight();

  // check if max_line_pixel_count is big enough
  for (int x = 0; x < image_width; x++) {
    for (int y = 0; x < image_width; x++) { // <<< wtf, is this correct?
      for (int i = 0; i < line_count_; i++) {
        lines[i] = new int[max_line_pixel_count];
        weights[i] = new float[max_line_pixel_count];
        float line_anle = static_cast<float>(i) * (M_PI / line_count_);
        int line_pixels_count = LinePixels(x, y, line_anle, image_width, image_height,
            line_length_, line_strength_,
            lines[i], weights[i]);
        if (line_pixels_count > max_line_pixel_count) {
          char x_string[16], y_string[16];
          sprintf(x_string, "%d", x);
          sprintf(y_string, "%d", y);
          (*message) = std::string("ERROR: more LinePixels returnt to many pixels for position (") +
            x_string + "," + y_string + ")!";
          return false;
        }
      }
    }
  }

  // show lines for some pixels
  for (int i = 0; i < line_count_; i++) {
    lines[i] = new int[max_line_pixel_count];
    weights[i] = new float[max_line_pixel_count];
    float line_anle = static_cast<float>(i) * (M_PI / line_count_);
    int line_pixels_count = LinePixels(100, 100, line_anle, image_width, image_height,
        line_length_, line_strength_,
        lines[i], weights[i]);

    // create an image for the line, where all line pixels are black, rest white
    unsigned char *line_data = new unsigned char[image_width*image_height*3];
    memset(line_data, 255, image_width*image_height*3);
    for (int j = 0; j < line_pixels_count; j++) {
      int pixel_index = lines[i][j];
      line_data[3 * pixel_index + 0] = 0;
      line_data[3 * pixel_index + 1] = 0;
      line_data[3 * pixel_index + 2] = 0;
    }
    char line_no[16];
    sprintf(line_no, "%d", i);
    std::string outfilename = std::string("resources/line") + line_no + "_pixel(100,100).jpg";
    if(!jpge::compress_image_to_jpeg_file(outfilename.c_str(), image_width, image_height, 3, line_data))
    {
      (*message) = "Error while writing image to disk";
      return false;
    }
    delete[] line_data;
  }
  // cleanup
  for (int i = 0; i < line_count_; i++) {
    delete[] lines[i];
    delete[] weights[i];
  }
  delete[] lines;
  delete[] weights;

  return true;
}
