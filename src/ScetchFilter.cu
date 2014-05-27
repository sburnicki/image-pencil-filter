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
	*x = c * (*x) - s* (*y);
	*y = s * (*x) - c * (*y);
}



__device__ __host__ int LinePixels(int x, int y, float line_angle, int image_width, int image_height,
	    							int line_length, float line_radius,
	    							int *indices, float *weights) {
	int line_pixel_count = 0;
	int halve_length = line_length / 2;
	for (int j = y - line_radius; j < y + line_radius; j++) {
		for (int i = x - halve_length; i <= x; i++) {
			float rotated_x = i - x;
			float rotated_y = j - x;
			RotatedCoordinate(&rotated_x, &rotated_y, line_angle);
			float mirrored_x = -rotated_x;
			float mirrored_y = -rotated_y;
			rotated_x += x;
			rotated_y += y;
			mirrored_x += x;
			mirrored_y += y;
			if (IsInImage(rotated_x, rotated_y, image_width, image_height)) {
				indices[line_pixel_count] = PixelIndexOf(rotated_x, rotated_y, image_width);
				weights[line_pixel_count] = 1;
				line_pixel_count++;
			}
			if (IsInImage(mirrored_x, mirrored_y, image_width, image_height)) {
				indices[line_pixel_count] = PixelIndexOf(mirrored_x, mirrored_y, image_width);
				weights[line_pixel_count] = 1;
				line_pixel_count++;
			}
		}
	}
	return line_pixel_count;
}

// very basic convolution kernel (no optimizations)
__global__ void SimpleScetchKernel(
		float *image,
		float *result,
		int image_width, int image_height,
		int line_length, float line_radius, int line_count) {
	// some neat index calculations:
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;
	int pixel_index = PixelIndexOf(x, y, image_width);

	// the number of pixels in a line equals the number of pixels in a rectangle
	// the true number of pixels might be smaller due to image boundaries
	int max_line_pixel_count = line_radius * 2 * line_length;

	// allocate some memory
	int* line_pixel_indices = new int[max_line_pixel_count];
	float* weights = new float[max_line_pixel_count];
	float max_value = 0.f;
	for (int line = 0; line < line_count; line++) {
		float line_angle = static_cast<float>(line) * (M_PI / line_count);
		int line_pixel_count = LinePixels(x, y, line_angle, image_width, image_height,
										   line_length, line_radius,
										   line_pixel_indices, weights);
		float convolution_result = 0;
		for  (int i = 0; i < line_pixel_count; i++) {
			float line_pixel_value = image[line_pixel_indices[i]];
			convolution_result += line_pixel_value * weights[i];
		}
		max_value = max(max_value, convolution_result);
	}

	delete[] line_pixel_indices;
	delete[] weights;
	result[pixel_index] = max_value;
}


ScetchFilter::ScetchFilter() {
	image_width_ = image_height_ = 0;
	line_length_ = 20;
	line_radius_ = 0.5;
	free_image_ = false;
}


ScetchFilter::~ScetchFilter() {
	cudaFree(gpu_result_data_);
	if (free_image_)
		cudaFree(gpu_image_data_);
}

void ScetchFilter::SetImage(float* cpu_image_data, int image_width,
		int image_height) {
	image_width_ = image_width;
	image_height_ = image_height;
	// allocate gpu memory
	cudaMalloc((void**) &gpu_image_data_, image_byte_count());
	cudaMalloc((void**) &gpu_result_data_, image_byte_count());
	// copy data to gpu
	cudaMemcpy(gpu_image_data_, cpu_image_data, image_byte_count(),
			cudaMemcpyHostToDevice);
}

void ScetchFilter::UseImage(float* gpu_image_data, int image_width,
		int image_height) {
	image_width_ = image_width;
	image_height_ = image_height;
	gpu_image_data_ = gpu_image_data;
	cudaMalloc((void**) &gpu_result_data_, image_byte_count());
}


void ScetchFilter::set_line_strength(int line_strength) {
	line_radius_ = static_cast<float>(line_strength) / 2.f;
}

void ScetchFilter::set_line_length(int line_length) {
	line_length_ = line_length;
}

void ScetchFilter::set_line_count(int line_count) {
	line_count_ = line_count;
}

int ScetchFilter::image_pixel_count() {
	return image_height_ * image_width_;
}

int ScetchFilter::image_byte_count() {
	return image_pixel_count() * sizeof(float);
}

void ScetchFilter::Run() {
	dim3 thread_block_size(64, 64, 1);
	dim3 block_grid_size(1 + image_width_ / thread_block_size.x,
						 1 + image_height_ / thread_block_size.y,
						 1);
	SimpleScetchKernel<<<block_grid_size, thread_block_size>>>(
			gpu_image_data_,
			gpu_result_data_,
			image_width_, image_height_,
			line_length_, line_radius_, line_count_);
}
