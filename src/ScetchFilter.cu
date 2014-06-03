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


ScetchFilter::ScetchFilter() {
	image_width_ = image_height_ = 0;
	line_length_ = 20;
	line_strength_ = 1;
	line_count_  = 4;
	free_image_  = false;
	gamma_ = 1.f;
	gpu_image_data_ = gpu_result_data_ = NULL;
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


void ScetchFilter::set_line_strength(float line_strength) {
	line_strength_ = line_strength;
}

void ScetchFilter::set_line_length(int line_length) {
	line_length_ = line_length;
}

void ScetchFilter::set_line_count(int line_count) {
	line_count_ = line_count;
}

void ScetchFilter::set_gamma(float gamma) {
	gamma_ = gamma;
}


float *ScetchFilter::get_gpu_result_data() {
	return gpu_result_data_;
}

float *ScetchFilter::GetCpuResultData() {
	float *image = new float[image_pixel_count()];
	cudaMemcpy(image, gpu_result_data_, image_byte_count(), cudaMemcpyDeviceToHost);
	return image;
}

int ScetchFilter::image_pixel_count() {
	return image_height_ * image_width_;
}

int ScetchFilter::image_byte_count() {
	return image_pixel_count() * sizeof(float);
}

void ScetchFilter::Run() {
	dim3 thread_block_size(32, 32, 1);
	dim3 block_grid_size(1 + image_width_ / thread_block_size.x,
			1 + image_height_ / thread_block_size.y,
			1);
	SimpleScetchKernel<<<block_grid_size, thread_block_size>>>(
			gpu_image_data_,
			gpu_result_data_,
			image_width_, image_height_,
			line_length_, line_strength_, line_count_,
			gamma_);
}

bool ScetchFilter::TestGpuFunctions(std::string *message) {
	int** lines = new int*[line_count_];
	float** weights = new float*[line_count_];
	int max_line_pixel_count = line_strength_ * line_length_;

	// check if max_line_pixel_count is big enough
	for (int x = 0; x < image_width_; x++) {
		for (int y = 0; x < image_width_; x++) {
			for (int i = 0; i < line_count_; i++) {
				lines[i] = new int[max_line_pixel_count];
				weights[i] = new float[max_line_pixel_count];
				float line_anle = static_cast<float>(i) * (M_PI / line_count_);
				int line_pixels_count = LinePixels(x, y, line_anle, image_width_, image_height_,
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
		int line_pixels_count = LinePixels(100, 100, line_anle, image_width_, image_height_,
				line_length_, line_strength_,
				lines[i], weights[i]);

		// create an image for the line, where all line pixels are black, rest white
		unsigned char *line_data = new unsigned char[image_width_*image_height_*3];
		memset(line_data, 255, image_width_*image_height_*3);
		for (int j = 0; j < line_pixels_count; j++) {
			int pixel_index = lines[i][j];
			line_data[3 * pixel_index + 0] = 0;
			line_data[3 * pixel_index + 1] = 0;
			line_data[3 * pixel_index + 2] = 0;
		}
		char line_no[16];
		sprintf(line_no, "%d", i);
		std::string outfilename = std::string("resources/line") + line_no + "_pixel(100,100).jpg";
		if(!jpge::compress_image_to_jpeg_file(outfilename.c_str(), image_width_, image_height_, 3, line_data))
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
