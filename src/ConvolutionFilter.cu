/*
 * ConvolutionFilter.cpp
 *
 *  Created on: May 12, 2014
 *      Author: braunra
 */

#include "ConvolutionFilter.h"
#include <stdio.h>
#include <stdlib.h>

ConvolutionFilter::ConvolutionFilter() {
	image_width_ = image_height_ = 0;
	free_image_ = free_kernel_ = false;
}


ConvolutionFilter::~ConvolutionFilter() {
	cudaFree(gpu_result_data_);
	if (free_image_)
		cudaFree(gpu_image_data_);
	if (free_kernel_)
		cudaFree(gpu_kernel_data_);
}

void ConvolutionFilter::SetImage(float* cpu_image_data, int image_width,
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

void ConvolutionFilter::UseImage(float* gpu_image_data, int image_width,
		int image_height) {
	image_width_ = image_width;
	image_height_ = image_height;
	gpu_image_data_ = gpu_image_data;
	cudaMalloc((void**) &gpu_result_data_, image_byte_count());
}

void ConvolutionFilter::SetKernel(float* cpu_kernel_data,
		int kernel_width, int kernel_height) {
	kernel_width_ = kernel_width;
	kernel_height_ = kernel_height;
	// allocate gpu memory
	cudaMalloc((void**) &gpu_kernel_data_, image_byte_count());
	// copy data to gpu
	cudaMemcpy(gpu_kernel_data_, cpu_kernel_data, kernel_byte_count(),
			cudaMemcpyHostToDevice);
}

void ConvolutionFilter::UseKernel(float* gpu_kernel_data,
		int kernel_width, int kernel_height) {
	kernel_width_ = kernel_width;
	kernel_height_ = kernel_height;
	gpu_kernel_data_ = gpu_kernel_data;
}

int ConvolutionFilter::image_pixel_count() {
	return image_height_ * image_width_;
}

int ConvolutionFilter::kernel_pixel_count() {
	return kernel_height_ * kernel_width_;
}

int ConvolutionFilter::image_byte_count() {
	return image_pixel_count() * sizeof(float);
}

void ConvolutionFilter::Run() {
	SimpleConvolutionKernel(gpu_image_data_,
							gpu_kernel_data_,
							gpu_result_data_,
							image_width_, image_height_,
							kernel_width_, kernel_height_);
}

int ConvolutionFilter::kernel_byte_count() {
	return kernel_pixel_count() * sizeof(float);
}

__global__ void ConvolutionFilter::SimpleConvolutionKernel(const float* source_image,
		const flaot* kernel, float* result, int image_width, int image_height,
		int kernel_width, int kernel_height) {

}
