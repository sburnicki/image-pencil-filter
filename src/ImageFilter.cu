/*
 * ImageFilter.cpp
 *
 *  Created on: 15.06.2014
 *      Author: stefan
 */

#include "ImageFilter.h"
#include <iostream>

ImageFilter::ImageFilter() {
	image_width_ = image_height_ = 0;
	free_image_  = free_result_ = false;
	gpu_image_data_ = gpu_result_data_ = NULL;
}

ImageFilter::~ImageFilter() {
	if (free_result_)
	{
		cudaFree(gpu_result_data_);
	}
	if (free_image_)
	{
		cudaFree(gpu_image_data_);
	}
}

void ImageFilter::SetImageFromCpu(float* cpu_image_data, int image_width,
		int image_height)
{
	SetImageFromCpu(cpu_image_data, image_width, image_height, NULL);
}

void ImageFilter::SetImageFromCpu(float* cpu_image_data, int image_width,
		int image_height, float *gpu_result_buffer) {
	image_width_ = image_width;
	image_height_ = image_height;
	// allocate gpu memory
	std::cerr << "Automatically allocating buffer for input image" << std::endl;
	cudaMalloc((void**) &gpu_image_data_, image_byte_count());
	set_result_buffer(gpu_result_buffer);
	// copy data to gpu
	cudaMemcpy(gpu_image_data_, cpu_image_data, image_byte_count(),
			cudaMemcpyHostToDevice);
}

void ImageFilter::SetImageFromGpu(float* gpu_image_data, int image_width,
		int image_height)
{
	SetImageFromGpu(gpu_image_data, image_width, image_height, NULL);
}

void ImageFilter::SetImageFromGpu(float* gpu_image_data, int image_width,
		int image_height, float *gpu_result_buffer) {
	image_width_ = image_width;
	image_height_ = image_height;
	gpu_image_data_ = gpu_image_data;
	set_result_buffer(gpu_result_buffer);
}


float *ImageFilter::GetGpuResultData() {
	return gpu_result_data_;
}

float *ImageFilter::GetCpuResultData() {
	float *image = new float[image_pixel_count()];
	cudaMemcpy(image, gpu_result_data_, image_byte_count(), cudaMemcpyDeviceToHost);
	return image;
}

void ImageFilter::set_result_buffer(float *gpu_result_buffer)
{
	if (gpu_result_buffer == NULL)
	{
		std::cout << "WARNING: Automatically allocating buffer for result image" << std::endl;
		cudaMalloc((void**) &gpu_result_data_, image_byte_count());
		free_result_ = true;
	}
	else
	{
		gpu_result_data_ = gpu_result_buffer;
		free_result_ = false;
	}
}

int ImageFilter::image_pixel_count() {
	return image_height_ * image_width_;
}

int ImageFilter::image_byte_count() {
	return image_pixel_count() * sizeof(float);
}
