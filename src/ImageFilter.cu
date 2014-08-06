/*
 * ImageFilter.cpp
 *
 *  Created on: 15.06.2014
 *      Author: stefan
 */

#include "ImageFilter.h"

ImageFilter::ImageFilter() {
	image_width_ = image_height_ = 0;
	free_image_  = false;
	gpu_image_data_ = gpu_result_data_ = NULL;
}

ImageFilter::~ImageFilter() {
	cudaFree(gpu_result_data_);
	if (free_image_)
		cudaFree(gpu_image_data_);
}

void ImageFilter::SetImageFromCpu(float* cpu_image_data, int image_width,
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

void ImageFilter::SetImageFromGpu(float* gpu_image_data, int image_width,
		int image_height) {
	image_width_ = image_width;
	image_height_ = image_height;
	gpu_image_data_ = gpu_image_data;
	cudaMalloc((void**) &gpu_result_data_, image_byte_count());
}


float *ImageFilter::GetGpuResultData() {
	return gpu_result_data_;
}

float *ImageFilter::GetCpuResultData() {
	float *image = new float[image_pixel_count()];
	cudaMemcpy(image, gpu_result_data_, image_byte_count(), cudaMemcpyDeviceToHost);
	return image;
}


int ImageFilter::image_pixel_count() {
	return image_height_ * image_width_;
}

int ImageFilter::image_byte_count() {
	return image_pixel_count() * sizeof(float);
}