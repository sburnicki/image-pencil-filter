/*
 * ConvolutionFilter.h
 *
 *  Created on: May 12, 2014
 *      Author: braunra
 */

#ifndef CONVOLUTIONFILTER_H_
#define CONVOLUTIONFILTER_H_

class ConvolutionFilter {
public:
	ConvolutionFilter();
	virtual ~ConvolutionFilter();

	// Creates and uploads the image data from the given cpu_image
	void SetImage(float *cpu_image_data,
				  int image_width, int image_height);

	// Uses the given gpu image data. Expects the data to be already
	// allocated and uploaded.
	// Useful to process an image that is already on the device
	void UseImage(float *gpu_image_data,
				  int image_width, int image_height);

	void SetKernel(float *cpu_kernel_data,
				   int kernel_width, int kernel_height);

	void UseKernel(float *gpu_kernel_data,
				   int kernel_width, int kernel_height);

private:
	int image_pixel_count();
	int kernel_pixel_count();
	int image_byte_count();
	int kernel_byte_count();

	int image_width_, image_height_;
	int kernel_width_, kernel_height_;
	float *gpu_image_data_;
	float *gpu_kernel_data_;
	float *gpu_result_data_;
};

#endif /* CONVOLUTIONFILTER_H_ */
