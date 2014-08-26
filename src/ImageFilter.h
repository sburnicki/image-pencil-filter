/*
 * ImageFilter.h
 *
 *  Created on: 15.06.2014
 *      Author: stefan
 */

#ifndef IMAGEFILTER_H_
#define IMAGEFILTER_H_

#include <string>

class ImageFilter {
public:
	ImageFilter();
	virtual ~ImageFilter();



	// Creates and uploads the image data from the given cpu_image
	// Optional: Set an existing result buffer
	void SetImageFromCpu(float *cpu_image_data,
				  int image_width, int image_height);
	void SetImageFromCpu(float *cpu_image_data,
				  int image_width, int image_height, float *gpu_result_buffer);

	// Uses the given gpu image data. Expects the data to be already
	// allocated and uploaded to the gpu.
	// Useful to process an image that is already on the device
	// Optional: Set an existing result buffer
	void SetImageFromGpu(float *gpu_image_data,
				  int image_width, int image_height);
	void SetImageFromGpu(float *gpu_image_data,
				  int image_width, int image_height, float *gpu_result_buffer);

	// returns the pointer to the result image
	// expects that Run() was already called
	float *GetGpuResultData();

	// Downloads the result from the GPU and returns a pointer to the data.
	// Expectst that Run() was already called.
	// the caller has to do the clean up!
	float *GetCpuResultData();

	// Applies the  the filter.
	// An image must have been uploaded beforehand either with SetImageFromGpu or SetImageFromCpu
	// Result can be gathered with GetGpuResultData() or GetCpuResultData()
	virtual void Run() = 0;

protected:
	// getters for use in derived classes
	int GetImageWidth() { return image_width_; }
	int GetImageHeight() { return image_height_; }
	float *GetGpuImageData() { return gpu_image_data_; }

private:
	void set_result_buffer(float *gpu_result_buffer);
	// little helper functions
	int image_pixel_count();
	int image_byte_count();

	int image_width_, image_height_;
	bool free_image_;
	bool free_result_;
	float *gpu_image_data_;
	float *gpu_result_data_;
};

#endif /* IMAGEFILTER_H_ */
