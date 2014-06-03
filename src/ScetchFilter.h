/*
 * ConvolutionFilter.h
 *
 *  Created on: May 12, 2014
 *      Author: braunra
 */

#ifndef CONVOLUTIONFILTER_H_
#define CONVOLUTIONFILTER_H_

#include <string>

class ScetchFilter {
public:
	ScetchFilter();
	virtual ~ScetchFilter();

	// Creates and uploads the image data from the given cpu_image
	void SetImage(float *cpu_image_data,
				  int image_width, int image_height);

	// Uses the given gpu image data. Expects the data to be already
	// allocated and uploaded to the gpu.
	// Useful to process an image that is already on the device
	void UseImage(float *gpu_image_data,
				  int image_width, int image_height);

	// set the strength of the scetch lines in pixels
	void set_line_strength(float line_strength);

	// set the length of the scetch lines in pixels
	void set_line_length(int line_length);

	// set the number of used lines for the scetch algorithm.
	// The angle between all possible lines will be 360/linecount
	void set_line_count(int line_count);

	// set the gamma value for the lines
	// usefull to darken lines for images with weak gradient magnitudes
	void set_gamma(float gamma);

	// returns the pointer to the result image
	// expects that Run() was already called
	float *get_gpu_result_data();

	// Downloads the result from the GPU and returns a pointer to the data.
	// Expectst that Run() was already called.
	// the caller has to do the clean up!
	float *GetCpuResultData();

	// Applies the  the filter.
	// An image must have been uploaded beforehand either with SetImage or UseImage
	// Result can be gathered with get_gpu_result_data() or GetCpuResultData()
	void Run();

	// For debugging
	bool TestGpuFunctions(std::string *message);

private:
	// little helper functions
	int image_pixel_count();
	int image_byte_count();

	int image_width_, image_height_;
	int line_length_, line_count_;
	float line_strength_;
	bool free_image_;
	float *gpu_image_data_;
	float *gpu_result_data_;
	float gamma_;
};

#endif /* CONVOLUTIONFILTER_H_ */
