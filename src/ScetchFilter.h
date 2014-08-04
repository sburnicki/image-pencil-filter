/*
 * ConvolutionFilter.h
 *
 *  Created on: May 12, 2014
 *      Author: braunra
 */

#ifndef CONVOLUTIONFILTER_H_
#define CONVOLUTIONFILTER_H_

#include <string>
#include "ImageFilter.h"

class ScetchFilter : public ImageFilter {
public:
	ScetchFilter();

	// set the strength of the scetch lines in pixels
	void set_line_strength(float line_strength);

	// set the length of the scetch lines in pixels
	void set_line_length(int line_length);

	// set the number of used lines for the scetch algorithm.
	// The angle between all possible lines will be 360/linecount
	void set_line_count(int line_count);

  // sets the offset angle of the lines that are used for convolution
  // the given angle is in RADIANTS, not degree!
  // default ist 0: the first line is horizontal.
  void set_line_rotation_offset(float offste_angle);

	// set the gamma value for the lines
	// usefull to darken lines for images with weak gradient magnitudes
	void set_gamma(float gamma);

	// overriden
	void Run();

	// For debugging
	bool TestGpuFunctions(std::string *message);

private:
	int line_length_, line_count_;
	float line_strength_;
	float gamma_;
  float rotation_offset_;
};

#endif /* CONVOLUTIONFILTER_H_ */
