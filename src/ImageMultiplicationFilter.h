/*
 * ImageMultiplicationFilter.h
 *
 *  Created on: Aug 12, 2014
 *      Author: burnicki
 */

#ifndef IMAGEMULTIPLICATIONFILTER_H_
#define IMAGEMULTIPLICATIONFILTER_H_

#include "ImageFilter.h"

class ImageMultiplicationFilter: public ImageFilter {
public:
	ImageMultiplicationFilter(float *gpu_base_img);
	virtual ~ImageMultiplicationFilter();

	// overriden
	void Run();
private:
	float *gpu_base_img_;
};

#endif /* IMAGEMULTIPLICATIONFILTER_H_ */
