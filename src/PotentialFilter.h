/*
 * PotentialFilter.h
 *
 *  Created on: Aug 12, 2014
 *      Author: altergot
 */

#ifndef POTENTIALFILTER_H_
#define POTENTIALFILTER_H_

#include "ImageFilter.h"

// This Filter can work inplace: Input Picture can be ouput picture
class PotentialFilter: public ImageFilter {
public:
	PotentialFilter(float *gpu_beta);

	// overwritten
	void Run();

private:
	float *gpu_beta_;
};
#endif /* POTENTIALFILTER_H_ */
