/*
 * PotentialFilter.h
 *
 *  Created on: Aug 12, 2014
 *      Author: altergot
 */

#ifndef POTENTIALFILTER_H_
#define POTENTIALFILTER_H_

#include "ImageFilter.h"

class PotentialFilter: public ImageFilter {
public:
	PotentialFilter(float *beta);

	// overwritten
	void Run();

private:
	float *beta;
};
#endif /* POTENTIALFILTER_H_ */
