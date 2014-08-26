/*
 * ToneMappingFilter.h
 *
 *  Created on: 01.07.2014
 *      Author: stefan
 */
#ifndef TONEMAPPINGFILTER_H_
#define TONEMAPPINGFILTER_H_

#include "ImageFilter.h"
#include "ToneMap.h"

/*
 * Important: This filter does not only perform tone mapping,
 * but also normalizes the value and applies the log2f to it!
 */
class ToneMappingFilter: public ImageFilter {
public:
	ToneMappingFilter(ToneMap &destinationMap, int *gpuCumHistogram);
	~ToneMappingFilter();

	// overriden
	void Run();

private:
	int *gpu_histogram_;
	float *gpu_tonemap_array_;
};

#endif /* TONEMAPPINGFILTER_H_ */

