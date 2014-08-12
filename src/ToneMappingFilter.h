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

class ToneMappingFilter: public ImageFilter {
public:
	ToneMappingFilter(int numTones, int *gpuCumHistogram);
	~ToneMappingFilter();

	const std::vector<float> &GetCpuTonemap();

	// overriden
	void Run();

private:
	int num_tones_;
	int *gpu_histogram_;
	float *gpu_tonemap_array_;
	const ToneMap &cpu_tonemap_;
};

#endif /* TONEMAPPINGFILTER_H_ */

