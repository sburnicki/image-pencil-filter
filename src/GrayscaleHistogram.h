/*
 * GrayscaleHistogram.h
 *
 *  Created on: Aug 21, 2014
 *      Author: burnicki
 */

#ifndef GRAYSCALEHISTOGRAM_H_
#define GRAYSCALEHISTOGRAM_H_

#define COLOR_DEPTH 256

class GrayscaleHistogram {
public:
	GrayscaleHistogram(float *gpuGrayscale, int imageSize);
	virtual ~GrayscaleHistogram();

	int *GpuHistogram() { return _histogram; }
	int *GpuCummulativeHistogram() { return _cummulativeHistogram; }

	void Run();

private:
	float *_gpuGrayscale;
	int _grayscaleSize;
	int *_histogram;
	int *_cummulativeHistogram;
};

#endif /* GRAYSCALEHISTOGRAM_H_ */
