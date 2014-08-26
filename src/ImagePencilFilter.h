/*
 * ImagePencilFilterr.h
 *
 *  Created on: Aug 26, 2014
 *      Author: burnicki
 */

#ifndef IMAGEPENCILFILTERR_H_
#define IMAGEPENCILFILTERR_H_

#include "macros.h"

#define MAX_BLOCKS 256
#define MAX_THREADS 256

class IPFConfiguration
{
public:
	IPFConfiguration()
	{
		ScetchLineCount = 13;
		ScetchLineLength = 10;
		ScetchLineStrength = 1;
		ScetchLineRotationOffset = 5 * ONE_DEGREE;
		ScetchGamma = 1.1;

		ToneMapBrightSigma = 9.0f;
		ToneMapBrightWeight = 31.0f; // 11.0f;

		ToneMapMiddleMuLower = 130.0f;
		ToneMapMiddleMuUpper = 200.0f;
		ToneMapMiddleWeight = 2.0f;

		ToneMapDarkMu = 90.0f;
		ToneMapDarkSigma = 11.0f;
		ToneMapDarkWeight = 1.0f;

		TextureRenderingSmoothness = 2.0f;

		UseColors = true;
	}

	int ScetchLineCount;
	int ScetchLineLength;
	float ScetchLineStrength;
	float ScetchLineRotationOffset;
	float ScetchGamma;

	float ToneMapBrightSigma;
	float ToneMapBrightWeight;

	float ToneMapMiddleMuLower;
	float ToneMapMiddleMuUpper;
	float ToneMapMiddleWeight;

	float ToneMapDarkMu;
	float ToneMapDarkSigma;
	float ToneMapDarkWeight;

	float TextureRenderingSmoothness;

	bool UseColors;
};


#endif /* IMAGEPENCILFILTERR_H_ */
