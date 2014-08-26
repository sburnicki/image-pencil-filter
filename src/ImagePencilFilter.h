/*
 * ImagePencilFilterr.h
 *
 *  Created on: Aug 26, 2014
 *      Author: burnicki
 */

#ifndef IMAGEPENCILFILTERR_H_
#define IMAGEPENCILFILTERR_H_

#define MAX_BLOCKS 256
#define MAX_THREADS 256

#define COLOR_DEPTH 256
#define MAX_COLOR_VALUE COLOR_DEPTH - 1

#define YUV_COMPONENTS 3
#define RGB_COMPONENTS 3

#define PENCIL_TEXTURE_PATH "resources/texture4.jpg"
#define ONE_DEGREE 0.0174532925


#define RGB_TO_Y(R, G, B) \
	(R*0.299 + G*0.587 + B*0.114)

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
