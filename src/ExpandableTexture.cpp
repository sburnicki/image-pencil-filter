/*
 * ExpandableTexture.cpp
 *
 *  Created on: Aug 21, 2014
 *      Author: burnicki
 */

#include "ExpandableTexture.h"
#include <stdlib.h>
#include <cmath>

ExpandableTexture::ExpandableTexture(const char *filename) : _texture(filename)
{
	_expandedWidth = 0;
	_expandedHeight = 0;
	_expandedBuffer = NULL;
	_logBuffer = NULL;
}

ExpandableTexture::~ExpandableTexture()
{
	if (_expandedBuffer != NULL)
	{
		free(_expandedBuffer);
	}
	if (_logBuffer != NULL)
	{
		free(_logBuffer);
	}
}

void ExpandableTexture::Expand(int width, int height)
{
	_expandedWidth = width;
	_expandedHeight = height;

	int expandedSize = width * height;
    int textureWidth = _texture.Width();

	_logBuffer = (float *) malloc(sizeof(float) * expandedSize);
    _expandedBuffer = (float *) malloc(sizeof(float) * expandedSize);

    for (int y = 0; y < height; y++) {
    	for (int x = 0; x < width; x++) {
    		int destIdx = x + y * width;
    		int srcX = x % textureWidth;
    		int srcY = y % _texture.Height();

    		int srcIdx = (srcX + srcY * textureWidth) * RGB_COMPONENTS;

    		// The following line should be the same as:
    		// expanded_text[destIdx] = RGB_TO_Y(cpuPencilTexture[srcIdx], cpuPencilTexture[srcIdx+1], cpuPencilTexture[srcIdx+2]) / 255.0f;
    		// However, using the macro leads to a runtime error in the cusp solver... wtf.
    		_expandedBuffer[destIdx] =  _texture[srcIdx] * 0.299;
    		_expandedBuffer[destIdx] += _texture[srcIdx+1] * 0.587;
    		_expandedBuffer[destIdx] += _texture[srcIdx+2] * 0.114;
    		_expandedBuffer[destIdx] /= 255.f;

    		_logBuffer[destIdx] = logf(_expandedBuffer[destIdx]);
    	}
    }
}

