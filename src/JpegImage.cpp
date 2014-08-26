/*
 * JpegImage.cpp
 *
 *  Created on: Aug 21, 2014
 *      Author: burnicki
 */

#include "JpegImage.h"

#include <exception>
#include <string>

#include "../lib/jpge.h"
#include "../lib/jpgd.h"

JpegImage::JpegImage(const char *filename)
{
	int comps;

	_buffer = jpgd::decompress_jpeg_image_from_file(filename, &_width, &_height, &comps, RGB_COMPONENTS);
	_size = _width * _height;
	if (comps != RGB_COMPONENTS)
	{
		free(_buffer);
		_buffer = NULL;
		if (comps == 0)
		{
			throw (std::string("Loading the image") + filename + " failed! Wrong path?.").c_str();
		}
		else
		{
			throw "Currently only images with 3 components are supported.";
		}
	}
}

JpegImage::~JpegImage()
{
	free(_buffer);
}

void JpegImage::Save(const char *filename)
{
	bool success = jpge::compress_image_to_jpeg_file(filename, _width, _height, RGB_COMPONENTS, _buffer);
	if(!success)
	{
		throw "Error writing the image.";
	}
}

