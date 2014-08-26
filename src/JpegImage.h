/*
 * JpegImage.h
 *
 *  Created on: Aug 21, 2014
 *      Author: burnicki
 */

#ifndef JPEGIMAGE_H_
#define JPEGIMAGE_H_

#include "ImagePencilFilter.h"

class JpegImage {
public:
	JpegImage(const char *filename);
	virtual ~JpegImage();

    unsigned char& operator[] (int idx) { return _buffer[idx]; }

	unsigned char *Buffer() { return _buffer; }
	int Width() const { return _width; }
	int Height() const { return _height; }
	int PixelSize() const  { return _size; }
	int ByteSize() const  { return _size * RGB_COMPONENTS * sizeof(unsigned char); }

	void Save(const char *filename);

private:
	unsigned char *_buffer;
	int _width;
	int _height;
	int _size;
};

#endif /* JPEGIMAGE_H_ */
