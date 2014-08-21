/*
 * ExpandableTexture.h
 *
 *  Created on: Aug 21, 2014
 *      Author: burnicki
 */

#ifndef EXPANDABLETEXTURE_H_
#define EXPANDABLETEXTURE_H_

#include "JpegImage.h"

class ExpandableTexture {
public:
	ExpandableTexture(const char *filename);
	virtual ~ExpandableTexture();
	float *ExpandedBuffer() { return _expandedBuffer; }
	float *LogBuffer() { return _logBuffer; }
	unsigned char *Buffer() { return _texture.Buffer(); }

	void Expand(int width, int height);

private:
	JpegImage _texture;
	int _expandedWidth;
	int _expandedHeight;
	float *_expandedBuffer;
	float *_logBuffer;
};

#endif /* EXPANDABLETEXTURE_H_ */
