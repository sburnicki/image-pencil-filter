/*
 * TextureExpander.h
 *
 *  Created on: Aug 26, 2014
 *      Author: burnicki
 */

#ifndef TEXTUREEXPANDER_H_
#define TEXTUREEXPANDER_H_

class TextureExpander {
public:
	TextureExpander(unsigned char *gpuColoredTexture, int origWidth, int origHeight);
	void ExpandDesaturateAndLogTo(float *gpuResultBuffer, int resultWidth, int resultHeight);

private:
	unsigned char *gpu_colored_texture_;
	int orig_width_;
	int orig_height_;
};

#endif /* TEXTUREEXPANDER_H_ */
