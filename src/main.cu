/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <iostream>

#include "../lib/jpge.h"
#include "../lib/jpgd.h"

int main(int argc, char* argv[]) {
	int width, height, comps;
	if (argc < 3)
	{
		std::cout << "Please provide input and output filenames as arguments." << std::endl;
		return 1;
	}
	char *infilename = argv[1];
	char *outfilename = argv[2];

	unit8 * image = jpgd::decompress_jpeg_image_from_file(infilename, &width, &height, &comps, 3);

	if(!jpge::compress_image_to_jpeg_file(outfilename, width, height, comps, image))
	{
		std::cout << "Error writing the image." << std::endl;
	}

	return 0;
}
