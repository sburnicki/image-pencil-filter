/*
 * macros.h
 *
 *  Created on: Aug 26, 2014
 *      Author: burnicki
 */

#ifndef MACROS_H_
#define MACROS_H_

#define DUMMYOP() do {} while(false)

//#define PROFILING

#ifdef PROFILING
  #include <nvToolsExt.h>
  #define PROF_RANGE_PUSH(s) nvtxRangePushA(s)
  #define PROF_RANGE_POP() nvtxRangePop()
#else
  #define PROF_RANGE_PUSH(s) DUMMYOP()
  #define PROF_RANGE_POP() DUMMYOP()
#endif

#define COLOR_DEPTH 256
#define MAX_COLOR_VALUE COLOR_DEPTH - 1

#define YUV_COMPONENTS 3
#define RGB_COMPONENTS 3

#define PENCIL_TEXTURE_PATH "resources/texture4.jpg"
#define ONE_DEGREE 0.0174532925


#define RGB_TO_Y(R, G, B) \
	(R*0.299 + G*0.587 + B*0.114)

#define PIXEL_INDEX_OF(x, y, width) (x + y * width)
#define IS_IN_IMAGE(x, y, width, height) (x >= 0 && x < width && y >= 0 && y < height)

#endif /* MACROS_H_ */
