/*
 * macros.h
 *
 *  Created on: Aug 26, 2014
 *      Author: burnicki
 */

#ifndef MACROS_H_
#define MACROS_H_

//#define PROFILING

#ifdef PROFILING
  #include <nvToolsExt.h>
  #define PROF_RANGE_PUSH(s) nvtxRangePushA(s)
  #define PROF_RANGE_POP() nvtxRangePop()
#else
  #define PROF_RANGE_PUSH(s) do {} while(false)
  #define PROF_RANGE_POP() do {} while(false)
#endif



#endif /* MACROS_H_ */
