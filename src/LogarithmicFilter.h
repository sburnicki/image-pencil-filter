/*
 * LogarithmicFilter.h
 *
 *  Created on: Aug 12, 2014
 *      Author: burnicki
 */

#ifndef LOGARITHMICFILTER_H_
#define LOGARITHMICFILTER_H_

#include "ImageFilter.h"

class LogarithmicFilter: public ImageFilter {
public:
	LogarithmicFilter();

	// overwritten
	void Run();
};
#endif /* LOGARITHMICFILTER_H_ */

