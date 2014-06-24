#include "ToneMap.h"
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.01
#define DEF_SIGMA_B 1
#define DEF_MU_A 1
#define DEF_MU_B 1
#define DEF_MU_D 1
#define DEF_SIGMA_D 1

ToneMap::ToneMap(int numTones)
{
	num_tones_ = numTones;
	for (int i = 0; i < num_tones_ - 1; i++)
	{
		tonemap_.push_back(getTone(i));
	    if (i > 0)
	    {
	      tonemap_[i] += tonemap_[i-1];
	    }
	}
	setP1Params(DEF_SIGMA_B);
	setP2Params(DEF_MU_A, DEF_MU_B);
	setP3Params(DEF_SIGMA_D, DEF_MU_D);
}

float ToneMap::getTone(int v)
{
  return getP1Tone(v) + getP2Tone(v) + getP3Tone(v);
}

int ToneMap::findTone(float prob)
{
    int maxidx = num_tones_;
    int minidx = 0;
    int pivot = -1;
    while(true)
    {
    	pivot = (maxidx - minidx) / 2 + minidx;
    	if (maxidx == minidx)
    	{
    		return minidx;
    	}
    	float mapval = tonemap_[pivot];
    	if (abs(prob - mapval) < EPSILON)
    	{
    		return pivot;
    	}
    	if (prob < mapval)
    	{
    		maxidx = pivot - 1;
    	}
    	else if (prob > mapval)
    	{
    		minidx = pivot + 1;
    	}
	}
    return pivot;
}

void ToneMap::setP1Params(float sigmaB)
{
	param_sigma_b_ = sigmaB;
}

void ToneMap::setP2Params(float muA, float muB)
{
	param_mu_a_ = muA;
	param_mu_b_ = muB;
}

void ToneMap::setP3Params(float sigmaD, float muD)
{
	param_mu_d_ = muD;
	param_sigma_d_ = sigmaD;
}

float ToneMap::getP1Tone(int v)
{
	if (v <= 1)
	{
		return 1.0 / param_sigma_b_ * exp(-(1.0f-(float) v) / param_sigma_b_);
	}
	return 0.0f;
}

float ToneMap::getP2Tone(int v)
{
	if (v >= param_mu_a_ && v <= param_mu_b_)
	{
		return 1.0 / (param_mu_b_ - param_mu_a_);
	}
	return 0.0f;
}

float ToneMap::getP3Tone(int v)
{
	float t1 = ((float) v) - param_mu_d_;
	return 1.0/sqrt(2*M_PI*param_sigma_d_)*exp(-(t1*t1)/(2*param_sigma_d_*param_sigma_d_));
}
