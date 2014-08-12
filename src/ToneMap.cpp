#include "ToneMap.h"
#include <cmath>

#define DEF_SIGMA_B 9.0f
#define DEF_MU_A 105.0f
#define DEF_MU_B 225.0f
#define DEF_MU_D 90.0f
#define DEF_SIGMA_D 11.0f
#define DEF_OMEGA_1 11.0f
#define DEF_OMEGA_2 37.0f
#define DEF_OMEGA_3 52.0f

ToneMap::ToneMap(int numTones)
{
	num_tones_ = numTones;
	setP1Params(DEF_SIGMA_B);
	setP2Params(DEF_MU_A, DEF_MU_B);
	setP3Params(DEF_SIGMA_D, DEF_MU_D);
	setPWeigths(DEF_OMEGA_1, DEF_OMEGA_2, DEF_OMEGA_3);
	initMap();
}

void ToneMap::initMap()
{
	float sum = 0;
	for (int i = 0; i < num_tones_; i++)
	{
		float val = getProbability(i);
		tonemap_.push_back(val);
	    sum += tonemap_[i];
	    if (i > 0)
	    {
	      tonemap_[i] += tonemap_[i-1];
	    }
	}
	float normalization = 1.0 / sum;
	for (int i = 0; i < num_tones_; i++)
	{
		tonemap_[i] = normalization * tonemap_[i];
	}
}

float ToneMap::getProbability(int v)
{
    return (param_omegas_[0] * getP1(v) + param_omegas_[1] * getP2(v) + param_omegas_[2] * getP3(v));
}

const std::vector<float> &ToneMap::getTonemap() const
{
	return tonemap_;
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

void ToneMap::setPWeigths(int omega1, int omega2, int omega3)
{
	param_omegas_[0] = omega1;
	param_omegas_[1] = omega2;
	param_omegas_[2] = omega3;
}

float ToneMap::getP1(int v)
{
	if (v <= 255)
	{
		return 1.0 / param_sigma_b_ * expf(-(255.0f-(float) v) / param_sigma_b_);
	}
	return 0.0f;
}

float ToneMap::getP2(int v)
{
	if (v >= param_mu_a_ && v <= param_mu_b_)
	{
		return 1.0 / (param_mu_b_ - param_mu_a_);
	}
	return 0.0f;
}

float ToneMap::getP3(int v)
{
	float t1 = ((float) v) - param_mu_d_;
	return 1.0/sqrt(2*M_PI*param_sigma_d_)*exp(-(t1*t1)/(2*param_sigma_d_*param_sigma_d_));
}
