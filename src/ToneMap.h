/*
 * ToneMap.h
 *
 *  Created on: 24.06.2014
 *      Author: stefan
 */
#include <vector>

#ifndef TONEMAP_H_
#define TONEMAP_H_

#define DEF_NUM_TONES 256

class ToneMap
{

public:
	ToneMap(int numTones = DEF_NUM_TONES);

	void setP1Params(float sigmaB);
	void setP2Params(float muA, float muB);
	void setP3Params(float sigmaD, float muD);
	void setPWeigths(int omega1, int omega2, int omega3);
	const std::vector<float> &getTonemap() const;

private:
	float getP1(int v);
	float getP2(int v);
	float getP3(int v);
	float getProbability(int v);
	void initMap();

	int num_tones_;

	float param_sigma_b_;
	float param_mu_a_;
	float param_mu_b_;
	float param_sigma_d_;
	float param_mu_d_;
	float param_omegas_[3];

	std::vector<float> tonemap_;
};


#endif /* TONEMAP_H_ */
