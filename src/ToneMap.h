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

	float getTone(int v);
	int findTone(float prob);
	void setP1Params(float sigmaB);
	void setP2Params(float muA, float muB);
	void setP3Params(float sigmaD, float muD);

private:
	float getP1Tone(int v);
	float getP2Tone(int v);
	float getP3Tone(int v);

	int num_tones_;

	float param_sigma_b_;
	float param_mu_a_;
	float param_mu_b_;
	float param_sigma_d_;
	float param_mu_d_;

	std::vector<float> tonemap_;
};


#endif /* TONEMAP_H_ */
