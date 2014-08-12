#include "ToneMap.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
	ToneMap tm;
	int errors = 0;
	float sum = 0;
	const std::vector<float> &tonemap = tm.getTonemap();
	for (int i = 0; i < DEF_NUM_TONES; i++)
	{

		float prob = tm.getProbability(i);
		int res = tm.findTone(tonemap[i]);
		sum += prob;
		fprintf(stderr, "TonemapVal: %f, Value: %u, Prob: %f, Resolved: %u\n", tonemap[i], i, prob, res);
	}
	printf("Found %u errors. Sum is %f\n", errors, sum);
	float find = 0.792300;
	int expected = 108;
	int res = tm.findTone(find);
	printf("Trying to find %f (which is the value for %u), but it's %u \n", find, expected, res);
	return 0;
}
