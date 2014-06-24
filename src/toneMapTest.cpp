#include "ToneMap.h"
#include <stdio.h>

int main(int argc, char* argv[]) {
	ToneMap tm;
	for (int i = 0; i < DEF_NUM_TONES; i++)
	{
		float tone = tm.getTone(i);
		int res = tm.findTone(tone);
		if (i != res)
		{
			fprintf(stderr, "Tone is wrong. Orig: %u, Tone: %f, Res: %u", i, tone, res);
			return -1;
		}
	}
	printf("Everything fine");
	return 0;
}
