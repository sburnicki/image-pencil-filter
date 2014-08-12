/*
 * EquationSolver.h
 *
 *  Created on: Aug 12, 2014
 *      Author: braunra
 */

#ifndef EQUATIONSOLVER_H_
#define EQUATIONSOLVER_H_

#include <cusp/dia_matrix.h>

class EquationSolver {
public:
	EquationSolver(float *log_texture, float *log_tonemap, int width, int height);
	void Run();
	float *GetResult();
	~EquationSolver();

private:
	void CopyTextureToMatrix(cusp::dia_matrix<int,float,cusp::host_memory> *A_cpu);
	void FillDeltaMatrixWithGradientOperation(cusp::dia_matrix<int,float,cusp::host_memory> *Delta_cpu);

	float *log_texture_;
	float *log_tonemap_;
	float *result_;
	int width_, height_, dimension_;
};

#endif /* EQUATIONSOLVER_H_ */
