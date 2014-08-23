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
	EquationSolver(float *log_texture, float *log_tonemap,  int width, int height, float lambda);
	void Run();
	float *GetResult();
	~EquationSolver();

private:
	void CopyTextureToMatrix(cusp::dia_matrix<int,float,cusp::host_memory> *A_cpu);
	void FillDeltaMatrixWithGradientOperation(cusp::dia_matrix<int,float,cusp::host_memory> *Delta_cpu);
	void FillLambdaMatrix(cusp::dia_matrix<int,float,cusp::host_memory> *lambda);

  // Create the Matrix on the left side of the Tikhonov-regularization normal equation
  // (A^T * A + lambda * Delta^T * Delta)
  // With A being the log_texture and Delta being the Gradient Operator Matrix
  void CreateMatrix_A(cusp::dia_matrix<int,float,cusp::host_memory> *A_cpu);

	// Create Vector for the right side of the Tikhonov-regularization normal equation
  // b = log_texture^T * log_tonemap = (A^T * b)
  void CreateVector_b(cusp::array1d<float, cusp::host_memory> *b_cpu);

	float *log_texture_;
	float *log_tonemap_;
	float *result_;
	float lambda_;
	int width_, height_, dimension_;
};

#endif /* EQUATIONSOLVER_H_ */
