/*
 * EquationSolver.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: braunra
 */

#include "EquationSolver.h"
#include <cusp/krylov/cg.h>
#include <cusp/dia_matrix.h>
#include <cusp/transpose.h>
#include <cusp/elementwise.h>
#include <cusp/blas.h>
#include <cusp/linear_operator.h>
#include <vector>

#define MATRIX_ON_CPU

EquationSolver::EquationSolver(float *log_texture, float *log_tonemap,
		int width, int height,
		float lambda) {
	log_texture_ = log_texture;
	log_tonemap_ = log_tonemap;
	lambda_ = lambda;
	width_ = width;
	height_ = height;
	dimension_ = width_ * height_;
	result_ = new float[dimension_];
}

EquationSolver::~EquationSolver() {
	delete[] result_;
}

void EquationSolver::Run() {

#ifdef MATRIX_ON_CPU // faster! yields a total speedup of 2 for the entire calculation
	// allocate storage for matrix A (dimension x dimension)
  // with dimension nonzero values in three diagonals
	cusp::dia_matrix<int,float,cusp::host_memory> A_cpu(dimension_, dimension_, dimension_, 3);

	// set the offsets of the diagonals
	A_cpu.diagonal_offsets[0] = -2;
	A_cpu.diagonal_offsets[1] = 0;
	A_cpu.diagonal_offsets[2] = 2;

  CreateMatrix_A(&A_cpu);
  // Copy to GPU
	cusp::dia_matrix<int,float,cusp::device_memory> A_gpu = A_cpu;

	// allocate storage for right hand side (b = log_texture^T * log_tonemap)
	cusp::array1d<float, cusp::host_memory> b_cpu(dimension_, 0);
  CreateVector_b(&b_cpu);
  // Copy to GPU
	cusp::array1d<float, cusp::device_memory> b1_gpu = b_cpu;

	// allocate storage for result (beta)
	cusp::array1d<float, cusp::device_memory> beta_gpu(dimension_, 0);

#else
	// allocate storage for matrix A (dimension x dimension) with dimension nonzero values in one diagonal
	cusp::dia_matrix<int,float,cusp::host_memory> A_cpu(dimension_, dimension_, dimension_, 1);
	// allocate storage for the lamba matrix (dimension x dimension) with dimension nonzero values in one diagonal
	cusp::dia_matrix<int,float,cusp::host_memory> lambda_cpu(dimension_, dimension_, dimension_, 1);
	// allocate storage for matrix Delta (dimension x dimension) with 2*dimension nonzero values in two diagonals
	cusp::dia_matrix<int,float,cusp::host_memory> Delta_cpu(dimension_, dimension_, 2 * dimension_, 2);

	// set the offsets of the diagonals
	A_cpu.diagonal_offsets[0] = 0;
	lambda_cpu.diagonal_offsets[0] = 0;
	Delta_cpu.diagonal_offsets[0] = -1;
	Delta_cpu.diagonal_offsets[1] = 1;

	CopyTextureToMatrix(&A_cpu);
	FillDeltaMatrixWithGradientOperation(&Delta_cpu);
	FillLambdaMatrix(&lambda_cpu);

	// copy to GPU
	cusp::dia_matrix<int,float,cusp::device_memory> A_gpu = A_cpu;
	cusp::dia_matrix<int,float,cusp::device_memory> lambda_gpu = lambda_cpu;

	cusp::dia_matrix<int,float,cusp::device_memory> A1_gpu;
	cusp::dia_matrix<int,float,cusp::device_memory> Delta_gpu = Delta_cpu;
	cusp::dia_matrix<int,float,cusp::device_memory> Delta1_gpu;
	cusp::dia_matrix<int,float,cusp::device_memory> DeltaT_gpu;

	// allocate storage for right hand side (b = log_texture^T * log_tonemap)
	cusp::array1d<float, cusp::host_memory> b_cpu(dimension_, 0);

	// copy tonemap to b_cpu
	std::vector<float> b_vec( log_tonemap_, log_tonemap_ + dimension_ ) ;
	b_cpu = b_vec;

	cusp::array1d<float, cusp::device_memory> b_gpu = b_cpu;
	cusp::array1d<float, cusp::device_memory> b1_gpu = b_cpu;

	cusp::multiply(A_gpu, b_gpu, b1_gpu);


	// allocate storage for result (beta)
	cusp::array1d<float, cusp::device_memory> beta_gpu(dimension_, 0);

	// Calculate the Matrix A = (A^T * A + lambda * Delta^T * Delta)
	cusp::multiply(A_gpu, A_gpu, A1_gpu); // A = A^T * A
	cusp::transpose(Delta_gpu, DeltaT_gpu); // Delta^T
	cusp::multiply(DeltaT_gpu, Delta_gpu, Delta1_gpu); // Delta^T * Delta
	cusp::multiply(lambda_gpu, Delta1_gpu, Delta_gpu); // lambda * Delta^T * Delta
	cusp::add(A1_gpu, Delta_gpu, A_gpu);  // (A^T * A + lambda * Delta^T * Delta)

#endif

	cusp::default_monitor<float> monitor(b1_gpu, 20, 1e-1);
	// solve equation
  //(A^T * A + lambda Delta^T * Delta) * x = b <-(log_texture^T * log_tonemap)
	cusp::krylov::cg(A_gpu,
			beta_gpu,
			b1_gpu,
			monitor);

	// copy to host memory
	cusp::array1d<float, cusp::host_memory> beta_cpu = beta_gpu;

	// extract data
	for (int i = 0; i < dimension_; i++) {
		result_[i] = beta_cpu[i];
	}
}

float *EquationSolver::GetResult() {
	return result_;
}


void EquationSolver::CopyTextureToMatrix(cusp::dia_matrix<int,float,cusp::host_memory> *A_cpu){
	for (int i = 0; i < dimension_; i++) {
		(*A_cpu).values(i, 0) = log_texture_[i];
	}
}

void EquationSolver::FillDeltaMatrixWithGradientOperation(cusp::dia_matrix<int,float,cusp::host_memory> *Delta_cpu){
	float lambda_half = lambda_ / 2;
	for (int i = 0; i < dimension_; i++) {
		(*Delta_cpu).values(i, 0) = -lambda_half;
		(*Delta_cpu).values(i, 1) = lambda_half;
	}
}

void EquationSolver::FillLambdaMatrix(cusp::dia_matrix<int,float,cusp::host_memory> *lambda) {
	for (int i = 0; i < dimension_; i++) {
		(*lambda).values(i, 0) = lambda_;
	}
}

void EquationSolver::CreateMatrix_A(cusp::dia_matrix<int,float,cusp::host_memory> *A_cpu) {
  float neg_lambda = -lambda_;
  float two_lambda = 2 * lambda_;
  // first line is lambda * (a, 0, -1, ...) with a being the first pixel in the log_texture
  (*A_cpu).values(0, 1) = lambda_ + log_texture_[0] * log_texture_[0];
  (*A_cpu).values(0, 2) = neg_lambda;
  
  // the lines in between are lambda * (..., -1, 0, 2*a[i], 0, -1, ...)
  for (int i = 1; i < dimension_ -1; i++) {
    (*A_cpu).values(i, 0) = neg_lambda;
    (*A_cpu).values(i, 1) = two_lambda + log_texture_[i] * log_texture_[i];
    (*A_cpu).values(i, 2) = neg_lambda;
  }

  // the last line is lambda * (...,0,-1,0,a) with a being the last pixel in log_texture
  (*A_cpu).values(dimension_ - 1, 0) = neg_lambda;
  (*A_cpu).values(dimension_ - 1, 1) = lambda_ +
    log_texture_[dimension_ - 1] * log_texture_[dimension_ - 1];
}

void EquationSolver::CreateVector_b(cusp::array1d<float, cusp::host_memory> *b_cpu) {
  for (int i = 0; i < dimension_; i++) {
    (*b_cpu)[i] = log_texture_[i] * log_tonemap_[i];
  }
}
