/*
 * EquationSolver.cpp
 *
 *  Created on: Aug 12, 2014
 *      Author: braunra
 */

#include "EquationSolver.h"
#include <cusp/krylov/cg.h>
#include <cusp/dia_matrix.h>
#include <vector>

EquationSolver::EquationSolver(float *log_texture, float *log_tonemap, int width, int height) {
	log_texture_ = log_texture;
	log_tonemap_ = log_tonemap;
	width_ = width;
	height_ = height;
	dimension_ = width_ * height_;
	result_ = new float[dimension_];
}

EquationSolver::~EquationSolver() {
	delete[] result_;
}

void EquationSolver::Run() {
	// allocate storage for matrix A (dimension x dimension) with dimension nonzero values in one diagonal
	cusp::dia_matrix<int,float,cusp::host_memory> A_cpu(dimension_, dimension_, dimension_, 1);
	// allocate storage for matrix Delta (dimension x dimension) with 2*dimension nonzero values in two diagonals
	cusp::dia_matrix<int,float,cusp::host_memory> Delta_cpu(dimension_, dimension_, 2 * dimension_, 2);

	// set the offsets of the diagonals
	A_cpu.diagonal_offsets[0] = 0;
	Delta_cpu.diagonal_offsets[0] = -1;
	Delta_cpu.diagonal_offsets[1] = 1;

	CopyTextureToMatrix(&A_cpu);
	FillDeltaMatrixWithGradientOperation(&Delta_cpu);

	// copy to GPU
	cusp::dia_matrix<int,float,cusp::device_memory> A_gpu = A_cpu;
	cusp::dia_matrix<int,float,cusp::device_memory> Delta_gpu;

	// allocate storage for right hand side (b = log_texture^T * log_tonemap)
	cusp::array1d<float, cusp::host_memory> b_cpu(dimension_, 0);

	// copy tonemap to b_cpu
	std::vector<float> b_vec( log_tonemap_, log_tonemap_ + dimension_ ) ;
	b_cpu = b_vec;

	cusp::array1d<float, cusp::device_memory> b_gpu = b_cpu;

	cusp::multiply(A_gpu, b_gpu, b_gpu);


	// allocate storage for result (beta)
	cusp::array1d<float, cusp::device_memory> beta_gpu(dimension_, 0);

	// solve equation
	cusp::krylov::cg(A_gpu,
			beta_gpu,
			b_gpu);

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
	for (int i = 0; i < dimension_; i++) {
		(*Delta_cpu).values(i, 0) = -1;
		(*Delta_cpu).values(i, 1) = 1;
	}

}
