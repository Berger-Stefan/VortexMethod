Use this to compile the code:

g++ solver.cpp -o solver -O3 -fopenmp -llapack -lblas -larmadillo


Try out several settings of h and sigma. If you fix sigma and let h go to 0 
you should observe convergence, at least for the Wendland kernel. However, 
the bad condition numbers of the matrices using the Gaussian and Matern kernel
(Wendland kernel to some extend as well) make solving the system relatively hard.
Thus in practice you will not converge to 0, as floating point errors enter the
equation beforehands. 
Still results should be much better compared to the blob-based-ansatz even with 
very low resolution.