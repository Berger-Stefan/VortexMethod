// Sample solution for homework 3


#include <chrono>
#include <cmath>
#include <iostream>

#include <armadillo>

#define KERNEL_NUM 2

// Kernel definitions:
double gaussian(const arma::vec& x, const arma::vec& y, double sigma)
{
	double r = arma::norm(x - y);
	r /= sigma;
	return std::exp(- r*r );
}

// double matern(const arma::vec& x, const arma::vec& y, double sigma, double s = 2)
// {
// 	double r = arma::norm(x - y);
// 	r /= sigma;
// 	double nu = s - 1;

// 	if(r < 1e-15)
// 	{
// 		r = 1e-15;
// 	}

// 	return std::cyl_bessel_k(nu, r) * std::pow(r, nu) * std::pow(2,-nu) / tgamma(s);
// }

double wendland(const arma::vec& x, const arma::vec& y, double sigma)
{
	double r = arma::norm(x - y);
	r /= sigma;
	if(r > 1)
	{
		return 0;
	}
	
	return std::pow(1-r, 6) * (35*r*r + 18*r + 3);
}


// Mathematical function definitions:

double u0(const arma::vec& x)
{
	double abs_x = arma::norm(x,2);

	if(abs_x < 1)
	{
		return std::exp(-1.0/(1-abs_x*abs_x));
	}

	return 0;
}

arma::vec velocity_field(const arma::vec& x)
{
	return arma::vec(std::vector<double>{-x(1), x(0)});
}

double u_t(const arma::vec& x, const arma::mat& points, const arma::vec& coeffs, double sigma)
{
	double u = 0;
	size_t N = points.n_cols;
	
	#pragma omp parallel for reduction(+:u)
	for(size_t i = 0; i < N; i++)
	{
		#if KERNEL_NUM==0
		u += coeffs(i) * gaussian(x, points.col(i), sigma);
		#elif KERNEL_NUM==1
		u += coeffs(i) * matern(x, points.col(i), sigma);
		#else
		u += coeffs(i) * wendland(x, points.col(i), sigma);
		#endif
	}

	return u;
}

int main()
{
	// First set the simulation parameters:
	size_t n_1d = 10;
	double h = 2.0 / n_1d;
	double sigma = 0.5;
//	double sigma = h/4;
	double mu = 0;
	double T = 10;
	size_t NT = 40;
	double dt = T / NT;
	
	// Initialize points:	
	size_t N = n_1d*n_1d;
	arma::mat points(2, N);
	arma::vec values(N);
	for(size_t i = 0; i < n_1d; i++)
	{
		for(size_t j = 0; j < n_1d; j++)
		{
			size_t curr = i + j * n_1d;
			points(0, curr) = -1 + i * h;
			points(1, curr) = -1 + j * h;
			values(curr) = u0(points.col(curr));
		}
	}

	// Plot parameters:
	double border_plot = 1.1;
	size_t Nx_plot = 100;
	double hx_plot = 2.0 * border_plot / Nx_plot;
	size_t Ny_plot = 100;
	double hy_plot = 2.0 * border_plot / Ny_plot;
	size_t Nt_plot = 10;

	// Plot initial data:
	std::ofstream data_0("u0.txt");
	for(size_t i = 0; i < Nx_plot; i++)
	{
		for(size_t j = 0; j < Ny_plot; j++)
		{
			double x = -border_plot + i * hx_plot;
			double y = -border_plot + j * hy_plot;
			double u = u0(arma::vec(std::vector<double>{x,y}));
			data_0 << x << " " << y << " " << u << std::endl;
		}
		data_0 << std::endl;
	}
	std::cout << "Started." << std::endl;

	arma::mat sys_mat(N, N, arma::fill::zeros);
	arma::mat Q(N, N, arma::fill::zeros);
	arma::mat R(N, N, arma::fill::zeros);
	arma::vec coeffs(N, arma::fill::zeros);

	// Run time-loop:
	for(size_t nt = 1; nt <= NT; nt++)
	{
		auto start = std::chrono::steady_clock::now();
		// Advance the particles along their trajectories:
		#pragma omp parallel for
		for(size_t i = 0; i < N; i++)
		{
			// Use classical 4-Runge-Kutta for time-integration
			arma::vec x = points.col(i);
			arma::vec k1 = velocity_field(x);
			arma::vec k2 = velocity_field(x + 0.5*h*k1);
			arma::vec k3 = velocity_field(x + 0.5*h*k2);
			arma::vec k4 = velocity_field(x + h*k3);
			
			points.col(i) = x + h * (k1 + 2*k2 + 2*k3 + k4) / 6.0;
		}
			
		// Interpolate:
		// First compute system matrix:
		#pragma omp parallel for
		for(size_t i = 0; i < N; i++ )
		{
			for(size_t j = 0; j < i; j++)
			{
				#if KERNEL_NUM==0
				sys_mat(i, j) = gaussian(points.col(i), points.col(j), sigma);
				#elif KERNEL_NUM==1
				sys_mat(i, j) = matern(points.col(i), points.col(j), sigma);
				#else
				sys_mat(i, j) = wendland(points.col(i), points.col(j), sigma);
				#endif
			}
			#if KERNEL_NUM==0
			sys_mat(i,i) = gaussian(points.col(i), points.col(i), sigma) + mu;
			#elif KERNEL_NUM==1
			sys_mat(i,i) = matern(points.col(i), points.col(i), sigma) + mu;
			#else
			sys_mat(i,i) = wendland(points.col(i), points.col(i), sigma) + mu;
			#endif
			for(size_t j = i+1; j < N; j++)
			{
				sys_mat(i, j) = sys_mat(j, i);
			}
		}
		std::cout << "Condition system matrix:\t" << arma::cond(sys_mat) << std::endl;
		
		// Solve linear system:
//		arma::qr(Q,R,sys_mat);
//		coeffs = arma::solve(R, Q.t()*values);
		coeffs = arma::solve(sys_mat, values);

		// Plot u_t:
		if(nt % Nt_plot == 0)
		{
			double t = nt * dt;
			std::ofstream data_t("u" + std::to_string(t) + ".txt");
			for(size_t i = 0; i < Nx_plot; i++)
			{
				for(size_t j = 0; j < Ny_plot; j++)
				{
					double x = -border_plot + i * hx_plot;
					double y = -border_plot + j * hy_plot;
					double u = u_t(arma::vec(std::vector<double>{x,y}), points, coeffs, sigma);
					data_t << x << " " << y << " " << u << std::endl;
				}
				data_t << std::endl;
			}
			/*
			std::ofstream points_t("points" + std::to_string(t) + ".txt");
			for(size_t i = 0; i < points.n_cols; i++)
			{
				points_t << points(0, i) << " " << points(1,i) << std::endl;
			}
			*/
		}
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed_seconds = end-start;
		std::cout << "t = " << nt * dt << " :  comp time = " << elapsed_seconds.count() << " s. "
			  << "Error in linear solve: " << arma::norm(sys_mat*coeffs - values) << std::endl;
	}	
	std::cout << "Ended." << std::endl;
	return 0;
}