#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <random>

double Delta_t = 0.01;
double Tf = 8.0;
double Sigma = 10.0;
double Rho = 28.0;
double Beta = 8.0 / 3.0;
unsigned int N = Tf / Delta_t; // Number of time steps

arma::vec T = arma::linspace(0.0, Tf, N); // Time vector

double lorenz_deriv(double x, double y, double z, int i); // i defines which variable to return
double ode_solver(double x, double y, double z, int i);   // Runge-Kutta 4th order

int main()
{
    // use random from merssene twister engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 30.0);

    // Initialize vectors
    arma::vec x_t = arma::zeros(N);
    arma::vec y_t = arma::zeros(N);
    arma::vec z_t = arma::zeros(N);

    // Initial conditions
    double x_0 = dis(gen) - 15.0;
    double y_0 = dis(gen) - 15.0;
    double z_0 = dis(gen) - 15.0;

    // evolve the system
    for (unsigned int i = 0; i < N; i++)
    {
        x_t(i) = ode_solver(x_0, y_0, z_0, 0);
        y_t(i) = ode_solver(x_0, y_0, z_0, 1);
        z_t(i) = ode_solver(x_0, y_0, z_0, 2);

        x_0 = x_t(i);
        y_0 = y_t(i);
        z_0 = z_t(i);
    }

    // write to file
    std::ofstream ofile;
    ofile.open("lorenz.csv");
    int prec = 10;
    int width = 15;

    // header
    ofile << std::setw(width) << std::setprecision(prec) << "t";
    ofile << std::setw(width) << std::setprecision(prec) << "x";
    ofile << std::setw(width) << std::setprecision(prec) << "y";
    ofile << std::setw(width) << std::setprecision(prec) << "z" << std::endl;

    // data
    for (int i = 0; i < N; i++)
    {
        ofile << std::setw(width) << std::setprecision(prec) << T(i) << ",";
        ofile << std::setw(width) << std::setprecision(prec) << x_t(i) << ",";
        ofile << std::setw(width) << std::setprecision(prec) << y_t(i) << ",";
        ofile << std::setw(width) << std::setprecision(prec) << z_t(i) << std::endl;
    }

    return 0;
}

double lorenz_deriv(double x, double y, double z, int i)
{
    if (i == 0) // x
    {
        return Sigma * (y - x);
    }
    else if (i == 1) // y
    {
        return x * (Rho - z) - y;
    }
    else if (i == 2) // z
    {
        return x * y - Beta * z;
    }
    else
    {
        return 0.0;
    }
}

double ode_solver(double x, double y, double z, int i)
{ // Runge-Kutta 4th order
    double k1 = lorenz_deriv(x, y, z, i);
    double k2 = lorenz_deriv(x + 0.5 * Delta_t * k1, y + 0.5 * Delta_t * k1, z + 0.5 * Delta_t * k1, i);
    double k3 = lorenz_deriv(x + 0.5 * Delta_t * k2, y + 0.5 * Delta_t * k2, z + 0.5 * Delta_t * k2, i);
    double k4 = lorenz_deriv(x + Delta_t * k3, y + Delta_t * k3, z + Delta_t * k3, i);

    double evolve = Delta_t * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    if (i == 0)
        return x + evolve;
    else if (i == 1)
        return y + evolve;
    else if (i == 2)
        return z + evolve;
    else
        return 0.0;
}