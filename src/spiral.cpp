
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <random>
#include <iomanip>

double spiral_deriv(double x, double y, int i);          // i defines which variable to return
double ode_solver(double x, double y, int i, double dt); // Runge-Kutta 4th order

// global parameters
double a = 0.2;
double b = 1.0;
double c = 1.0;
double d = 0.2;

int main(int argc, char *argv[])
{
    // sysargs
    int n_ptc = atoi(argv[1]);
    double tf = atof(argv[2]);
    double dt = atof(argv[3]);

    unsigned int N = tf / dt;                 // Number of time steps
    arma::vec T = arma::linspace(0.0, tf, N); // Time vector

    // use random from merssene twister engine
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 30.0);

    // X_t, Y_t, Z_t will hold n_ptc time series
    arma::mat X_t(N, n_ptc);
    arma::mat Y_t(N, n_ptc);

    // vector to hold the 100 initial conditions
    arma::vec X_0(n_ptc);
    arma::vec Y_0(n_ptc);

    // Fill Initial conditions
    for (int i = 0; i < n_ptc; i++)
    {
        X_0(i) = dis(gen) - 15.0;
        Y_0(i) = dis(gen) - 15.0;
    }

    // evolve the system
    double X_0_ptc;
    double Y_0_ptc;
    for (unsigned int step = 0; step < N; step++)
    {
        for (int ptc = 0; ptc < n_ptc; ptc++) // loop over particles = different trajectories
        {
            // there is probably a better way to do the following
            X_0_ptc = X_0(ptc);
            Y_0_ptc = Y_0(ptc);

            X_t(step, ptc) = ode_solver(X_0_ptc, Y_0_ptc, 0, dt);
            Y_t(step, ptc) = ode_solver(X_0_ptc, Y_0_ptc, 1, dt);

            X_0(ptc) = X_t(step, ptc);
            Y_0(ptc) = Y_t(step, ptc);
        }
    }

    // write to file
    std::ofstream ofile;
    ofile.open("spiral.csv");
    int prec = 10;
    int width = 15;

    // header
    ofile << std::setw(width) << std::setprecision(prec) << "t,";
    ofile << std::setw(width) << std::setprecision(prec) << "x,";
    ofile << std::setw(width) << std::setprecision(prec) << "y,";
    ofile << std::setw(width) << std::setprecision(prec) << "particle"
          << std::endl;

    // data
    for (int ptc = 0; ptc < n_ptc; ptc++)
    {
        for (int step = 0; step < N; step++)
        {
            ofile << std::setw(width) << std::setprecision(prec) << T(step) << ",";
            ofile << std::setw(width) << std::setprecision(prec) << X_t(step, ptc) << ",";
            ofile << std::setw(width) << std::setprecision(prec) << Y_t(step, ptc) << ",";
            ofile << std::setw(width) << std::setprecision(prec) << ptc << std::endl;
        }
    }

    return 0;
}

double spiral_deriv(double x, double y, int i)
{
    if (i == 0) // x
    {
        return a * x - b * y;
    }
    else if (i == 1) // y
    {
        return c * x + d * y;
    }
    else
    {
        return 0.0;
    }
}

double ode_solver(double x, double y, int i, double dt)
{ // Runge-Kutta 4th order
    double k1 = spiral_deriv(x, y, i);
    double k2 = spiral_deriv(x + 0.5 * dt * k1, y + 0.5 * dt * k1, i);
    double k3 = spiral_deriv(x + 0.5 * dt * k2, y + 0.5 * dt * k2, i);
    double k4 = spiral_deriv(x + dt * k3, y + dt * k3, i);

    double evolve = dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    if (i == 0)
        return x + evolve;
    else if (i == 1)
        return y + evolve;
    else
        return 0.0;
}