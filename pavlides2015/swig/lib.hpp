#ifndef LIB_HPP
#define LIB_HPP

#include "omp.h"
#include <cmath>
#include <chrono>
#include <random>
#include <string>
#include <vector>
#include <time.h>
#include <iomanip>
#include <fstream>
#include <fstream>
#include <complex>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <sys/time.h>
#include <sys/stat.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define SQUARE(X) ((X) * (X))

typedef std::vector<double> dim1;
typedef std::vector<float> dim1f;
typedef std::vector<std::vector<int>> dim2I;
typedef std::vector<std::vector<double>> dim2;
typedef std::vector<std::vector<float>> dim2f;

class DDE
{
private:
    int N = 4;
    double dt;
    double T_SG = 6.0;
    double T_GS = 6.0;
    double T_GG = 4.0;
    double T_CS = 5.5;
    double T_SC = 21.5;
    double tau_S = 12.8; //ms
    double tau_G = 20.0;
    double Ms = 0.3; // spk / s
    double Mg = 0.4;
    double Bs = 0.01;
    double Bg = 0.02;

    // parameters ---------------------
    // double wSG = 4.87;
    // double wGS = 1.33;
    // double wCS = 9.98;
    // double wSC = 8.93;
    // double wGG = 0.53;
    // double wCC = 6.17;
    double wSG;
    double wGS;
    double wCS;
    double wSC;
    double wGG;
    double wCC;

    // double C = 172.18 / 1000.0;
    // double Str = 8.46 / 1000.0;
    
    double I_C;
    double I_Str;

    double Be = 17.85 / 1000.0;
    double Bi = 9.87 / 1000.0;

    double Me = 75.77 / 1000.0;
    double Mi = 205.72 / 1000.0;

    double T_CC = 4.65;

    double tau_E = 11.59;
    double tau_I = 13.02;
    // --------------------------------
    double maxdelay;
    int nstart = 50;
    bool PARAMETERs_SET = false;
    bool HISTORY_SET = false;
    
    std::vector<long unsigned> plag;

public:
    dim2 y;
    dim2 yp;
    dim1 t_ar;
    double t_simulation;
    size_t num_iteration;

    virtual ~DDE() {}
    void set_params(
        const double t_simulation,
        const double dt,
        const double I_C,
        const double I_Str,
        double wSG,
        double wGS,
        double wCS,
        double wSC,
        double wGG,
        double wCC
        // const int nstart
        // const double maxdelay,
    );

    void set_history(const dim1 &hist);
    void euler_integrator();

    double FS(const double x);
    double FG(const double x);
    double FE(const double x);
    double FI(const double x);
    void sys_eqs(dim1 &dxdt,
                 const double t,
                 long unsigned n,
                 std::vector<long unsigned> &plag);
    double interp_x(double t, size_t i, long unsigned &n0);
    double hermite_x(double t, double tn, double Xn, double Vn,
                     double tnp1, double Xnp1, double Vnp1);
};

#endif // LIB_HPP