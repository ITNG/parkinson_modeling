#include "lib.hpp"

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
    double wSG = 4.87;
    double wGS = 1.33;
    double wCS = 9.98;
    double wSC = 8.93;
    double wGG = 0.53;
    double wCC = 6.17;

    double C = 172.18 / 1000.0;
    double Str = 8.46 / 1000.0;

    double Be = 17.85 / 1000.0;
    double Bi = 9.87 / 1000.0;

    double Me = 75.77 / 1000.0;
    double Mi = 205.72 / 1000.0;

    double T_CC = 4.65;

    double tau_E = 11.59;
    double tau_I = 13.02;

    dim1 t_ar;
    dim2 y;
    int seed;
    double maxdelay;
    double t_sim;
    int nstart;
    long unsigned num_iteration;
    vector<long unsigned> plag;

public:
    virtual ~DDE() {}
    //--------------------------------------------------------//
    void set_params(
        const double t_sim,
        const double dt,
        const int nstart);
    //--------------------------------------------------------//
    double FS(const double x);
    double FG(const double x);
    double FE(const double x);
    double FI(const double x);
    //--------------------------------------------------------//
    void set_history(const dim1 &);
    void euler_integrator();

    void dde_sys(dim1 &,
                 const double,
                 const long unsigned,
                 vector<long unsigned> &);

    double hermite_x(double t, double tn, double Xn, double Vn,
                     double tnp1, double Xnp1, double Vnp1);

    double interp_y(const double,
                    const size_t,
                    long unsigned &);

    dim1 get_times()
    {
        return t_ar;
    }
    dim2 get_coordinates()
    {
        return y;
    }
};
