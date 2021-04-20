#include "lib.hpp"

void DDE::set_params(
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
)
{
    this->dt = dt;
    this->t_simulation = t_simulation;

    this->I_C = I_C;
    this->I_Str = I_Str;

    this->wSG = wSG;
    this->wGS = wGS;
    this->wCS = wCS;
    this->wSC = wSC;
    this->wGG = wGG;
    this->wCC = wCC;
    // this->nstart = nstart;
    // this->maxdelay = maxdelay;

    dim1 delays = {T_CC, T_CS, T_GG, T_GS, T_SC, T_SG};
    plag.resize(delays.size(), nstart);
    /* 
     * 0 : T_CC
     * 1 : T_CS
     * 2 : T_GG
     * 3 : T_GS
     * 4 : T_SC
     * 5 : T_SG
    */

    maxdelay = *std::max_element(delays.begin(), delays.end());
    num_iteration = (int)(round(t_simulation / dt));

    t_ar.reserve(num_iteration + nstart + 1);

    y.resize(N);
    for (size_t i = 0; i < N; i++)
        y[i].reserve(num_iteration + nstart + 1);

    yp.resize(N);
    for (size_t i = 0; i < N; i++)
        yp[i].reserve(num_iteration + nstart + 1);

    PARAMETERs_SET = true;
}
// //---------------------------------------------------------------------------//
void DDE::set_history(const dim1 &hist)
{
    for (int i = 0; i < (nstart + 1); i++)
        t_ar.push_back(-(nstart - i) / (double)nstart * maxdelay);

    // p_x_ar: N x nstart
    for (int i = 0; i < N; i++)
        for (int j = 0; j < (nstart + 1); j++)
            y[i].push_back(hist[i]);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < (nstart + 1); j++)
            yp[i].push_back(0.0);

    HISTORY_SET = true;
}
// //---------------------------------------------------------------------------//
double DDE::FS(const double x)
{
    return Ms / (1.0 + (Ms - Bs) / Bs * exp(-4.0 * x / Ms));
}
double DDE::FG(const double x)
{
    return Mg / (1.0 + (Mg - Bg) / Bg * exp(-4.0 * x / Mg));
}
double DDE::FE(const double x)
{
    return Me / (1.0 + (Me - Be) / Be * exp(-4.0 * x / Me));
}
double DDE::FI(const double x)
{
    return Mi / (1.0 + (Mi - Bi) / Bi * exp(-4.0 * x / Mi));
}

void DDE::sys_eqs(dim1 &dxdt,
                  const double t,
                  size_t n,
                  std::vector<long unsigned> &plag)
{
    double dS = 1.0 / tau_S * (FS(wCS * interp_x(t - T_CS, 2, plag[1]) - wGS * interp_x(t - T_GS, 1, plag[3])) - y[0][n]);
    double dG = 1.0 / tau_G * (FG(wSG * interp_x(t - T_SG, 0, plag[5]) - wGG * interp_x(t - T_GG, 1, plag[2]) - I_Str) - y[1][n]);
    double dE = 1.0 / tau_E * (FE(-wSC * interp_x(t - T_SC, 0, plag[4]) - wCC * interp_x(t - T_CC, 3, plag[0]) + I_C) - y[2][n]);
    double dI = 1.0 / tau_I * (FI(wCC * interp_x(t - T_CC, 2, plag[0])) - y[3][n]);

    dxdt = {dS, dG, dE, dI};
}
//---------------------------------------------------------------------------//
void DDE::euler_integrator()
{
    assert(PARAMETERs_SET);
    assert(HISTORY_SET);

    dim1 f(N);
    for (size_t itr = nstart; itr < (num_iteration + nstart); itr++)
    {
        sys_eqs(f, t_ar[itr], itr, plag);
        for (size_t j = 0; j < N; j++)
        {
            y[j].push_back(dt * f[j] + y[j][itr]);
            yp[j].push_back(f[j]);
        }

        t_ar.push_back((itr - nstart + 1) * dt);
    }
}
//---------------------------------------------------------------------------//
// double DDE::interp_x(double t, size_t i, long unsigned &n0)
// {
//     // assert(n0 >= 0);
//     assert(n0 < t_ar.size());
//     assert(t >= t_ar[0]);
//     while (n0 < t_ar.size() - 1 && t_ar[n0 + 1] < t)
//         n0++;
//     while (t_ar[n0] > t)
//         n0--;
//     return hermite_x(t, t_ar[n0], y[i][n0], yp[i][n0],
//                      t_ar[n0 + 1], y[i][n0 + 1], yp[i][n0 + 1]);
// }

// double DDE::hermite_x(double t, double tn, double Xn, double Vn,
//                       double tnp1, double Xnp1, double Vnp1)
// {
//     double h = tnp1 - tn;
//     double s = (t - tn) / h;
//     double s1 = SQUARE(s - 1.0);
//     double s2 = SQUARE(s);
//     return (1.0 + 2.0 * s) * s1 * Xn + (3.0 - 2.0 * s) * s2 * Xnp1 + h * s * s1 * Vn + h * (s - 1) * s2 * Vnp1;
// }
//---------------------------------------------------------------------------//
double lerp(double v0, double v1, double t)
{
    return (1 - t) * v0 + t * v1;
}

double DDE::interp_x(const double t0,
                     const size_t index,
                     long unsigned &n0)
{
    assert(n0 < t_ar.size());
    while ((n0 < t_ar.size() - 1) && (t_ar[n0 + 1] < t0))
        n0++;
    while (t_ar[n0] > t0)
        n0--;

    double a = y[index][n0];
    double b = y[index][n0 + 1];
    double t = (t0 - t_ar[n0]) / dt;

    return lerp(a, b, t);
}
//---------------------------------------------------------------------------//
