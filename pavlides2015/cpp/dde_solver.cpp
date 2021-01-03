#include "dde_solver.hpp"
// ------------------------------------------------------------------
void DDE::set_params(
    const double t_sim,
    const double dt,
    const int nstart)
{
    this->dt = dt;
    this->t_sim = t_sim;
    this->nstart = nstart;

    dim1 delays {T_CC, T_CS, T_GG, T_GS, T_SC, T_SG};
    
    maxdelay = *std::max_element(delays.begin(), delays.end());
    num_iteration = (int)(t_sim / dt);

    plag.resize(delays.size());
    for (int i = 0; i < delays.size(); ++i)
        plag[i]= nstart;

    t_ar.reserve(num_iteration + nstart + 1);
    y.resize(N);
    for (size_t i = 0; i < N; i++)
        y[i].reserve(num_iteration + nstart + 1);
    
}
// ------------------------------------------------------------------
void DDE::set_history(const dim1 &hist)
{
    for (int i = 0; i < (nstart + 1); i++)
        t_ar.push_back(-(nstart - i) / (double)nstart * maxdelay);

    // p_x_ar: N x nstart
    for (int i = 0; i < N; i++)
        for (int j = 0; j < (nstart + 1); j++)
            y[i].push_back(hist[i]);

}

// ------------------------------------------------------------------

void DDE::dde_sys(dim1 &dxdt,
                  const double t,
                  const long unsigned n,
                  std::vector<long unsigned> &plag)
{
    double dS = 1.0 / tau_S * (FS(wCS * interp_y(t - T_CS, 2, plag[0]) - wGS * interp_y(t-T_GS, 1, plag[1])) - y[0][n]);
    double dG = 1.0 / tau_G * (FG(wSG * interp_y(t - T_SG, 0, plag[2]) - wGG * interp_y(t - T_GG, 1, plag[3]) - Str) - y[1][n]);
    double dE = 1.0 / tau_E * (FE(-wSC * interp_y(t - T_SC, 0, plag[4]) - wCC * interp_y(t - T_CC, 3, plag[5]) + C) - y[2][n]);
    double dI = 1.0 / tau_I * (FI(wCC * interp_y(t-T_CC, 2, plag[5])) - y[3][n]);

    dxdt = {dS, dG, dE, dI};
}

// ------------------------------------------------------------------
void DDE::euler_integrator()
{
    dim1 f(N);
    for (long unsigned itr = nstart; itr < (num_iteration + nstart); itr++)
    {
        dde_sys(f, t_ar[itr], itr, plag);
        for (size_t j = 0; j < N; j++)
            y[j].push_back(dt * f[j] + y[j][itr]);

        t_ar.push_back((itr - nstart + 1) * dt);
    }
}
// ------------------------------------------------------------------

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
// ------------------------------------------------------------------
double DDE::interp_y(const double t0,
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
// ------------------------------------------------------------------