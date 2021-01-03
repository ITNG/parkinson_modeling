#include "lib.hpp"
#include "dde_solver.hpp"

int main(int argc, char const *argv[])
{
    double wtime = get_wall_time();
    constexpr int N = 4;
    constexpr double dt = 0.01;
    dim1 y0{0., 0.0, 0.0, 0.0};
    constexpr int nstart = 50;
    constexpr double t_sim = 200.0;

    DDE dde;
    dde.set_params(t_sim, dt, nstart);
    dde.set_history(y0);
    dde.euler_integrator();

    auto t = dde.get_times();
    auto y = dde.get_coordinates();

    FILE *oy = fopen("data/euler.txt", "w");

    for (long unsigned i = 0; i < y[0].size(); i++)
    {
        fprintf(oy, "%18.9f", t[i]);
        for (size_t j = 0; j < y.size(); j++)
            fprintf(oy, "%18.9f", y[j][i]);
        fprintf(oy, "\n");
    }

    fclose(oy);
    wtime = get_wall_time() - wtime;
    display_timing(wtime, 0.0);
}
