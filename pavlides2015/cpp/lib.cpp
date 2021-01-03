#include "lib.hpp"

/*------------------------------------------------------------*/
bool fileExists(const std::string &filename)
{
    struct stat buf;
    if (stat(filename.c_str(), &buf) != -1)
    {
        return true;
    }
    return false;
}
/*------------------------------------------------------------*/
std::vector<double> arange(const double start, const double end,
                           const double step)
{
    int nstep = round((end - start) / step);
    std::vector<double> arr(nstep);

    for (int i = 0; i < nstep; i++)
        arr[i] = start + i * step;
    return arr;
}
//-----------------------------------------------------------//
double get_wall_time()
{
    /*measure real passed time */
    struct timeval time;
    if (gettimeofday(&time, NULL))
    {
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
//------------------------------------------------------------//
double get_cpu_time()
{
    /*measure cpu passed time*/
    return (double)clock() / CLOCKS_PER_SEC;
}
//------------------------------------------------------------//
void display_timing(double wtime, double cptime)
{
    int wh;      //, ch;
    int wmin;    //, cpmin;
    double wsec; //, csec;
    wh = (int)wtime / 3600;
    // ch = (int)cptime / 3600;
    wmin = ((int)wtime % 3600) / 60;
    // cpmin = ((int)cptime % 3600) / 60;
    wsec = wtime - (3600. * wh + 60. * wmin);
    // csec = cptime - (3600. * ch + 60. * cpmin);
    printf("Wall Time : %d hours and %d minutes and %.4f seconds.\n", wh, wmin, wsec);
    // printf ("CPU  Time : %d hours and %d minutes and %.4f seconds.\n",ch,cpmin,csec);
}

double interpolate(const std::vector<double> &x,
                   const std::vector<double> &y,
                   const double xnew, int &n0,
                   const std::string kind)
{
    int len = x.size();
    if (xnew > x[len - 1] || xnew < x[0])
    {
        std::cerr << "warning : out of interval [X_min, X_max]"
                  << "\n";
        std::cout << x[0] << " < " << xnew << " < " << x[len - 1] << "\n";
    }
    /* find left end of interval for interpolation */
    while (x[n0 + 1] < xnew)
        n0++;
    while (x[n0] > xnew)
        n0--;

    /* linear interpolation*/
    std::vector<double> xp(2);
    std::vector<double> yp(2);

    for (int i = 0; i < 2; i++)
    {
        xp[i] = x[n0 + i];
        yp[i] = y[n0 + i];
    }
    double dydx = (yp[1] - yp[0]) / (xp[1] - xp[0]);
    return (yp[0] + dydx * (xnew - xp[0]));
}

double lerp(double v0, double v1, double t)
{
    return (1 - t) * v0 + t * v1;
}