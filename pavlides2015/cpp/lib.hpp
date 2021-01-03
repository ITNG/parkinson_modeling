#ifndef LIB_HPP
#define LIB_HPP

#include <time.h>
#include <random>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <valarray>
#include <string>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <assert.h>

#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#define SQUARE(X) ((X) * (X))

#define RANDOM gsl_rng_uniform(gsl_rng_r)
#define RANDOM_INT(A) gsl_rng_uniform_int(gsl_rng_r, A)
#define RANDOM_GAUSS(S) gsl_ran_gaussian(gsl_rng_r, S)
#define RANDOM_POISSON(M) gsl_ran_poisson(gsl_rng_r, M)
#define INITIALIZE_RANDOM_CLOCK(seed)              \
    {                                              \
        gsl_rng_env_setup();                       \
        if (!getenv("GSL_RNG_SEED"))               \
            gsl_rng_default_seed = time(0) + seed; \
        gsl_rng_T = gsl_rng_default;               \
        gsl_rng_r = gsl_rng_alloc(gsl_rng_T);      \
    }
#define INITIALIZE_RANDOM_F(seed)             \
    {                                         \
        gsl_rng_env_setup();                  \
        if (!getenv("GSL_RNG_SEED"))          \
            gsl_rng_default_seed = seed;      \
        gsl_rng_T = gsl_rng_default;          \
        gsl_rng_r = gsl_rng_alloc(gsl_rng_T); \
    }
#define FREE_RANDOM gsl_rng_free(gsl_rng_r)

static const gsl_rng_type *gsl_rng_T;
static gsl_rng *gsl_rng_r;

using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;

extern unsigned seed;

typedef std::vector<double> dim1;
typedef std::vector<std::vector<double>> dim2;
typedef std::vector<std::vector<long unsigned int>> dim2I;

bool fileExists(const std::string &filename);
std::vector<double> arange(const double start, const double end, const double step);
double get_wall_time();
double get_cpu_time();
void display_timing(double wtime, double cptime);

template <class T>
double average(const vector<T> &v, const int idx = 0)
{ /*average the vector from element "id" to end of the vector */
    assert(v.size() > idx);
    double result = accumulate(v.begin() + idx, v.end(), 0.0) / (v.size() - idx);
    return result;
}

double lerp(double v0, double v1, double t);
#endif
