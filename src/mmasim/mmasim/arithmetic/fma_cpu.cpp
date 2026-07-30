#include <cmath>

extern "C" void fma_f32(const float *__restrict a, const float *__restrict b,
                        const float *__restrict c, float *__restrict o, int n)
{
#pragma omp parallel for if (n > 1024)
    for (int i = 0; i < n; ++i)
        o[i] = std::fma(a[i], b[i], c[i]);
}

extern "C" void fma_f64(const double *__restrict a, const double *__restrict b,
                        const double *__restrict c, double *__restrict o, int n)
{
#pragma omp parallel for if (n > 1024)
    for (int i = 0; i < n; ++i)
        o[i] = std::fma(a[i], b[i], c[i]);
}
