#define TRIVIAL_SOLUTION

__kernel void matrix_multiplication(
  __global const float* as,
  __global const float* bs,
  __global float* cs,
  const unsigned int M,
  const unsigned int K,
  const unsigned int N)
{
  const unsigned int
    l_x = get_local_id(0),
    l_y = get_local_id(1),
    g_x = get_global_id(0),
    g_y = get_global_id(1);

#ifdef TRIVIAL_SOLUTION
  cs[g_x + g_y * N] = 0;
  for (unsigned int i = 0; i < K; ++i)
    cs[g_y * N + g_x] += as[g_y * K + i] * bs[i * N + g_x];
#else
#error not implemented yet
#endif
}