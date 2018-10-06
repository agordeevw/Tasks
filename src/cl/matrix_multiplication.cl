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
  float sum = 0.0f;
  for (unsigned int i = 0; i < K; ++i)
    sum += as[g_y * K + i] * bs[i * N + g_x];
  cs[g_y * N + g_x] = sum;
#else
  __local float tile_a[TILE_SIZE][TILE_SIZE];
  __local float tile_b[TILE_SIZE][TILE_SIZE];
  __local float tmp[TILE_SIZE];

  float sum = 0.0f;
  for (int tile_k = 0; tile_k * TILE_SIZE < K; ++tile_k) {
    tile_a[l_y][l_x] = as[g_y * K + (tile_k * TILE_SIZE + l_x)];
    tile_b[l_y][l_x] = bs[(tile_k * TILE_SIZE + l_y) * N + g_x];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TILE_SIZE; ++k) {
      sum += tile_a[l_y][k] * tile_b[k][l_x];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  cs[g_y * N + g_x] = sum;
#endif
}