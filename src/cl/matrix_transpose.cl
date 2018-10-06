__kernel void matrix_transpose(
  __global const float* as,
  __global float* as_t,
  const unsigned int rows,
  const unsigned int cols)
{
  const unsigned int 
    l_i = get_local_id(0),
    l_j = get_local_id(1),
    g_i = get_global_id(0),
    g_j = get_global_id(1);

#ifdef TRIVIAL_SOLUTION
  as_t[g_j + rows * g_i] = as[g_i + cols * g_j];
#else
  __local float l_as[TILE_SIZE * TILE_SIZE];

  // assuming WARP_SIZE=32 and CACHE_LINE_SIZE=128
  // in one warp:
  // g_j is constant, while g_i varies in range
  // [g_i * TILE_SIZE / TILE_SIZE, g_i * TILE_SIZE / TILE_SIZE + TILE_SIZE)
  // l_j is constant, while l_i varies in range [0, 32)

  // coalesced memory access:
  // g_i + cols * g_j requests 128 sequential bytes
  l_as[l_i + TILE_SIZE * l_j] = as[g_i + cols * g_j];
  
  barrier(CLK_LOCAL_MEM_FENCE);

  // trivial approach from lecture leads to bank conflicts
  // each thread from one warp writes to one column of the small matrix
  // to avoid that, each warp should access not a row of the small matrix,
  //  but a diagonal
  // which diagonal is chosen is defined by l_j
  // using this approach accesses to memory banks can be more uniformly distributed
  if ((l_j > 0 && l_j < TILE_SIZE / 2) 
      || (l_j == TILE_SIZE / 2 && l_i < TILE_SIZE / 2))
  {
    const unsigned int i = l_i;
    const unsigned int j = (l_i + l_j) % TILE_SIZE;
    float tmp = l_as[i + TILE_SIZE * j];
    l_as[i + TILE_SIZE * j] = l_as[j + TILE_SIZE * i];
    l_as[j + TILE_SIZE * i] = tmp;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  const unsigned int group_id_i = g_i / TILE_SIZE;
  const unsigned int group_id_j = g_j / TILE_SIZE;

  // coalesced memory access:
  // to achieve coalescence a block of output matrix must be chosen so that
  // small matrix can be flushed to the block directly, row by row
  // as established earlier, l_j is constant through the warp,
  // while l_i changes in range [0, 32), which leads to coalesced memory access
  const unsigned int out_g_i = group_id_j * TILE_SIZE + l_i;
  const unsigned int out_g_j = group_id_i * TILE_SIZE + l_j;
  as_t[out_g_i + rows * out_g_j] = l_as[l_i + TILE_SIZE * l_j];
#endif
}