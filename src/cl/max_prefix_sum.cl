__kernel void zero_buffer(
    __global int* buffer
)
{
    buffer[get_global_id(0)] = 0;
}

// --------------------------------------------------------------

__kernel void scan(
    __global const int* input,
    __global int* output,
    __global int* sums,
    __local int* locals)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int group_size = get_local_size(0);
    const unsigned int global_id = get_global_id(0);

    locals[local_id] = input[global_id];

    int offset = 1;

    for (int level = group_size / 2; level > 0; level /= 2) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < level) {
        int dst_i = (2 * local_id + 2) * offset - 1;
        int src_i = (2 * local_id + 1) * offset - 1;
        locals[dst_i] += locals[src_i];
      }
      offset *= 2;
    }

    if (local_id == 0) {
      if (sums)
        sums[get_group_id(0)] = locals[group_size - 1];
      locals[group_size - 1] = 0;
    }

    for (int level = 1; level < group_size; level *= 2) {
      offset /= 2;
      barrier(CLK_LOCAL_MEM_FENCE);
      if (local_id < level) {
        int dst_i = (2 * local_id + 2) * offset - 1;
        int src_i = (2 * local_id + 1) * offset - 1;
        int t = locals[src_i];
        locals[src_i] = locals[dst_i];
        locals[dst_i] += t;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id + 1 < group_size)
      output[global_id] = locals[local_id + 1];
    else
      output[global_id] = locals[local_id] + input[global_id];
}

// --------------------------------------------------------------

__kernel void add(
    __global int* scanned_input,
    __global const int* sums
)
{
    if (get_group_id(0) != 0)
      scanned_input[get_global_id(0)] += sums[get_group_id(0) - 1];
}

// --------------------------------------------------------------

__kernel void get_max_value(
    __global const int* data,
    __global int* ret,
    __local int* idxs
)
{
    const unsigned int local_id = get_local_id(0);

    idxs[local_id] = get_global_id(0);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int stride = get_local_size(0) / 2; stride > 0; stride /= 2) {
        if (local_id < stride) {
            if (data[idxs[local_id]] < data[idxs[local_id + stride]])
                idxs[local_id] = idxs[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        atomic_max(ret, data[idxs[0]]);
}

// --------------------------------------------------------------

__kernel void get_idx_of_max(
    __global const int* data,
    __global const int* max,
    __global int* ret
) 
{
    if (data[get_global_id(0)] == max[0]) {
        atomic_min(ret, (int)get_global_id(0) + 1);
    }
}