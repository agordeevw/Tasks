__kernel void sum(
    __global const unsigned int* values,
    const unsigned int n,
    __global unsigned int* result,
    __local unsigned int* local_values)
{
    const unsigned int local_id = get_local_id(0);
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_size = get_local_size(0);

    if (global_id == 0)
        *result = 0;

    local_values[local_id] = (global_id < n) ? values[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int stride = group_size / 2; stride > 0; stride /= 2) {
        if (local_id < stride)
            local_values[local_id] += local_values[local_id + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0)
        atomic_add(result, local_values[0]);
}