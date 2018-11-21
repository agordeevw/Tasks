// http://www.heterogeneouscompute.org/wordpress/wp-content/uploads/2011/06/RadixSort.pdf

#define BITS    4
#define BITS2   16
#define WG_SIZE 128

__kernel void count(
    __global const uint * input,
    const uint input_size,
    __global uint * global_counts,
    __global uint * global_offsets,
    __local uint * local_counts,
    const uint low_bit
)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    if (lid < BITS2) {
        local_counts[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < input_size) {
        const uint value = input[gid];
        uint bucket = (value >> low_bit) & ((1 << BITS) - 1);
        atomic_inc(&local_counts[bucket]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < BITS2) {
        global_counts[BITS2 * get_group_id(0) + lid] = local_counts[lid];
        if (get_group_id(0) == get_num_groups(0) - 1) {
            global_offsets[lid] = local_counts[lid];
        }
    }
}

__kernel void scan(
    __global const uint * block_offsets,
    __global uint * global_offsets,
    const uint block_count
)
{
    __global const uint * last_block_offsets =
        block_offsets + BITS2 * (block_count - 1);

    uint sum = 0;
    for (uint i = 0; i < BITS2; ++i) {
        uint x = global_offsets[i] + last_block_offsets[i];
        global_offsets[i] = sum;
        sum += x;
    }
}

__kernel void scatter(
    __global const uint * input,
    const uint input_size,
    const uint low_bit,
    __global const uint * counts,
    __global const uint * global_offsets,
    __global uint * output
)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    uint value;
    uint bucket;
    __local uint local_input[WG_SIZE];
    if (gid < input_size) {
        value = input[gid];
        bucket = (value >> low_bit) & ((1 << BITS) - 1);
        local_input[lid] = bucket;
    }

    __local uint local_counts[1 << BITS];
    if (lid < BITS2) {
        local_counts[lid] = counts[get_group_id(0) * BITS2 + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= input_size)
        return;

    uint offset = global_offsets[bucket] + local_counts[bucket];

    uint local_offset = 0;
    for (uint i = 0; i < lid; i++) {
        if (local_input[i] == bucket)
            local_offset++;
    }

    output[offset + local_offset] = value;
}

__kernel void exclusive_scan_pre(
    __global const uint16 * in,
    __global uint16 * out,
    __global uint16 * sums,
    __local uint16 * shared
)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);

    int offset = 1;
    shared[2 * lid] = in[2 * gid];
    shared[2 * lid + 1] = in[2 * gid + 1];

    for (int d = wg; d > 0; d /= 2)
    {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d)
        {
            int ai = offset * (2 * lid + 1) - 1;
            int bi = offset * (2 * lid + 2) - 1;
            shared[bi] += shared[ai];
        }
        offset *= 2;
    }

    if (lid == 0) {
        if (sums)
            sums[get_group_id(0)] = shared[2 * wg - 1];
        shared[2 * wg - 1] = 0;
    }

    for (int d = 1; d < 2 * wg; d *= 2)
    {
        offset /= 2;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < d)
        {
            int ai = offset * (2 * lid + 1) - 1;
            int bi = offset * (2 * lid + 2) - 1;
            uint16 t = shared[ai];
            shared[ai] = shared[bi];
            shared[bi] += t;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    out[2 * gid] = shared[2 * lid];
    out[2 * gid + 1] = shared[2 * lid + 1];
}

__kernel void exclusive_scan_post(
    __global const uint16 * sums,
    __global uint16 * scanned
)
{
    const uint gid = get_global_id(0);

    uint16 value = sums[get_group_id(0)];
    scanned[2 * gid] += value;
    scanned[2 * gid + 1] += value;
}
