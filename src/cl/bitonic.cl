inline void order_vals(float * arr, const int i, const int j, const bool reverse) {
    const bool to_swap = (arr[i] > arr[j]) ^ reverse;
    if (to_swap) {
        const float tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

__kernel void bitonic_local(
    __global float* as,
    const uint size,
    __local float* shared
)
{
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wg = get_local_size(0);
    const uint wg_idx = gid / wg;

    if (gid < size)
        shared[lid] = as[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint left = lid;
    const uint max_length = min(wg, size);

    for (uint length = 2; length <= max_length; length *= 2)
    {
        for (uint stride = length / 2; stride > 0; stride /= 2)
        {
            const bool reverse = (wg_idx % 2 != 0) ^ ((left & length) != 0);
            const uint right = left ^ stride;

            if (right > left)
            {
                float vals[2];
                vals[0] = shared[left];
                vals[1] = shared[right];
                order_vals(vals, 0, 1, reverse);
                shared[left]  = vals[0];
                shared[right] = vals[1];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (gid < size)
        as[gid] = shared[lid];
}

inline void bitonic_subsort_2(float * vals, int idx, const int reverse) {
    order_vals(vals, idx + 0, idx + 1, reverse);
}

inline void bitonic_subsort_4(float * vals, int idx, const int reverse) {
    const unsigned int k = 2;

    #pragma unroll
    for (int i = 0; i < k; ++i)
        order_vals(vals, idx + i, idx + i + k, reverse);

    bitonic_subsort_2(vals, idx, reverse);
    bitonic_subsort_2(vals, idx + k, reverse);
}

inline void bitonic_subsort_8(float * vals, int idx, const int reverse) {
    const unsigned int k = 4;

    #pragma unroll
    for (int i = 0; i < k; ++i)
        order_vals(vals, idx + i, idx + i + k, reverse);

    bitonic_subsort_4(vals, idx, reverse);
    bitonic_subsort_4(vals, idx + k, reverse);
}

inline void bitonic_subsort_16(float * vals, int idx, const int reverse) {
    const unsigned int k = 8;

    #pragma unroll
    for (int i = 0; i < k; ++i)
        order_vals(vals, idx + i, idx + i + k, reverse);

    bitonic_subsort_8(vals, idx, reverse);
    bitonic_subsort_8(vals, idx + k, reverse);
}

inline void bitonic_subsort_32(float * vals, int idx, const int reverse) {
    const unsigned int k = 16;

    #pragma unroll
    for (int i = 0; i < k; ++i)
        order_vals(vals, idx + i, idx + i + k, reverse);

    bitonic_subsort_16(vals, idx, reverse);
    bitonic_subsort_16(vals, idx + k, reverse);
}

#ifndef BITONIC_ORDER
#define BITONIC_ORDER 2
#endif

__kernel void bitonic(
    __global float * as,
    const uint size,
    const uint length,
    uint stride
)
{
    stride /= (BITONIC_ORDER / 2);
    const uint gid = get_global_id(0);
    const uint low = gid & (stride - 1);
    const uint i = BITONIC_ORDER == 2 ?
        ((gid * BITONIC_ORDER) - low) :
        (((gid - low) * BITONIC_ORDER) + low);
    const bool reverse = (i & length) != 0;
    as += i;

    float vals[BITONIC_ORDER];

    #pragma unroll
    for (int k = 0; k < BITONIC_ORDER; ++k)
        vals[k] = as[k * stride];

    switch (BITONIC_ORDER) {
        case 2  : bitonic_subsort_2(vals, 0, reverse);
        case 4  : bitonic_subsort_4(vals, 0, reverse);
        case 8  : bitonic_subsort_8(vals, 0, reverse);
        case 16 : bitonic_subsort_16(vals, 0, reverse);
        case 32 : bitonic_subsort_32(vals, 0, reverse);
    }

    #pragma unroll
    for (int k = 0; k < BITONIC_ORDER; ++k)
        as[k * stride] = vals[k];
}
