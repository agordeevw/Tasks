#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

unsigned int roundup(const unsigned int value, const unsigned int n) {
    return (value + n - 1) / n * n;
}

unsigned int wg_count(const unsigned int gw_size, const unsigned int wg_size) {
    return roundup(gw_size, wg_size) / wg_size;
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    constexpr unsigned int RADIX_BITS = 4;
    constexpr unsigned int RADIX_BITS2 = 16;

    const unsigned int workGroupSize = 128;
    const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u output;
    output.resizeN(n);

    gpu::gpu_mem_32u counts;
    counts.resizeN(global_work_size / workGroupSize * RADIX_BITS2);

    gpu::gpu_mem_32u offsets;
    offsets.resizeN(RADIX_BITS2);

    std::vector<gpu::gpu_mem_32u> exclusive_scan_buffers;
    unsigned int scan_work_size = counts.number();
    while (scan_work_size > 2 * workGroupSize) {
        scan_work_size = 16 * wg_count(scan_work_size / 2, workGroupSize);
        gpu::gpu_mem_32u buffer;
        buffer.resizeN(scan_work_size);
        exclusive_scan_buffers.push_back(std::move(buffer));
    }

    gpu::gpu_mem_32u * input_buffer = &as_gpu;
    gpu::gpu_mem_32u * output_buffer = &output;

    {
        ocl::Kernel count_kernel(radix_kernel, radix_kernel_length, "count");
        count_kernel.compile();

        ocl::Kernel scan_kernel(radix_kernel, radix_kernel_length, "scan");
        scan_kernel.compile();

        ocl::Kernel scatter_kernel(radix_kernel, radix_kernel_length, "scatter");
        scatter_kernel.compile();

        ocl::Kernel exclusive_scan_pre_kernel(radix_kernel, radix_kernel_length, "exclusive_scan_pre");
        exclusive_scan_pre_kernel.compile();

        ocl::Kernel exclusive_scan_post_kernel(radix_kernel, radix_kernel_length, "exclusive_scan_post");
        exclusive_scan_post_kernel.compile();

        timer t;
        for (int iter = 0; iter < 1; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            for (int b = 0; b < sizeof(unsigned int) * 8 / RADIX_BITS; b++) {
                // local histograms
                count_kernel.exec(gpu::WorkSize{ workGroupSize, global_work_size },
                    *input_buffer,
                    n,
                    counts,
                    offsets,
                    ocl::LocalMem{ sizeof(unsigned int) * RADIX_BITS2 },
                    b * RADIX_BITS);

                // scan histograms
                exclusive_scan_pre_kernel.exec(
                    gpu::WorkSize{
                        workGroupSize / 16,
                        (uint)counts.number() / 2 / 16
                    },
                    counts,
                    counts,
                    exclusive_scan_buffers[0],
                    ocl::LocalMem{ 2 * sizeof(unsigned int) * workGroupSize });

                for (uint i = 0; i < exclusive_scan_buffers.size() - 1; ++i) {
                    exclusive_scan_pre_kernel.exec(
                        gpu::WorkSize{
                            workGroupSize / 16,
                            (uint)exclusive_scan_buffers[i].number() / 2 / 16
                        },
                        exclusive_scan_buffers[i],
                        exclusive_scan_buffers[i],
                        exclusive_scan_buffers[i+1],
                        ocl::LocalMem{ 2 * sizeof(unsigned int) * workGroupSize });
                }

                const uint last_buffer = exclusive_scan_buffers.size() - 1;
                exclusive_scan_pre_kernel.exec(
                    gpu::WorkSize{
                        workGroupSize / 16,
                        (uint)exclusive_scan_buffers[last_buffer].number() / 2 / 16
                    },
                    exclusive_scan_buffers[last_buffer],
                    exclusive_scan_buffers[last_buffer],
                    gpu::gpu_mem_32u{},
                    ocl::LocalMem{ 2 * sizeof(unsigned int) * workGroupSize });

                for (uint i = last_buffer; i > 0; i--) {
                    exclusive_scan_post_kernel.exec(
                        gpu::WorkSize{
                            workGroupSize / 16,
                            (uint)exclusive_scan_buffers[i-1].number() / 2 / 16
                        },
                        exclusive_scan_buffers[i],
                        exclusive_scan_buffers[i-1]);
                }
                exclusive_scan_post_kernel.exec(
                    gpu::WorkSize{
                        workGroupSize / 16,
                        (uint)counts.number() / 2 / 16
                    },
                    exclusive_scan_buffers[0],
                    counts);

                // calc global offsets
                scan_kernel.exec(gpu::WorkSize{ 1, 1 },
                    counts,
                    offsets,
                    global_work_size / workGroupSize);

                // scatter according to global offsets
                scatter_kernel.exec(gpu::WorkSize{ workGroupSize, global_work_size },
                    *input_buffer,
                    n,
                    b * RADIX_BITS,
                    counts,
                    offsets,
                    *output_buffer);

                std::swap(input_buffer, output_buffer);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        input_buffer->readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
