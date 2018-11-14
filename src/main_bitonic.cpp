#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

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


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
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

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel bitonic_local(bitonic_kernel, bitonic_kernel_length, "bitonic_local");
        bitonic_local.compile();

        ocl::Kernel bitonic_2(bitonic_kernel, bitonic_kernel_length, "bitonic", "-DBITONIC_ORDER=2");
        bitonic_2.compile();

        ocl::Kernel bitonic_4(bitonic_kernel, bitonic_kernel_length, "bitonic", "-DBITONIC_ORDER=4");
        bitonic_4.compile();

        ocl::Kernel bitonic_8(bitonic_kernel, bitonic_kernel_length, "bitonic", "-DBITONIC_ORDER=8");
        bitonic_8.compile();

        ocl::Kernel bitonic_16(bitonic_kernel, bitonic_kernel_length, "bitonic", "-DBITONIC_ORDER=16");
        bitonic_16.compile();

        ocl::Kernel bitonic_32(bitonic_kernel, bitonic_kernel_length, "bitonic", "-DBITONIC_ORDER=32");
        bitonic_32.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            bitonic_local.exec(gpu::WorkSize(workGroupSize, global_work_size),
                         as_gpu, n, ocl::LocalMem{ workGroupSize * sizeof(as[0]) });

            for (unsigned int length = 2 * workGroupSize; length <= n; length *= 2) {
                unsigned int stride = length / 2;
                while (stride >= 16) {
                    bitonic_32.exec(gpu::WorkSize(workGroupSize, global_work_size / 32),
                        as_gpu, n, length, stride);
                    stride /= 32;
                }
                while (stride >= 8) {
                    bitonic_16.exec(gpu::WorkSize(workGroupSize, global_work_size / 16),
                        as_gpu, n, length, stride);
                    stride /= 16;
                }
                while (stride >= 4) {
                    bitonic_8.exec(gpu::WorkSize(workGroupSize, global_work_size / 8),
                        as_gpu, n, length, stride);
                    stride /= 8;
                }
                while (stride >= 2) {
                    bitonic_4.exec(gpu::WorkSize(workGroupSize, global_work_size / 4),
                        as_gpu, n, length, stride);
                    stride /= 4;
                }
                while (stride >= 1) {
                    bitonic_2.exec(gpu::WorkSize(workGroupSize, global_work_size / 2),
                        as_gpu, n, length, stride);
                    stride /= 2;
                }
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
