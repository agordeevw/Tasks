#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/sum_cl.h"


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
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    try
    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum");
        kernel.compile(true);

        const unsigned int work_group_size = 128;

        auto round_up_work_size = [&work_group_size](unsigned int work_size) {
            return (work_size + work_group_size - 1) / work_group_size * work_group_size;
        };

        auto work_groups_count = [&round_up_work_size, &work_group_size](unsigned int work_size) {
            return round_up_work_size(work_size) / work_group_size;
        };

        gpu::gpu_mem_32u buffer_data;
        buffer_data.resizeN(n);
        buffer_data.writeN(as.data(), as.size());
        gpu::gpu_mem_32u buffer_result;
        buffer_result.resizeN(work_groups_count(n));

        std::vector<unsigned int> results(work_groups_count(n));
        unsigned int sum;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            gpu::WorkSize work_size{ work_group_size, round_up_work_size(n) };
            ocl::LocalMem local_values{ sizeof(unsigned int) * work_group_size };
            kernel.exec(work_size, buffer_data, n, buffer_result, local_values);

            sum = 0;
            buffer_result.readN(results.data(), results.size());
            for (auto result : results)
                sum += result;
            EXPECT_THE_SAME(reference_sum, sum, "GPU result is wrong");
                     
            t.nextLap();
        }

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}