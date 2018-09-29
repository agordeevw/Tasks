#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/max_prefix_sum_cl.h"
#include <fstream>

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

    ocl::Kernel scan, add, zero_buffer, get_max_value, get_idx_of_max;
    try {
        scan.init(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "scan");
        scan.compile(true);

        add.init(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "add");
        add.compile(true);

        zero_buffer.init(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "zero_buffer");
        zero_buffer.compile(true);

        get_max_value.init(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "get_max_value");
        get_max_value.compile(true);

        get_idx_of_max.init(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "get_idx_of_max");
        get_idx_of_max.compile(true);
    }  catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }    

    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        try {
            // TODO: implement on OpenCL
            // Reference:
            // https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

            const unsigned int work_group_size = 128;

            auto round_up_work_size = [&work_group_size](unsigned int work_size) {
                return (work_size + work_group_size - 1) / work_group_size * work_group_size;
            };

            auto work_groups_count = [&round_up_work_size, &work_group_size](unsigned int work_size) {
                return round_up_work_size(work_size) / work_group_size;
            };

            gpu::gpu_mem_32i buffer_input;
            buffer_input.resizeN(round_up_work_size(n));
            zero_buffer.exec(gpu::WorkSize{ work_group_size, buffer_input.number() }, buffer_input);
            buffer_input.writeN(as.data(), as.size());

            gpu::gpu_mem_32i buffer_output;
            buffer_output.resizeN(round_up_work_size(n));

            std::vector<gpu::gpu_mem_32i> buffers_sums, buffers_scanned_sums;
            unsigned int buffer_size = n;
            do {
                buffer_size = round_up_work_size(work_groups_count(buffer_size));

                buffers_sums.emplace_back();
                auto& buffer_sums = buffers_sums.back();
                buffer_sums.resizeN(buffer_size);
                zero_buffer.exec(gpu::WorkSize{ work_group_size, buffer_sums.number() }, buffer_sums);

                buffers_scanned_sums.emplace_back();
                auto& buffer_scanned_sums = buffers_scanned_sums.back();
                buffer_scanned_sums.resizeN(buffer_size);
            } while (buffer_size > work_group_size);

            gpu::gpu_mem_32i max_value;
            max_value.resizeN(1);
            gpu::gpu_mem_32i max_idx;
            max_idx.resizeN(1);

            int max_sum;
            int result;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                scan.exec(
                    gpu::WorkSize{ work_group_size, buffer_input.number() },
                    buffer_input, buffer_output, buffers_sums[0],
                    ocl::LocalMem{ sizeof(int) * work_group_size });
                if (buffer_input.number() > work_group_size) {
                    for (int i = 0; i < buffers_sums.size() - 1; ++i) {
                        scan.exec(
                            gpu::WorkSize{ work_group_size, buffers_sums[i].number() },
                            buffers_sums[i], buffers_scanned_sums[i], buffers_sums[i + 1],
                            ocl::LocalMem{ sizeof(int) * work_group_size });
                    }
                    scan.exec(
                        gpu::WorkSize{ work_group_size, buffers_sums[buffers_sums.size() - 1].number() },
                        buffers_sums[buffers_sums.size() - 1],
                        buffers_scanned_sums[buffers_sums.size() - 1],
                        gpu::gpu_mem_32i{},
                        ocl::LocalMem{ sizeof(int) * work_group_size });
                    for (int i = buffers_sums.size() - 1; i > 0; --i) {
                        add.exec(
                            gpu::WorkSize{ work_group_size, buffers_scanned_sums[i - 1].number() },
                            buffers_scanned_sums[i - 1],
                            buffers_scanned_sums[i]);
                    }
                    add.exec(
                        gpu::WorkSize{ work_group_size, buffer_output.number() },
                        buffer_output, buffers_scanned_sums[0]);
                }

                max_sum = 0;
                result = 0;

                
                int initial_max_value = 0;
                max_value.writeN(&initial_max_value, 1);

                get_max_value.exec(
                    gpu::WorkSize{ work_group_size, buffer_output.number() },
                    buffer_output, max_value,
                    ocl::LocalMem{ sizeof(int) * work_group_size });

                max_value.readN(&max_sum, 1);

                if (max_sum > 0) {
                    int initial_idx_value = round_up_work_size(n) + 1;
                    max_idx.writeN(&initial_idx_value, 1);

                    get_idx_of_max.exec(
                        gpu::WorkSize{ work_group_size, buffer_output.number() },
                        buffer_output, max_value, max_idx);

                    max_idx.readN(&result, 1);
                }

                EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result diverges");
                EXPECT_THE_SAME(reference_result, result, "GPU result diverges");
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << e.what() << std::endl;
            return EXIT_FAILURE;
        }
    }
}
