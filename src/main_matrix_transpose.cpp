#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int M = 1 << 12;
    unsigned int K = 1 << 12;

    std::vector<float> as(M*K, 0);
    std::vector<float> as_t(M*K, 0);

    FastRandom r(M+K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);

    as_gpu.writeN(as.data(), M*K);

    constexpr auto TILE_SIZE = 32u;
    std::string defines;
    defines += " -DTILE_SIZE=";
    defines += std::to_string(TILE_SIZE);

#ifdef NDEBUG
    const bool show_compilation_log = false;
#else
    const bool show_compilation_log = true;
#endif

    try {
      std::string trivial_defines = defines;
      trivial_defines += " -DTRIVIAL_SOLUTION";
      ocl::Kernel matrix_transpose_kernel_trivial(matrix_transpose, matrix_transpose_length,
        "matrix_transpose", trivial_defines);
      matrix_transpose_kernel_trivial.compile(show_compilation_log);

      ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length,
        "matrix_transpose", defines);
      matrix_transpose_kernel.compile(show_compilation_log);

      auto run_kernel = [TILE_SIZE, M, K, benchmarkingIters, &as_gpu, &as_t_gpu, &as_t](
        ocl::Kernel& kernel, const std::string& name) 
      {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
          const auto rows_count = M;
          const auto cols_count = K;

          kernel.exec(
            gpu::WorkSize{ TILE_SIZE, TILE_SIZE, cols_count, rows_count },
            as_gpu, as_t_gpu, rows_count, cols_count);

          t.nextLap();
        }
        std::cout << name << std::endl;
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M * K / 1000.0 / 1000.0 / t.lapAvg() << " millions/s" << std::endl;
      
        as_t_gpu.readN(as_t.data(), M * K);
      };

      auto check_correctness = [M, K, &as, &as_t]() -> bool 
      {
        for (unsigned j = 0; j < M; ++j) {
          for (unsigned i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b)
              return false;
          }
        }
        return true;
      };

      run_kernel(matrix_transpose_kernel_trivial, "Trivial solution");
      if (!check_correctness()) {
        std::cerr << "Not the same!" << std::endl;
        return EXIT_FAILURE;
      }

      run_kernel(matrix_transpose_kernel, "Local memory solution");
      if (!check_correctness()) {
        std::cerr << "Not the same!" << std::endl;
        return EXIT_FAILURE;
      }

    } catch (const std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
    }

    return 0;
}
