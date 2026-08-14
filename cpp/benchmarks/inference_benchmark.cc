#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "mini_ort/model.h"
#include "mini_ort/model_format.h"
#include "mini_ort/session.h"
#include "mini_ort/tensor.h"


namespace {

    using Clock = std::chrono::steady_clock;

    struct Options final {
        std::filesystem::path model_path;
        std::size_t iterations = 1000;
        std::size_t warmup_iterations = 100;
    }

    double Milliseconds(const Clock::duration duration) {
        return std::chrono::duration<double, std::milli>(duration).count();
    }

    std::size_t ParseCount(
        const std::string_view text,
        const char* name,
        const bool allow_zero
    ) {
        std::size_t value = 0;
        const auto result = std::from_chars(
            text.data(),
            text.data() + text.size(),
            value
        );

        if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || (!allow_zero && value == 0)) {
            throw std::invalid_argument(
                std::string(name) + " must be " + (allow_zero ? "a non-negative integer" : "a positive integer")
            );
        }

        return value;
    }

    Options ParseOptions(
        const int argc,
        const char* const* argv
    ) {
        if (argc < 2 || argc > 4) {
            throw std::invalid_argument("usage: mini_ort_benchmark <model.mer> [iterations] [warmup]");
        }

        Options options {
            std::filesystem::path(argv[1])
        };

        if (argc >= 3) {
            options.iterations = ParseCount(
                argv[2],
                "iterations",
                false
            );
        }

        if (argc == 4) {
            options.warmup_iterations = ParseCount(
                argv[3],
                "warmup",
                true
            );
        }

        return options;
    }


    std::size_t InputFeatures(
        const mini_ort::SequentialModel& model
    ) {
        for (const auto& layer : model.layers()) {
            if (const auto* linear = std::get_if<mini_ort::LinearLayer>(&layer)) {
                return linear->in_features;
            }
        }
        throw std::invalid_argument(
            "benchmark requires a model containing at least one Linear layer"
        );
    }


    mini_ort::Tensor MakeInput(const std::size_t features) {
        std::vector<float> values(features);

        for (std::size_t index = 0; index < features; ++index) {
            const auto centered = static_cast<std::int64_t>(index % 17) - 8;
            values[index] = static_cast<float>(centered) / 8.0F;
        }

        return mini_ort::Tensor(
            {
                1,
                static_cast<std::int64_t>(features)
            },
            std::move(values)
        );
    }


    double Median(const std::vector<double>& sorted) {
        const auto middle = sorted.size() / 2;

        if (sorted.size() % 2 == 0) {
            return (
                sorted[middle - 1] + sorted[middle]
            ) / 2.0;
        }

        return sorted[middle];
    }

    
    double NearestRankPercentile(
        const std::vector<double>& sorted,
        const double percentile
    ) {
        const auto rank = static_cast<std::size_t>(
            std::ceil(
                percentile * static_cast<double>(sorted.size())
            )
        );

        return sorted[
            std::max<std::size_t>(1, rank) - 1
        ];
    }

    void PrintResults(
        const Options& options,
        const std::size_t input_features,
        const double load_ms,
        const double session_init_ms,
        std::vector<double> latencies_ms,
        const double checksum,
        const mini_ort::TensorMemoryStats& baseline_memory,
        const mini_ort::TensorMemoryStats& memory
    ) {
        std::sort(
            latencies_ms.begin(),
            latencies_ms.end()
        );
        const double total_ms = std::accumulate(
            latencies_ms.begin(),
            latencies_ms.end(),
            0.0
        );
        const double mean_ms = total_ms / static_cast<double>(latencies_ms.size());
        const double throughput = static_cast<double>(latencies_ms.size()) * 1000.0 / total_ms;
        const auto peak_growth = memory.peak_live_bytes >= baseline_memory.live_bytes ? memory.peak_live_bytes - baseline_memory.live_bytes : 0;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "mini_ort native benchmark\n";
        std::cout << "model=" << options.model_path.string() << '\n';
        std::cout << "input_shape=[1," << input_features << "]\n";
        std::cout << "iterations=" << options.iterations << '\n';
        std::cout << "warmup_iterations=" << options.warmup_iterations << '\n';
        std::cout << "model_load_ms=" << load_ms << '\n';
        std::cout << "session_init_ms=" << session_init_ms << '\n';
        std::cout << "latency_min_ms=" << latencies_ms.front() << '\n';
        std::cout << "latency_mean_ms=" << mean_ms << '\n';
        std::cout << "latency_median_ms=" << Median(latencies_ms) << '\n';
        std::cout << "latency_p95_ms="
                    << NearestRankPercentile(latencies_ms, 0.95) << '\n';
        std::cout << "latency_max_ms=" << latencies_ms.back() << '\n';
        std::cout << "throughput_inferences_per_second=" << throughput << '\n';
        std::cout << "tensor_buffer_allocations=" << memory.buffer_allocations
                    << '\n';
        std::cout << "tensor_buffer_allocations_per_inference="
                    << static_cast<double>(memory.buffer_allocations) /
                        static_cast<double>(options.iterations)
                    << '\n';
        std::cout << "tensor_allocated_bytes=" << memory.allocated_bytes << '\n';
        std::cout << "tensor_allocated_bytes_per_inference="
                    << static_cast<double>(memory.allocated_bytes) /
                        static_cast<double>(options.iterations)
                    << '\n';
        std::cout << "tensor_copies=" << memory.tensor_copies << '\n';
        std::cout << "tensor_copied_bytes=" << memory.copied_bytes << '\n';
        std::cout << "tensor_baseline_live_bytes=" << baseline_memory.live_bytes
                    << '\n';
        std::cout << "tensor_peak_live_bytes=" << memory.peak_live_bytes << '\n';
        std::cout << "tensor_peak_growth_bytes=" << peak_growth << '\n';
        std::cout << "output_checksum=" << checksum << '\n';
    }
}


