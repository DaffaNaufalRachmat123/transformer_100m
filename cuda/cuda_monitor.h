#pragma once
// =============================================================================
// cuda_monitor.h — NVML-based GPU monitoring (replaces IOKit gpu_monitor)
// =============================================================================
#include <cstddef>

namespace cuda {

struct GpuStats {
    int gpu_util_pct;
    int mem_util_pct;
    size_t mem_used_bytes;
    size_t mem_total_bytes;
    size_t mem_peak_bytes;
    int temperature_c;
    int power_w;
};

void nvml_init();
void nvml_shutdown();
GpuStats get_gpu_stats(int device = 0);
size_t get_active_memory(int device = 0);
size_t get_peak_memory(int device = 0);
float get_gpu_utilization(int device = 0);

void timer_begin(int device = 0);
void timer_end(int device = 0);

} // namespace cuda
