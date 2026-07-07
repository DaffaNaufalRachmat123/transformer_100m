#include "cuda_monitor.h"
#include <cstdio>

// NVML is optional: if USE_NVML is not defined (library not found at configure
// time), we compile stub implementations that report zeros. This keeps the
// executable linkable on machines without libnvidia-ml.so (some Kaggle images).
#ifdef USE_NVML
#include <nvml.h>
#endif

namespace cuda {

#ifdef USE_NVML

static bool nvml_ok = false;

void nvml_init() {
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
        printf("[NVML] Failed to init (code %d) — GPU monitoring unavailable\n", r);
        nvml_ok = false;
        return;
    }
    nvml_ok = true;

    unsigned int count = 0;
    nvmlDeviceGetCount(&count);
    printf("[NVML] %u GPU(s) detected\n", count);
}

void nvml_shutdown() {
    if (nvml_ok) nvmlShutdown();
}

GpuStats get_gpu_stats(int device) {
    GpuStats s = {};
    if (!nvml_ok) return s;

    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(device, &dev) != NVML_SUCCESS) return s;

    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
        s.gpu_util_pct = util.gpu;
        s.mem_util_pct = util.memory;
    }

    nvmlMemory_t mem;
    if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
        s.mem_used_bytes = mem.used;
        s.mem_total_bytes = mem.total;
    }

    unsigned int temp = 0, power = 0;
    nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp);
    nvmlDeviceGetPowerUsage(dev, &power);
    s.temperature_c = (int)temp;
    s.power_w = (int)(power / 1000);

    return s;
}

size_t get_active_memory(int device) {
    if (!nvml_ok) return 0;
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(device, &dev) != NVML_SUCCESS) return 0;
    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(dev, &mem);
    return mem.used;
}

size_t get_peak_memory(int device) {
    return get_active_memory(device);
}

float get_gpu_utilization(int device) {
    GpuStats s = get_gpu_stats(device);
    return (float)s.gpu_util_pct;
}

#else // !USE_NVML — fallbacks using cudaMemGetInfo + nvidia-smi

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>

// Query GPU utilization via `nvidia-smi` (always installed on driver-present
// systems). We keep it lightweight: one popen call per log tick, parse a
// single integer. Result cached briefly to avoid spamming.
static int query_nvidia_smi_util(int device) {
    char cmd[128];
    snprintf(cmd, sizeof(cmd),
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i %d 2>/dev/null",
        device);
    FILE* p = popen(cmd, "r");
    if (!p) return 0;
    char buf[32] = {0};
    if (!fgets(buf, sizeof(buf), p)) { pclose(p); return 0; }
    pclose(p);
    return atoi(buf);
}

void nvml_init() {
    printf("[NVML] disabled — using cudaMemGetInfo + nvidia-smi fallback\n");
}
void nvml_shutdown() {}

GpuStats get_gpu_stats(int device) {
    GpuStats s = {};
    // Memory: cudaMemGetInfo is always available (queries current device).
    // The caller sets device via cudaSetDevice earlier; we just read here.
    int prev_dev = 0;
    cudaGetDevice(&prev_dev);
    if (device != prev_dev) cudaSetDevice(device);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    if (device != prev_dev) cudaSetDevice(prev_dev);
    s.mem_total_bytes = total_b;
    s.mem_used_bytes  = (total_b > free_b) ? (total_b - free_b) : 0;
    s.mem_util_pct    = total_b > 0 ? (int)(100 * s.mem_used_bytes / total_b) : 0;
    // GPU utilization via nvidia-smi
    s.gpu_util_pct = query_nvidia_smi_util(device);
    return s;
}

size_t get_active_memory(int device) {
    return get_gpu_stats(device).mem_used_bytes;
}
size_t get_peak_memory(int device) { return get_active_memory(device); }
float get_gpu_utilization(int device) { return (float)get_gpu_stats(device).gpu_util_pct; }

#endif

void timer_begin(int) {}
void timer_end(int) {}

} // namespace cuda
