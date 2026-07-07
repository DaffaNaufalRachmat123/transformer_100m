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

#else // !USE_NVML — stubs

void nvml_init() {
    printf("[NVML] disabled at build time (libnvidia-ml not found)\n");
}
void nvml_shutdown() {}
GpuStats get_gpu_stats(int) { return {}; }
size_t get_active_memory(int) { return 0; }
size_t get_peak_memory(int) { return 0; }
float get_gpu_utilization(int) { return 0.0f; }

#endif

void timer_begin(int) {}
void timer_end(int) {}

} // namespace cuda
