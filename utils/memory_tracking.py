import tracemalloc
import pynvml
import time

def start_gpu_monitoring():
    """Initialize NVML to monitor GPU memory."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(3)  # Assuming a single GPU. Adjust the index for multiple GPUs.
    return handle

def get_gpu_memory_usage(handle):
    """Retrieve the current GPU memory usage."""
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return info.used / (1024 ** 2)  # Convert bytes to MB

def monitor_function_cpu_gpu_memory(func, *args, **kwargs):
    """Monitor and report the CPU and GPU memory usage of a function."""
    # Initialize GPU monitoring
    gpu_handle = start_gpu_monitoring()

    # Start monitoring CPU memory
    tracemalloc.start()

    # Record CPU and GPU memory before function execution
    cpu_start, _ = tracemalloc.get_traced_memory()
    gpu_start = get_gpu_memory_usage(gpu_handle)

    # Execute the function
    result = func(*args, **kwargs)

    # Record CPU and GPU memory after function execution
    cpu_end, _ = tracemalloc.get_traced_memory()
    gpu_end = get_gpu_memory_usage(gpu_handle)

    # Stop the CPU memory monitoring
    tracemalloc.stop()

    # Calculate the memory usage
    cpu_memory_used = (cpu_end - cpu_start) / 1024  # Convert to KB
    gpu_memory_used = gpu_end - gpu_start

    print(f"CPU Memory Used: {cpu_memory_used} KB")
    print(f"GPU Memory Used: {gpu_memory_used} MB")

    # Don't forget to shutdown NVML
    pynvml.nvmlShutdown()

    return result
