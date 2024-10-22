import jax

def get_gpu_memory_usage():
    devices = jax.local_devices()
    if not devices:
        return 0  # Return 0 if no devices are available

    peak_memory = max(
        device.memory_stats()["peak_bytes_in_use"] for device in devices
    )
    return peak_memory / (1024 * 1024 * 1024)  
