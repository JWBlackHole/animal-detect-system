import psutil

def process_cpu_percent():
    process = psutil.Process()
    process.cpu_percent(interval=None)
    return process