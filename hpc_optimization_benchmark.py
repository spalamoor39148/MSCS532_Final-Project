import time
import random
import numpy as np
import json

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

# -------------------------------
# Utility: Convert NumPy objects to Python types
# -------------------------------
def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

# -------------------------------
# Data generation functions
# -------------------------------
def generate_list_of_tuples(N, seed=42):
    """Generate list of (x, y) tuples."""
    random.seed(seed)
    return [(random.random(), random.random()) for _ in range(N)]

def generate_numpy_separate_arrays(N, seed=42):
    """Generate two separate NumPy arrays for x and y."""
    np.random.seed(seed)
    x = np.random.rand(N)
    y = np.random.rand(N)
    return x, y

def generate_numpy_structured_array(N, seed=42):
    """Generate NumPy structured array with fields x and y."""
    np.random.seed(seed)
    data = np.zeros(N, dtype=[('x', 'f8'), ('y', 'f8')])
    data['x'] = np.random.rand(N)
    data['y'] = np.random.rand(N)
    return data

# -------------------------------
# Filtering operations
# -------------------------------
def filter_list_of_tuples(data):
    """Filter list of tuples for points inside unit circle."""
    return [pt for pt in data if pt[0]**2 + pt[1]**2 < 1]

def filter_numpy_separate_arrays(x, y):
    """Filter separate NumPy arrays for points inside unit circle."""
    mask = x**2 + y**2 < 1
    return x[mask], y[mask]

def filter_numpy_structured_array(data):
    """Filter NumPy structured array for points inside unit circle."""
    mask = data['x']**2 + data['y']**2 < 1
    return data[mask]

# Numba-accelerated version for separate arrays
if NUMBA_AVAILABLE:
    @jit(nopython=True)
    def filter_numpy_separate_arrays_numba(x, y):
        mask = np.empty(len(x), dtype=np.bool_)
        for i in range(len(x)):
            mask[i] = x[i]**2 + y[i]**2 < 1
        return x[mask], y[mask]
else:
    def filter_numpy_separate_arrays_numba(x, y):
        raise RuntimeError("Numba not available.")

# -------------------------------
# Benchmark function
# -------------------------------
def benchmark(N=1_000_000, trials=5, seed=42, warmup=True, use_numba=False):
    results = {}

    # Data setup
    list_data = generate_list_of_tuples(N, seed)
    x_sep, y_sep = generate_numpy_separate_arrays(N, seed)
    structured_data = generate_numpy_structured_array(N, seed)

    # Warmup
    if warmup:
        filter_list_of_tuples(list_data)
        filter_numpy_separate_arrays(x_sep, y_sep)
        filter_numpy_structured_array(structured_data)
        if use_numba and NUMBA_AVAILABLE:
            filter_numpy_separate_arrays_numba(x_sep, y_sep)

    # List of tuples benchmark
    list_times = []
    for _ in range(trials):
        start = time.perf_counter()
        filter_list_of_tuples(list_data)
        end = time.perf_counter()
        list_times.append(end - start)
    results["list_of_tuples"] = {
        "mean_time_sec": np.mean(list_times),
        "min_time_sec": np.min(list_times),
        "max_time_sec": np.max(list_times),
    }

    # NumPy separate arrays benchmark
    numpy_sep_times = []
    for _ in range(trials):
        start = time.perf_counter()
        filter_numpy_separate_arrays(x_sep, y_sep)
        end = time.perf_counter()
        numpy_sep_times.append(end - start)
    results["numpy_separate_arrays"] = {
        "mean_time_sec": np.mean(numpy_sep_times),
        "min_time_sec": np.min(numpy_sep_times),
        "max_time_sec": np.max(numpy_sep_times),
    }

    # NumPy structured array benchmark
    numpy_struct_times = []
    for _ in range(trials):
        start = time.perf_counter()
        filter_numpy_structured_array(structured_data)
        end = time.perf_counter()
        numpy_struct_times.append(end - start)
    results["numpy_structured_array"] = {
        "mean_time_sec": np.mean(numpy_struct_times),
        "min_time_sec": np.min(numpy_struct_times),
        "max_time_sec": np.max(numpy_struct_times),
    }

    # Numba-accelerated benchmark
    if use_numba and NUMBA_AVAILABLE:
        numba_times = []
        for _ in range(trials):
            start = time.perf_counter()
            filter_numpy_separate_arrays_numba(x_sep, y_sep)
            end = time.perf_counter()
            numba_times.append(end - start)
        results["numba_numpy_separate_arrays"] = {
            "mean_time_sec": np.mean(numba_times),
            "min_time_sec": np.min(numba_times),
            "max_time_sec": np.max(numba_times),
        }

    return results

# -------------------------------
# Main
# -------------------------------
def main():
    res = benchmark(N=1_000_000, trials=5, seed=42, warmup=True, use_numba=True)
    res_clean = convert_to_serializable(res)

    print("\nBenchmark Results (JSON):")
    print(json.dumps(res_clean, indent=2))

    print("\nBenchmark Results (Table):")
    for method, stats in res_clean.items():
        print(f"{method:30} | mean: {stats['mean_time_sec']:.6f} s | "
              f"min: {stats['min_time_sec']:.6f} s | max: {stats['max_time_sec']:.6f} s")

if __name__ == "__main__":
    main()
