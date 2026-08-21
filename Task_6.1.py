import numpy as np

import numpy as np
def build_array(values: list[float]) -> np.ndarray:
    reading = np.array(values, dtype="float32")
    return reading

def reshape_to_hours(arr, window_size) -> np.ndarray:
    reshaped = arr.reshape(-1, window_size)
    return reshaped

def get_window(arr, idx) -> np.ndarray:
    return arr[idx]

def get_window_safe(arr, idx) -> np.ndarray:
    mCopy = arr[idx].copy()
    return mCopy

def is_view(sub_arr: np.ndarray, parent_arr: np.ndarray) -> bool:
    return sub_arr.base is not None

readings = [68.5, 67.9, 67.2, 66.8, 66.5, 66.9,
            68.0, 70.5, 73.2, 75.8, 77.1, 78.4,
            79.5, 80.1, 80.6, 80.2, 79.4, 77.8,
            75.0, 73.1, 71.5, 70.2, 69.4, 68.8]

arr = build_array(readings)
print(f"{arr, arr.shape}\n")

window_size = 6
reeshape = reshape_to_hours(arr, window_size)
print(f"{reeshape}\n")

view_result = get_window(reeshape, 1)
copy_result = get_window_safe(reeshape, 1)

print(f"{view_result}\n")
print(f"{copy_result}\n")

print(is_view(view_result, reeshape))   
print(is_view(copy_result, reeshape))   

view_result[0] = -999
print("After mutating view:")
print(reeshape)   

copy_result[0] = -999
print("After mutating copy:")
print(reeshape)   