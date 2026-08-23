import numpy as np

data = np.array([
    [10, 200, 1.5],   
    [12, 180, 1.7],   
    [9,  220, 1.4]    
])

def zscore_normalize(data: np.ndarray) -> np.ndarray:
    normalized = (data - data.mean(axis=0)) / data.std(axis=0)
    return normalized

normalized = zscore_normalize(data)
print(normalized)

def feature_stats(data: np.ndarray) -> dict[str, np.ndarray]:
    column_means = data.mean(axis=0)
    column_std = data.std(axis=0)

    result = {
        "mean": column_means,
        "std":column_std
    }
    return result

dict_result = feature_stats(data)
print(dict_result)

def most_extreme_sample(normalized: np.ndarray) -> int:
    abs_score = np.abs(normalized)
    row_totals = abs_score.sum(axis=1)
    worst_idx = np.argmax(row_totals)
    return worst_idx

print(most_extreme_sample(normalized))
