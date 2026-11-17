import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    data = np.array(list(range(n_samples)))

    if shuffle:
        np.random.shuffle(data)

    val_length = n_samples // k
    quota_list = [val_length] * k

    if n_samples % k != 0:
        remain = n_samples % k

        for i in range(remain):
            quota_list[i] += 1

    first_idx = 0
    last_idx = quota_list[0]

    answer = []
    while(True):

        val_indices = data[first_idx:last_idx]
        train_indices = np.concatenate([data[:first_idx], data[last_idx:]])
        
        answer.append((train_indices.tolist(), val_indices.tolist()))

        first_idx += quota_list.pop(0)
        last_idx += quota_list[0] if quota_list else 0

        if first_idx == n_samples:
            break

    return answer