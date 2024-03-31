# Structural Complexity Metrics 

## Introduction
Structural Complexity Metrics is a tool that provides two efficient and accurate complexity metrics: Mutual Neighbor Consistency (MNC) and Pairwise Distance Shift (PDS). These metrics take arbitrary high-dimensional datasets as input and produce complexity scores that are comparable across datasets.

1. **Mutual Neighbor Consistency(MNC)**

MNC measures the complexity of the local neighborhood structure by quantifying inconsistency between two different similarity functions—k-Nearest Neighbors (kNN) and Shared Nearest Neighbors (SNN). These functions have different granularity in examining the neighborhood structure.

2. **Pairwise Distance Shift(PDS)**

PDS quantifies how much the global structure of a given dataset suffers from the shift of distance, which is a widely known indicator of the curse of dimensionality.

These two complexity metrics complement each other in comprehensively examining high-dimensional datasets. For more detailed information, please refer to the related academic papers: (paper link).


## API
### `mutual_neighbor_consistency(data, k)`

This function measures the Mutual Neighbor Consistency (MNC) of the given data.

#### Parameters:

- `data` : array-like, shape (n_samples, n_features)
    - Input data. High-dimensional data points are represented as rows that correspond to samples and columns that correspond to features.

- `k` : int
    - The number of nearest neighbors to consider. This is the 'k' in k-Nearest Neighbors (kNN).

#### Returns:

- `mnc_score` : float
    - The MNC score of the data. This score quantifies the complexity of the local neighborhood structure.

#### Example:

```python
from mnc import mutual_neighbor_consistency
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
k = 2

mnc_score = mutual_neighbor_consistency(data, k)
print(mnc_score)
```

### `pairwise_distance_shift(data)`

This function measures the Pairwise Distance Shift (PDS) of the given data.

#### Parameters:

- `data` : array-like, shape (n_samples, n_features)
    - Input data. High-dimensional data points are represented as rows that correspond to samples and columns that correspond to features.

#### Returns:

- `pds_score` : float
    - The PDS score of the data. This score quantifies how much the global structure of a given dataset suffers from the shift of distance.

#### Example:

```python
from pds import pairwise_distance_shift
import numpy as np

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

pds_score = pairwise_distance_shift(data)
print(pds_score)
```

## Installation
To use Structural Complexity Metrics, clone the repository and install the dependencies. Here's how you can do it:

```bash
git clone https://github.com/hj-n/structural-complexity.git
cd structural-complexity
pip install -r requirements.txt
```