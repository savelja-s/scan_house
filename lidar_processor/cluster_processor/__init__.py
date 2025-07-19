from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class Building:
    polygon: List[Tuple[float, float]]
    height: float
    cluster_id: int
