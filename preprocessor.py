from utils import Utils
from typing import NamedTuple
from operator import attrgetter
import statistics
import random

class Preprocessor:

    @staticmethod
    # removing outliers using 3-sigma rule
    # using int and float fields only (ignoring strings, bools etc.)
    def outlier_indexes[T : NamedTuple](data: list[T]) -> list[int]:
        if len(data) <= 1:
            return []
        
        continuous_names = Utils.get_attrs_of(data[0], (int, float))
        getters = {f: attrgetter(f) for f in continuous_names}

        means = {
            field: statistics.mean(getters[field](entry) for entry in data)
            for field in continuous_names
            }
        
        stdevs = {
            field: statistics.stdev(getters[field](entry) for entry in data)
            for field in continuous_names
            }
        
        def is_outlier(entry: T) -> bool:
            return any(
                abs(getters[field](entry) - means[field]) > 3 * stdevs[field]
                for field in continuous_names
            )
        
        return [i for i, entry in enumerate(data) if is_outlier(entry)]

    @staticmethod
    def divide_data_randomly(data: list, ratio: float) -> tuple[list, list]:
        data = data[:]
        random.shuffle(data)
        split_index = int(len(data) * ratio)
        return data[:split_index], data[split_index:]