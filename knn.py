import math
import statistics
from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

class Knn(BaseModel):
    def __init__(self, k: int = 7):
        self.k = k
        self.training_data: list[RussianFlat] = []
        self.continuous_scales: tuple[float, ...] = ()

    def train(self, training_data: list[RussianFlat]):
        self.training_data = training_data[:]
        self.continuous_scales = self._compute_continuous_scales(self.training_data)

    def predict(self, entry: RussianFlatProps) -> str:
        if not self.training_data:
            return "unknown"

        entry_features = self._extract_features(entry)
        distances = []

        for flat in self.training_data:
            train_features = self._extract_features(flat.data)
            dist = self._distance(entry_features, train_features)
            distances.append((dist, flat.price_range))

        distances.sort(key=lambda item: item[0])
        nearest = distances[: self.k]

        if not nearest:
            return "unknown"

        return self._vote(nearest)

    def _vote(self, neighbors: list[tuple[float, str]]) -> str:
        weights: dict[str, float] = {}
        for dist, label in neighbors:
            if dist <= 1e-9:
                return label
            weights[label] = weights.get(label, 0.0) + 1.0 / (dist + 1e-9)

        return max(weights.items(), key=lambda item: item[1])[0]

    def _distance(
        self,
        first: tuple[tuple[float, ...], bool, str, str],
        second: tuple[tuple[float, ...], bool, str, str],
    ) -> float:
        first_cont, first_new_building, first_region, first_building = first
        second_cont, second_new_building, second_region, second_building = second

        squared_sum = 0.0
        for value_a, value_b, scale in zip(first_cont, second_cont, self.continuous_scales):
            diff = (value_a - value_b) / scale
            squared_sum += diff * diff

        numeric_distance = math.sqrt(squared_sum)

        binary_distance = 0.4 if first_new_building != second_new_building else 0.0
        region_distance = 1.0 if first_region != second_region else 0.0
        building_distance = 0.5 if first_building != second_building else 0.0

        return numeric_distance + binary_distance + region_distance + building_distance

    def _compute_continuous_scales(self, training_data: list[RussianFlat]) -> tuple[float, ...]:
        if not training_data:
            return (1.0,) * 9

        examples = [_ for _ in training_data]
        columns = list(zip(*(self._extract_features(flat.data)[0] for flat in examples)))

        scales: list[float] = []
        for column in columns:
            stdev = statistics.stdev(column) if len(column) > 1 else 0.0
            if stdev <= 0.0:
                stdev = 1.0
            scales.append(stdev)

        return tuple(scales)

    @staticmethod
    def _extract_features(entry: RussianFlatProps) -> tuple[tuple[float, ...], bool, str, str]:
        lat_km = float(entry.geo_lat) * 111.0
        lon_km = float(entry.geo_lon) * 111.0 * math.cos(math.radians(float(entry.geo_lat)))

        continuous = (
            float(entry.publish_year),
            float(entry.publish_month),
            float(entry.level),
            float(entry.levels),
            float(entry.rooms),
            math.log(float(entry.area) + 1e-5),
            math.log(float(entry.kitchen_area) + 1e-5),
            lat_km,
            lon_km,
        )

        return continuous, entry.new_building, entry.region_id, entry.building_type
