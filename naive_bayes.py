import math
from collections import defaultdict
from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

class NaiveBayes(BaseModel):
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.class_priors = {}
        self.feature_stats = defaultdict(list)

    def train(self, training_data: list[RussianFlat]):
        separated_data = defaultdict(list)
        total_samples = len(training_data)

        for flat in training_data:
            label = flat.price_range
            features = self.to_tuple(flat.data)
            
            separated_data[label].append(features)
            self.class_counts[label] += 1

        for label, features_list in separated_data.items():
            self.class_priors[label] = self.class_counts[label] / total_samples
            
            feature_columns = zip(*features_list)
            
            stats = []
            for column in feature_columns:
                mean = sum(column) / len(column)
                if len(column) > 1:
                    variance = sum((x - mean) ** 2 for x in column) / (len(column) - 1)
                else:
                    variance = 0.0
                
                stats.append((mean, variance + 1e-9))
                
            self.feature_stats[label] = stats

    def predict(self, entry: RussianFlatProps) -> str:
        features = self.to_tuple(entry)
        
        best_label = "unknown"
        max_log_prob = -float('inf')
        
        for label, stats in self.feature_stats.items():
            log_prob = math.log(self.class_priors[label])
            
            for x, (mean, var) in zip(features, stats):
                log_pdf = -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
                log_prob += log_pdf
                
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label
                
        return best_label
    
    @staticmethod
    def to_tuple(entry: RussianFlatProps) -> tuple[float, ...]:
        return (
            float(entry.publish_year),
            float(entry.publish_month),
            entry.geo_lat,
            entry.geo_lon,
            float(entry.level),
            float(entry.levels),
            float(entry.rooms),
            entry.area,
            entry.kitchen_area,
            1.0 if entry.building_type == "other" else 0.0,
            1.0 if entry.building_type == "panel" else 0.0,
            1.0 if entry.building_type == "monolithic" else 0.0,
            1.0 if entry.building_type == "brick" else 0.0,
            1.0 if entry.building_type == "block" else 0.0,
            1.0 if entry.building_type == "wood" else 0.0,
            1.0 if entry.new_building else 0.0
        )