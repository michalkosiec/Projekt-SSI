import math
from collections import defaultdict
from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

class NaiveBayes(BaseModel):
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.class_priors = {}
        # Słownik przechowujący listę krotek (średnia, wariancja) dla każdej cechy w danej klasie
        self.feature_stats = defaultdict(list)

    def train(self, training_data: list[RussianFlat]):
        separated_data = defaultdict(list)
        total_samples = len(training_data)

        # 1. Grupowanie danych według klasy docelowej
        for flat in training_data:
            # Zakładamy, że 'flat' posiada atrybut 'label' określający klasę (np. "drogi", "tani")
            # oraz że właściwości można zrzucić przez 'to_tuple'
            label = flat.price_range
            features = self.to_tuple(flat.data)
            
            separated_data[label].append(features)
            self.class_counts[label] += 1

        # 2. Obliczanie średniej i wariancji dla każdej cechy w obrębie każdej klasy
        for label, features_list in separated_data.items():
            self.class_priors[label] = self.class_counts[label] / total_samples
            
            # zip(*...) transponuje listę list, grupując te same cechy (kolumny) ze wszystkich rekordów
            feature_columns = zip(*features_list)
            
            stats = []
            for column in feature_columns:
                mean = sum(column) / len(column)
                # Obliczanie wariancji z próby
                if len(column) > 1:
                    variance = sum((x - mean) ** 2 for x in column) / (len(column) - 1)
                else:
                    variance = 0.0
                
                # Dodajemy mały epsilon (1e-9) do wariancji, aby uniknąć dzielenia przez zero w predict
                stats.append((mean, variance + 1e-9))
                
            self.feature_stats[label] = stats

    def predict(self, entry: RussianFlatProps) -> str:
        features = self.to_tuple(entry)
        
        best_label = "unknown"
        max_log_prob = -float('inf')
        
        # 3. Klasyfikacja na podstawie sumy logarytmów prawdopodobieństw
        for label, stats in self.feature_stats.items():
            # Zaczynamy od logarytmu prawdopodobieństwa a priori
            log_prob = math.log(self.class_priors[label])
            
            for x, (mean, var) in zip(features, stats):
                # Obliczanie logarytmu gęstości prawdopodobieństwa rozkładu normalnego (Gaussa)
                log_pdf = -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
                log_prob += log_pdf
                
            # Wybieramy klasę z najwyższym prawdopodobieństwem a posteriori
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