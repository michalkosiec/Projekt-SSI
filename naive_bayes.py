import math
from collections import defaultdict
from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

class NaiveBayes(BaseModel):
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.class_priors = {}
        
        self.continuous_stats = defaultdict(list)
        self.binary_probs = defaultdict(list)
        self.cat_probs = defaultdict(dict)
        self.cat_unseen_probs = defaultdict(dict)
        
        self.num_classes = 0

    def train(self, training_data: list[RussianFlat]):
        separated_data_cont = defaultdict(list)
        separated_data_bin = defaultdict(list)
        separated_data_cat = defaultdict(list)
        
        all_continuous_features = []
        global_cat_uniques = defaultdict(set)
        
        total_samples = len(training_data)

        # 1. Zbieranie i grupowanie danych
        for flat in training_data:
            label = flat.price_range
            cont_features, bin_features, cat_features = self.extract_features(flat.data)
            
            separated_data_cont[label].append(cont_features)
            separated_data_bin[label].append(bin_features)
            separated_data_cat[label].append(cat_features)
            all_continuous_features.append(cont_features)
            self.class_counts[label] += 1
            
            for i, val in enumerate(cat_features):
                global_cat_uniques[i].add(val)

        self.num_classes = len(self.class_counts)

        # 2. Obliczanie globalnego wygładzania wariancji dla cech ciągłych
        global_variances = []
        if total_samples > 1 and all_continuous_features:
            global_columns = list(zip(*all_continuous_features))
            for col in global_columns:
                mean = sum(col) / len(col)
                var = sum((x - mean) ** 2 for x in col) / (len(col) - 1)
                global_variances.append(var)
        else:
            global_variances = [0.0] * len(all_continuous_features[0])
        
        epsilons = [max(1e-4 * var, 1e-9) for var in global_variances]

        # 3. Trening parametrów dla każdej klasy
        for label in self.class_counts:
            self.class_priors[label] = (self.class_counts[label] + 1) / (total_samples + self.num_classes)
            class_size = self.class_counts[label]
            
            # 1. Trening: Ciągłe (Gauss)
            cont_columns = zip(*separated_data_cont[label])
            cont_stats = []
            for i, column in enumerate(cont_columns):
                mean = sum(column) / len(column)
                if len(column) > 1:
                    variance = sum((x - mean) ** 2 for x in column) / (len(column) - 1)
                else:
                    variance = 0.0
                cont_stats.append((mean, variance + epsilons[i]))
            self.continuous_stats[label] = cont_stats

            # 2. Trening: Binarne (Bernoulli)
            bin_columns = zip(*separated_data_bin[label])
            bin_probs = []
            alpha_bin = 1.0
            for column in bin_columns:
                ones_count = sum(column)
                prob_one = (ones_count + alpha_bin) / (class_size + 2 * alpha_bin)
                bin_probs.append(prob_one)
            self.binary_probs[label] = bin_probs

            # 3. Trening: Kategoryczne (Multinomial)
            cat_columns = zip(*separated_data_cat[label])
            self.cat_probs[label] = {}
            self.cat_unseen_probs[label] = {}
            alpha_cat = 1.0
            
            for i, column in enumerate(cat_columns):
                counts = defaultdict(int)
                for val in column:
                    counts[val] += 1
                
                v_total = len(global_cat_uniques[i])
                self.cat_probs[label][i] = {}
                
                for val in global_cat_uniques[i]:
                    prob = (counts[val] + alpha_cat) / (class_size + alpha_cat * v_total)
                    self.cat_probs[label][i][val] = prob
                
                self.cat_unseen_probs[label][i] = alpha_cat / (class_size + alpha_cat * v_total)

    def predict(self, entry: RussianFlatProps) -> str:
        cont_features, bin_features, cat_features = self.extract_features(entry)
        
        best_label = "unknown"
        max_log_prob = -float('inf')
        
        for label in self.class_counts:
            log_prob = math.log(self.class_priors[label])
            
            # 1. Ciągłe (Gauss)
            stats = self.continuous_stats[label]
            for x, (mean, var) in zip(cont_features, stats):
                log_pdf = -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
                log_prob += log_pdf
                
            # 2. Binarne (Bernoulli)
            probs = self.binary_probs[label]
            for x, p_one in zip(bin_features, probs):
                if x == 1.0:
                    log_prob += math.log(p_one)
                else:
                    log_prob += math.log(1.0 - p_one)
                    
            # 3. Kategoryczne (Multinomial)
            for i, val in enumerate(cat_features):
                prob = self.cat_probs[label][i].get(val, self.cat_unseen_probs[label][i])
                log_prob += math.log(prob)
                
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label
                
        return best_label
    
    @staticmethod
    def extract_features(entry: RussianFlatProps) -> tuple[tuple[float, ...], tuple[float, ...], tuple]:
        # Wartości ciągłe (logarytmicznie przekształcone)
        continuous = (
            math.log(float(entry.area) + 1e-5), 
            math.log(float(entry.kitchen_area) + 1e-5),
        )
        
        # Wartości binarne
        binary = (
           1.0 if entry.new_building else 0.0,
        )
        
        # Wartości kategoryczne
        categorical = (
            int(entry.publish_year),
            int(entry.publish_month),
            int(entry.level),
            int(entry.levels),
            int(entry.rooms),
            int(entry.region_id),
            str(entry.building_type),
            f"{round(float(entry.geo_lat), 1)}_{round(float(entry.geo_lon), 1)}", # 11km x 11km
        )
        
        return continuous, binary, categorical