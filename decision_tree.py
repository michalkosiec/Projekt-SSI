import math
import random
from collections import Counter
from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

MAX_TRAIN_SAMPLES = 5_000
MAX_DEPTH = 13
MIN_SAMPLES_LEAF = 10


class _Node:
    __slots__ = ('feat', 'threshold', 'cat_val', 'is_cat', 'left', 'right', 'label')

    def __init__(self):
        self.feat = None
        self.threshold = None
        self.cat_val = None
        self.is_cat = False
        self.left = None
        self.right = None
        self.label = None


class DecisionTree(BaseModel):
    def __init__(self, max_depth: int = MAX_DEPTH, min_samples: int = MIN_SAMPLES_LEAF):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: _Node | None = None

    def name(self) -> str:
        return "Drzewo decyzyjne"

    def train(self, training_data: list[RussianFlat]):
        print(f"Trenuję drzewo na {min(len(training_data), MAX_TRAIN_SAMPLES)} próbkach...")
        if len(training_data) > MAX_TRAIN_SAMPLES:
            data = random.Random(42).sample(training_data, MAX_TRAIN_SAMPLES)
        else:
            data = list(training_data)

        X = [_extract(flat.data) for flat in data]
        y = [flat.price_range for flat in data]
        self.root = self._build(X, y, 0)

    def predict(self, entry: RussianFlatProps) -> str:
        if self.root is None:
            return "unknown"
        return _traverse(self.root, _extract(entry))

    def _build(self, X: list, y: list, depth: int) -> _Node:
        node = _Node()
        counts = Counter(y)
        node.label = counts.most_common(1)[0][0]

        if depth >= self.max_depth or len(y) <= self.min_samples or len(counts) == 1:
            return node

        best = self._best_split(X, y, len(y))
        if best is None:
            return node

        feat, threshold, cat_val, is_cat = best
        node.feat = feat
        node.threshold = threshold
        node.cat_val = cat_val
        node.is_cat = is_cat

        if is_cat:
            mask = [x[feat] == cat_val for x in X]
        else:
            mask = [x[feat] <= threshold for x in X]

        X_l = [x for x, m in zip(X, mask) if m]
        y_l = [lbl for lbl, m in zip(y, mask) if m]
        X_r = [x for x, m in zip(X, mask) if not m]
        y_r = [lbl for lbl, m in zip(y, mask) if not m]

        if not X_l or not X_r:
            return node

        node.left = self._build(X_l, y_l, depth + 1)
        node.right = self._build(X_r, y_r, depth + 1)
        node.label = None
        return node

    def _best_split(self, X: list, y: list, n: int):
        base_gini = _gini_counter(Counter(y), n)
        best_gain = 1e-10
        best = None

        for feat in range(len(X[0])):
            vals = [x[feat] for x in X]

            if isinstance(vals[0], str):
                for cat_val in set(vals):
                    l_labels = [lbl for v, lbl in zip(vals, y) if v == cat_val]
                    r_labels = [lbl for v, lbl in zip(vals, y) if v != cat_val]
                    nl, nr = len(l_labels), len(r_labels)
                    if nl < self.min_samples or nr < self.min_samples:
                        continue
                    gain = base_gini - (nl / n * _gini(l_labels) + nr / n * _gini(r_labels))
                    if gain > best_gain:
                        best_gain = gain
                        best = (feat, None, cat_val, True)
            else:
                pairs = sorted(zip(vals, y))
                right_cnt = Counter(y)
                left_cnt: Counter = Counter()
                nl, nr = 0, n
                prev = None

                for val, lbl in pairs:
                    if prev is not None and val != prev and nl >= self.min_samples and nr >= self.min_samples:
                        gain = base_gini - (
                            nl / n * _gini_counter(left_cnt, nl) +
                            nr / n * _gini_counter(right_cnt, nr)
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best = (feat, (prev + val) / 2.0, None, False)
                    left_cnt[lbl] += 1
                    right_cnt[lbl] -= 1
                    if right_cnt[lbl] == 0:
                        del right_cnt[lbl]
                    nl += 1
                    nr -= 1
                    prev = val

        return best


def _gini_counter(counter: Counter, n: int) -> float:
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in counter.values())


def _gini(labels: list) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in Counter(labels).values())


def _extract(entry: RussianFlatProps) -> list:
    return [
        math.log(float(entry.area) + 1e-5),
        math.log(float(entry.kitchen_area) + 1e-5),
        float(entry.geo_lat),
        float(entry.geo_lon),
        float(entry.level),
        float(entry.levels),
        float(entry.rooms),
        float(entry.publish_year),
        float(entry.publish_month),
        1.0 if entry.new_building else 0.0,
        float(entry.region_id),
        str(entry.building_type),
    ]


def _traverse(node: _Node, features: list) -> str:
    if node.label is not None:
        return node.label
    val = features[node.feat]
    go_left = (val == node.cat_val) if node.is_cat else (val <= node.threshold)
    return _traverse(node.left if go_left else node.right, features)
