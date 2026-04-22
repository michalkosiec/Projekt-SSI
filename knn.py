from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

class Knn(BaseModel):
    def train(self, training_data: list[RussianFlat]):
        pass

    def predict(self, entry: RussianFlatProps) -> str:
        return "unknown"