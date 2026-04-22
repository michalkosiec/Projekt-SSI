from russian_flat import RussianFlat, RussianFlatProps
from base_model import BaseModel

class NaiveBayes(BaseModel):
    def train(self, training_data: list[RussianFlat]):
        pass

    def predict(self, entry: RussianFlatProps) -> str:
        return "unknown"