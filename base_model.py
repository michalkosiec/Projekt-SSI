from abc import ABC, abstractmethod
from russian_flat import RussianFlat, RussianFlatProps

class BaseModel(ABC):
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def train(self, training_data: list[RussianFlat]):
        pass

    @abstractmethod
    def predict(self, entry: RussianFlatProps) -> str:
        pass