import os
from utils import Utils
from russian_flat import RussianFlat
from preprocessor import Preprocessor
from data_analysis import DataAnalysis
from base_model import BaseModel
from dummy_model import DummyModel
from knn import Knn
from naive_bayes import NaiveBayes
from decision_tree import DecisionTree

BORDER = 60 * "-"

DATAFILE = "all_v2.csv"
OUT_DIR_RAW = "output_raw/"
OUT_DIR_CLEAN = "output_clean/"

TESTMODE = True # enable running on smaller dataset (10000 lines) for testing

LIMIT = 50000 if TESTMODE else None
EXPECTED_ROWS = 50000 if TESTMODE else 5477006

# currently trained model
current_model: BaseModel = DummyModel()

# collected data
raw_flats: list[RussianFlat]
clean_flats: list[RussianFlat]
training_data: list[RussianFlat]
test_data: list[RussianFlat]

def pause(clear: bool = False):
    print("Naciśnij Enter, aby kontynuować...")
    input()
    if clear:
        os.system("cls" if os.name == "nt" else "clear")

def menu() -> bool:
    global current_model
    
    print(BORDER)
    print("Obecnie wytrenowany model: " + current_model.name())
    print(BORDER)
    print("1 - Uruchom analizę danych")
    print("2 - Trenuj model (KNN)")
    print("3 - Trenuj model (Naiwny Bayes)")
    print("4 - Trenuj model (Drzewo decyzyjne)")
    print("5 - Uruchom analizę modelu (" + current_model.name() + ")")
    print("6 - Zakończ program")
    choice = input("> ")

    def train_and_report(model):
        global current_model
        current_model = model
        current_model.train(training_data)
        print(f"Model {current_model.name()} wytrenowany.")

    if choice == "1":
        DataAnalysis.analyse_data(raw_flats, OUT_DIR_RAW)
        DataAnalysis.analyse_data(clean_flats, OUT_DIR_CLEAN)
    if choice == "2": train_and_report(Knn())
    if choice == "3": train_and_report(NaiveBayes())
    if choice == "4": train_and_report(DecisionTree())
    if choice == "5": DataAnalysis.analyse_results(
        [current_model.predict(flat.data) for flat in test_data], # predictions
        [flat.price_range for flat in test_data] # real price ranges
    )
    
    abort = choice == "6"
    if not abort:
        pause(clear=True)
    
    return not abort

def main():
    global raw_flats, clean_flats, training_data, test_data

    print(BORDER)
    print("Rozpoczęto wstępne przetwarzanie danych...")

    raw_flats = RussianFlat.from_csv_file(DATAFILE, skip=1, record_limit=LIMIT)
    print(f"Załadowano mieszkania w liczbie {len(raw_flats)}.")

    if len(raw_flats) != EXPECTED_ROWS:
        print(f"UWAGA: Oczekiwano {EXPECTED_ROWS} wierszy, ale załadowano {len(raw_flats)}.")

    if len(raw_flats) == 0:
        print("Nie można kontynuować bez danych. Zamykam program.")
        return

    outlier_indexes = Preprocessor.outlier_indexes([flat.data for flat in raw_flats])
    delete_count = len(outlier_indexes)
    delete_ratio = (delete_count / len(raw_flats)) * 100
    print(f"Zidentyfikowano {len(outlier_indexes)} outlierów ({delete_ratio:.2f}%).")

    clean_flats = Utils.list_without_indexes(raw_flats, outlier_indexes)
    print(f"Usunięto outliery. Pozostało {len(clean_flats)} mieszkań.")

    training_data, test_data = Preprocessor.divide_data_randomly(clean_flats, ratio=0.8)
    print(f"Podzielono na zbiory treningowy ({len(training_data)}) oraz testowy ({len(test_data)}).")

    pause(clear=True)

    while menu():
        pass

if __name__ == "__main__":
    main()