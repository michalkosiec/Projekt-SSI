from russian_flat import RussianFlat
from table_builder import TableBuilder

class DataAnalysis:

    @staticmethod
    def analyse_data(data: list[RussianFlat], output_dir: str):
        # create some graphs here etc. - save them to files in output_dir
        # average, stdevs, medians of columns etc. etc. - print them to console

        # IMPORTANT: ensure that the directory exists
        # IMPORTANT: delete all existing files in output_dir before saving new ones

        print(f"Analiza danych zakończona.")
        print(f"Dane graficzne zapisano do katalogu \"{output_dir}\".")

    @staticmethod
    def analyse_results(predicted: list[str], real: list[str]):
        # build some tables here (you may use TableBuilder class for that)
        # calculate model efficiency etc.

        # use builder like this:
        # you may use any object that can be converted to string
        builder = TableBuilder(["C1", "C2", "C3"])
        builder.add_row([11, 12, 13])
        builder.add_row([21, 22, 23])
        builder.add_row([31, 32, 33])
        print(builder.build())

        print("Analiza wyników modelu zakończona.")