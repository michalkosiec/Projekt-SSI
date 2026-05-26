import os
import shutil
import statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from russian_flat import RussianFlat
from table_builder import TableBuilder
from utils import Utils

class DataAnalysis:

    @staticmethod
    def analyse_data(data: list[RussianFlat], output_dir: str):
        if not data:
            print("Brak danych do analizy.")
            return

        # 1. Zarządzanie katalogiem wyjściowym
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        # 2. Analiza rozkładu klas cenowych
        print(f"\n--- Analiza danych ({output_dir}) - liczba rekordów: {len(data)} ---")
        distribution = {}
        for flat in data:
            price_range = flat.price_range
            distribution[price_range] = distribution.get(price_range, 0) + 1
        
        builder_dist = TableBuilder(["Przedział cenowy", "Liczba mieszkań", "Procentowo"])
        total = len(data)
        for price_range, count in sorted(distribution.items()):
            percentage = (count / total) * 100
            builder_dist.add_row([price_range, count, f"{percentage:.1f}%"])
        print(builder_dist.build())

        # POBRANIE CECH CIĄGŁYCH
        continuous_names = [
            field for field in Utils.get_attrs_of(data[0].data, (int, float))
            if type(getattr(data[0].data, field)) is not bool
        ]

        # 3. Analiza statystyczna cech ciągłych
        print("\n--- Statystyki cech ciągłych ---")
        builder_stats = TableBuilder(["Cecha", "Min", "Max", "Średnia", "Odch. Stand."])
        for field in continuous_names:
            values = [getattr(entry.data, field) for entry in data]
            min_v = round(min(values), 4)
            max_v = round(max(values), 4)
            avg_v = round(statistics.mean(values), 4)
            stdev_v = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
            builder_stats.add_row([field, min_v, max_v, avg_v, stdev_v])
        print(builder_stats.build())

        # 4. MACIERZ KORELACJI (Pearson)
        print("\n--- Macierz Korelacji (Cechy ciągłe) ---")

        def color_by_corr(value: float, text: str) -> str:
            value = max(-1.0, min(1.0, value))
            intensity = abs(value)
            base = int(255 * (1 - intensity))
            if value >= 0:
                r, g, b = 255, base, base
            else:
                r, g, b = base, base, 255
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            fg = 30 if brightness > 128 else 97
            return f"\x1b[{fg}m\x1b[48;2;{r};{g};{b}m{text}\x1b[0m"

        builder_corr = TableBuilder(["Cecha"] + continuous_names)
        
        for f1 in continuous_names:
            row = [f1]
            vals1 = [getattr(entry.data, f1) for entry in data]
            
            for f2 in continuous_names:
                vals2 = [getattr(entry.data, f2) for entry in data]
                
                if f1 == f2:
                    row.append(color_by_corr(1.0, "1.00"))
                else:
                    try:
                        corr = statistics.correlation(vals1, vals2)
                        row.append(color_by_corr(corr, f"{corr:.2f}"))
                    except statistics.StatisticsError:
                        row.append("N/A")
                        
            builder_corr.add_row(row)
            
        print(builder_corr.build())

        # 5. Wykresy
        print("\n--- Generowanie wykresów ---")
        try:
            # Wykres rozkładu klas cenowych
            fig, ax = plt.subplots(figsize=(8, 5))
            ranges = [price_range for price_range, _ in sorted(distribution.items())]
            counts = [distribution[price_range] for price_range in ranges]
            ax.bar(ranges, counts, color="#4c72b0")
            ax.set_xlabel("Przedział cenowy")
            ax.set_ylabel("Liczba mieszkań")
            ax.set_title("Rozkład liczby mieszkań według przedziałów cenowych")
            ax.set_xticklabels(ranges, rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
            plt.close(fig)

            # Wykres macierzy korelacji
            fig, ax = plt.subplots(figsize=(max(6, len(continuous_names) * 0.8), max(6, len(continuous_names) * 0.8)))
            corr_matrix = []
            for f1 in continuous_names:
                row = []
                vals1 = [getattr(entry.data, f1) for entry in data]
                for f2 in continuous_names:
                    vals2 = [getattr(entry.data, f2) for entry in data]
                    if f1 == f2:
                        row.append(1.0)
                    else:
                        try:
                            row.append(statistics.correlation(vals1, vals2))
                        except statistics.StatisticsError:
                            row.append(0.0)
                corr_matrix.append(row)

            im = ax.imshow(corr_matrix, cmap="bwr", vmin=-1, vmax=1)
            ax.set_xticks(range(len(continuous_names)))
            ax.set_yticks(range(len(continuous_names)))
            ax.set_xticklabels(continuous_names, rotation=45, ha="right")
            ax.set_yticklabels(continuous_names)
            ax.set_title("Macierz korelacji cech ciągłych")
            fig.colorbar(im, ax=ax, label="Korelacja Pearsona")

            # Dodaj wartości procentowe do kwadracików
            for i in range(len(continuous_names)):
                for j in range(len(continuous_names)):
                    corr_value = corr_matrix[i][j]
                    percent_text = f"{corr_value * 100:.0f}%"
                    ax.text(j, i, percent_text,
                            ha="center", va="center",
                            color="black" if abs(corr_value) < 0.5 else "white",
                            fontsize=8)

            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=150)
            plt.close(fig)

            # 6. Wykresy punktowe pojedynczych cech
            print("Generowanie wykresów punktowych cech ciągłych...")
            for field in continuous_names:
                values = [getattr(entry.data, field) for entry in data]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(range(len(values)), values, s=10, color="#2a9d8f", alpha=0.7)
                ax.set_xlabel("Indeks rekordu")
                ax.set_ylabel(field)
                ax.set_title(f"Wykres punktowy cechy: {field}")
                fig.tight_layout()
                safe_name = field.replace(" ", "_").replace("/", "_")
                fig.savefig(os.path.join(output_dir, f"scatter_{safe_name}.png"), dpi=150)
                plt.close(fig)

            print(f"Wykresy zapisane do katalogu: \"{output_dir}\".")
        except Exception as e:
            print("Błąd przy generowaniu wykresów:", e)

        print("\nAnaliza danych zakończona.")

    @staticmethod
    def analyse_results(predicted: list[str], real: list[str]):
        # [Istniejący kod analyse_results pozostaje bez zmian]
        if not predicted or not real or len(predicted) != len(real):
            print("Błąd: Dane przewidywane i rzeczywiste nie pasują do siebie.")
            return

        print("\n--- Podsumowanie wyników modelu ---")
        correct = sum(1 for p, r in zip(predicted, real) if p == r)
        total = len(predicted)
        accuracy = (correct / total) * 100

        builder = TableBuilder(["Metryka", "Wynik"])
        builder.add_row(["Poprawne predykcje", f"{correct} / {total}"])
        builder.add_row(["Skuteczność (Accuracy)", f"{accuracy:.2f}%"])
        print(builder.build())

        print("\n--- Macierz korelacji wyników (confusion matrix) ---")
        labels = sorted(set(real) | set(predicted))
        confusion = {real_label: {pred_label: 0 for pred_label in labels} for real_label in labels}
        for p, r in zip(predicted, real):
            confusion[r][p] += 1

        max_count = max(
            confusion[r][p]
            for r in labels
            for p in labels
        ) if labels else 0

        def color_by_value(count: int, text: str) -> str:
            intensity = count / max_count if max_count else 0.0
            base = int(255 * (1 - intensity))
            r, g, b = 255, base, base
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            fg = 30 if brightness > 128 else 97
            return f"\x1b[{fg}m\x1b[48;2;{r};{g};{b}m{text}\x1b[0m"

        builder_conf = TableBuilder(["Rzeczywiste\\Przewidziane"] + labels)
        for real_label in labels:
            row = [real_label]
            for pred_label in labels:
                count = confusion[real_label][pred_label]
                row.append(color_by_value(count, str(count)))
            builder_conf.add_row(row)
        print(builder_conf.build())

        print("Analiza wyników modelu zakończona.")