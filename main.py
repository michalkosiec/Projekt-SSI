from parser import parse_csv
from russian_flat import RussianFlat
from table_builder import TableBuilder

DATAFILE = "all_v2.csv"
TESTMODE = True # enable running on smaller dataset (10000 lines) for testing

LIMIT = 10000 if TESTMODE else None

flats = parse_csv(DATAFILE, RussianFlat.from_csv_line, skip=1, record_limit=LIMIT)
print(f"Number of flats: {len(flats)} (should be 5477006)")
print("Avg. area: ", sum(flat.area for flat in flats) / len(flats))

builder = TableBuilder(["", "color", "age", "very long header"])
for i in range(5):
    builder.add_row([f"{i}", f" tata {i}", f"mama {i}", f"{i ^ 5}"])

print(builder.build())