SIZE = 200_000

lines = []
with open("all_v2.csv", "r") as f:
    for line in f:
        lines.append(line.strip())

header_line = lines[0]
lines = lines[1:]

import random
random.shuffle(lines)

lines = lines[:SIZE]

with open("filtered_v3.csv", "w") as f:
    f.write(header_line + "\n")
    for line in lines:
        f.write(line + "\n")

print("Done!")