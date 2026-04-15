from typing import Callable

def parse_csv[T](filename: str, parser: Callable[[str], T], skip: int=0, record_limit: int | None=None) -> list[T]:
    flats = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if record_limit is not None and i >= skip + record_limit:
                break
            flat = parser(line)
            flats.append(flat)
    return flats