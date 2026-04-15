from typing import NamedTuple
from datetime import date, datetime, time
from price_range import PriceRange

class RussianFlat(NamedTuple):
    price: int
    price_range: PriceRange
    date_value: date
    time_value: time
    geo_lat: float
    geo_lon: float
    region: int
    building_type: int
    level: int
    levels: int
    rooms: int
    area: float
    kitchen_area: float
    object_type: int

    @staticmethod
    def from_csv_line(line: str) -> 'RussianFlat':
        parts = line.strip().split(",")

        dt = datetime.strptime(
            f"{parts[1]} {parts[2]}",
            "%Y-%m-%d %H:%M:%S"
        )

        price = int(parts[0])
        price_range = PriceRange.from_price(price)

        return RussianFlat(
            price=price,
            price_range=price_range,
            date_value=dt.date(),
            time_value=dt.time(),
            geo_lat=float(parts[3]),
            geo_lon=float(parts[4]),
            region=int(parts[5]),
            building_type=int(parts[6]),
            level=int(parts[7]),
            levels=int(parts[8]),
            rooms=int(parts[9]),
            area=float(parts[10]),
            kitchen_area=float(parts[11]),
            object_type=int(parts[12])
        )