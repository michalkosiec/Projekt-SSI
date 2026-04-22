from typing import NamedTuple
from datetime import date, datetime, time

# building_type: 0 - other | 1 - panel | 2 - monolithic | 3 - brick | 4 - block | 5 - wood
# object_type: 1 - secondary real estate market | 11 - new building

# a raw csv record as a class, for internal use only
class RussianFlatCsv(NamedTuple):
    price: int
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
    def from_csv_line(line: str) -> 'RussianFlatCsv':
        parts = line.strip().split(",")

        dt = datetime.strptime(
            f"{parts[1]} {parts[2]}",
            "%Y-%m-%d %H:%M:%S"
        )

        return RussianFlatCsv(
            price=int(parts[0]),
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

# processed data, contains russian flat properties (except price)
class RussianFlatProps(NamedTuple):
    publish_year: int
    publish_month: int
    geo_lat: float
    geo_lon: float
    region_id: str
    building_type: str
    level: int
    levels: int
    rooms: int
    area: float
    kitchen_area: float
    new_building: bool

# a record with price range and russian flat data (basically label-data pair)
class RussianFlat(NamedTuple):
    price_range: str
    data: RussianFlatProps

    @staticmethod
    def from_csv_file(filename: str, skip: int, record_limit: int | None=None) -> list['RussianFlat']:
        flats = []
        with open(filename, "r") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                if record_limit is not None and i >= skip + record_limit:
                    break
                flat = RussianFlat.from_csv_line(line)
                flats.append(flat)
        return flats

    @staticmethod
    def from_csv_line(line: str) -> 'RussianFlat':
        flat_csv = RussianFlatCsv.from_csv_line(line)
        return RussianFlat(
            price_range=RussianFlat.price_string(flat_csv.price),
            data=RussianFlatProps(
                publish_year=flat_csv.date_value.year,
                publish_month=flat_csv.date_value.month,
                geo_lat=flat_csv.geo_lat,
                geo_lon=flat_csv.geo_lon,
                region_id=str(flat_csv.region), # this is not a string, but should be treated as string
                building_type=RussianFlat.building_type_string(flat_csv.building_type),
                level=flat_csv.level,
                levels=flat_csv.levels,
                rooms=flat_csv.rooms,
                area=flat_csv.area,
                kitchen_area=flat_csv.kitchen_area,
                new_building=flat_csv.object_type == 11
            )
        )

    # converts price into string
    @staticmethod
    def price_string(price: int) -> str: # TODO: adjust based on data analysis
        if price < 0: return "negative"
        if price < 1_000_000: return "0-1M"
        if price < 2_000_000: return "1M-2M"
        if price < 3_000_000: return "2M-3M"
        if price < 4_000_000: return "3M-4M"
        if price < 5_000_000: return "4M-5M"
        return "5M+"

    # converts building type into string
    @staticmethod
    def building_type_string(building_type: int) -> str:
        if building_type == 0: return "other"
        if building_type == 1: return "panel"
        if building_type == 2: return "monolithic"
        if building_type == 3: return "brick"
        if building_type == 4: return "block"
        if building_type == 5: return "wood"
        return "unknown"