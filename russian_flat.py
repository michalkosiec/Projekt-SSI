import statistics
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

    price_dividers: list[float] = []

    @classmethod
    def set_price_dividers(cls, prices: list[int], num_bins: int = 5):
        cls.price_dividers = statistics.quantiles(prices, n=num_bins)

    @classmethod
    def get_dynamic_price_range(cls, price: int) -> str:
        if not cls.price_dividers:
            return "unknown"

        divs = cls.price_dividers
        if price <= divs[0]:
            return f"<= {divs[0]/1_000_000:.1f}M"
            
        for i in range(1, len(divs)):
            if price <= divs[i]:
                return f"{divs[i-1]/1_000_000:.1f}M - {divs[i]/1_000_000:.1f}M"
                
        return f"> {divs[-1]/1_000_000:.1f}M"

    @classmethod
    def from_csv_file(cls, filename: str, skip: int, record_limit: int | None=None, num_bins: int = 5) -> list['RussianFlat']:
        csv_records = []
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                if record_limit is not None and i >= skip + record_limit:
                    break
                csv_records.append(RussianFlatCsv.from_csv_line(line))
        
        if not csv_records:
            return []

        all_prices = [record.price for record in csv_records]
        cls.set_price_dividers(all_prices, num_bins)

        flats = []
        for record in csv_records:
            flats.append(
                cls(
                    price_range=cls.get_dynamic_price_range(record.price),
                    data=RussianFlatProps(
                        publish_year=record.date_value.year,
                        publish_month=record.date_value.month,
                        geo_lat=record.geo_lat,
                        geo_lon=record.geo_lon,
                        region_id=str(record.region),
                        building_type=cls.building_type_string(record.building_type),
                        level=record.level,
                        levels=record.levels,
                        rooms=record.rooms,
                        area=record.area,
                        kitchen_area=record.kitchen_area,
                        new_building=record.object_type == 11
                    )
                )
            )
            
        return flats

    @classmethod
    def from_csv_line(cls, line: str) -> 'RussianFlat':
        flat_csv = RussianFlatCsv.from_csv_line(line)
        return cls(
            price_range=cls.get_dynamic_price_range(flat_csv.price),
            data=RussianFlatProps(
                publish_year=flat_csv.date_value.year,
                publish_month=flat_csv.date_value.month,
                geo_lat=flat_csv.geo_lat,
                geo_lon=flat_csv.geo_lon,
                region_id=str(flat_csv.region),
                building_type=cls.building_type_string(flat_csv.building_type),
                level=flat_csv.level,
                levels=flat_csv.levels,
                rooms=flat_csv.rooms,
                area=flat_csv.area,
                kitchen_area=flat_csv.kitchen_area,
                new_building=flat_csv.object_type == 11
            )
        )

    @staticmethod
    def building_type_string(building_type: int) -> str:
        if building_type == 0: return "other"
        if building_type == 1: return "panel"
        if building_type == 2: return "monolithic"
        if building_type == 3: return "brick"
        if building_type == 4: return "block"
        if building_type == 5: return "wood"
        return "unknown"