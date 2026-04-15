from enum import Enum

class PriceRange(Enum):
    RANGE_0_1M = "0-1M"
    RANGE_1M_2M = "1M-2M"
    RANGE_2M_3M = "2M-3M"
    RANGE_3M_4M = "3M-4M"
    RANGE_4M_5M = "4M-5M"
    RANGE_5M_PLUS = "5M+"

    @staticmethod
    def from_price(price: int) -> 'PriceRange':
        if price < 1_000_000:
            return PriceRange.RANGE_0_1M
        elif price < 2_000_000:
            return PriceRange.RANGE_1M_2M
        elif price < 3_000_000:
            return PriceRange.RANGE_2M_3M
        elif price < 4_000_000:
            return PriceRange.RANGE_3M_4M
        elif price < 5_000_000:
            return PriceRange.RANGE_4M_5M
        else:
            return PriceRange.RANGE_5M_PLUS