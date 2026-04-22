class Utils:

    @staticmethod
    def list_without_indexes[T](lst: list[T], indexes: list[int]) -> list[T]:
        return [item for i, item in enumerate(lst) if i not in indexes]

    @staticmethod
    # returns list of attribute names of given record that are of given types
    def get_attrs_of(record, types: tuple[type, ...]) -> list[str]:
        all_names = record._fields if hasattr(record, "_fields") else record.__dict__.keys()
        return [
            field for field in all_names
            if isinstance(getattr(record, field), types)
        ]