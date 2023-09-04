import enum
import typing

_CACHE_PREFIX = "-+cached+-"

T = typing.TypeVar("T", bound="EnumIdx")

class EnumIdx(enum.Enum):

    @classmethod
    def _idx_map(cls: typing.Type[T]) -> typing.Mapping[int, T]:
        attr_name = _CACHE_PREFIX + "-idx-map"

        if hasattr(cls, attr_name):
            return getattr(cls, attr_name)

        value = {i: e for i, e in enumerate(cls)}
        setattr(cls, attr_name, value)
        return value

    @classmethod
    def _idx_map_rev(cls: typing.Type[T]) -> typing.Mapping[T, int]:
        attr_name = _CACHE_PREFIX + "-idx-map-rev"

        if hasattr(cls, attr_name):
            return getattr(cls, attr_name)

        value = {e: i for i, e in enumerate(cls)}
        setattr(cls, attr_name, value)
        return value

    def idx(self):
        return type(self)._idx_map_rev()[self]

    @classmethod
    def all_indices(cls: typing.Type[T]):
        return list(range(len(cls)))

    @classmethod
    def from_idx(cls: typing.Type[T], idx: int):
        return cls._idx_map()[idx]

    @classmethod
    def from_indices(cls: typing.Type[T], indices: typing.Iterable[int]):
        mapper = cls._idx_map()
        return [mapper[idx] for idx in indices]
