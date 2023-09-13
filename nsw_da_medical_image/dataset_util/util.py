import enum
import typing

_CACHE_PREFIX = "-+cached+-"

T = typing.TypeVar("T", bound="EnumIdx")


class EnumIdx(enum.Enum):
    """All enums deriving from 'EnumIdx' have the methods

        def idx(self) -> int:
            ...
        def all_indices() -> list[int]:
            ...
        def from_idx(idx: int) -> T:
            ...

        def from_indices(indices: typing.Iterable[int]) -> list[T]:
            ...
    """

    @classmethod
    def _idx_map(cls: typing.Type[T]) -> typing.Mapping[int, T]:
        """_idx_map : a (private) function that returns a map from the position
        of each value in the enum to the actual enum value as an enum object.

        Returns:
            typing.Mapping[int, T]: mapping from position to object
        """

        # use an identifier that is impossible to type for the cached attribute
        attr_name = _CACHE_PREFIX + "-idx-map"

        # cache-hit
        if hasattr(cls, attr_name):
            return getattr(cls, attr_name)

        # generate the mapping
        value = {i: e for i, e in enumerate(cls)}

        # cache-miss : store the value at the right attribute and return it
        setattr(cls, attr_name, value)
        return value

    @classmethod
    def _idx_map_rev(cls: typing.Type[T]) -> typing.Mapping[T, int]:
        """_idx_map_rev : a (private) function that returns a map from the enum
        object to its position in the enum.

        Returns:
            typing.Mapping[T, int]: mapping from object to position
        """

        # use an identifier that is impossible to type for the cached attribute
        attr_name = _CACHE_PREFIX + "-idx-map-rev"

        # cache-hit
        if hasattr(cls, attr_name):
            return getattr(cls, attr_name)

        # generate the mapping
        value = {e: i for i, e in enumerate(cls)}

        # cache-miss : store the value at the right attribute and return it
        setattr(cls, attr_name, value)
        return value

    def idx(self):
        "equivalent list(type(self)).index(self), but O(1)"
        return type(self)._idx_map_rev()[self]

    @classmethod
    def all_indices(cls: typing.Type[T]):
        "list of all valid indices for this enum"
        return list(range(len(cls)))

    @classmethod
    def from_idx(cls: typing.Type[T], idx: int):
        "the value at position `idx` in T"
        return cls._idx_map()[idx]

    @classmethod
    def from_indices(cls: typing.Type[T], indices: typing.Iterable[int]):
        "equivalent to [T.from_idx(i) for i in indices]"
        mapper = cls._idx_map()
        return [mapper[idx] for idx in indices]
