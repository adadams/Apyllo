from typing import Final

REARTH_IN_CM: Final[float] = 6.371e8
RJUPITER_IN_REARTH: Final[float] = 11.2
PARSEC_IN_REARTH: Final[float] = 4.838e9


def radius(radius_in_earth_radii: float) -> float:
    return radius


def radius_distance_ratio(
    radius_to_distance_ratio: float, distance_in_parsecs: float
) -> float:
    return (
        10**radius_to_distance_ratio
        * distance_in_parsecs
        * PARSEC_IN_REARTH
        * REARTH_IN_CM
    )


def squared_radius_distance_ratio(
    squared_radius_to_distance_ratio: float,
) -> float:
    raise NotImplementedError
