"""Constants for IMAP-Lo."""

from dataclasses import dataclass
from typing import ClassVar, NamedTuple


class PivotAngleSpec(NamedTuple):
    """
    Pivot angle [degrees] and associated settings for a nominal pivot index.

    Attributes
    ----------
    nominal : float
        Nominal (commanded) pivot angle.
    min : float
        Lower bound of the acceptable pivot angle range.
    max : float
        Upper bound of the acceptable pivot angle range.
    bg_rate_ram : float, optional
        RAM background-rate threshold [counts/s] for this pivot angle. ``None``
        if no pivot-specific value is known, in which case
        ``LoConstants.THRESHOLD_BG_RATE_RAM_DEFAULT`` applies.
    bg_rate_anti_ram : float, optional
        Anti-RAM background-rate threshold [counts/s] for this pivot angle.
        ``None`` if no pivot-specific value is known, in which case
        ``LoConstants.THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT`` applies.
    """

    nominal: float
    min: float
    max: float
    bg_rate_ram: float | None = None
    bg_rate_anti_ram: float | None = None


@dataclass(frozen=True)
class LoConstants:
    """Constants for Lo which can be used across different levels."""

    # Expected pivot angle [degrees] for pointing sets for generating map products.
    PSET_PIVOT_ANGLE: float = 90.0

    # Ion species tracked. "H" is mandatory (and should be the first element);
    # any others for which we have histrates may be added here.
    ELEMS = ("H", "O")

    # Hours into the day (UTC) for HK data to calculate median for pivot angle
    # estimation.
    PIVOT_HK_HOUR_RANGE: tuple[float, float] = (0.5, 22.5)

    N_CYCLE_SUM: int = 1  # Granularity of goodtime boundaries
    N_CYCLE_AVE: int = 7  # Cycles to average over when estimating background rates
    N_ESA_LEVELS: int = 7  # Total number of ESA levels
    N_SPINS_PER_ESA_LEVEL: int = 4  # Spins per ESA step within one histogram cycle
    N_SPIN_ANGLE_BINS: int = 60  # Number of angular bins within a spin

    # Nominal spin period [s]. True spin duration is NOT 15 seconds.
    NOMINAL_SPIN_PERIOD_SEC: float = 15.0

    # One histogram accumulation cycle duration [s]
    HISTOGRAM_CYCLE_EPOCHS: int = (
        N_ESA_LEVELS * N_SPINS_PER_ESA_LEVEL * int(NOMINAL_SPIN_PERIOD_SEC)
    )
    RAM_ESA_LEVELS: tuple[int, ...] = (
        6,
        7,
    )  # ESA levels for RAM estimation (1-indexed)

    # Histogram angular bins (0-indexed) corresponding to the RAM and anti-RAM look
    # directions
    RAM_HISTOGRAM_BINS: tuple[slice, ...] = (slice(0, 20), slice(50, 60))
    ANTI_RAM_HISTOGRAM_BINS: tuple[slice, ...] = (slice(20, 50),)

    # Nominal background rates [counts/s] for each species
    BG_RATES: ClassVar[dict[str, float]] = {"H": 0.0014925, "O": 0.000136635}
    # When no exposure is available, scale the nominal rate down as a conservative
    # estimate.
    BG_RATE_FALLBACK_SCALE: ClassVar[dict[str, float]] = {"H": 1.0, "O": 0.3}
    # Minimum non-zero background rate floor = nominal / divisor
    BG_RATE_FLOOR_DIVISOR: ClassVar[dict[str, float]] = {"H": 50.0, "O": 150.0}

    # Pivot angle range [degrees] keyed by nominal pivot index, along with the
    # background-rate thresholds [counts/s] that apply within that range.
    # The first range containing the pivot angle (min <= pivot <= max) is used;
    # where no thresholds are given (or no range matches),
    # THRESHOLD_BG_RATE_RAM_DEFAULT / THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT apply.
    PIVOT_ANGLES: ClassVar[dict[int, PivotAngleSpec]] = {
        1: PivotAngleSpec(60.0, 55.0, 65.0),
        2: PivotAngleSpec(75.0, 70.0, 80.0, 0.035, 0.0175),
        3: PivotAngleSpec(90.0, 85.0, 95.0, 0.028, 0.014),
        4: PivotAngleSpec(105.0, 100.0, 110.0, 0.0224, 0.0112),
        5: PivotAngleSpec(120.0, 115.0, 125.0),
        6: PivotAngleSpec(135.0, 130.0, 140.0),
        7: PivotAngleSpec(148.0, 143.0, 153.0),
        8: PivotAngleSpec(160.0, 155.0, 165.0),
    }

    # Default background-rate thresholds [counts/s] when no pivot range matches.
    THRESHOLD_BG_RATE_RAM_DEFAULT: float = 0.028
    THRESHOLD_BG_RATE_ANTI_RAM_DEFAULT: float = 0.014

    # Maximum time gap [s] between consecutive histogram epochs before treating them as
    # separate intervals.
    DELAY_MAX: int = 100
    # Fraction of each cycle duration that contributes actual exposure.
    EXPOSURE_FACTOR: float = 0.5
    # Padding [s] added to begin/end of each goodtime interval to ensure complete
    # cycles are covered at interval edges.
    GOODTIME_PADDING: float = 2.0

    # Star-sensor spin-angle binning offset (fractional bin-index shift used when
    # computing sample centers), keyed by the IFB star-sync housekeeping state
    # (ifb_ctrl_star_sync). Flight software 4.8 enabled star sync ("EN"),
    # switching from binning to the bin center (+0.5) to the left edge (+0.0).
    STAR_BIN_OFFSET_BY_SYNC: ClassVar[dict[str | None, float]] = {
        "DS": 0.5,  # star sync disabled (pre FSW 4.8)
        "EN": 0.0,  # star sync enabled (FSW 4.8+)
    }

    # Number of ending bins to exclude from each star-sensor profile average.
    STAR_END_BINS_TO_EXCLUDE: int = 2
    # Minimum COUNT value for a star-sensor record to be considered valid.
    STAR_MIN_COUNT_THRESHOLD: int = 700
