"""IMAP-Lo L2 data processing."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import generic_filter

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps.ena_maps import (
    PointingSet,
    RectangularSkyMap,
    SkyTilingType,
)
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.corrections import PowerLawFluxCorrector
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo import lo_ancillary
from imap_processing.lo.constants import EsaCalibration
from imap_processing.lo.constants import LoConstants as c  # noqa: N813
from imap_processing.lo.l1c.lo_l1c import compute_pointing_directions
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_spacecraft_to_instrument_spin_phase_offset,
)
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

logger = logging.getLogger(__name__)

# The descriptors of the L1B products a map is built from, one set per pointing.
REQUIRED_PRODUCTS = ("goodtimes", "bgrates", "histrates")

# The map variables accumulated directly from the pointings, before any rates
# or intensities are derived from them.
ACCUMULATED_VARIABLES = ("ena_count", "exposure_factor", "bg_rate_exposure")

# The map variables the ISN mask blanks out: everything derived from the counts
# of the mapped species.
ISN_MASKED_VARIABLES = (
    "ena_count",
    "ena_count_rate",
    "ena_count_rate_stat_uncert",
    "ena_intensity",
    "ena_intensity_stat_uncert",
    "ena_intensity_sys_err",
    "ena_intensity_sys_err_plus",
    "ena_intensity_sys_err_minus",
)

# The fill value of every floating point map variable for the L2 map.
# A pixel holding this is one the map has no measurement for.
FILLVAL_FLOAT = -1.0e31

# The map variables that are filled with FILLVAL_FLOAT where the map was never exposed.
FILLED_VARIABLES = (
    "ena_count_rate",
    "ena_count_rate_stat_uncert",
    "ena_intensity",
    "ena_intensity_stat_uncert",
    "ena_intensity_sys_err",
    "ena_intensity_sys_err_plus",
    "ena_intensity_sys_err_minus",
    "bg_rate",
    "bg_rate_stat_uncert",
    "bg_intensity",
    "bg_intensity_stat_uncert",
)

# The calibration ancillaries shipped with the package.
ANCILLARY_DATA_DIR = Path(__file__).parent.parent / "ancillary_data"


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def lo_l2(
    sci_dependencies: dict[int, dict[str, xr.Dataset]],
    anc_dependencies: list,
    descriptor: str,
) -> list[xr.Dataset]:
    """
    Process IMAP-Lo L1B data into an L2 sky map.

    A map accumulates the histogram counts and exposure of every pointing in
    its window, binned by the sky direction each spin-angle bin was looking in,
    and converts the accumulated counts into an intensity with the instrument's
    geometric factors.

    The inputs are expected to have already been filtered down to the pivot
    angle of the map being made, which is done in pre-processing (see
    ``cli.Lo.pre_processing``) so that the map records only the files it was
    made from as its parents. A combined map, written "ilo" rather than with a
    pivot angle of its own, is filtered by nothing and is given the pointings
    of every pivot angle.

    A combined map is not built by pooling those pointings. Each pivot angle is
    calibrated, corrected and masked on a map of its own, exactly as it would
    be if it had been asked for by its own descriptor, and the finished maps
    are combined onto one grid at the end. Pooling first would force one
    pivot's geometric factors onto all of them, tune the ISN mask on a mixed
    intensity distribution, and blend the RAM projections of pivot angles that
    never looked the same way.

    Parameters
    ----------
    sci_dependencies : dict[int, dict[str, xr.Dataset]]
        The input datasets covering the pointings of the map window, keyed by
        repointing and then by product descriptor.
    anc_dependencies : list
        List of ancillary file paths, read for the efficiency and correction
        factors. The geometric factors and ESA level energies come from the
        ancillaries shipped with the package, in ``ANCILLARY_DATA_DIR``.
    descriptor : str
        The map descriptor to be produced
        (e.g., "l090-ena-h-sf-nsp-ram-hae-6deg-3mo").

    Returns
    -------
    list[xr.Dataset]
        List containing the processed L2 map.

    Raises
    ------
    NotImplementedError
        If a HEALPix map is requested (only rectangular maps supported for Lo),
        or if the map is of a species other than hydrogen.
    ValueError
        If the map is to be Compton-Getting corrected but the ancillary
        dependencies hold no ESA eta fit factors to correct it with, if it is
        to be ISN masked at a pivot angle the mask has no tuning for, or if the
        pivot angles of a combined map were not all taken in the same ESA mode.
    """
    logger.info("Starting IMAP-Lo L2 processing pipeline")

    map_descriptor = MapDescriptor.from_string(descriptor)
    logger.info(f"Processing map for species: {map_descriptor.species}")

    # The Compton-Getting correction reads the source spectrum of each pixel
    # through the ESA transmission factors of the eta fit ancillary. Read up
    # front, so that a map missing it fails before anything is accumulated.
    flux_corrector = (
        _flux_corrector(anc_dependencies) if map_descriptor.cg_corrected else None
    )

    logger.info("Step 1: Loading ancillary data")
    _efficiency_data = load_efficiency_data(anc_dependencies)

    # Only hydrogen maps are supported end to end for now.
    if map_descriptor.species != "h":
        raise NotImplementedError(
            f"Cannot make a map of species {map_descriptor.species} for "
            f"{descriptor}. Only hydrogen maps are supported."
        )

    pointings = _complete_pointings(sci_dependencies)
    logger.info(f"Building {descriptor} from {len(pointings)} pointings")

    # One group per pivot angle, each of which becomes a map of its own. A map
    # named for a pivot angle has only ever the one group.
    groups = _group_pointings_by_pivot(map_descriptor, pointings)

    built = {
        pivot_angle: _build_pivot_map(
            group, map_descriptor, pivot_angle, flux_corrector
        )
        for pivot_angle, group in sorted(groups.items())
    }

    if len(built) == 1:
        sky_map, variables, calibration = next(iter(built.values()))
    else:
        logger.info(f"Combining the maps of pivot angles {sorted(built)}")
        sky_map, variables, calibration = _combine_pivot_maps(built)

    dataset = _build_map_dataset(sky_map, variables, calibration)

    logger.info("IMAP-Lo L2 processing pipeline completed successfully")
    return [
        sky_map.build_cdf_dataset(
            instrument="lo",
            level="l2",
            descriptor=descriptor,
            external_map_dataset=dataset,
        )
    ]


def _build_pivot_map(
    pointings: dict[int, tuple],
    map_descriptor: MapDescriptor,
    pivot_angle: int,
    flux_corrector: PowerLawFluxCorrector | None,
) -> tuple[RectangularSkyMap, dict[str, np.ndarray], EsaCalibration]:
    """
    Build one finished map from the pointings of a single pivot angle.

    This is the whole of a map named for a pivot angle, and one contribution to
    a combined map: the pointings are accumulated onto the grid and every
    correction the descriptor asks for is made, leaving map variables that need
    nothing further done to them.

    Parameters
    ----------
    pointings : dict[int, tuple]
        The (goodtimes, bgrates, histrates) datasets of the pointings taken at
        this pivot angle, keyed by repointing.
    map_descriptor : MapDescriptor
        The parsed descriptor of the map being made.
    pivot_angle : int
        The nominal pivot angle [degrees] of these pointings, which the ISN
        mask is tuned by.
    flux_corrector : PowerLawFluxCorrector | None
        The ESA transmission factors the Compton-Getting correction reads, or
        None if the map is not to be corrected.

    Returns
    -------
    tuple[RectangularSkyMap, dict[str, np.ndarray], EsaCalibration]
        The map the pointings were projected onto, its variables each of shape
        (epoch, esa level, pixel), and the energy response it is binned in.

    Raises
    ------
    NotImplementedError
        If a HEALPix map is requested (only rectangular maps supported for Lo).
    """
    sky_map = map_descriptor.to_empty_map()
    if not isinstance(sky_map, RectangularSkyMap):
        raise NotImplementedError("HEALPix map output not supported for Lo")

    # The mask is tuned per pivot angle. Resolved before anything is
    # accumulated, so that a map the mask cannot be tuned for fails before the
    # work is done.
    isn_mask_parameters = (
        _isn_mask_parameters(pivot_angle) if map_descriptor.isn_masked else None
    )

    # Every pointing of a pivot angle is taken in the same ESA mode, so the
    # last one sets the energy response this map is binned in.
    esa_mode = _get_esa_mode(pointings[max(pointings)][2]) if pointings else 0
    calibration = _esa_calibration(map_descriptor.species, esa_mode)

    # The species sputtering into this map, if it is to be sputter corrected,
    # and the ESA levels it sputters into. Its counts are accumulated on the
    # same grid, alongside the map's own.
    sputter_source, sputter_matrix = (
        _sputter_correction(map_descriptor.species)
        if map_descriptor.sputter_corrected
        else (None, None)
    )
    accumulators = (
        ACCUMULATED_VARIABLES
        + (("sputter_source_count",) if sputter_source else ())
        + (("cos_alpha_exposure",) if map_descriptor.cg_corrected else ())
    )

    _initialize_accumulators(sky_map, calibration.energy, accumulators)

    for repointing, (goodtimes, bgrates, histrates) in sorted(pointings.items()):
        logger.debug(f"Accumulating repoint{repointing:05d}")
        _accumulate_pointing(
            goodtimes,
            bgrates,
            histrates,
            sky_map,
            map_descriptor,
            calibration.energy,
            sputter_source,
        )

    bootstrap_matrix = (
        _bootstrap_correction() if map_descriptor.bootstrap_corrected else None
    )

    variables = _calculate_rates_and_intensities(
        sky_map,
        calibration,
        sputter_matrix,
        bootstrap_matrix,
        flux_corrector,
        isn_mask_parameters,
    )
    return sky_map, variables, calibration


def _combine_pivot_maps(
    built: dict[int, tuple[RectangularSkyMap, dict[str, np.ndarray], EsaCalibration]],
) -> tuple[RectangularSkyMap, dict[str, np.ndarray], EsaCalibration]:
    """
    Combine the finished maps of each pivot angle onto one grid.

    Counts and exposure add. Intensities combine by inverse variance, weighting
    each pivot by what it actually measured, and everything else is exposure
    weighted. A pivot contributes to a pixel only where it has a value to
    report there: one it was exposed for, and that the ISN mask left in place.
    A pixel no pivot reports is left at the fill value rather than averaged
    with one.

    Parameters
    ----------
    built : dict[int, tuple[RectangularSkyMap, dict[str, np.ndarray], EsaCalibration]]
        The map, variables and calibration of each nominal pivot angle, as
        ``_build_pivot_map`` returned them.

    Returns
    -------
    tuple[RectangularSkyMap, dict[str, np.ndarray], EsaCalibration]
        The map carrying the combined result, the combined variables, and the
        energy response they are binned in.

    Raises
    ------
    ValueError
        If the pivot angles were not all taken in the same ESA mode, leaving
        them binned in energies that cannot be combined onto one axis.
    """
    pivot_angles = sorted(built)
    maps = [built[pivot][0] for pivot in pivot_angles]
    per_pivot = [built[pivot][1] for pivot in pivot_angles]
    calibration = built[pivot_angles[0]][2]

    # The pivots have to share an energy axis to be combined onto one. They do
    # when they were flown in the same ESA mode, which is what sets it.
    for pivot in pivot_angles[1:]:
        energy = built[pivot][2].energy
        if not np.array_equal(energy, calibration.energy):
            raise ValueError(
                f"The pivot angles of the combined map were taken in different "
                f"ESA modes: {pivot_angles[0]} is binned in energies "
                f"{calibration.energy.tolist()} keV and {pivot} in "
                f"{energy.tolist()} keV, which cannot be combined."
            )

    exposure = [variables["exposure_factor"] for variables in per_pivot]
    total_exposure = np.sum(exposure, axis=0)

    # A pivot has a background to report wherever it was exposed, and a signal
    # to report where the ISN mask also left the pixel in place.
    exposed = [pivot_exposure > 0 for pivot_exposure in exposure]
    measured = [variables["ena_intensity"] > FILLVAL_FLOAT for variables in per_pivot]
    any_exposed = np.logical_or.reduce(exposed)
    any_measured = np.logical_or.reduce(measured)

    # The exposure each pivot brings to a pixel it reports, which is the weight
    # of every exposure weighted mean below.
    signal_weight = [
        np.where(m, e, 0.0) for m, e in zip(measured, exposure, strict=True)
    ]
    background_weight = [
        np.where(m, e, 0.0) for m, e in zip(exposed, exposure, strict=True)
    ]
    total_signal_weight = np.sum(signal_weight, axis=0)
    total_background_weight = np.sum(background_weight, axis=0)

    def _mean(name: str, weight: list[np.ndarray], total: np.ndarray) -> np.ndarray:
        """
        Take the weighted mean of a variable over the pivots reporting it.

        Parameters
        ----------
        name : str
            The variable to combine.
        weight : list[np.ndarray]
            The weight of each pivot, zero where it reports nothing.
        total : np.ndarray
            The summed weight, which the mean is divided by.

        Returns
        -------
        np.ndarray
            The weighted mean, zero where no pivot reports the pixel.
        """
        weighted = np.sum(
            [w * v[name] for w, v in zip(weight, per_pivot, strict=True)], axis=0
        )
        return np.divide(weighted, total, out=np.zeros_like(total), where=total > 0)

    def _propagate(
        name: str, weight: list[np.ndarray], total: np.ndarray
    ) -> np.ndarray:
        """
        Propagate an uncertainty through the weighted mean of its variable.

        Parameters
        ----------
        name : str
            The uncertainty to combine.
        weight : list[np.ndarray]
            The weight of each pivot, zero where it reports nothing.
        total : np.ndarray
            The summed weight, which the mean is divided by.

        Returns
        -------
        np.ndarray
            The uncertainty on the weighted mean, taking the pivots as
            independent, zero where no pivot reports the pixel.
        """
        variance = np.sum(
            [(w * v[name]) ** 2 for w, v in zip(weight, per_pivot, strict=True)], axis=0
        )
        return np.divide(
            np.sqrt(variance), total, out=np.zeros_like(total), where=total > 0
        )

    # The intensity is combined by inverse variance, so that a pivot that
    # barely saw a pixel does not pull the pixel it barely saw. A pivot with no
    # uncertainty to weight by has no variance either, which happens where it
    # counted nothing; those pixels fall back to the exposure weighted mean, so
    # that a genuine zero is reported as zero rather than lost.
    uncertainty = [variables["ena_intensity_stat_uncert"] for variables in per_pivot]
    inverse_variance = [
        np.where(m & (u > 0), 1.0 / np.where(u > 0, u, 1.0) ** 2, 0.0)
        for m, u in zip(measured, uncertainty, strict=True)
    ]
    total_inverse_variance = np.sum(inverse_variance, axis=0)
    weighted = total_inverse_variance > 0

    intensity = np.where(
        weighted,
        _mean("ena_intensity", inverse_variance, total_inverse_variance),
        _mean("ena_intensity", signal_weight, total_signal_weight),
    )
    intensity_stat_uncert = np.where(
        weighted,
        np.sqrt(
            np.divide(
                1.0,
                total_inverse_variance,
                out=np.zeros_like(total_inverse_variance),
                where=weighted,
            )
        ),
        _mean("ena_intensity_stat_uncert", signal_weight, total_signal_weight),
    )

    # Counts add over the pivots that report the pixel. One no pivot reports is
    # masked out if it was exposed at all, and simply unseen if it was not.
    counts = np.sum(
        [
            np.where(m, v["ena_count"], 0.0)
            for m, v in zip(measured, per_pivot, strict=True)
        ],
        axis=0,
    )
    counts = np.where(any_measured, counts, np.where(any_exposed, FILLVAL_FLOAT, 0.0))

    sys_err_plus = _mean(
        "ena_intensity_sys_err_plus", signal_weight, total_signal_weight
    )
    sys_err_minus = _mean(
        "ena_intensity_sys_err_minus", signal_weight, total_signal_weight
    )

    variables = {
        "ena_count": counts,
        "exposure_factor": total_exposure,
        "ena_count_rate": _mean("ena_count_rate", signal_weight, total_signal_weight),
        "ena_count_rate_stat_uncert": _propagate(
            "ena_count_rate_stat_uncert", signal_weight, total_signal_weight
        ),
        "ena_intensity": intensity,
        "ena_intensity_stat_uncert": intensity_stat_uncert,
        "ena_intensity_sys_err": np.sqrt(sys_err_plus * sys_err_minus),
        "ena_intensity_sys_err_plus": sys_err_plus,
        "ena_intensity_sys_err_minus": sys_err_minus,
        "bg_rate": _mean("bg_rate", background_weight, total_background_weight),
        "bg_rate_stat_uncert": _propagate(
            "bg_rate_stat_uncert", background_weight, total_background_weight
        ),
        "bg_intensity": _mean(
            "bg_intensity", background_weight, total_background_weight
        ),
        "bg_intensity_stat_uncert": _propagate(
            "bg_intensity_stat_uncert", background_weight, total_background_weight
        ),
    }

    # A pixel no pivot reports has nothing to say about it. The counts carry
    # their own distinction between masked and unseen, and the exposure is real
    # wherever the map looked, so neither is filled here.
    for name, value in variables.items():
        if name in ("ena_count", "exposure_factor"):
            continue
        reported = any_exposed if name.startswith("bg_") else any_measured
        variables[name] = np.where(reported, value, FILLVAL_FLOAT)

    # The combined map covers the whole window its pivots were taken over, not
    # the window of whichever one happens to carry the result.
    sky_map = maps[0]
    sky_map.min_epoch = min(pivot_map.min_epoch for pivot_map in maps)
    sky_map.max_epoch = max(pivot_map.max_epoch for pivot_map in maps)

    return sky_map, variables, calibration


# =============================================================================
# SETUP AND INITIALIZATION HELPERS
# =============================================================================


def load_efficiency_data(anc_dependencies: list) -> pd.DataFrame:
    """
    Load efficiency factor data from ancillary files.

    Parameters
    ----------
    anc_dependencies : list
        List of ancillary file paths to search for efficiency factor files.

    Returns
    -------
    pd.DataFrame
        Concatenated efficiency factor data from all matching files.
        Returns empty DataFrame if no efficiency files found.
    """
    efficiency_files = [
        anc_file
        for anc_file in anc_dependencies
        if "efficiency-factor" in str(anc_file)
    ]

    if not efficiency_files:
        logger.warning("No efficiency factor files found in ancillary dependencies")
        return pd.DataFrame()

    logger.debug(f"Loading {len(efficiency_files)} efficiency factor files")
    return pd.concat(
        [lo_ancillary.read_ancillary_file(anc_file) for anc_file in efficiency_files],
        ignore_index=True,
    )


def load_sputter_correction_data() -> pd.DataFrame:
    """
    Load the sputter correction factors shipped with the package.

    Returns
    -------
    pd.DataFrame
        The ancillary data, with columns: source_species, target_species,
        target_esa, source_esa, sputter_factor.

    Raises
    ------
    ValueError
        If no sputter correction ancillary is shipped with the package.
    """
    sputter_files = sorted(ANCILLARY_DATA_DIR.glob("*sputter-correction-factors*"))

    if not sputter_files:
        raise ValueError("No sputter correction files found")

    return lo_ancillary.read_ancillary_file(sputter_files[-1])


def _sputter_correction(species: str) -> tuple[str, np.ndarray]:
    """
    Load the sputter correction of a species from the ancillary.

    Parameters
    ----------
    species : str
        The species being mapped, whose counts are to be corrected.

    Returns
    -------
    tuple[str, np.ndarray]
        The species sputtering into the mapped species, and the (target level,
        source level) fraction of its counts to remove from the mapped
        species' counts, zero where a source level does not sputter into a
        target level.

    Raises
    ------
    ValueError
        If the ancillary has nothing sputtering into the mapped species, or
        names more than one species doing it, which the correction cannot
        choose between.
    """
    factors = load_sputter_correction_data()
    factors = factors[factors["target_species"] == species]
    sources = factors["source_species"].unique()

    if len(sources) == 0:
        raise ValueError(
            f"The map asks for a sputter correction, but the ancillary has no "
            f"factors for {species}"
        )
    if len(sources) > 1:
        raise ValueError(
            f"More than one species sputters into {species}: {sorted(sources)}"
        )

    matrix = np.zeros((c.N_ESA_LEVELS, c.N_ESA_LEVELS))
    matrix[
        factors["target_esa"].to_numpy() - 1, factors["source_esa"].to_numpy() - 1
    ] = factors["sputter_factor"].to_numpy()
    return str(sources[0]), matrix


def load_bootstrap_correction_data() -> pd.DataFrame:
    """
    Load bootstrap correction factors from an ancillary file.

    Returns
    -------
    pd.DataFrame
        Bootstrap correction factors with columns: esa_step_i, esa_step_k,
        bootstrap_factor. Indices are 1-based ESA step numbers where esa_step_k=8
        refers to the virtual E8 channel.

    Raises
    ------
    ValueError
        If no bootstrap correction ancillary is shipped with the package.
    """
    bootstrap_files = sorted(ANCILLARY_DATA_DIR.glob("*bootstrap-correction-factors*"))

    if not bootstrap_files:
        raise ValueError("No bootstrap correction factor files found")

    return lo_ancillary.read_ancillary_file(bootstrap_files[-1])


def _bootstrap_correction() -> np.ndarray:
    """
    Load the bootstrap coefficients from the ancillary.

    Returns
    -------
    np.ndarray
        The (target level, source level) fraction of a source level's intensity
        to remove from a target level below it, zero where a source level does
        not bleed into a target level. The source axis carries one level more
        than the target axis, for the virtual ESA level above the top of the
        map. These are the nominal coefficients; the correction scales them
        itself, once for the correction and once for each of its bounds.
    """
    factors = load_bootstrap_correction_data()

    matrix = np.zeros((c.N_ESA_LEVELS, c.N_ESA_LEVELS + 1))
    matrix[
        factors["esa_step_i"].to_numpy() - 1, factors["esa_step_k"].to_numpy() - 1
    ] = factors["bootstrap_factor"].to_numpy()
    return matrix


def _flux_corrector(anc_dependencies: list) -> PowerLawFluxCorrector:
    """
    Load the ESA transmission factors the Compton-Getting correction reads.

    Parameters
    ----------
    anc_dependencies : list
        List of ancillary file paths, searched for the ESA eta fit factors.

    Returns
    -------
    PowerLawFluxCorrector
        The transmission factors, which the correction recovers the source
        spectrum of a pixel through.

    Raises
    ------
    ValueError
        If the ancillary dependencies hold no ESA eta fit factors.
    """
    try:
        flux_factors = next(
            x for x in anc_dependencies if "esa-eta-fit-factors" in str(x)
        )
    except StopIteration:
        raise ValueError(
            "A heliospheric frame map needs the ESA eta fit factors to be "
            "Compton-Getting corrected, and none were found in the ancillary "
            "dependencies"
        ) from None

    return PowerLawFluxCorrector(flux_factors)


def load_isn_mask_parameters() -> pd.DataFrame:
    """
    Load the ISN mask tuning parameters shipped with the package.

    Returns
    -------
    pd.DataFrame
        The ancillary data, with columns: pivot_angle, esa_step,
        intensity_threshold_fraction, angular_width_deg, outlier_percentile.

    Raises
    ------
    ValueError
        If no ISN mask parameter ancillary is shipped with the package.
    """
    mask_files = sorted(ANCILLARY_DATA_DIR.glob("*isn-mask-parameters*"))

    if not mask_files:
        raise ValueError("No ISN mask parameter files found")

    return lo_ancillary.read_ancillary_file(mask_files[-1])


def _isn_mask_parameters(pivot_angle: int) -> pd.DataFrame:
    """
    Get the ISN mask tuning of a pivot angle, in ascending ESA level order.

    The mask is tuned per pivot angle, and a map is masked one pivot angle at a
    time: a combined map masks each of its pivots on that pivot's own map,
    before the maps are combined, so there is never a mixed intensity
    distribution for a tuning to be applied to.

    Parameters
    ----------
    pivot_angle : int
        The nominal pivot angle [degrees] being masked.

    Returns
    -------
    pd.DataFrame
        The tuning of each ESA level, indexed by 1-based ESA step.

    Raises
    ------
    ValueError
        If the ancillary has no tuning for the pivot angle.
    """
    parameters = load_isn_mask_parameters()

    if pivot_angle not in set(parameters["pivot_angle"]):
        raise ValueError(
            f"The map asks for the ISN band to be masked out, but the ancillary "
            f"has no mask tuning for the {[pivot_angle]} degree pivot angle(s) "
            f"it was built from"
        )

    tuning = parameters[parameters["pivot_angle"] == pivot_angle].set_index("esa_step")

    # Select the ESA levels, in order. Raises if the ancillary is missing one.
    return tuning.loc[list(range(1, c.N_ESA_LEVELS + 1))]


def _isn_mask(
    intensity: np.ndarray, elevation: np.ndarray, parameters: pd.DataFrame
) -> np.ndarray:
    """
    Find the pixels the interstellar neutral flow dominates the map in.

    The ISN hydrogen the instrument sees is not the heliospheric ENA signal the
    map is of, and it is bright enough to swamp it. It arrives as a band along
    the ecliptic plane, so the mask is the pixels of a level that are both
    bright and close enough to the plane, plus the brightest few pixels of the
    level wherever they are.

    Parameters
    ----------
    intensity : np.ndarray
        The uncorrected intensity of every ESA level, of shape
        (epoch, esa level, pixel).
    elevation : np.ndarray
        The ecliptic latitude [degrees] of each pixel, of shape (pixel,). The
        ISN band lies along zero elevation.
    parameters : pd.DataFrame
        The mask tuning of each ESA level, in ascending level order.

    Returns
    -------
    np.ndarray
        Whether each pixel of each level is masked, of the shape of
        ``intensity``.
    """
    threshold = parameters["intensity_threshold_fraction"].to_numpy()[:, np.newaxis]
    angular_width = parameters["angular_width_deg"].to_numpy()[:, np.newaxis]
    percentile = parameters["outlier_percentile"].to_numpy()

    # The band: the bright pixels of a level, near enough to the ecliptic. A
    # level that saw nothing has no brightest pixel to take a fraction of.
    peak = np.nanmax(intensity, axis=-1, keepdims=True)
    bright = np.where(peak > 0, intensity >= threshold * peak, False)
    mask = bright & (np.abs(elevation) <= angular_width)

    # The outliers: the top tail of a level's own intensity distribution, which
    # catches the ISN pixels that sit off the plane.
    cutoff = np.stack(
        [
            np.nanpercentile(intensity[:, level], level_percentile, axis=-1)
            for level, level_percentile in enumerate(percentile)
        ],
        axis=1,
    )
    return mask | (intensity > cutoff[..., np.newaxis])


def finalize_dataset(dataset: xr.Dataset, descriptor: str) -> xr.Dataset:
    """
    Add attributes and perform final dataset preparation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to finalize with attributes.
    descriptor : str
        The descriptor for this map dataset.

    Returns
    -------
    xr.Dataset
        The finalized dataset with all attributes added.
    """
    # Initialize the attribute manager
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-rectangular")

    # Add global and variable attributes
    dataset.attrs.update(attr_mgr.get_global_attributes("imap_lo_l2_enamap"))

    # Our global attributes have placeholders for descriptor
    # so iterate through here and fill that in with the map-specific descriptor
    for key in ["Data_type", "Logical_source", "Logical_source_description"]:
        dataset.attrs[key] = dataset.attrs[key].format(descriptor=descriptor)
    for var in dataset.data_vars:
        try:
            dataset[var].attrs = attr_mgr.get_variable_attributes(var)
        except KeyError:
            # If no attributes found, try without schema validation
            try:
                dataset[var].attrs = attr_mgr.get_variable_attributes(
                    var, check_schema=False
                )
            except KeyError:
                logger.warning(f"No attributes found for variable {var}")

    return dataset


# =============================================================================
# INPUT HANDLING
# =============================================================================


def _nominal_pivot_angle(pivot_angle: float) -> int | None:
    """
    Snap a measured pivot angle onto the nominal pivot angle it was flown at.

    Parameters
    ----------
    pivot_angle : float
        The pivot angle [degrees] a pointing's goodtimes report.

    Returns
    -------
    int | None
        The nominal pivot angle [degrees] whose range contains it, or None if
        it falls in none of them.
    """
    for nominal, spec in c.PIVOT_ANGLES.items():
        if spec.min <= pivot_angle <= spec.max:
            return nominal
    return None


def _group_pointings_by_pivot(
    map_descriptor: MapDescriptor, pointings: dict[int, tuple]
) -> dict[int, dict[int, tuple]]:
    """
    Split the pointings of a map into one group per nominal pivot angle.

    A Lo map carries its pivot angle as its sensor, e.g. the 90 of "l090", and
    its inputs were filtered down to that pivot in pre-processing, so it has
    the one group. A map that combines every pivot angle instead of selecting
    one is written without a sensor, as "ilo", and is grouped by the pivot
    angles its pointings were actually flown at, which their goodtimes report.
    Each group becomes a map of its own.

    Parameters
    ----------
    map_descriptor : MapDescriptor
        The parsed descriptor of the map being made.
    pointings : dict[int, tuple]
        The (goodtimes, bgrates, histrates) datasets of each mappable pointing.

    Returns
    -------
    dict[int, dict[int, tuple]]
        The pointings of each nominal pivot angle [degrees], keyed by
        repointing within each group.

    Raises
    ------
    ValueError
        If the map combines pivot angles but none of its pointings reports one
        that is recognisably nominal, leaving nothing to group them by.
    """
    if isinstance(map_descriptor.sensor, int):
        return {map_descriptor.sensor: pointings}

    groups: dict[int, dict[int, tuple]] = {}
    unrecognised = set()
    for repointing, products in pointings.items():
        measured = float(np.atleast_1d(products[0]["pivot"].values)[0])
        pivot_angle = _nominal_pivot_angle(measured)
        if pivot_angle is None:
            unrecognised.add(measured)
            continue
        groups.setdefault(pivot_angle, {})[repointing] = products

    if unrecognised:
        logger.warning(
            f"Ignoring the pivot angles {sorted(unrecognised)} of "
            f"{map_descriptor.instrument_descriptor}, they match none of the "
            f"nominal pivot angles."
        )

    if not groups:
        raise ValueError(
            f"None of the pointings of "
            f"{map_descriptor.instrument_descriptor} reports a nominal pivot "
            f"angle to build a combined map from"
        )

    return groups


def _complete_pointings(
    sci_dependencies: dict[int, dict[str, xr.Dataset]],
) -> dict[int, tuple]:
    """
    Reduce the grouped inputs to the pointings that can be mapped.

    Parameters
    ----------
    sci_dependencies : dict[int, dict[str, xr.Dataset]]
        The input datasets of each pointing, keyed by repointing and then by
        product descriptor.

    Returns
    -------
    dict[int, tuple]
        The (goodtimes, bgrates, histrates) datasets of each mappable pointing,
        keyed by repointing.

    Raises
    ------
    KeyError
        If any of the three required products is missing entirely.
    """
    found = {product for products in sci_dependencies.values() for product in products}
    missing_products = set(REQUIRED_PRODUCTS) - found
    if missing_products:
        raise KeyError(f"No input files for {sorted(missing_products)}")

    pointings = {}
    for repointing, products in sci_dependencies.items():
        missing = set(REQUIRED_PRODUCTS) - set(products)
        if missing:
            logger.warning(
                f"Dropping repoint{repointing:05d}, it has no {sorted(missing)}"
            )
            continue
        pointings[repointing] = tuple(products[p] for p in REQUIRED_PRODUCTS)

    return pointings


def _get_esa_mode(histrates: xr.Dataset) -> int:
    """
    Read the ESA mode of a pointing, defaulting to HiRes.

    Parameters
    ----------
    histrates : xr.Dataset
        The L1B histogram rates of the pointing.

    Returns
    -------
    int
        The ESA mode, 0 for HiRes and 1 for HiThr.
    """
    if "esa_mode" not in histrates:
        return 0
    return int(np.atleast_1d(histrates["esa_mode"].values)[0])


# =============================================================================
# SKY MAP ACCUMULATION
# =============================================================================


class LoSpinAnglePointingSet(PointingSet):
    """
    The spin-angle bins of one pointing, as an in-memory pointing set.

    Lo builds its maps straight from the L1B products of a pointing rather than
    from a written L1C pointing set, so the sky direction of each spin-angle
    bin and the values looking in it are assembled here.

    Parameters
    ----------
    epoch : int
        The time [TTJ2000 ns] the pointing is projected from.
    pivot_angle : float
        The pivot angle [degrees] of the pointing.
    spin_angles : np.ndarray
        The IMAP_DPS azimuth [degrees] of each spin-angle bin.
    values : dict[str, np.ndarray]
        The values of the pointing, each of shape (esa level, spin angle).
    frame : SpiceFrame
        The frame to compute the sky directions in, i.e. the map's frame.
    energy : np.ndarray
        The energy [keV] of each ESA level.
    """

    tiling_type: SkyTilingType = SkyTilingType.RECTANGULAR

    def __init__(
        self,
        epoch: int,
        pivot_angle: float,
        spin_angles: np.ndarray,
        values: dict[str, np.ndarray],
        frame: SpiceFrame,
        energy: np.ndarray,
    ):
        dims = [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, "spin_angle"]
        super().__init__(
            xr.Dataset(
                {
                    name: (dims, value[np.newaxis, ...])  # add epoch axis
                    for name, value in values.items()
                },
                coords={
                    CoordNames.TIME.value: [epoch],
                    CoordNames.ENERGY_L2.value: energy,
                },
            ),
            spice_reference_frame=frame,
        )
        self.spatial_coords = ("spin_angle",)

        az_el = compute_pointing_directions(
            epoch,
            pivot_angle,
            spin_angles=spin_angles,
            off_angles=np.array([0.0]),
            to_frame=frame,
        )
        self.az_el_points = xr.DataArray(
            np.asarray(az_el),
            dims=[CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
        )

    @property
    def midpoint_j2000_et(self) -> float:
        """
        The time the pointing is projected from.

        The base class derives this from an ``epoch_delta``; a pointing built
        here is handed the single epoch it is projected from directly.

        Returns
        -------
        float
            The epoch of the pointing set [J2000 ET].
        """
        return float(ttj2000ns_to_et(self.epoch))


def _initialize_accumulators(
    sky_map: RectangularSkyMap, energy: np.ndarray, names: tuple[str, ...]
) -> None:
    """
    Seed the map with the empty accumulators each pointing is added into.

    ``project_pset_values_to_map`` creates a map variable the first time it
    projects one, so seeding them is what lets the rest of the pipeline read
    the accumulators unconditionally, however many pointings turn out to be
    usable.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map being built, modified in place.
    energy : np.ndarray
        The energy [keV] of each ESA level.
    names : tuple[str, ...]
        The accumulators to seed.
    """
    for name in names:
        sky_map.data_1d[name] = xr.DataArray(
            np.zeros((1, c.N_ESA_LEVELS, sky_map.num_points)),
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_L2.value,
                CoordNames.GENERIC_PIXEL.value,
            ],
            coords={CoordNames.ENERGY_L2.value: energy},
        )


def _accumulate_pointing(
    goodtimes: xr.Dataset,
    bgrates: xr.Dataset,
    histrates: xr.Dataset,
    sky_map: RectangularSkyMap,
    map_descriptor: MapDescriptor,
    energy: np.ndarray,
    sputter_source: str | None = None,
) -> None:
    """
    Add one pointing's counts and exposure to the map.

    Parameters
    ----------
    goodtimes : xr.Dataset
        The L1B goodtimes of the pointing, giving its pivot angle and the
        good-time windows its histograms are accepted within.
    bgrates : xr.Dataset
        The L1B background rates of the pointing, one rate per ESA level.
    histrates : xr.Dataset
        The L1B histogram rates of the pointing, giving the counts and exposure
        of each spin-angle bin.
    sky_map : RectangularSkyMap
        The map being built, modified in place.
    map_descriptor : MapDescriptor
        The parsed descriptor of the map being made.
    energy : np.ndarray
        The energy [keV] of each ESA level.
    sputter_source : str | None
        The species sputtering into the mapped species, whose counts are
        accumulated alongside. None if this map is not sputter corrected.
    """
    species = map_descriptor.species
    pivot_angle = float(np.atleast_1d(goodtimes["pivot"].values)[0])
    gt_start = np.atleast_1d(goodtimes["gt_start_met"].values)
    gt_end = np.atleast_1d(goodtimes["gt_end_met"].values)

    histogram_met = ttj2000ns_to_met(histrates["epoch"].values)
    in_goodtime = np.any(
        (histogram_met[:, np.newaxis] >= gt_start)
        & (histogram_met[:, np.newaxis] <= gt_end),
        axis=1,
    )
    if not in_goodtime.any():
        logger.warning("No histogram epochs fall within the good-time windows.")
        return

    spin_angles = _dps_spin_angles()
    keep = _spin_phase_mask(spin_angles, pivot_angle, map_descriptor.spin_phase)
    if not keep.any():
        return

    pointing_counts = histrates[f"{species}_counts"].values[in_goodtime].sum(axis=0)
    pointing_exposure = histrates["exposure_time_6deg"].values[in_goodtime].sum(axis=0)
    background_rates = np.atleast_2d(bgrates[f"{species}_background_rates"].values)[0]

    # The whole pointing is projected from the middle of its good times, which
    # is where the despun frame is sampled.
    epoch = met_to_ttj2000ns((gt_start.min() + gt_end.max()) / 2.0)
    values = {
        "ena_count": pointing_counts,
        "exposure_factor": pointing_exposure,
        # Background is a rate per ESA level per pointing, so it is
        # accumulated weighted by exposure and divided by the total
        # exposure at the end.
        "bg_rate_exposure": background_rates[:, np.newaxis] * pointing_exposure,
    }
    if sputter_source:
        values["sputter_source_count"] = (
            histrates[f"{sputter_source}_counts"].values[in_goodtime].sum(axis=0)
        )
    if map_descriptor.cg_corrected:
        # The RAM projection is a property of the bin, not of the counts in it,
        # so like the background it is accumulated weighted by exposure and
        # divided by the total exposure at the end.
        values["cos_alpha_exposure"] = (
            _ram_projection(spin_angles, pivot_angle) * pointing_exposure
        )

    pointing_set = LoSpinAnglePointingSet(
        epoch,
        pivot_angle,
        spin_angles,
        values,
        sky_map.spice_reference_frame,
        energy,
    )
    # The projection sums the spin-angle bins that land in the same map pixel,
    # and adds this pointing on top of what the earlier pointings left there.
    sky_map.project_pset_values_to_map(
        pointing_set,
        value_keys=list(values),
        pset_valid_mask=keep,
    )

    sky_map.min_epoch = min(sky_map.min_epoch, int(met_to_ttj2000ns(gt_start.min())))
    sky_map.max_epoch = max(sky_map.max_epoch, int(met_to_ttj2000ns(gt_end.max())))


def _dps_spin_angles() -> np.ndarray:
    """
    Get the despun-frame azimuth of each histogram spin-angle bin center.

    The L1B histogram spin bins are hardware spin-phase bins referenced to the
    spacecraft spin pulse, NOT the instrument (DPS) spin angle. A bin center is
    converted to the IMAP_DPS azimuth by adding the spacecraft to instrument
    spin-phase offset, exactly as the L1B star-sensor product does.

    Returns
    -------
    np.ndarray
        The IMAP_DPS azimuth [degrees] of each of the histogram spin bins.
    """
    bin_width = 360.0 / c.N_SPIN_ANGLE_BINS
    bin_centers = (np.arange(c.N_SPIN_ANGLE_BINS) + 0.5) * bin_width
    offset = get_spacecraft_to_instrument_spin_phase_offset(SpiceFrame.IMAP_LO) * 360.0
    return np.mod(bin_centers + offset, 360.0)


def _spin_phase_mask(
    spin_angles: np.ndarray, pivot_angle: float, spin_phase: str
) -> np.ndarray:
    """
    Get the spin-angle bins belonging on a map of the given spin phase.

    A bin's RAM projection factor is ``sin(pivot) * sin(spin angle)``, positive
    looking into the RAM direction and negative looking away from it.

    Parameters
    ----------
    spin_angles : np.ndarray
        The IMAP_DPS azimuth [degrees] of each spin-angle bin.
    pivot_angle : float
        The pivot angle [degrees] of the pointing.
    spin_phase : str
        The spin phase of the map, "ram", "anti" or "full".

    Returns
    -------
    np.ndarray
        Boolean mask of the bins to keep.

    Raises
    ------
    ValueError
        If the spin phase is not one of "ram", "anti" or "full".
    """
    if spin_phase == "full":
        return np.ones(spin_angles.size, dtype=bool)
    if spin_phase not in ("ram", "anti"):
        raise ValueError(
            f"Invalid spin phase: {spin_phase}. Must be 'ram', 'anti' or 'full'."
        )

    ram_projection = _ram_projection(spin_angles, pivot_angle)
    return ram_projection > 0 if spin_phase == "ram" else ram_projection < 0


def _ram_projection(spin_angles: np.ndarray, pivot_angle: float) -> np.ndarray:
    """
    Project the look direction of each spin-angle bin onto the RAM direction.

    The projection is the cosine of the angle between the bin's look direction
    and the spacecraft's velocity, which is what tells the RAM half of the spin
    from the anti-RAM half, and what the Compton-Getting correction is a
    function of.

    Parameters
    ----------
    spin_angles : np.ndarray
        The IMAP_DPS azimuth [degrees] of each spin-angle bin.
    pivot_angle : float
        The pivot angle [degrees] of the pointing.

    Returns
    -------
    np.ndarray
        The projection factor of each bin, positive looking into the RAM
        direction and negative looking away from it.
    """
    return np.sin(np.radians(pivot_angle + c.PIVOT_RAM_OFFSET)) * np.sin(
        np.radians(spin_angles)
    )


# =============================================================================
# GEOMETRIC FACTORS
# =============================================================================


def load_geometric_factor_data(species: str) -> pd.DataFrame:
    """
    Load geometric factor data for the specified species.

    Parameters
    ----------
    species : str
        The species to load geometric factors for ("h" or "o").

    Returns
    -------
    pd.DataFrame
        Geometric factor dataframe for the specified species.

    Raises
    ------
    ValueError
        If species is not "h" or "o".
    """
    if species not in ["h", "o"]:
        raise ValueError(
            f"Geometric factors only available for 'h' and 'o', got '{species}'"
        )

    if species == "h":
        gf_file = sorted(ANCILLARY_DATA_DIR.glob("*hydrogen-geometric-factor*"))[-1]
    else:  # species == "o"
        gf_file = sorted(ANCILLARY_DATA_DIR.glob("*oxygen-geometric-factor*"))[-1]

    return lo_ancillary.read_ancillary_file(gf_file)


def reduce_geometric_factor_data(species: str, esa_mode: int) -> pd.DataFrame:
    """
    Get geometric factor data for a specific species and ESA mode.

    This helper function loads geometric factor data, filters by ESA mode, and
    selects the row of each of the 7 energy steps, in ascending step order.

    Parameters
    ----------
    species : str
        The species to load geometric factors for ("h" or "o").
    esa_mode : int
        ESA mode (0 for HiRes, 1 for HiThr).

    Returns
    -------
    pd.DataFrame
        Geometric factor data indexed by Observed_E-Step (1-7), containing all
        columns from the geometric factor CSV file.
    """
    # Load geometric factor data for this species
    gf_data = load_geometric_factor_data(species)

    # Filter for the specific ESA mode
    if "esa_mode" in gf_data.columns:
        gf_data = gf_data[gf_data["esa_mode"] == esa_mode]

    # Lo Instrument team: Use only geometric factors where
    # incident_E-Step == Observed_E-Step
    diagonal = gf_data["incident_E-Step"] == gf_data["Observed_E-Step"]
    gf_data = gf_data[diagonal].set_index("Observed_E-Step")

    # Select the energy steps, in order. Raises if the file is missing one.
    return gf_data.loc[list(range(1, c.N_ESA_LEVELS + 1))]


def _esa_calibration(species: str, esa_mode: int) -> EsaCalibration:
    """
    Get the ESA level calibration one map is built from.

    The ancillary names its two geometric factor uncertainty columns for the
    direction the intensity derived from the factor moves in, which is the
    opposite of the direction the factor itself moves in: intensity goes as
    1/G, so a smaller factor gives a larger intensity. Its ``_unc_plus`` is
    therefore the downward excursion of the factor, and its ``_unc_minus`` the
    upward one.

    Parameters
    ----------
    species : str
        The species of the map ("h" or "o").
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr.

    Returns
    -------
    EsaCalibration
        The energies, passband half-widths and geometric factors of every ESA
        level, in ascending level order.
    """
    gf_data = reduce_geometric_factor_data(species, esa_mode).astype(float)

    factor = f"GF_Trpl_{species.upper()}"
    geometric_factor = gf_data[factor].to_numpy()

    return EsaCalibration(
        energy=gf_data["Cntr_E"].to_numpy(),
        energy_delta_minus=gf_data["Cntr_E_delta_minus"].to_numpy(),
        energy_delta_plus=gf_data["Cntr_E_delta_plus"].to_numpy(),
        geometric_factor=geometric_factor,
        geometric_factor_low=geometric_factor
        - gf_data[f"{factor}_unc_plus"].to_numpy(),
        geometric_factor_high=geometric_factor
        + gf_data[f"{factor}_unc_minus"].to_numpy(),
    )


# =============================================================================
# RATES AND INTENSITIES CALCULATIONS
# =============================================================================


def _sputter_correct_counts(
    counts: np.ndarray, source_counts: np.ndarray, sputter_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove the counts sputtered into the mapped species from another species.

    Parameters
    ----------
    counts : np.ndarray
        The accumulated counts of the mapped species, of shape
        (epoch, esa level, pixel).
    source_counts : np.ndarray
        The accumulated counts of the sputtering species, same shape and grid.
    sputter_matrix : np.ndarray
        The (target level, source level) fraction of the source counts to
        remove from the target counts.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The corrected counts and their variance. Counting in both species is
        Poisson, so each source term contributes its counts scaled by the
        square of its factor, and the variance only ever grows.
    """
    logger.info("Applying the sputter correction to the accumulated counts")
    corrected = counts - np.einsum("ts,esp->etp", sputter_matrix, source_counts)
    variance = counts + np.einsum("ts,esp->etp", sputter_matrix**2, source_counts)
    return corrected, variance


def _local_median_spectral_index(
    spectral_index: np.ndarray, grid_shape: tuple[int, ...]
) -> np.ndarray:
    """
    Fill in the spectral index of a pixel from the pixels around it.

    Parameters
    ----------
    spectral_index : np.ndarray
        The measured spectral index of every pixel, of shape (epoch, pixel),
        NaN where the pixel has none.
    grid_shape : tuple[int, ...]
        The (azimuth, elevation) shape the pixel axis unwraps to, which is what
        makes two pixels neighbors.

    Returns
    -------
    np.ndarray
        The median of the measured indices in each pixel's neighborhood, of the
        same shape, NaN where the whole neighborhood is unmeasured. The map
        does not wrap around in azimuth, so the pixels at its edges take the
        median of the neighbors they have.
    """

    def median_of_measured(neighborhood: np.ndarray) -> float:
        """
        Take the median of the pixels of a neighborhood that have an index.

        Parameters
        ----------
        neighborhood : np.ndarray
            The spectral indices of the pixels around (and including) one
            pixel, NaN where a pixel has none.

        Returns
        -------
        float
            The median, or NaN if no pixel of the neighborhood has an index.
        """
        measured = neighborhood[~np.isnan(neighborhood)]
        return float(np.median(measured)) if measured.size else float(np.nan)

    return np.stack(
        [
            generic_filter(
                epoch_index.reshape(grid_shape),
                median_of_measured,
                size=c.BOOTSTRAP_SPECTRAL_INDEX_FILTER_SIZE,
                mode="constant",
                cval=np.nan,
            ).ravel()
            for epoch_index in spectral_index
        ]
    )


def _extrapolate_top_intensity(
    intensity: np.ndarray, energy: np.ndarray, grid_shape: tuple[int, ...]
) -> np.ndarray:
    """
    Extrapolate the intensity of the virtual ESA level above the top of the map.

    The top ESA levels have nothing above them in the map to be bootstrap
    corrected against, so a virtual level is extrapolated from the top two
    levels of each pixel, taking the spectrum between them as a power law.

    Parameters
    ----------
    intensity : np.ndarray
        The intensity of every ESA level, of shape (epoch, esa level, pixel).
    energy : np.ndarray
        The energy [keV] of each ESA level.
    grid_shape : tuple[int, ...]
        The (azimuth, elevation) shape the pixel axis unwraps to.

    Returns
    -------
    np.ndarray
        The intensity of the virtual level, of shape (epoch, pixel), zero in
        the pixels the top level saw nothing in.

    Notes
    -----
    The spectrum is taken as a power law in energy, ``I(E) = A * E ** -gamma``,
    where ``gamma`` is the spectral index. The normalization ``A`` never has to
    be evaluated: writing the law at the top two levels and dividing one by the
    other cancels it, leaving::

        I_top / I_second = (E_top / E_second) ** -gamma

    Taking logs of both sides gives the index the code solves for::

        log(I_top / I_second) = -gamma * log(E_top / E_second)
        gamma = -log(I_top / I_second) / log(E_top / E_second)

    The virtual level is then read off the same law anchored on the top level,
    which keeps ``A`` cancelled, so a pixel needs only a spectral index and one
    measured point to be extrapolated::

        I_virtual = I_top * (E_virtual / E_top) ** -gamma

    That is what lets a pixel with no spectrum of its own borrow an index from
    its neighbors and still use its own top level as the anchor.
    """
    second, top = energy[-2], energy[-1]
    virtual = top * c.ESA_8_ENERGY_RATIO
    second_intensity, top_intensity = intensity[:, -2], intensity[:, -1]

    # The spectral index the two levels of a pixel imply, where it has both.
    measured = (second_intensity > 0) & (top_intensity > 0)
    spectral_index = np.zeros_like(top_intensity)
    spectral_index[measured] = -np.log(
        top_intensity[measured] / second_intensity[measured]
    ) / np.log(top / second)

    extrapolated = np.zeros_like(top_intensity)
    extrapolated[measured] = top_intensity[measured] * (virtual / top) ** (
        -spectral_index[measured]
    )

    # A pixel the second level saw nothing in has no spectrum of its own to
    # extrapolate along, so it borrows one from its neighbors, falling back to
    # the whole map and then to a nominal index.
    borrowing = (top_intensity > 0) & ~measured
    if borrowing.any():
        local = _local_median_spectral_index(
            np.where(measured, spectral_index, np.nan), grid_shape
        )
        global_index = (
            float(np.median(spectral_index[measured]))
            if measured.any()
            else c.BOOTSTRAP_DEFAULT_SPECTRAL_INDEX
        )
        borrowed = np.where(np.isfinite(local), local, global_index)
        extrapolated[borrowing] = top_intensity[borrowing] * (virtual / top) ** (
            -borrowed[borrowing]
        )

    return extrapolated


def _bootstrap_correct_intensity(
    intensity: np.ndarray,
    variance: np.ndarray,
    calibration: EsaCalibration,
    bootstrap_matrix: np.ndarray,
    grid_shape: tuple[int, ...],
    valid_gf_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove the intensity that bled into each ESA level from the levels above it.

    Every level is corrected against the intensities as they were measured, so
    the correction of one level does not feed the correction of the next.

    Parameters
    ----------
    intensity : np.ndarray
        The intensity of every ESA level, of shape (epoch, esa level, pixel).
    variance : np.ndarray
        The statistical variance of those intensities, same shape.
    calibration : EsaCalibration
        The energy response the map is binned in, read for the energies the
        virtual level is extrapolated along and the geometric factor bounds the
        systematic error is taken from.
    bootstrap_matrix : np.ndarray
        The (target level, source level) nominal bootstrap coefficients.
    grid_shape : tuple[int, ...]
        The (azimuth, elevation) shape the pixel axis unwraps to.
    valid_gf_bounds : np.ndarray
        Whether the lower geometric factor bound of each ESA level is usable,
        of shape (esa level, 1). The systematic error is zero where it is not.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The corrected intensity, its statistical uncertainty, and the upward
        and downward systematic excursions of the correction.
    """
    logger.info("Applying the bootstrap correction to the intensities")

    # The correction subtracts the level above the top of the map as well, so
    # the intensities are extended by the virtual level it is extrapolated to.
    # Its variance is approximated by that of the level it was extrapolated
    # from, which contributes little to the levels it is subtracted from.
    top_intensity = _extrapolate_top_intensity(
        intensity, calibration.energy, grid_shape
    )
    extended = np.concatenate([intensity, top_intensity[:, np.newaxis]], axis=1)
    extended_variance = np.concatenate([variance, variance[:, -1:]], axis=1)

    def subtract(scale: float) -> np.ndarray:
        """
        Subtract the bled intensity, at one scaling of the coefficients.

        Parameters
        ----------
        scale : float
            The scaling of the nominal coefficients.

        Returns
        -------
        np.ndarray
            The corrected intensity, which the subtraction can take below zero
            in a faint pixel, floored there.
        """
        return np.maximum(
            intensity - np.einsum("ik,ekp->eip", scale * bootstrap_matrix, extended),
            0.0,
        )

    corrected = subtract(c.BOOTSTRAP_SCALE)
    corrected_variance = variance + np.einsum(
        "ik,ekp->eip", (c.BOOTSTRAP_SCALE * bootstrap_matrix) ** 2, extended_variance
    )

    # The systematic error spans the two scalings the correction is bracketed
    # by, each moved on by the geometric factor bound of the same direction.
    geometric_factor = calibration.geometric_factor[:, np.newaxis]
    gf_high = calibration.geometric_factor_high[:, np.newaxis]
    gf_low = np.where(
        valid_gf_bounds, calibration.geometric_factor_low[:, np.newaxis], 1.0
    )
    lower = subtract(c.BOOTSTRAP_SCALE_INTENSITY_LOW) * geometric_factor / gf_high
    upper = subtract(c.BOOTSTRAP_SCALE_INTENSITY_HIGH) * geometric_factor / gf_low

    return (
        corrected,
        np.sqrt(corrected_variance),
        np.where(valid_gf_bounds, upper - corrected, 0.0),
        np.where(valid_gf_bounds, corrected - lower, 0.0),
    )


def _power_law_slopes(intensity: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Estimate the spectral index of every ESA level of every pixel.

    Parameters
    ----------
    intensity : np.ndarray
        The intensity of each ESA level, of shape (epoch, esa level, pixel),
        NaN where a pixel saw nothing at a level.
    energy : np.ndarray
        The energy of each ESA level, in any unit.

    Returns
    -------
    np.ndarray
        The index of the power law through each level and the one above it, of
        the same shape, NaN where either level is unmeasured. The top level has
        no level above it, so it keeps the index of the one below it.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = (
            np.log(intensity[:, 1:] / intensity[:, :-1])
            / np.log(energy[1:] / energy[:-1])[:, np.newaxis]
        )
    return np.concatenate([slopes, slopes[:, -1:]], axis=1)


def _source_intensity(
    intensity: np.ndarray,
    energy: np.ndarray,
    flux_corrector: PowerLawFluxCorrector,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Undo the ESA transmission bias in the observed intensities.

    An ESA level integrates over a passband rather than sampling a single
    energy, so what it observes depends on the spectrum falling through it. The
    transmission factor that relates the two is itself a function of the
    spectral index, so the source spectrum is recovered by iterating: estimate
    the index, undo the transmission, re-estimate the index, until the
    intensities settle.

    Parameters
    ----------
    intensity : np.ndarray
        The observed intensity of each ESA level, of shape
        (epoch, esa level, pixel).
    energy : np.ndarray
        The energy of each ESA level, in any unit.
    flux_corrector : PowerLawFluxCorrector
        The ESA transmission factors, read from the eta fit ancillary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The source intensity and its spectral index, both NaN in the pixels
        that saw nothing.
    """
    levels: np.ndarray = np.arange(c.N_ESA_LEVELS) + 1

    def transmission(spectral_index: np.ndarray) -> np.ndarray:
        """
        Get the ESA transmission factor of each level of each pixel.

        Parameters
        ----------
        spectral_index : np.ndarray
            The index of the power law through each ESA level of each pixel.

        Returns
        -------
        np.ndarray
            The transmission factor, of the same shape.
        """
        # The shared corrector takes the ESA level on the leading axis.
        return np.moveaxis(
            flux_corrector.eta_esa(levels, np.moveaxis(spectral_index, 1, 0)), 0, 1
        )

    # A pixel that saw nothing at a level has no spectrum through it; the NaN
    # carries through the iteration and out to the corrected map.
    observed = np.where(intensity > 0, intensity, np.nan)

    index = _power_law_slopes(observed, energy)
    source = observed / transmission(index)

    for iteration in range(c.CG_MAX_ITERATIONS):
        predicted = 0.5 * (index + _power_law_slopes(source, energy))
        corrected = 0.5 * (
            index + _power_law_slopes(observed / transmission(predicted), energy)
        )

        previous, source = source, observed / transmission(corrected)
        index = corrected

        with np.errstate(divide="ignore", invalid="ignore"):
            change = np.sqrt(np.nanmean((source / previous) ** 2)) - 1.0
        if np.isfinite(change) and abs(change) < c.CG_CONVERGENCE_TOLERANCE:
            logger.debug(f"Source spectrum converged after {iteration + 1} iterations")
            break
    else:
        logger.warning(
            f"Source spectrum did not converge in {c.CG_MAX_ITERATIONS} iterations"
        )

    return source, index


def _spacecraft_frame_energy(cos_alpha: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """
    Get the energy an ENA of a given heliospheric energy arrives with.

    An ENA arriving at the spacecraft is seen at a different energy than it has
    in the heliosphere, by the spacecraft's own motion through it: the same ENA
    is faster in the spacecraft frame when the spacecraft is moving into it and
    slower when it is moving away.

    Parameters
    ----------
    cos_alpha : np.ndarray
        The cosine of the angle between the look direction of each pixel and
        the spacecraft's velocity, of shape (epoch, esa level, pixel).
    energy : np.ndarray
        The heliospheric-frame energy [eV] of each ESA level.

    Returns
    -------
    np.ndarray
        The spacecraft-frame energy [eV] of each ESA level of each pixel, of
        the shape of ``cos_alpha``.
    """
    energy_u = c.CG_ENA_ENERGY_AT_SPACECRAFT_SPEED_EV
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)

    # The speed of the ENA in the spacecraft frame, in units of the
    # spacecraft's own speed: x = cos(a) + sqrt(y^2 - sin^2(a)), y^2 = E / E_u.
    ratio = (energy / energy_u)[:, np.newaxis]
    speed = cos_alpha + np.sqrt(np.maximum(ratio + cos_alpha**2 - 1.0, 0.0))
    return speed**2 * energy_u


def _compton_getting_correct_intensity(
    intensity: np.ndarray,
    uncertainties: tuple[np.ndarray, ...],
    cos_alpha: np.ndarray,
    calibration: EsaCalibration,
    flux_corrector: PowerLawFluxCorrector,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """
    Move the intensities from the spacecraft frame into the heliospheric one.

    The intensity a pixel reports is of ENAs at their spacecraft-frame energy,
    which the spacecraft's motion has shifted away from the heliospheric-frame
    energy the map is binned in. The correction reads the source spectrum of
    the pixel off its own power law at the shifted energy, and scales the
    intensity back onto the map's energy.

    Parameters
    ----------
    intensity : np.ndarray
        The intensity of every ESA level, of shape (epoch, esa level, pixel).
    uncertainties : tuple[np.ndarray, ...]
        The uncertainties on that intensity, each of the same shape. They are
        scaled by the same factor as the intensity they belong to.
    cos_alpha : np.ndarray
        The cosine of the angle between the look direction of each pixel and
        the spacecraft's velocity, same shape.
    calibration : EsaCalibration
        The energy response the map is binned in, read for the energies the
        correction shifts between.
    flux_corrector : PowerLawFluxCorrector
        The ESA transmission factors, read from the eta fit ancillary.

    Returns
    -------
    tuple[np.ndarray, tuple[np.ndarray, ...]]
        The corrected intensity and its correspondingly scaled uncertainties,
        zero in the pixels the correction has nothing to say about.
    """
    # The kinematics are in eV; the map is binned in keV.
    energy = calibration.energy * 1e3

    source, spectral_index = _source_intensity(intensity, energy, flux_corrector)
    energy_sc = _spacecraft_frame_energy(cos_alpha, energy)

    with np.errstate(divide="ignore", invalid="ignore"):
        corrected = source * (energy_sc / energy[:, np.newaxis]) ** (
            spectral_index + 1.0
        )
        # The uncertainties are fractionally unchanged, so they move with the
        # intensity. This folds in the transmission factor as well, which the
        # source intensity was already divided by.
        scaling = np.where(intensity > 0, corrected / intensity, np.nan)

    return np.nan_to_num(corrected), tuple(
        np.nan_to_num(uncertainty * scaling) for uncertainty in uncertainties
    )


def _calculate_rates_and_intensities(
    sky_map: RectangularSkyMap,
    calibration: EsaCalibration,
    sputter_matrix: np.ndarray | None = None,
    bootstrap_matrix: np.ndarray | None = None,
    flux_corrector: PowerLawFluxCorrector | None = None,
    isn_mask_parameters: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """
    Turn the accumulated counts and exposure into rates and intensities.

    Every derived quantity is filled in with a fill value in the pixels
    that were never exposed.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map the pointings were projected onto, read for its accumulators.
    calibration : EsaCalibration
        The energy response the map is binned in, read for the energies and
        geometric factors the intensities are derived with.
    sputter_matrix : np.ndarray | None
        The (target level, source level) sputter correction factors, applied
        to the counts before the rate is taken. None leaves the counts as
        they were observed.
    bootstrap_matrix : np.ndarray | None
        The (target level, source level) bootstrap correction coefficients,
        applied to the intensities once they are derived. None leaves the
        intensities as the counts gave them.
    flux_corrector : PowerLawFluxCorrector | None
        The ESA transmission factors the Compton-Getting correction recovers
        the source spectrum with. None leaves the intensities in the
        spacecraft frame.
    isn_mask_parameters : pd.DataFrame | None
        The tuning of the ISN mask, which blanks out the pixels the
        interstellar neutral flow dominates once every correction has been
        made. None leaves the whole map in place.

    Returns
    -------
    dict[str, np.ndarray]
        The map variables, each of shape (epoch, esa level, pixel).
    """
    counts = sky_map.data_1d["ena_count"].values
    exposure = sky_map.data_1d["exposure_factor"].values
    bg_rate_exposure = sky_map.data_1d["bg_rate_exposure"].values

    if sputter_matrix is None:
        rate_counts, rate_counts_var = counts, counts
    else:
        rate_counts, rate_counts_var = _sputter_correct_counts(
            counts, sky_map.data_1d["sputter_source_count"].values, sputter_matrix
        )

    # Every ESA level quantity gets a pixel axis to broadcast over the map.
    energy = calibration.energy[:, np.newaxis]
    geometric_factor = calibration.geometric_factor[:, np.newaxis]
    gf_low = calibration.geometric_factor_low[:, np.newaxis]
    gf_high = calibration.geometric_factor_high[:, np.newaxis]

    exposed = exposure > 0

    def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """
        Divide only where the map was exposed, zero elsewhere.

        Parameters
        ----------
        numerator : np.ndarray
            The array being divided.
        denominator : np.ndarray
            The array to divide it by.

        Returns
        -------
        np.ndarray
            The quotient, zero in the pixels that were never exposed.
        """
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(exposure),
            where=exposed,
        )

    # Removing the sputtered counts can take a low-count pixel below zero,
    # which is not a rate the instrument can have observed.
    count_rate = np.maximum(_divide(rate_counts, exposure), 0.0)
    # Poisson uncertainty on the counts, propagated to the rate
    count_rate_stat_uncert = _divide(np.sqrt(rate_counts_var), exposure)

    intensity = _divide(count_rate, geometric_factor * energy)
    intensity_stat_uncert = _divide(count_rate_stat_uncert, geometric_factor * energy)

    # The ISN mask reads the intensity as the instrument observed it, with none
    # of the corrections below made, so that a map is masked the same way
    # regardless of the map descriptor.
    isn_mask = None
    if isn_mask_parameters is not None:
        logger.info("Masking the ISN band out of the map")
        isn_mask = _isn_mask(
            _divide(_divide(counts, exposure), geometric_factor * energy),
            sky_map.az_el_points.values[:, 1],
            isn_mask_parameters,
        )

    # The systematic error is the intensity excursion from the recalibrated
    # G-factor bounds, and the symmetric error is the geometric mean of the two.
    # Intensity goes as 1/G, so the lower G-factor bound gives the upper
    # intensity. It is undefined where that bound is not positive.
    valid = gf_low > 0
    if not valid.all():
        logger.warning(
            "The geometric factor of ESA levels "
            f"{(np.flatnonzero(~valid[:, 0]) + 1).tolist()} is below its lower "
            f"error bound; their systematic errors are left at zero."
        )
    intensity_upper = _divide(count_rate, np.where(valid, gf_low, 1.0) * energy)
    intensity_lower = _divide(count_rate, gf_high * energy)
    intensity_sys_err_plus = np.where(valid, intensity_upper - intensity, 0.0)
    intensity_sys_err_minus = np.where(valid, intensity - intensity_lower, 0.0)

    if bootstrap_matrix is not None:
        (
            intensity,
            intensity_stat_uncert,
            intensity_sys_err_plus,
            intensity_sys_err_minus,
        ) = _bootstrap_correct_intensity(
            intensity,
            intensity_stat_uncert**2,
            calibration,
            bootstrap_matrix,
            sky_map.binning_grid_shape,
            valid,
        )

    bg_rate = _divide(bg_rate_exposure, exposure)
    bg_rate_stat_uncert = np.sqrt(_divide(bg_rate, exposure))
    bg_intensity = _divide(bg_rate, geometric_factor * energy)
    bg_intensity_stat_uncert = _divide(bg_rate_stat_uncert, geometric_factor * energy)

    # The Compton-Getting correction comes last, moving the intensities out of
    # the frame the instrument observed them in. The background is a spectrum
    # of its own, so it is corrected on its own terms rather than with the
    # map's.
    if flux_corrector is not None:
        cos_alpha = _divide(sky_map.data_1d["cos_alpha_exposure"].values, exposure)
        logger.info("Applying the Compton-Getting correction to the intensities")
        (
            intensity,
            (
                intensity_stat_uncert,
                intensity_sys_err_plus,
                intensity_sys_err_minus,
            ),
        ) = _compton_getting_correct_intensity(
            intensity,
            (
                intensity_stat_uncert,
                intensity_sys_err_plus,
                intensity_sys_err_minus,
            ),
            cos_alpha,
            calibration,
            flux_corrector,
        )
        bg_intensity, (bg_intensity_stat_uncert,) = _compton_getting_correct_intensity(
            bg_intensity,
            (bg_intensity_stat_uncert,),
            cos_alpha,
            calibration,
            flux_corrector,
        )

    variables = {
        "ena_count": counts,
        "exposure_factor": exposure,
        "ena_count_rate": count_rate,
        "ena_count_rate_stat_uncert": count_rate_stat_uncert,
        "ena_intensity": intensity,
        "ena_intensity_stat_uncert": intensity_stat_uncert,
        "ena_intensity_sys_err": np.sqrt(
            intensity_sys_err_plus * intensity_sys_err_minus
        ),
        "ena_intensity_sys_err_plus": intensity_sys_err_plus,
        "ena_intensity_sys_err_minus": intensity_sys_err_minus,
        "bg_rate": bg_rate,
        "bg_rate_stat_uncert": bg_rate_stat_uncert,
        "bg_intensity": bg_intensity,
        "bg_intensity_stat_uncert": bg_intensity_stat_uncert,
    }

    # A masked pixel has no ENA measurement to report.
    if isn_mask is not None:
        for name in ISN_MASKED_VARIABLES:
            variables[name] = np.where(isn_mask, FILLVAL_FLOAT, variables[name])

    for name in FILLED_VARIABLES:
        variables[name] = np.where(exposed, variables[name], FILLVAL_FLOAT)

    return variables


def _build_map_dataset(
    sky_map: RectangularSkyMap,
    variables: dict[str, np.ndarray],
    calibration: EsaCalibration,
) -> xr.Dataset:
    """
    Lay the map variables out on the map's sky grid.

    The variables are handed to the map as 1D pixel arrays, which the map
    rewraps onto its longitude/latitude grid and adds its solid angles to.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map being built.
    variables : dict[str, np.ndarray]
        The map variables, each of shape (epoch, esa level, pixel).
    calibration : EsaCalibration
        The energy response the map is binned in, read for the widths of the
        ESA energy passbands.

    Returns
    -------
    xr.Dataset
        The map variables on the (epoch, energy, longitude, latitude) grid,
        with the energy coordinate and its widths.
    """
    dims = sky_map.data_1d["ena_count"].dims
    for name, values in variables.items():
        sky_map.data_1d[name] = xr.DataArray(values.astype(np.float32), dims=dims)
    # These are accumulators, not map variables.
    sky_map.data_1d = sky_map.data_1d.drop_vars(
        ["bg_rate_exposure", "sputter_source_count", "cos_alpha_exposure"],
        errors="ignore",
    )

    dataset = sky_map.to_dataset()

    dataset["energy_delta_minus"] = xr.DataArray(
        calibration.energy_delta_minus, dims=["energy"]
    )
    dataset["energy_delta_plus"] = xr.DataArray(
        calibration.energy_delta_plus, dims=["energy"]
    )

    return dataset
