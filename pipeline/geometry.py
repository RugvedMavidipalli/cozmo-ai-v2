from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GravityEstimate:
    """The result of figuring out which way is "up" in a reconstruction, and
    where the floor and ceiling sit.

    A raw point cloud has no built-in sense of up, down, floor, or ceiling
    -- it's just a cloud of 3D coordinates. This dataclass holds the answer
    to that question for one reconstruction: which direction is up, and how
    far along that direction the floor and ceiling are.

    Attributes:
        up: A unit vector, in world coordinates, that points away from the
            floor and toward the ceiling.
        floor_height: How far along `up` the floor sits, measured as a
            signed distance from the world origin. It's a signed number
            rather than a plain height because it depends on where the
            world's origin happens to be.
        ceiling_height: The same kind of measurement for the ceiling, or
            `None` if no ceiling could be found -- for example, in an
            outdoor capture, or one where the ceiling was never in view.
        inlier_fraction: What share of all the points fall close to either
            the floor or the ceiling. A low value is a sign that the
            floor/ceiling heights found here might not be very trustworthy.
    """

    up: np.ndarray
    floor_height: float
    ceiling_height: float | None
    inlier_fraction: float

    @property
    def room_height(self) -> float | None:
        """The floor-to-ceiling distance, in metres, or `None` if no ceiling
        height was found for this reconstruction."""
        if self.ceiling_height is None:
            return None
        return self.ceiling_height - self.floor_height


def estimate_gravity(
    points: np.ndarray,
    hint: np.ndarray,
    normals: np.ndarray | None = None,
    cone_degrees: float = 25.0,
) -> GravityEstimate:
    """Works out which direction is "up" for this reconstruction, and finds
    the floor and ceiling heights along that direction.

    The starting point is a rough guess at up -- usually taken from the
    phone's own motion sensors -- which gets refined using the actual
    surfaces in the point cloud. Flat surfaces that are roughly horizontal,
    like floors, ceilings, and countertops, should all point in the same
    direction if the rough guess was close, so averaging them together
    gives a more accurate answer. Once up is known, the floor and ceiling
    heights are found by looking for the two most crowded "slices" of the
    room along that direction, since most points in a typical indoor scan
    end up belonging to one of those two surfaces.

    Args:
        points: The reconstructed 3D points, as an (N, 3) array in world
            coordinates.
        hint: A rough starting guess for the up direction, as a unit
            vector. This is typically `CaptureBundle.gravity_up`, which
            comes from the phone's accelerometer.
        normals: The surface normal at each point, as an (N, 3) array,
            used to refine `hint`. If this is `None` or empty, `hint` is
            used as-is without any refinement.
        cone_degrees: How far, in degrees, a surface normal is allowed to
            lean away from `hint` and still be counted as "roughly
            horizontal" when refining the up direction.

    Returns:
        A `GravityEstimate` holding the refined up direction along with the
        floor and ceiling heights found along it.
    """
    up = hint / np.linalg.norm(hint)
    if normals is not None and len(normals):
        up = _refine_up(normals, up, cone_degrees)

    heights = points @ up
    floor, ceiling, inliers = _floor_and_ceiling(heights)
    return GravityEstimate(
        up=up,
        floor_height=floor,
        ceiling_height=ceiling,
        inlier_fraction=inliers,
    )


def _refine_up(
    normals: np.ndarray, hint: np.ndarray, cone_degrees: float
) -> np.ndarray:
    """Improves a rough guess at "up" by averaging the surface normals that
    are roughly horizontal.

    A surface normal is a vector that points straight out from a surface --
    for a floor or ceiling, that vector points close to straight up or
    straight down. This keeps only the normals that point close enough to
    the rough guess (or its exact opposite, since a ceiling's normal points
    down while a floor's points up), flips the downward-pointing ones
    around so everything faces the same way, and averages what's left to
    get a cleaner estimate.

    Args:
        normals: The surface normal at each point, as an (N, 3) array.
        hint: The rough starting guess for the up direction, as a unit
            vector.
        cone_degrees: How far, in degrees, a normal is allowed to lean
            away from `hint` (in either direction) and still be kept.

    Returns:
        A refined unit vector pointing up, or `hint` unchanged if fewer
        than 100 normals survived the filter -- that's treated as too
        little evidence to trust a refined answer.
    """
    unit = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    alignment = unit @ hint
    threshold = np.cos(np.radians(cone_degrees))
    selected = unit[np.abs(alignment) > threshold]
    if len(selected) < 100:
        return hint
    folded = selected * np.sign(selected @ hint)[:, None]
    refined = folded.mean(axis=0)
    return refined / np.linalg.norm(refined)


def _floor_and_ceiling(
    heights: np.ndarray, bin_size: float = 0.02
) -> tuple[float, float | None, float]:
    """Finds the floor and ceiling heights by looking for the two most
    crowded height values in the point cloud.

    This works by sorting every point's height into a histogram -- a set
    of bins, like a bar chart -- and looking for two tall spikes: one near
    the bottom of the room, and one near the top. That works well because
    in a typical indoor scan, a huge share of points end up belonging to
    either the floor or the ceiling, while the walls in between spread
    their points out much more thinly across many different heights.

    Args:
        heights: How far along the up axis each point sits, as an (N,)
            array of signed distances.
        bin_size: How tall each histogram bin is, in metres.

    Returns:
        A tuple of `(floor_height, ceiling_height, inlier_fraction)`. If no
        plausible ceiling could be found, `ceiling_height` is `None`.
    """
    bins = max(16, int(np.ptp(heights) / bin_size))
    counts, edges = np.histogram(heights, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Only treat a bin as a real spike if it holds at least a quarter as
    # many points as the single busiest bin -- this filters out the thin,
    # noisy bins that come from walls and furniture rather than the floor
    # or ceiling.
    strong = counts > 0.25 * counts.max()
    if not strong.any():
        return float(np.min(heights)), None, 0.0
    indices = np.flatnonzero(strong)

    # The lowest surviving spike is taken as the floor, and the highest as
    # the ceiling.
    floor = float(centers[indices[0]])
    ceiling = float(centers[indices[-1]])
    if ceiling - floor < 1.8:
        # A real room is at least 1.8 m tall floor-to-ceiling. If the gap
        # here is smaller than that, the "ceiling" spike found above is
        # probably just another floor-height surface, like a tabletop, so
        # there's no reliable ceiling to report.
        ceiling_value = None
        inliers = counts[indices[0]] / counts.sum()
    else:
        ceiling_value = ceiling
        # Otherwise, count how many points sit close (within 10 cm) to
        # either surface, as a rough measure of how trustworthy these
        # heights are.
        near_floor = np.abs(heights - floor) < 0.1
        near_ceiling = np.abs(heights - ceiling) < 0.1
        inliers = float((near_floor | near_ceiling).mean())
    return floor, ceiling_value, float(inliers)
