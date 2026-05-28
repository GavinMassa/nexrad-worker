"""
One-time terrain setup for BACI (Bay Area Convection Index).
Downloads SRTM 30m DEM tiles from OpenTopography, computes terrain aspect
and slope on the BACI domain, saves to /app/sidecar-out/baci_terrain.npz.

Called once at sidecar startup if baci_terrain.npz does not exist.
Runtime: ~30-60 seconds (download + compute). Subsequent startups skip this.
"""
import logging
import numpy as np
from pathlib import Path
import httpx

log = logging.getLogger(__name__)

OUT_DIR      = Path('/app/sidecar-out')
TERRAIN_PATH = OUT_DIR / 'baci_terrain.npz'

# BACI domain
LAT_MIN, LAT_MAX =  36.5,  38.5
LON_MIN, LON_MAX = -123.0, -120.0
GRID_RES         =  0.01   # degrees (~1km)

# Terrain smoothing — smooth the DEM before computing gradients to suppress
# SRTM noise artifacts. sigma=2 gridcells = ~2km smoothing, appropriate for
# resolving ridgeline-scale features without over-smoothing bay/valley boundaries.
SMOOTH_SIGMA = 2.0


def terrain_already_exists() -> bool:
    return TERRAIN_PATH.exists() and TERRAIN_PATH.stat().st_size > 10_000


async def build_terrain() -> bool:
    """
    Download SRTM DEM and compute terrain aspect/slope for the BACI domain.
    Returns True on success, False on failure.
    """
    if terrain_already_exists():
        log.info('[terrain] baci_terrain.npz already exists — skipping build')
        return True

    log.info('[terrain] Building BACI terrain reference (one-time setup)...')
    OUT_DIR.mkdir(exist_ok=True)

    # Build output lat/lon grids
    lats_1d = np.arange(LAT_MIN, LAT_MAX + GRID_RES / 2, GRID_RES, dtype=np.float32)
    lons_1d = np.arange(LON_MIN, LON_MAX + GRID_RES / 2, GRID_RES, dtype=np.float32)
    ny, nx  = len(lats_1d), len(lons_1d)
    log.info(f'[terrain] domain: {ny}×{nx} grid ({LAT_MIN}–{LAT_MAX}°N, {LON_MIN}–{LON_MAX}°E)')

    # Download SRTM 30m DEM from OpenTopography public API (no auth required).
    # Uses the SRTMGL3 (90m) product — sufficient for ridge-scale resolution
    # at our 1km output grid, faster download than 30m.
    # URL format: /API/globaldem?demtype=SRTMGL3&south=&north=&west=&east=&outputFormat=GTiff
    url = (
        'https://portal.opentopography.org/API/globaldem'
        f'?demtype=SRTMGL3'
        f'&south={LAT_MIN}&north={LAT_MAX}'
        f'&west={LON_MIN}&east={LON_MAX}'
        f'&outputFormat=GTiff'
        f'&API_Key=demoapikeyot2022'  # OpenTopography demo key — public, rate limited
    )

    dem_path = OUT_DIR / 'srtm_baci.tif'
    log.info('[terrain] downloading SRTM DEM...')
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream('GET', url) as r:
                r.raise_for_status()
                with open(dem_path, 'wb') as f:
                    async for chunk in r.aiter_bytes(65536):
                        f.write(chunk)
        log.info(f'[terrain] SRTM downloaded: {dem_path.stat().st_size/1e3:.0f}KB')
    except Exception as e:
        log.error(f'[terrain] SRTM download failed: {e}')
        log.info('[terrain] using synthetic terrain fallback')
        return _build_synthetic_terrain(lats_1d, lons_1d)

    # Read GeoTIFF and reproject to our 1km grid
    try:
        import rasterio
        from scipy.ndimage import zoom as ndimage_zoom

        with rasterio.open(str(dem_path)) as src:
            dem_raw = src.read(1).astype(np.float32)
            nodata  = src.nodata
            if nodata is not None:
                dem_raw[dem_raw == nodata] = 0.0
            dem_raw = np.maximum(dem_raw, 0.0)

        zoom_y = ny / dem_raw.shape[0]
        zoom_x = nx / dem_raw.shape[1]
        dem    = ndimage_zoom(dem_raw, (zoom_y, zoom_x), order=1).astype(np.float32)
        dem    = dem[:ny, :nx]   # crop rounding errors

        log.info(f'[terrain] DEM resampled: {dem_raw.shape} → {dem.shape}, '
                 f'max elevation={dem.max():.0f}m')

    except ImportError:
        log.warning('[terrain] rasterio not installed — using synthetic terrain')
        dem_path.unlink(missing_ok=True)
        return _build_synthetic_terrain(lats_1d, lons_1d)
    except Exception as e:
        log.error(f'[terrain] DEM read failed: {e}')
        dem_path.unlink(missing_ok=True)
        return _build_synthetic_terrain(lats_1d, lons_1d)
    finally:
        dem_path.unlink(missing_ok=True)

    return _compute_and_save(dem, lats_1d, lons_1d)


def _build_synthetic_terrain(lats_1d: np.ndarray, lons_1d: np.ndarray) -> bool:
    """
    Synthetic terrain fallback using known Bay Area ridge geometry.
    Approximates the Santa Cruz Mountains and Diablo Range as Gaussian ridges.
    Used when SRTM download fails.
    """
    ny, nx = len(lats_1d), len(lons_1d)
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)
    dem = np.zeros((ny, nx), dtype=np.float32)

    # Santa Cruz Mountains — NW-SE ridge, crest ~1200m, runs from SF to Monterey
    # Approximate crest line: lat = -2.5 * (lon + 122) + 37.4
    scm_dist = np.abs(lats_2d - (-2.5 * (lons_2d + 122.0) + 37.4)) / 0.15
    dem += 1200.0 * np.exp(-0.5 * scm_dist**2)

    # Diablo Range — roughly N-S, crest ~1000m, centered near lon=-121.5
    dr_dist = np.abs(lons_2d - (-121.5)) / 0.12
    dem += 900.0 * np.exp(-0.5 * dr_dist**2)

    # Central Valley floor — flat, near sea level
    valley_mask = (lons_2d > -121.3) & (lats_2d < 38.0) & (lats_2d > 37.0)
    dem[valley_mask] = np.minimum(dem[valley_mask], 30.0)

    log.info(f'[terrain] synthetic DEM built: max={dem.max():.0f}m')
    return _compute_and_save(dem, lats_1d, lons_1d)


def _compute_and_save(dem: np.ndarray, lats_1d: np.ndarray, lons_1d: np.ndarray) -> bool:
    """
    Compute aspect and slope from DEM, save to baci_terrain.npz.
    """
    ny, nx = dem.shape

    from scipy.ndimage import gaussian_filter

    # Smooth DEM to suppress noise before gradient computation
    dem_smooth = gaussian_filter(dem, sigma=SMOOTH_SIGMA)

    # Grid spacing in meters (approximate, valid for Bay Area latitudes)
    lat_center = float((lats_1d[0] + lats_1d[-1]) / 2.0)
    dy_m = GRID_RES * 111_000.0                                         # ~1110m per 0.01°
    dx_m = GRID_RES * 111_000.0 * np.cos(np.radians(lat_center))       # ~900m at 37°N

    # Gradient (dz/dy, dz/dx) in m/m — use central differences
    dz_dy, dz_dx = np.gradient(dem_smooth, dy_m, dx_m)

    # Slope: arctan of gradient magnitude (degrees)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))).astype(np.float32)

    # Aspect: compass bearing of the steepest uphill direction (degrees, 0=N)
    # atan2(dz_dx, dz_dy): dz_dx=east component, dz_dy=north component
    # Result is 0=N, 90=E, 180=S, 270=W (meteorological convention)
    aspect_rad = np.arctan2(dz_dx, dz_dy)
    aspect     = (np.degrees(aspect_rad) % 360).astype(np.float32)
    # Flat areas (slope < 1°) get aspect=0 (no preferred direction)
    aspect[slope < 1.0] = 0.0

    np.savez_compressed(
        str(TERRAIN_PATH),
        aspect=aspect,
        slope=slope,
        lats=lats_1d,
        lons=lons_1d,
    )
    log.info(f'[terrain] saved baci_terrain.npz: {ny}×{nx}, '
             f'slope max={slope.max():.1f}°, '
             f'aspect range=0–360°')
    return True
