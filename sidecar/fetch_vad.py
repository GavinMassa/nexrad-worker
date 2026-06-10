"""
fetch_vad.py — NEXRAD VAD Wind Profile ingest for observed low-level hodographs.

Fetches the most recent VAD Wind Profile (VWP, NIDS product 48) for each
NEXRAD site from the NWS TGFTP server. The product is a BINARY NEXRAD Level III
file (not ASCII) — it is decoded with MetPy's Level3File, whose tabular page
already provides U and V in m/s plus the radar lat/lon.

URL pattern (always serves the most recent volume scan):
  https://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/DS.48vwp/SI.{site}/sn.last
  where {site} is lowercase 4-char ICAO (e.g. ktlx, kfws, kdvn).

HYBRID design (see vad_worker / blend.py):
  VAD profiles are typically shallow (top ~2-3 km on quiet days), so site-level
  Bunkers storm motion (which needs the 5.5-6 km layer) is unreliable. We
  therefore do NOT compute storm motion here. Instead we provide the OBSERVED
  low-level hodograph by interpolating each profile to fixed AGL heights
  (500 m and 1000 m). blend.py combines these with the RTMA surface wind and
  the model's (RAP/RRFS) full-depth Bunkers storm motion to form an
  observation-anchored 0-1 km SRH, used only where VAD coverage exists.

Output of fetch_vad_profiles(): list of dicts, one per site with valid data:
  {
    'site': 'KTLX', 'lat': 35.333, 'lon': -97.278,
    'levels': [{'z_m': 488.0, 'u_ms': 0.8, 'v_ms': 11.0}, ...]  # ascending z
  }
compute_site_layer_winds() then adds u500/v500/u1000/v1000 (or None if the
profile does not reach that height).
"""
import asyncio
import io
import logging
import warnings as _warnings

import httpx
import numpy as np

# MetPy decodes the binary NIDS product 48 (VWP). Suppress pint warnings on import.
with _warnings.catch_warnings():
    _warnings.simplefilter('ignore')
    from metpy.io import Level3File

log = logging.getLogger(__name__)

TGFTP_BASE = ('https://tgftp.nws.noaa.gov/SL.us008001/DF.of/DC.radar/'
              'DS.48vwp/SI.{site}/sn.last')

# Standard AGL heights (m) at which the observed hodograph is sampled.
VAD_HEIGHTS = (500.0, 1000.0)

# All CONUS NEXRAD sites with lat/lon (WSR-88D locations). The decoded file
# carries its own lat/lon which we prefer; this list drives which sites to poll.
NEXRAD_SITES = [
    ('KABR', 45.4558, -98.4133), ('KABX', 35.1497, -106.8236),
    ('KAKQ', 36.9839, -77.0078), ('KAMA', 35.2333, -101.7092),
    ('KAMX', 25.6111, -80.4128), ('KAPX', 44.9072, -84.7197),
    ('KARX', 43.8228, -91.1914), ('KATX', 48.1944, -122.4958),
    ('KBBX', 39.4961, -121.6317), ('KBGM', 42.1997, -75.9847),
    ('KBHX', 40.4983, -124.2919), ('KBIS', 46.7708, -100.7603),
    ('KBLX', 45.8539, -108.6067), ('KBMX', 33.1722, -86.7700),
    ('KBOX', 41.9558, -71.1369), ('KBRO', 25.9161, -97.4189),
    ('KBUF', 42.9486, -78.7369), ('KBYX', 24.5975, -81.7033),
    ('KCAE', 33.9486, -81.1183), ('KCBW', 46.0392, -67.8064),
    ('KCBX', 43.4906, -116.2358), ('KCCX', 40.9228, -78.0036),
    ('KCLE', 41.4131, -81.8597), ('KCLX', 32.6556, -81.0428),
    ('KCRP', 27.7840, -97.5112), ('KCXX', 44.5111, -73.1664),
    ('KCYS', 41.1519, -104.8061), ('KDAX', 38.5011, -121.6775),
    ('KDDC', 37.7708, -99.9686), ('KDFX', 29.2731, -100.2803),
    ('KDGX', 32.2800, -89.9844), ('KDIX', 39.9469, -74.4108),
    ('KDLH', 46.8369, -92.2097), ('KDMX', 41.7311, -93.7228),
    ('KDOX', 38.8256, -75.4400), ('KDTX', 42.6997, -83.4717),
    ('KDVN', 41.6117, -90.5808), ('KDYX', 32.5383, -99.2542),
    ('KEAX', 38.8103, -94.2644), ('KEMX', 31.8931, -110.6306),
    ('KENX', 42.5864, -74.0639), ('KEOX', 31.4603, -85.4594),
    ('KEPZ', 31.8731, -106.6983), ('KESX', 35.7011, -114.8917),
    ('KEVX', 30.5644, -85.9219), ('KEWX', 29.7039, -98.0286),
    ('KEYX', 35.0978, -117.5606), ('KFCX', 37.0242, -80.2739),
    ('KFDR', 34.3622, -98.9764), ('KFDX', 34.6344, -103.6289),
    ('KFFC', 33.3636, -84.5658), ('KFSD', 43.5878, -96.7294),
    ('KFSX', 34.5744, -111.1983), ('KFTG', 39.7867, -104.5458),
    ('KFWS', 32.5731, -97.3031), ('KGGW', 48.2064, -106.6253),
    ('KGJX', 39.0622, -108.2139), ('KGLD', 39.3667, -101.7003),
    ('KGRB', 44.4986, -88.1111), ('KGRK', 30.7219, -97.3831),
    ('KGRR', 42.8939, -85.5447), ('KGSP', 34.8831, -82.2203),
    ('KGWX', 33.8967, -88.3289), ('KGYX', 43.8914, -70.2569),
    ('KHDX', 33.0769, -106.1233), ('KHGX', 29.4719, -95.0792),
    ('KHNX', 36.3142, -119.6317), ('KHPX', 36.7367, -87.2850),
    ('KHTX', 34.9306, -86.0833), ('KICT', 37.6544, -97.4422),
    ('KICX', 37.5908, -112.8628), ('KILN', 39.4203, -83.8217),
    ('KILX', 40.1506, -89.3367), ('KIND', 39.7075, -86.2803),
    ('KINX', 36.1750, -95.5647), ('KIWA', 33.2892, -111.6700),
    ('KIWX', 41.3589, -85.7000), ('KJAX', 30.4850, -81.7019),
    ('KJGX', 32.6758, -83.3511), ('KJKL', 37.5906, -83.3131),
    ('KLBB', 33.6541, -101.8142), ('KLCH', 30.1253, -93.2161),
    ('KLGX', 47.1173, -124.1073), ('KLNX', 41.9578, -100.5758),
    ('KLOT', 41.6044, -88.0847), ('KLRX', 40.7397, -116.8028),
    ('KLSX', 38.6986, -90.6828), ('KLTX', 33.9894, -78.4292),
    ('KLVX', 37.9753, -85.9439), ('KLWX', 38.9753, -77.4778),
    ('KLZK', 34.8365, -92.2622), ('KMAF', 31.9433, -102.1894),
    ('KMAX', 42.0811, -122.7178), ('KMBX', 48.3928, -100.8644),
    ('KMHX', 34.7761, -76.8764), ('KMKX', 42.9678, -88.5506),
    ('KMLB', 28.1131, -80.6544), ('KMOB', 30.6797, -88.2397),
    ('KMPX', 44.8489, -93.5653), ('KMQT', 46.5311, -87.5483),
    ('KMRX', 36.1686, -83.4017), ('KMSX', 47.0411, -113.9861),
    ('KMTX', 41.2628, -112.4478), ('KMUX', 37.1553, -121.8983),
    ('KMVX', 47.5278, -97.3253), ('KMXX', 32.5369, -85.7897),
    ('KNKX', 32.9189, -117.0419), ('KNQA', 35.3447, -89.8731),
    ('KOAX', 41.3203, -96.3667), ('KOHX', 36.2472, -86.5628),
    ('KOKX', 40.8656, -72.8644), ('KOTX', 47.6806, -117.6258),
    ('KPAH', 37.0681, -88.7719), ('KPBZ', 40.5317, -80.2181),
    ('KPDT', 45.6908, -118.8528), ('KPOE', 31.1556, -92.9758),
    ('KPUX', 38.4595, -104.1814), ('KRAX', 35.6656, -78.4897),
    ('KRGX', 39.7542, -119.4619), ('KRIW', 43.0661, -108.4772),
    ('KRLX', 38.3111, -81.7228), ('KRTX', 45.7153, -122.9653),
    ('KSFX', 43.1058, -112.6861), ('KSGF', 37.2353, -93.4003),
    ('KSHV', 32.4508, -93.8411), ('KSJT', 31.3714, -100.4922),
    ('KSOX', 33.8178, -117.6358), ('KSRX', 35.2908, -94.3617),
    ('KTBW', 27.7056, -82.4019), ('KTFX', 47.4597, -111.3853),
    ('KTLH', 30.3983, -84.3293), ('KTLX', 35.3331, -97.2778),
    ('KTWX', 38.9969, -96.2325), ('KUDX', 44.1250, -102.8294),
    ('KUEX', 40.3211, -98.4417), ('KVAX', 30.8903, -83.0019),
    ('KVBX', 34.8381, -120.3978), ('KVNX', 36.7408, -98.1278),
    ('KVTX', 34.4117, -119.1789), ('KVWX', 38.2603, -87.7247),
    ('KYUX', 32.4953, -114.6569),
]


def _parse_vwp_level3(raw: bytes):
    """
    Decode a binary NIDS product-48 VWP file into a level list.

    Returns (levels, lat, lon) where levels is a list of
    {'z_m', 'u_ms', 'v_ms'} sorted ascending by height, and lat/lon are the
    radar coordinates from the file header. Returns ([], None, None) on failure.

    The decoded tabular page is a fixed-column text table:
        ALT      U       V       W    DIR   SPD   RMS  ...
       100ft    m/s     m/s    cm/s   deg   kts   kts
        016     0.8    11.0     NA    184   021   4.1  ...
    ALT is in hundreds of feet above radar level; U/V are already in m/s.
    Rows with U or V == 'NA' (missing) are skipped.
    """
    try:
        with _warnings.catch_warnings():
            _warnings.simplefilter('ignore')
            f = Level3File(io.BytesIO(raw))
        if not getattr(f, 'tab_pages', None):
            return [], None, None
        text = ''.join(f.tab_pages[0])
        lat = float(f.lat) if getattr(f, 'lat', None) is not None else None
        lon = float(f.lon) if getattr(f, 'lon', None) is not None else None

        levels = []
        seen_alt = set()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                alt_100ft = int(parts[0])     # data rows begin with an integer ALT
            except ValueError:
                continue                       # header / units / date lines
            try:
                u_ms = float(parts[1])
                v_ms = float(parts[2])
            except ValueError:
                continue                       # U or V == 'NA' (missing wind)
            if alt_100ft in seen_alt:
                continue                       # dedupe repeated altitude rows
            z_m = alt_100ft * 100.0 * 0.3048   # hundreds of ft (ARL) → metres
            if not (0.0 <= z_m <= 18_000.0):
                continue
            seen_alt.add(alt_100ft)
            levels.append({'z_m': z_m, 'u_ms': u_ms, 'v_ms': v_ms})
        levels.sort(key=lambda x: x['z_m'])
        return levels, lat, lon
    except Exception:
        return [], None, None


async def _fetch_one_site(
    client: httpx.AsyncClient,
    site: str,
    fallback_lat: float,
    fallback_lon: float,
) -> dict | None:
    """Fetch and decode the VWP for one NEXRAD site. Returns None on failure."""
    url = TGFTP_BASE.format(site=site.lower())
    try:
        r = await client.get(url, timeout=10.0)
        if r.status_code != 200:
            return None
        levels, lat, lon = _parse_vwp_level3(r.content)
        if len(levels) < 3:
            return None                        # too few levels to be useful
        return {
            'site':   site,
            'lat':    lat if lat is not None else fallback_lat,
            'lon':    lon if lon is not None else fallback_lon,
            'levels': levels,
        }
    except Exception:
        return None                            # network error — silently skip


async def fetch_vad_profiles() -> list[dict]:
    """
    Fetch current VAD wind profiles for all NEXRAD sites concurrently.

    Returns list of site dicts (see module docstring). Empty list on failure.
    """
    async with httpx.AsyncClient(
        timeout=15.0,
        limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
    ) as client:
        tasks = [
            _fetch_one_site(client, site, lat, lon)
            for site, lat, lon in NEXRAD_SITES
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    profiles = [r for r in results if isinstance(r, dict) and r is not None]
    log.info(f'[vad] fetched {len(profiles)}/{len(NEXRAD_SITES)} sites '
             f'with valid profiles')
    return profiles


def _interp_uv(levels: list, z: float):
    """
    Interpolate observed (u, v) to height z (m AGL).

    Below the lowest observed level, np.interp clamps to the lowest value
    (acceptable for the small gap from ~450 m down to the 500 m target).
    Above the top observed level, returns (None, None) — we never fabricate
    winds above what the radar actually sampled.
    """
    zs = np.array([lv['z_m']  for lv in levels], dtype=np.float64)
    us = np.array([lv['u_ms'] for lv in levels], dtype=np.float64)
    vs = np.array([lv['v_ms'] for lv in levels], dtype=np.float64)
    if z > zs[-1] + 50.0:                       # above radar's top sample
        return None, None
    return float(np.interp(z, zs, us)), float(np.interp(z, zs, vs))


def compute_site_layer_winds(profiles: list[dict]) -> list[dict]:
    """
    Add observed hodograph winds at VAD_HEIGHTS (500 m, 1000 m AGL) to each
    profile: keys u500/v500/u1000/v1000 (None where the profile is too shallow).

    Storm motion is intentionally NOT computed here — VAD profiles are usually
    too shallow for a reliable 0-6 km Bunkers estimate, so blend.py uses the
    model's full-depth USTM/VSTM instead.
    """
    for prof in profiles:
        levels = prof['levels']
        u5, v5   = _interp_uv(levels, 500.0)
        u10, v10 = _interp_uv(levels, 1000.0)
        prof['u500'], prof['v500']   = u5, v5
        prof['u1000'], prof['v1000'] = u10, v10
    return profiles
