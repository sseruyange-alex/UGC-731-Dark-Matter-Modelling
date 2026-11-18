"""
UGC 731 - Dark Matter Modelling Pipeline
========================================
Input files (same folder as this script)
    UGC731_Optical_RC.txt          cols: R(arcsec) R(kpc) V(km/s) err(km/s)
    UGC731_HI_RC.txt               BBarolo 3DFIT output (cols 0 RAD_kpc, 1 RAD_arcs, 2 VROT, 4 INC, 5 PA,
                                   9 XPOS, 10 YPOS, 13 E_VROT1, 14 E_VROT2)
    velo_u731s30cl_regrid.fits     WHISP 30" cube (units W.U.; 1 W.U. = 5 mJy/beam)
    <WISE_FITS>                    WISE W1 cutout (DN)

Output goes to ./UGC731_output/
    Fig1 ... Fig9 (Fig9 = WISE W1 image of the galaxy), HI_surface_density_profile.txt, WISE_W1_stellar_profile.txt, vstars_profile.txt,
    vgas_profile.txt, UGC731_DM_results.txt, UGC731_sensitivity.txt, UGC731_moment0_from_cube.fits

FITS handling
-------------
All FITS files are read and written with astropy (astropy.io.fits + astropy.wcs):
  * BSCALE/BZERO, BITPIX and byte order are handled by astropy (no hand-written header parser);
  * extra degenerate axes (Stokes, 4-axis cubes) are squeezed away;
  * sky offsets (East, North) of every pixel come from the file's own WCS, through the pixel-scale
    matrix, so axis direction, pixel size and any rotation are taken from the header;
  * the galaxy centre used for the WISE image is the BBarolo centre converted to RA/Dec with the
    cube WCS and then to WISE pixels with the WISE WCS (both images therefore share one sky centre);
  * the moment-0 FITS written at the end carries a correct WCS for the cropped box.

WISE stellar profile
--------------------
The 40-point W1 profile is measured here from the cutout (3" elliptical annuli, median, sky from the cutout edge).
Every annulus gets an EMPIRICAL error (the same annuli placed at random blank-sky positions), the stellar disc is fitted
weighted by those errors, and Step 6 tests the sky level (+/- WISE_SKY_SYST), the profile statistic (star-masked mean),
the fit range and the weighting. The W1 disc is detected at >~3 sigma only inside ~80" (~4 kpc).

Requirements:  numpy, scipy, matplotlib, astropy
"""
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.special import i0, i1, k0, k1, ellipk
from scipy.ndimage import gaussian_filter, label, generate_binary_structure, zoom
from scipy.ndimage import label as ndi_label   # safe alias: the name `label` is reused as a loop variable in the plots
from astropy.io import fits
from astropy.wcs import WCS
warnings.filterwarnings("ignore")

HERE   = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "UGC731_output")
os.makedirs(OUTDIR, exist_ok=True)
def _writable(d):
    try:
        os.makedirs(d, exist_ok=True)
        t = os.path.join(d, "_write_test.tmp")
        with open(t, "wb") as f: f.write(b"x")
        os.remove(t)
        return True
    except OSError:
        return False

def _fallback_dirs():
    import tempfile
    return [os.path.join(os.path.expanduser("~"), "UGC731_output"),
            os.path.join(tempfile.gettempdir(), "UGC731_output")]

if not _writable(OUTDIR):                      # e.g. cloud-synced / locked / network folder
    for _d in _fallback_dirs():
        if _writable(_d):
            print(f"WARNING: cannot write to {OUTDIR}\n         using {_d} instead")
            OUTDIR = _d
            break

def outpath(fn): return os.path.join(OUTDIR, fn)

def safe_savefig(fig, fn, **kw):
    """Save a figure without ever crashing the pipeline. Tries (1) the normal path, (2) render to memory and write the
    bytes, (3) the same name with '_new' (if the old file is locked by a viewer), (4) the fallback folders."""
    import io
    stem, ext = os.path.splitext(fn)
    tries = [(OUTDIR, fn), (OUTDIR, stem + "_new" + ext)] + [(d, fn) for d in _fallback_dirs()]
    for d, name in tries:
        p = os.path.join(d, name)
        try:
            os.makedirs(d, exist_ok=True)
            fig.savefig(p, **kw)
            return p
        except OSError:
            try:
                buf = io.BytesIO(); fig.savefig(buf, format=ext[1:], **kw)
                with open(p, "wb") as f: f.write(buf.getvalue())
                return p
            except OSError:
                continue
    print(f"WARNING: could not save {fn} anywhere")
    return None
def inpath(fn):  return os.path.join(HERE, fn)

class Tee:                                   # print to screen AND to the results file
    def __init__(self, *fhs): self.fhs = [f for f in fhs if f is not None]
    def write(self, s):
        for f in self.fhs:
            try: f.write(s)
            except Exception: pass
    def flush(self):
        for f in self.fhs:
            try: f.flush()
            except Exception: pass
_log = open(outpath("UGC731_DM_results.txt"), "w")
sys.stdout = Tee(sys.stdout, _log)

# =====================================================================
# 1. CONSTANTS AND SETTINGS  (everything you may want to change is here)
# =====================================================================
G          = 4.3010e-6         # kpc (km/s)^2 / Msun
D_MPC      = 10.3
KPC_ARCS   = D_MPC / 206.265   # kpc per arcsec  (0.04994)
INC_DEG    = 58.0              # inclination used by the optical RC and for deprojection
ML_W1      = 0.5               # 3.4 um stellar M/L (Korsaga+2019)
M_SUN_W1   = 3.24              # Willmer (2018)
HELIUM     = 1.4               # gas = HELIUM * HI (He + metals); 1.33 = He only
SPLICE_KPC = 2.2               # optical for r < this, BBarolo HI for r >= this
BEAM_ARCS  = 30.0              # WHISP 30" maps are 30" x 30" (Swaters+2002), i.e. circular
WU_MJY     = 5.0               # 1 W.U. = 5 mJy/beam
VEL_RANGE  = (520.0, 750.0)    # km/s used for moments
MASK_FWHM  = 60.0              # arcsec: cube smoothed to this resolution to build the mask
MASK_SIGMA = 3.0               # clip level in units of the SMOOTHED noise
MASK_RMAX  = 400.0             # arcsec from the centre within which the mask may grow
RING_W     = 8.0               # arcsec, = BBarolo ring width
EXTRAP_OUT = 300.0             # arcsec, HI rings are measured out to here

print("=" * 66)
print("UGC 731 - Dark Matter Modelling Pipeline")
print("=" * 66)
print(f"D = {D_MPC} Mpc, scale = {KPC_ARCS*1000:.2f} pc/arcsec, i = {INC_DEG} deg, "
      f"M/L(W1) = {ML_W1}, gas factor = {HELIUM}")

# =====================================================================
# 2. UTILITIES
# =====================================================================
def find_input(name, patterns=()):
    """Path of an input file next to this script. If `name` is not there, try the glob `patterns` (e.g. a browser may have
    saved the file as '<name>_1.fits' or '<name> (1).fits'); if nothing matches, stop with a clear message."""
    import glob
    p = inpath(name)
    if os.path.exists(p):
        return p
    for pat in patterns:
        hits = sorted(glob.glob(inpath(pat)))
        if hits:
            print(f"  NOTE: '{name}' not found; using '{os.path.basename(hits[0])}' instead")
            return hits[0]
    have = ", ".join(sorted(os.path.basename(f) for f in glob.glob(inpath("*.fits")))) or "none"
    raise SystemExit(f"Input file not found: {name}\n  looked in: {HERE}\n  .fits files in that folder: {have}\n"
                     f"  -> copy the file there (or set the name at the top of the step that uses it).")

def open_fits(fn, ndim):
    """Read a FITS file with astropy. Returns (data float array squeezed to `ndim` dims, header, celestial WCS).

    astropy applies BSCALE/BZERO and handles BITPIX and byte order. Degenerate axes (e.g. a Stokes axis)
    are removed with squeeze(); NaNs are kept (the caller decides what to do with them).
    """
    with fits.open(fn, memmap=False) as hd:
        hdu = next((h for h in hd if getattr(h, "data", None) is not None and h.data.ndim >= ndim), None)
        if hdu is None:
            raise ValueError(f"{fn}: no image HDU with >= {ndim} axes")
        data = np.squeeze(np.asarray(hdu.data, dtype=np.float64 if ndim == 2 else np.float32))
        hdr = hdu.header.copy()
    if data.ndim != ndim:
        raise ValueError(f"{fn}: expected a {ndim}-D array after squeezing, got shape {data.shape}")
    return data, hdr, celestial_wcs(hdr)

def celestial_wcs(hdr):
    """Celestial WCS of a header. Some archive headers (e.g. WHISP cubes) carry unit strings astropy rejects, such as
    CUNIT3 = 'km/S' (capital S): the unit strings are normalised first, and if the 3-D WCS still fails only the two
    sky axes are used (the spectral axis is handled separately from CRVAL3/CDELT3)."""
    h2 = hdr.copy()
    fix = {"km/s": "km/s", "m/s": "m/s", "deg": "deg", "degree": "deg", "degrees": "deg", "arcsec": "arcsec"}
    for k in list(h2.keys()):
        if k.startswith("CUNIT") and isinstance(h2[k], str) and h2[k].strip().lower() in fix:
            h2[k] = fix[h2[k].strip().lower()]
    try:
        return WCS(h2).celestial
    except Exception:
        return WCS(h2, naxis=2)

def offsets_arcsec(wcs, dx, dy):
    """Sky offsets (East, North) in arcsec for pixel offsets (dx, dy) from a reference pixel, using the WCS
    pixel-scale matrix (= CDELT x PC): handles axis direction, pixel size and rotation of the image."""
    M = wcs.pixel_scale_matrix*3600.0           # [[dRA/dx, dRA/dy],[dDec/dx, dDec/dy]] in arcsec/pixel
    return M[0, 0]*dx + M[0, 1]*dy, M[1, 0]*dx + M[1, 1]*dy

def pix_scale(wcs):
    M = wcs.pixel_scale_matrix*3600.0
    return float(np.sqrt(abs(np.linalg.det(M))))   # arcsec per pixel (geometric mean)

def wise_profile_from_image(img, wcs, cx, cy, pa_deg, inc_deg, radii, dr, sky_rmin):
    """Azimuthally-averaged, sky-subtracted radial profile in elliptical annuli (median per annulus).

    img       2-D image in DN;  wcs  its celestial WCS;  cx, cy  galaxy centre in pixels (0-indexed)
    pa_deg    major-axis position angle [deg, North through East];  inc_deg  inclination
    radii     annulus centres along the MAJOR axis [arcsec];  dr  annulus width [arcsec]
    sky_rmin  the sky is the median of all pixels farther than this circular radius [arcsec]
    Returns (profile, sky): median DN/pixel per annulus minus the sky, and the sky in DN/pixel.
    """
    yy, xx = np.indices(img.shape)
    dE, dN = offsets_arcsec(wcs, xx - cx, yy - cy)         # East / North offsets from the centre [arcsec]
    t = np.radians(pa_deg)
    a = dE*np.sin(t) + dN*np.cos(t)                        # along the major axis
    b = dE*np.cos(t) - dN*np.sin(t)                        # along the minor axis
    rr = np.hypot(a, b/np.cos(np.radians(inc_deg)))
    sky = np.nanmedian(img[np.hypot(dE, dN) > sky_rmin])
    prof = np.array([np.nanmedian(img[(rr >= r - dr/2) & (rr < r + dr/2)]) for r in radii]) - sky
    return prof, sky

def wise_profile_masked_mean(img, wcs, cx, cy, pa_deg, inc_deg, radii, dr, sky_rmin, star_dn=15.0):
    """Alternative W1 profile (used only as a systematic check): MEAN of the pixels left after masking bright compact
    sources (> star_dn above the sky, grown by 4 pixels). The sky is taken with the SAME statistic (mean of the unmasked
    pixels beyond sky_rmin), because a masked mean compared with a median sky leaves a spurious positive tail
    (the mean of this image's sky is ~1 DN above its median because of faint stars)."""
    from scipy.ndimage import binary_dilation
    yy, xx = np.indices(img.shape)
    dE, dN = offsets_arcsec(wcs, xx - cx, yy - cy)
    t = np.radians(pa_deg)
    a = dE*np.sin(t) + dN*np.cos(t)
    b = dE*np.cos(t) - dN*np.sin(t)
    rr = np.hypot(a, b/np.cos(np.radians(inc_deg)))
    far = np.hypot(dE, dN) > sky_rmin
    bad = binary_dilation((img - np.nanmedian(img[far])) > star_dn, iterations=4) | ~np.isfinite(img)
    sky = np.mean(img[far & ~bad])
    prof = np.array([np.mean(img[(rr >= r - dr/2) & (rr < r + dr/2) & ~bad]) for r in radii]) - sky
    return prof, sky

def wise_annulus_noise(img, wcs, cx, cy, pa_deg, inc_deg, radii, dr, sky, n=200, min_sep=170.0, seed=1):
    """Empirical error of every annulus of wise_profile_from_image (median statistic).
    The SAME elliptical annuli are placed at n random blank-sky positions (>= min_sep arcsec from the galaxy; annuli cut by
    the image edge need >= 5 pixels) and their median minus `sky` is recorded, so the scatter contains the real
    noise + faint-star confusion + correlated pixels, which a pixel-noise formula would miss. The scatter is the robust
    1.4826*MAD (a plain standard deviation of the few-pixel inner annuli is dominated by the odd star and would make the
    bright core look 'noisy', so the fit would ignore it).
    Returns (sigma, pedestal) per annulus in DN/pixel; pedestal = mean offset of blank-sky annuli from `sky`."""
    yy, xx = np.indices(img.shape)
    t = np.radians(pa_deg); cosi_ = np.cos(np.radians(inc_deg)); rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < n:
        x0 = rng.uniform(40, img.shape[1] - 40); y0 = rng.uniform(40, img.shape[0] - 40)
        if np.hypot(*offsets_arcsec(wcs, x0 - cx, y0 - cy)) < min_sep:
            continue
        dE, dN = offsets_arcsec(wcs, xx - x0, yy - y0)
        a = dE*np.sin(t) + dN*np.cos(t)
        b = dE*np.cos(t) - dN*np.sin(t)
        rr = np.hypot(a, b/cosi_)
        row = []
        for r in radii:
            m = (rr >= r - dr/2) & (rr < r + dr/2)
            row.append(np.nanmedian(img[m]) if m.sum() >= 5 else np.nan)
        rows.append(row)
    off = np.array(rows) - sky
    med = np.nanmedian(off, axis=0)
    sig = 1.4826*np.nanmedian(np.abs(off - med), axis=0)       # robust (MAD) scatter: rare star hits do not inflate it
    return sig, np.nanmean(off, axis=0)

def fit_stellar_disc(flux, sig_flux, r_arcs, pix, magzp, rmin_kpc=0.5, rmax_kpc=4.0):
    """Face-on exponential stellar disc fitted to a W1 annulus profile (flux = sky-subtracted DN/pixel).
    DN -> mag/arcsec^2 -> L_sun/pc^2 -> M_sun/pc^2 (M/L = ML_W1) -> face-on (x cos i); this conversion is linear in the flux,
    so an error on the flux maps to an error on Sigma by the same factor. sig_flux = None gives the unweighted fit.
    Returns (Sigma0 [M_sun/pc^2], h [kpc], (err_Sigma0, err_h), number of points used)."""
    good = flux > 0
    sb = np.full_like(flux, np.nan); sb[good] = magzp - 2.5*np.log10(flux[good]/pix**2)
    S_obs = ML_W1*10**(0.4*(M_SUN_W1 + 21.572 - sb))
    S_face = S_obs*np.cos(np.radians(INC_DEG))
    rk = r_arcs*KPC_ARCS
    m = (rk > rmin_kpc) & (rk < rmax_kpc) & (S_obs > 1.0) & np.isfinite(S_face)
    sig = None if sig_flux is None else S_face[m]*sig_flux[m]/flux[m]
    p, cov = curve_fit(exp_disk, rk[m], S_face[m], p0=[30, 1.5], sigma=sig, absolute_sigma=sig is not None, maxfev=5000)
    return p[0], p[1], tuple(np.sqrt(np.diag(cov))), int(m.sum())

def lowess(x, y, frac=0.3, it=2):
    """Simple robust LOWESS (tricube, local linear). Returns smoothed y at x (x sorted)."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    k = max(int(np.ceil(frac*n)), 3); yfit = np.zeros(n); rw = np.ones(n)
    for _ in range(it + 1):
        for i in range(n):
            d = np.abs(x - x[i]); h = np.sort(d)[k-1]; h = max(h, 1e-12)
            w = np.clip(1 - (d/h)**3, 0, None)**3 * rw
            A = np.vstack([np.ones(n), x - x[i]]).T
            W = np.sqrt(w)[:, None]
            beta = np.linalg.lstsq(A*W, y*W[:, 0], rcond=None)[0]
            yfit[i] = beta[0]
        res = y - yfit; s = np.median(np.abs(res))
        rw = np.clip(1 - (res/(6*s + 1e-12))**2, 0, None)**2 if s > 0 else np.ones(n)
    return yfit

def exp_disk(r, S0, h):
    return S0 * np.exp(-r / h)

def sigma_hole(r, S0, h, A, w):
    """Exponential disc with a Gaussian central suppression: S0 exp(-r/h) [1 - A exp(-r^2/2w^2)]."""
    return S0 * np.exp(-r / h) * (1.0 - A * np.exp(-0.5 * (r / w)**2))

# ---- circular velocity of a razor-thin disc ------------------------
def freeman_vcirc(r, S0_pc2, h_kpc):
    """Freeman (1970) exponential thin disc.  S0 in Msun/pc^2, h in kpc -> km/s."""
    y = np.maximum(np.asarray(r, float) / (2.0*h_kpc), 1e-8)
    V2 = 4*np.pi*G*(S0_pc2*1e6)*h_kpc*y**2*(i0(y)*k0(y) - i1(y)*k1(y))
    return np.sqrt(np.maximum(V2, 0.0))

def thin_disc_vcirc(R_eval, a_tab, sig_tab_pc2, da=0.01):
    """Circular velocity of a razor-thin disc with ARBITRARY tabulated face-on Sigma(a) [Msun/pc^2], a [kpc].
    Exact ring-by-ring elliptic-integral potential; Sigma held constant inside the first tabulated radius
    and 0 beyond the last one.  Validated against freeman_vcirc (< 0.4 %)."""
    a_tab = np.asarray(a_tab, float); S = np.asarray(sig_tab_pc2, float)*1e6
    amax = a_tab.max()
    a  = (np.arange(int(amax/da)) + 0.5)*da
    Sa = np.interp(a, a_tab, S)
    dm = 2*np.pi*a*Sa*da
    Rg = np.arange(1, int(amax/da) + 1)*da
    Phi = np.empty(len(Rg))
    for j, R in enumerate(Rg):
        Phi[j] = -(2*G/np.pi)*np.sum(dm*ellipk(4*a*R/(a + R)**2)/(a + R))
    V2 = Rg*np.gradient(Phi, Rg)
    return np.sqrt(np.maximum(np.interp(R_eval, Rg, V2, left=0.0), 0.0))

# ---- halo circular velocities -------------------------------------
def V_Burkert(r, rho0, rc):
    """Burkert (1995). M = pi rho0 rc^3 [ln(1+x^2) + 2 ln(1+x) - 2 atan x]."""
    r = np.maximum(r, 1e-6); x = r/rc
    M = np.pi*rho0*rc**3*(np.log(1 + x**2) + 2*np.log(1 + x) - 2*np.arctan(x))
    return np.sqrt(np.maximum(G*M/r, 0.0))

def V_NFW(r, rhos, rs):
    """NFW (1996/97)."""
    r = np.maximum(r, 1e-6); x = r/rs
    M = 4*np.pi*rhos*rs**3*(np.log(1 + x) - x/(1 + x))
    return np.sqrt(np.maximum(G*M/r, 0.0))

from scipy.special import hyp2f1
def V_gNFW(r, rhos, rs, gam):
    """Generalised NFW: rho = rhos / [(r/rs)^gam (1+r/rs)^(3-gam)].  gam=1 -> NFW, gam=0 -> cored."""
    r = np.maximum(r, 1e-6); x = r/rs
    I = x**(3 - gam)/(3 - gam)*hyp2f1(3 - gam, 3 - gam, 4 - gam, -x)
    return np.sqrt(np.maximum(G*4*np.pi*rhos*rs**3*I/r, 0.0))

# self-tests of the halo formulas against numerical integration of the densities
from scipy.integrate import quad
_r0, _rho, _rc = 3.0, 1e8, 1.2
_Mb = quad(lambda s: 4*np.pi*s*s*_rho/((1 + s/_rc)*(1 + (s/_rc)**2)), 0, _r0)[0]
_Mn = quad(lambda s: 4*np.pi*s*s*_rho/((s/_rc)*(1 + s/_rc)**2), 0, _r0)[0]
assert abs(V_Burkert(_r0, _rho, _rc)/np.sqrt(G*_Mb/_r0) - 1) < 1e-6
assert abs(V_NFW(_r0, _rho, _rc)/np.sqrt(G*_Mn/_r0) - 1) < 1e-6
_Mg = quad(lambda s_: 4*np.pi*s_*s_*_rho/((s_/_rc)**0.4*(1 + s_/_rc)**2.6), 0, _r0)[0]
assert abs(V_gNFW(_r0, _rho, _rc, 0.4)/np.sqrt(G*_Mg/_r0) - 1) < 1e-6
assert abs(V_gNFW(_r0, _rho, _rc, 1.0)/V_NFW(_r0, _rho, _rc) - 1) < 1e-6
_S0, _h = 40., 2.
_a = np.linspace(0, 40, 2001)
assert np.max(np.abs(thin_disc_vcirc(np.array([1., 3., 6.]), _a, _S0*np.exp(-_a/_h))/
                     freeman_vcirc(np.array([1., 3., 6.]), _S0, _h) - 1)) < 0.01
print("Self-tests passed: Burkert & NFW formulas, thin-disc solver vs Freeman.\n")

# =====================================================================
# 3. LOAD ROTATION-CURVE DATA (gives the geometry used for the HI profile)
# =====================================================================
opt = np.loadtxt(inpath("UGC731_Optical_RC.txt"), comments="#")   # arcs, kpc, V, err
hi  = np.loadtxt(inpath("UGC731_HI_RC.txt"), comments="#")
PA_KIN  = float(np.median(hi[:, 5]))      # BBarolo PA (receding side, N through E)
INC_BB  = float(np.mean(hi[:, 4]))
XPOS, YPOS = float(hi[0, 9]), float(hi[0, 10])   # BBarolo centre (0-indexed pixels of the full cube)
print(f"BBarolo file: median PA = {PA_KIN:.1f} deg (range {hi[:,5].min():.0f}-{hi[:,5].max():.0f}), "
      f"mean INC = {INC_BB:.1f} deg (range {hi[:,4].min():.0f}-{hi[:,4].max():.0f}), centre = ({XPOS:.0f},{YPOS:.0f}) pix")
if hi[:, 4].std() > 0.5 or hi[:, 5].std() > 0.5:
    print("  NOTE: INC and PA vary ring-to-ring in UGC731_HI_RC.txt (they were free in the BBarolo run).")
print()

# =====================================================================
# 4. HI SURFACE DENSITY FROM THE CUBE
# =====================================================================
print("-" * 66); print("STEP 1 - HI surface density from the WHISP cube"); print("-" * 66)
cube, h, wcs_cube = open_fits(find_input("velo_u731s30cl_regrid.fits", ["velo_u731*.fits", "*u731*cl*.fits"]), 3)
nv = cube.shape[0]
pix = pix_scale(wcs_cube)                                              # arcsec / pixel from the WCS
cd3 = float(h["CDELT3"])                                               # SIGNED channel step
vel = h["CRVAL3"] + (np.arange(1, nv + 1) - h["CRPIX3"])*cd3
if np.nanmax(np.abs(vel)) > 1e4:                                       # header in m/s -> km/s
    vel = vel/1000.0; cd3 = cd3/1000.0
    print("  NOTE: velocity axis was in m/s, converted to km/s")
dv = abs(cd3)
cube = np.nan_to_num(cube, nan=0.0)
_bmaj, _bmin = h.get("BMAJ"), h.get("BMIN")
if _bmaj is not None and _bmin is not None and _bmaj > 0 and _bmin > 0 \
        and 10.0 < np.sqrt(_bmaj*_bmin)*3600 < 120.0:
    BEAM_ARCS = float(np.sqrt(_bmaj*_bmin))*3600.0
    print(f"beam from header: {_bmaj*3600:.1f} x {_bmin*3600:.1f} arcsec -> effective {BEAM_ARCS:.1f} arcsec")
else:
    print(f"no usable BMAJ/BMIN in header -> assuming {BEAM_ARCS:.0f} x {BEAM_ARCS:.0f} arcsec (WHISP 30 arcsec maps)")
print(f"cube {cube.shape}, pixel = {pix:.3f} arcsec, channel = {dv:.4f} km/s, effective beam = {BEAM_ARCS:.1f}\"")

BEAM_PIX = 1.1331*BEAM_ARCS**2/pix**2            # pixels per beam
Tb_per_WU = WU_MJY/(1.65e-3*BEAM_ARCS**2)         # K per W.U.
print(f"T_b per W.U. = {Tb_per_WU:.4f} K  (S[mJy/beam] = 1.65e-3 * theta^2 * T_b)")

# crop a 256x256 box around the galaxy
cx, cy = int(round(XPOS)), int(round(YPOS))
hw = 128
x0, y0 = cx - hw, cy - hw
sub = cube[:, y0:y0 + 2*hw, x0:x0 + 2*hw]
gx, gy = XPOS - x0, YPOS - y0
yy, xx = np.mgrid[0:2*hw, 0:2*hw]
dE_box, dN_box = offsets_arcsec(wcs_cube, xx - gx, yy - gy)           # sky offsets of every pixel of the box
r_sky = np.hypot(dE_box, dN_box)

line_free = ((vel < VEL_RANGE[0]) | ((vel > 760) & (vel < 860))) & (np.arange(nv) > 3) & (np.arange(nv) < 114)
noise_ch = np.median([np.std(sub[i]) for i in np.where(line_free)[0]])
print(f"channel noise (line-free)      : {noise_ch:.3f} W.U.")

def build_mask(fwhm=MASK_FWHM, nsig=MASK_SIGMA):
    k = np.sqrt(max(fwhm**2 - BEAM_ARCS**2, 0.0))/2.3548/pix
    s = np.array([gaussian_filter(ch, k) for ch in sub]) if k > 0 else sub.copy()
    s = gaussian_filter(s, sigma=(1.0, 0, 0))
    ns = np.median([np.std(s[i]) for i in np.where(line_free)[0]])
    inrange = ((vel >= VEL_RANGE[0]) & (vel <= VEL_RANGE[1]))[:, None, None]
    m = (s > nsig*ns) & (r_sky < MASK_RMAX)[None] & inrange
    lab, n = ndi_label(m, structure=generate_binary_structure(3, 3))
    peak = np.where(m & (r_sky < 150)[None], s, -np.inf)
    key = lab.flat[np.argmax(peak)]
    return lab == key, ns, n

def moment0(mask):
    return np.where(mask, sub, 0.0).sum(0)*dv                          # W.U. km/s per pixel

def mom0_to_sigma_obs(m0):
    TK = m0*Tb_per_WU                                                  # K km/s
    NHI = 1.823e18*TK                                                  # atoms / cm^2
    return NHI/1.248e20                                                # Msun/pc^2 (HI only, observed)

mask, ns, ncomp = build_mask()
m0 = moment0(mask)
sig_obs = mom0_to_sigma_obs(m0)

# write the moment-0 FITS with a correct WCS for the cropped box
whd = fits.Header()
whd["BUNIT"] = "W.U. km/s"
for k_ in ("CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
           "CRPIX1", "CRPIX2", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE"):
    if k_ in h: whd[k_] = h[k_]
whd["CRPIX1"] = h["CRPIX1"] - x0                                       # shift the reference pixel into the crop
whd["CRPIX2"] = h["CRPIX2"] - y0
for k_ in ("PC1_1", "PC1_2", "PC2_1", "PC2_2"):
    if k_ in h: whd[k_] = h[k_]
whd["COMMENT"] = "moment 0 built from the WHISP cube (3-D connected-region mask)"
fits.PrimaryHDU(m0.astype(np.float32), header=whd).writeto(outpath("UGC731_moment0_from_cube.fits"), overwrite=True)
print("moment-0 map computed from the FITS cube -> UGC731_moment0_from_cube.fits")

flux_Jykms = m0.sum()*(WU_MJY*1e-3)/BEAM_PIX
MHI_flux = 2.356e5*D_MPC**2*flux_Jykms
print(f"mask: {MASK_FWHM:.0f}\" smoothing, {MASK_SIGMA} sigma (smoothed noise {ns:.3f} W.U.), "
      f"{ncomp} 3-D regions found, galaxy region = {mask.sum()} voxels")
print(f"total HI flux = {flux_Jykms:.1f} Jy km/s  ->  M_HI = {MHI_flux:.3e} Msun")
print(f"peak observed Sigma_HI = {sig_obs.max():.2f} Msun/pc^2  (peak N_HI = {sig_obs.max()*1.248e20:.2e} cm^-2)")

def ring_profile(sig_map, pa_deg=PA_KIN, inc_deg=INC_DEG, centre=(gx, gy), width=RING_W,
                 rmax=EXTRAP_OUT, up=4):
    """Mean of sig_map in elliptical annuli (semi-major radii at BBarolo ring centres, width 8")."""
    z = zoom(sig_map, up, order=1)
    ny, nx = z.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    px = (X + 0.5)/up - 0.5; py = (Y + 0.5)/up - 0.5
    dE, dN = offsets_arcsec(wcs_cube, px - centre[0], py - centre[1])   # East / North offsets from the WCS
    pa = np.radians(pa_deg)
    xm = dE*np.sin(pa) + dN*np.cos(pa)
    ym = dE*np.cos(pa) - dN*np.sin(pa)
    rell = np.sqrt(xm**2 + (ym/np.cos(np.radians(inc_deg)))**2)
    centres = np.arange(width/2, rmax + 1e-6, width)
    out = []
    for rc_ in centres:
        sel = (rell >= rc_ - width/2) & (rell < rc_ + width/2)
        if sel.sum() < 4: continue
        vals = z[sel]
        area_arcs2 = sel.sum()*(pix/up)**2
        nbeam = max(area_arcs2/(1.1331*BEAM_ARCS**2), 1.0)
        out.append((rc_, vals.mean(), vals.std()/np.sqrt(nbeam), nbeam))
    return np.array(out)

cosi = np.cos(np.radians(INC_DEG))
rings = ring_profile(sig_obs)
r_as = rings[:, 0]; r_kpc = r_as*KPC_ARCS
sigHI_face = rings[:, 1]*cosi
sig_noise = mom0_to_sigma_obs(np.array(noise_ch*dv*np.sqrt(20.0)))
err_face = np.sqrt(rings[:, 2]**2 + (sig_noise/np.sqrt(rings[:, 3]))**2)*cosi
err_face = np.maximum(err_face, 0.05*sigHI_face)
valid = sigHI_face > 0
gas_face = HELIUM*sigHI_face

area = np.pi*((r_as + RING_W/2)**2 - (r_as - RING_W/2)**2)*(KPC_ARCS*1e3)**2
MHI_prof = np.sum(sigHI_face*area)
print(f"M_HI from face-on profile = {MHI_prof:.3e} Msun  (ratio to flux-based mass = {MHI_prof/MHI_flux:.3f})")

hdr = ("UGC 731 - HI surface density profile (WHISP 30\" cube)\n"
       f"mask: {MASK_FWHM:.0f}\" smoothing, {MASK_SIGMA} sigma, 3-D connected region, v = {VEL_RANGE} km/s\n"
       f"T_b = {Tb_per_WU:.4f} K/WU, N_HI = 1.823e18*int(T_b dv), 1 Msun/pc^2 = 1.248e20 cm^-2\n"
       f"ellipses: PA = {PA_KIN:.1f} deg, i = {INC_DEG} deg, ring width {RING_W} arcsec (= BBarolo rings)\n"
       "Sigma_HI is FACE-ON (observed * cos i); Sigma_gas includes the factor "
       f"{HELIUM}\nColumns: R_arcsec R_kpc Sigma_HI_faceon Err Sigma_gas_faceon")
np.savetxt(outpath("HI_surface_density_profile.txt"),
           np.column_stack([r_as, r_kpc, sigHI_face, err_face, gas_face]),
           header=hdr, fmt="%9.2f %9.4f %12.5f %12.5f %12.5f")
print(f"Saved HI_surface_density_profile.txt ({len(r_as)} rings, 0 - {r_kpc.max():.1f} kpc)")
peak_i = np.argmax(sigHI_face)
print(f"peak face-on Sigma_HI = {sigHI_face[peak_i]:.2f} Msun/pc^2 at {r_kpc[peak_i]:.2f} kpc")

sel = (r_kpc > 0.9) & (sigHI_face > 0)
w = np.maximum(err_face[sel], 1e-3)
p_exp, _ = curve_fit(exp_disk, r_kpc[sel], gas_face[sel], p0=[10, 5], sigma=HELIUM*w, maxfev=20000)
chi_exp = np.sum(((gas_face[sel] - exp_disk(r_kpc[sel], *p_exp))/(HELIUM*w))**2)
try:
    p_hole, _ = curve_fit(sigma_hole, r_kpc[sel], gas_face[sel], p0=[15, 4, 0.5, 2.5], sigma=HELIUM*w,
                          bounds=([0, 0.5, 0, 0.2], [3000, 50, 1, 10]), maxfev=80000)
    chi_hole = np.sum(((gas_face[sel] - sigma_hole(r_kpc[sel], *p_hole))/(HELIUM*w))**2)
except Exception as e:
    p_hole, chi_hole = None, np.nan
dof = sel.sum()
print(f"plain exponential fit : S0 = {p_exp[0]:.2f}, h = {p_exp[1]:.2f} kpc, chi2/N = {chi_exp/dof:.2f}")
if p_hole is not None:
    print(f"central-suppression   : S0 = {p_hole[0]:.1f}, h = {p_hole[1]:.2f} kpc, A = {p_hole[2]:.2f}, "
          f"w = {p_hole[3]:.2f} kpc, chi2/N = {chi_hole/dof:.2f}")
print("  -> the measured profile is used directly for V_gas.\n")

# =====================================================================
# 5. WISE W1 STELLAR PROFILE  (measured here from the W1 FITS cutout; sky-subtracted DN)
# =====================================================================
print("-" * 66); print("STEP 2 - WISE W1 stellar surface density"); print("-" * 66)
r_wise_arcs = np.arange(2, 120, 3, dtype=float)   # annulus centres: 2, 5, ..., 119 arcsec
WISE_FITS   = "0185p499_ac51-w1-int-3_ra17_6833359_dec49_602112_asec600_000.fits"
WISE_PA     = PA_KIN % 180.0                  # kinematic major-axis PA (265.5 deg = 85.5 deg); the faint W1 light does not fix a PA by itself
WISE_DR     = 3.0                                 # annulus width [arcsec]
WISE_SKY_R  = 250.0                               # sky = median of the cutout beyond this radius [arcsec]

WISE_PATH = find_input(WISE_FITS, ["0185p499*w1*asec600*.fits", "*w1-int-3*.fits", "*asec600*.fits"])
WISE_FITS = os.path.basename(WISE_PATH)
wimg, whdr, wcs_w = open_fits(WISE_PATH, 2)
pix_w  = pix_scale(wcs_w)                         # from the WCS (= 1.375")
magzp  = float(whdr.get("MAGZP", 20.5))
# --- galaxy centre for the STELLAR profile ---------------------------------------------------
# "radec"   : the fixed sky position GAL_RA, GAL_DEC converted with the WISE WCS (adopted centre).
# "centroid": iterative flux-weighted centroid of the sky-subtracted W1 light within WISE_CEN_R arcsec,
#             with bright compact sources (stars) masked - this is the stellar centre.
# "hi"      : BBarolo centre converted with the two WCS (the HI centre can differ from the stellar one by ~40").
# (x, y)    : a fixed pixel position (0-indexed), e.g. (226, 217).
WISE_CENTRE   = "radec"             # galaxy position adopted from the image (cutout target): see GAL_RA / GAL_DEC
GAL_RA, GAL_DEC = 17.6833359, 49.602112   # RA, Dec [deg] of the galaxy centre (= centre of the WISE cutout)
WISE_FIT_RMIN = 0.5               # kpc: inner limit of the exponential fit (the W1 PSF FWHM is 6.1\" = 0.3 kpc)
WISE_START    = (226.0, 217.0)    # starting guess for the centroid (pixel, 0-indexed)
WISE_CEN_R    = 60.0              # arcsec: aperture of the centroid
WISE_STAR_DN  = 15.0              # pixels brighter than this above the sky (and their surroundings) are masked
WISE_FIT_RMAX     = 4.0    # kpc: outer limit of the stellar-disc fit (3.0 kpc = 60 arcsec, where the profile is still >~4 sigma)
WISE_FIT_WEIGHTED = True   # True: weight the disc fit by the empirical annulus errors; False: old unweighted fit
WISE_SKY_SYST     = 0.1    # DN/pixel: sky-level systematic used in the Step 6 sensitivity tests

def wise_centroid(img, wcs, start, rad_arcs, star_dn, n_iter=30):
    from scipy.ndimage import binary_dilation
    yy_, xx_ = np.indices(img.shape); p = pix_scale(wcs)
    sk = np.nanmedian(img[np.hypot(xx_ - start[0], yy_ - start[1])*p > 250.0])
    b_ = img - sk
    star = binary_dilation(b_ > star_dn, iterations=4)
    sm = gaussian_filter(np.where(star | ~np.isfinite(b_), 0.0, b_), 2)
    c = np.array(start, float)
    for _ in range(n_iter):
        m = (np.hypot(xx_ - c[0], yy_ - c[1])*p < rad_arcs) & ~star
        f = np.clip(sm, 0, None)*m
        if f.sum() <= 0: break
        c = np.array([(f*xx_).sum()/f.sum(), (f*yy_).sum()/f.sum()])
    return float(c[0]), float(c[1])

if isinstance(WISE_CENTRE, (tuple, list)):
    WISE_CX, WISE_CY = float(WISE_CENTRE[0]), float(WISE_CENTRE[1]); src = "fixed pixel"
elif WISE_CENTRE == "radec":
    WISE_CX, WISE_CY = [float(v) for v in wcs_w.world_to_pixel_values(GAL_RA, GAL_DEC)]; src = "adopted RA/Dec"
elif WISE_CENTRE == "hi":
    try:
        ra_c, dec_c = wcs_cube.pixel_to_world_values(XPOS, YPOS)
        WISE_CX, WISE_CY = [float(v) for v in wcs_w.world_to_pixel_values(ra_c, dec_c)]; src = "BBarolo centre via the two WCS"
    except Exception:
        WISE_CX, WISE_CY = WISE_START; src = "fallback (WCS conversion failed)"
else:
    WISE_CX, WISE_CY = wise_centroid(wimg, wcs_w, WISE_START, WISE_CEN_R, WISE_STAR_DN); src = f"masked centroid within {WISE_CEN_R:.0f}\""
_ra, _dec = wcs_w.pixel_to_world_values(WISE_CX, WISE_CY)
print(f"stellar centre: pixel ({WISE_CX:.1f}, {WISE_CY:.1f}) = RA {float(_ra):.5f}, Dec {float(_dec):.5f}  [{src}]")
try:
    _hx, _hy = [float(v) for v in wcs_w.world_to_pixel_values(*wcs_cube.pixel_to_world_values(XPOS, YPOS))]
    print(f"  (BBarolo/HI centre falls at WISE pixel ({_hx:.1f}, {_hy:.1f}), "
          f"{np.hypot(_hx - WISE_CX, _hy - WISE_CY)*pix_w:.0f}\" from the stellar centre)")
except Exception:
    pass
if not (0 <= WISE_CX < wimg.shape[1] and 0 <= WISE_CY < wimg.shape[0]):
    raise SystemExit("The galaxy centre falls outside the WISE image - check the WCS / file.")

flux_dn, wise_sky = wise_profile_from_image(wimg, wcs_w, WISE_CX, WISE_CY, WISE_PA, INC_DEG,
                                            r_wise_arcs, WISE_DR, WISE_SKY_R)
_yy, _xx = np.indices(wimg.shape)
_rs = np.hypot(_xx - WISE_CX, _yy - WISE_CY)*pix_w
_sk = wimg[_rs > WISE_SKY_R]
print(f"sky region (> {WISE_SKY_R:.0f}\"): {_sk.size} px ({100*_sk.size/wimg.size:.0f}% of the image), median {np.nanmedian(_sk):.3f} DN, "
      f"robust sigma {1.4826*np.nanmedian(np.abs(_sk - np.nanmedian(_sk))):.3f} DN, "
      f"{100*np.mean(_sk > np.nanmedian(_sk) + 10):.1f}% of pixels >10 DN above the median (stars; the median is insensitive to them)")
print(f"WISE W1 profile from {WISE_FITS}: {len(flux_dn)} points, sky = {wise_sky:.3f} DN/pix, "
      f"pixel = {pix_w:.3f}\", MAGZP = {magzp}, PA = {WISE_PA:.1f} deg")

r_wise_kpc = r_wise_arcs*KPC_ARCS
good = flux_dn > 0                                  # log10 needs positive flux
sb_mag = np.full_like(flux_dn, np.nan); sb_mag[good] = magzp - 2.5*np.log10(flux_dn[good]/pix_w**2)
Sigma_L_obs = 10**(0.4*(M_SUN_W1 + 21.572 - sb_mag))     # constant 21.572 = 5 log10(206265/10), distance-independent
Sigma_M_obs = ML_W1*Sigma_L_obs
Sigma_M = Sigma_M_obs*cosi
np.savetxt(outpath("WISE_W1_stellar_profile.txt"),
           np.column_stack([r_wise_arcs, r_wise_kpc, flux_dn, sb_mag, Sigma_L_obs, Sigma_M_obs, Sigma_M]),
           header=("UGC 731 - WISE W1 profile (sky-subtracted)\n"
                   f"MAGZP={magzp}, pixel={pix_w:.3f}\", M_sun(W1)={M_SUN_W1}, M/L={ML_W1}, face-on = observed*cos(i), i={INC_DEG}\n"
                   "Columns: R_arcsec R_kpc Flux_DN SB(mag/arcsec2) Sigma_L_obs Sigma_M_obs Sigma_M_faceon"),
           fmt="%9.3f %9.5f %9.5f %10.5f %12.5f %12.5f %12.5f")

# --- empirical errors, alternative profile and the exponential-disc fit ------------------------------------
wise_err_dn, wise_pedestal = wise_annulus_noise(wimg, wcs_w, WISE_CX, WISE_CY, WISE_PA, INC_DEG,
                                                r_wise_arcs, WISE_DR, wise_sky)          # same annuli at 200 blank-sky positions
wise_err_dn = np.maximum(wise_err_dn, 0.05*np.abs(flux_dn))                              # 5 % floor, as for the HI profile
flux_mm, _ = wise_profile_masked_mean(wimg, wcs_w, WISE_CX, WISE_CY, WISE_PA, INC_DEG,
                                      r_wise_arcs, WISE_DR, WISE_SKY_R, WISE_STAR_DN)    # star-masked mean (sensitivity test only)
_ii = [i for i in (13, 19, 26, 32) if i < len(r_wise_arcs)]
print("W1 annulus S/N (galaxy / blank-sky scatter) at r = " + ", ".join(f"{r_wise_arcs[i]:.0f}\"" for i in _ii) + ": "
      + ", ".join(f"{flux_dn[i]/wise_err_dn[i]:.1f}" for i in _ii) + "   (the stellar disc is not detected beyond ~95\")")
print(f"  blank-sky annuli sit {np.nanmedian(wise_pedestal):+.3f} DN/pix above the adopted sky (pedestal; the sky systematic tested in Step 6 is +/-{WISE_SKY_SYST} DN)")
Sigma0_s, h_s, (_eS0, _eh), _nfit = fit_stellar_disc(flux_dn, wise_err_dn if WISE_FIT_WEIGHTED else None,
                                                      r_wise_arcs, pix_w, magzp, WISE_FIT_RMIN, WISE_FIT_RMAX)
print(f"stellar disc fit ({'weighted by the empirical errors' if WISE_FIT_WEIGHTED else 'unweighted'}; {_nfit} points, "
      f"{WISE_FIT_RMIN}-{WISE_FIT_RMAX} kpc): fit errors Sigma0 +/- {_eS0:.1f}, h +/- {_eh:.2f} kpc (statistical only)")
print(f"face-on stellar exponential disc: Sigma0* = {Sigma0_s:.2f} Msun/pc^2, h* = {h_s:.3f} kpc "
      f"(observed-plane value {Sigma0_s/cosi:.2f})")
M_star = 2*np.pi*Sigma0_s*1e6*h_s**2
print(f"stellar mass (exp. disc) = {M_star:.2e} Msun\n")

# =====================================================================
# 6. BARYONIC VELOCITIES
# =====================================================================
print("-" * 66); print("STEP 3 - Baryonic circular velocities"); print("-" * 66)
r_fine = np.linspace(0.02, 11.0, 550)
V_stars = freeman_vcirc(r_fine, Sigma0_s, h_s)
V_gas = thin_disc_vcirc(r_fine, r_kpc[valid], gas_face[valid])
V_gas_exp  = freeman_vcirc(r_fine, *p_exp)
V_gas_hole = thin_disc_vcirc(r_fine, np.linspace(0.05, 15, 300), sigma_hole(np.linspace(0.05, 15, 300), *p_hole)) \
             if p_hole is not None else None
V_bar = np.sqrt(V_stars**2 + V_gas**2)
print(f"peak V_stars = {V_stars.max():.1f} km/s at {r_fine[V_stars.argmax()]:.2f} kpc")
print(f"V_gas (measured profile): max {V_gas.max():.1f} km/s at {r_fine[V_gas.argmax()]:.2f} kpc; "
      f"at 7.8 kpc = {np.interp(7.8, r_fine, V_gas):.1f} km/s")
print(f"V_gas (plain exponential)                  at 7.8 kpc = {np.interp(7.8, r_fine, V_gas_exp):.1f} km/s")
if V_gas_hole is not None:
    print(f"V_gas (central-suppression exponential)    at 7.8 kpc = {np.interp(7.8, r_fine, V_gas_hole):.1f} km/s")
print(f"M_gas = {HELIUM*MHI_prof:.2e} Msun, M_star = {M_star:.2e} Msun")
np.savetxt(outpath("vstars_profile.txt"), np.column_stack([r_fine, V_stars]),
           header=f"V_stars (Freeman, face-on Sigma0={Sigma0_s:.3f}, h={h_s:.4f}, M/L={ML_W1})\nR_kpc V_km/s", fmt="%.5f")
np.savetxt(outpath("vgas_profile.txt"), np.column_stack([r_fine, V_gas, V_gas_exp]),
           header="V_gas from the measured face-on Sigma_gas (thin-disc solver); col3 = plain exponential\nR_kpc V_gas V_gas_exp", fmt="%.5f")
print()

# =====================================================================
# 7. ROTATION-CURVE DATA
# =====================================================================
print("-" * 66); print("STEP 4 - Rotation-curve data"); print("-" * 66)
ERR_FLOOR = 0.5*dv
r_opt_raw, v_opt_raw, e_opt_raw = opt[:, 1], opt[:, 2], np.maximum(opt[:, 3], ERR_FLOOR)
ok = r_opt_raw > 0
r_opt, v_opt, e_opt = r_opt_raw[ok], v_opt_raw[ok], e_opt_raw[ok]
r_hi_all, v_hi_all = hi[:, 0], hi[:, 2]
e_lo_all, e_up_all = np.abs(hi[:, 13]), np.abs(hi[:, 14])
e_hi_all = np.maximum(0.5*(e_lo_all + e_up_all), ERR_FLOOR)
mh = r_hi_all >= SPLICE_KPC
r_hi, v_hi, e_hi, e_lo, e_up = r_hi_all[mh], v_hi_all[mh], e_hi_all[mh], e_lo_all[mh], e_up_all[mh]
r_obs = np.concatenate([r_opt, r_hi]); v_obs = np.concatenate([v_opt, v_hi]); e_obs = np.concatenate([e_opt, e_hi])
o = np.argsort(r_obs); r_obs, v_obs, e_obs = r_obs[o], v_obs[o], e_obs[o]
N = len(r_obs)
print(f"optical: {len(r_opt)} pts ({r_opt.min():.2f}-{r_opt.max():.2f} kpc);  HI: {len(r_hi)} pts "
      f"({r_hi.min():.2f}-{r_hi.max():.2f} kpc);  N = {N}")
print(f"inner BBarolo points (r<{SPLICE_KPC} kpc) discarded: {(~mh).sum()}  "
      f"(V = {v_hi_all[~mh].min():.0f}-{v_hi_all[~mh].max():.0f} km/s vs optical "
      f"{np.interp(r_hi_all[~mh][0], r_opt, v_opt):.0f} km/s at {r_hi_all[~mh][0]:.1f} kpc: beam smearing)\n")

# =====================================================================
# 8. HALO FITS
# =====================================================================
def make_total(Vs, Vg):
    fs = interp1d(r_fine, Vs, bounds_error=False, fill_value=(0.0, Vs[-1]))
    fg = interp1d(r_fine, Vg, bounds_error=False, fill_value=(0.0, Vg[-1]))
    return fs, fg

def fit_halo(kind, fs, fg, r=None, v=None, e=None):
    r = r_obs if r is None else r; v = v_obs if v is None else v; e = e_obs if e is None else e
    Vh = V_Burkert if kind == "Burkert" else V_NFW
    Vb2 = fs(r)**2 + fg(r)**2
    def chi2_of(lrho, lr):
        return np.sum(((v - np.sqrt(Vb2 + Vh(r, 10**lrho, 10**lr)**2))/e)**2)
    lrho_g = np.linspace(5, 11, 241)
    lr_g = np.linspace(np.log10(0.05), np.log10(40), 241)
    LR, LL = np.meshgrid(lrho_g, lr_g, indexing="ij")
    Vb2c = Vb2[None, None, :]
    Vh_g = Vh(r[None, None, :], 10**LR[..., None], 10**LL[..., None])
    CH = np.sum(((v[None, None, :] - np.sqrt(Vb2c + Vh_g**2))/e[None, None, :])**2, axis=-1)
    i, j = np.unravel_index(np.argmin(CH), CH.shape)
    f = lambda rr, lrho, lr: np.sqrt(fs(rr)**2 + fg(rr)**2 + Vh(rr, 10**lrho, 10**lr)**2)
    popt, pcov = curve_fit(f, r, v, p0=[lrho_g[i], lr_g[j]], sigma=e, absolute_sigma=True,
                           bounds=([3, np.log10(0.02)], [12, np.log10(100)]), maxfev=20000)
    chi2 = chi2_of(*popt); k = 2
    rho, rs = 10**popt
    perr = np.sqrt(np.diag(pcov))
    dchi = CH - CH.min()
    prof_rho = dchi.min(axis=1); prof_r = dchi.min(axis=0)
    rho_rng = 10**lrho_g[prof_rho <= 1.0]; r_rng = 10**lr_g[prof_r <= 1.0]
    res = v - f(r, *popt)
    return dict(kind=kind, rho=rho, rs=rs, e_rho=rho*np.log(10)*perr[0], e_rs=rs*np.log(10)*perr[1],
                rho_rng=(rho_rng.min(), rho_rng.max()), r_rng=(r_rng.min(), r_rng.max()),
                chi2=chi2, chi2r=chi2/(N - k), AIC=chi2 + 2*k, BIC=chi2 + k*np.log(N),
                resid=res, popt=popt, at_bound=(abs(popt[1] - np.log10(100)) < 1e-3 or abs(popt[0] - 12) < 1e-3 or
                                               abs(popt[1] - np.log10(0.02)) < 1e-3),
                grid_edge=(j == 0 or j == CH.shape[1] - 1 or i == 0 or i == CH.shape[0] - 1))

fs, fg = make_total(V_stars, V_gas)
print("-" * 66); print("STEP 5 - Halo fitting"); print("-" * 66)
res_B = fit_halo("Burkert", fs, fg)
res_N = fit_halo("NFW", fs, fg)

def report(res):
    nm = "rho_0" if res["kind"] == "Burkert" else "rho_s"; rn = "r_c" if res["kind"] == "Burkert" else "r_s"
    print(f"\n  --- {res['kind']} ---")
    print(f"  {nm:6s}= {res['rho']:.4e} +/- {res['e_rho']:.2e} Msun/kpc^3   "
          f"[profile 1-sigma: {res['rho_rng'][0]:.2e} - {res['rho_rng'][1]:.2e}]")
    print(f"  {rn:6s}= {res['rs']:.4f} +/- {res['e_rs']:.4f} kpc          "
          f"[profile 1-sigma: {res['r_rng'][0]:.3f} - {res['r_rng'][1]:.3f}]")
    print(f"  chi2 = {res['chi2']:.3f}   chi2_red = {res['chi2r']:.3f} (dof = {N-2})   "
          f"AIC = {res['AIC']:.3f}   BIC = {res['BIC']:.3f}")
    if res["grid_edge"] or res["at_bound"]:
        print("  WARNING: best fit sits on the edge of the allowed parameter range -> parameter poorly constrained")
sep = "=" * 66
print(sep); print(f"{'RESULTS':^66}"); print(sep)
print(f"face-on stellar disc: Sigma0* = {Sigma0_s:.2f} Msun/pc^2, h* = {h_s:.3f} kpc, V*max = {V_stars.max():.1f} km/s")
print(f"gas (He-corrected): measured profile, peak Sigma_gas = {gas_face.max():.2f} Msun/pc^2, "
      f"V_gas(7.8 kpc) = {np.interp(7.8, r_fine, V_gas):.1f} km/s")
report(res_B); report(res_N)
dAIC = res_N["AIC"] - res_B["AIC"]; dBIC = res_N["BIC"] - res_B["BIC"]
win = "Burkert" if dAIC > 0 else "NFW"
ad = abs(dAIC)
strength = "decisive (>10)" if ad > 10 else "strong (6-10)" if ad > 6 else "positive (2-6)" if ad > 2 else "inconclusive (<2)"
print(f"\n  Delta AIC = AIC(NFW) - AIC(Burkert) = {dAIC:+.3f}      Delta BIC = {dBIC:+.3f}")
print(f"  Preferred model: {win}  -> evidence: {strength}")
rho0_rc = res_B["rho"]*1e-9*res_B["rs"]*1e3
print(f"  Burkert rho0*rc = {rho0_rc:.1f} Msun/pc^2   (Burkert 1995 / Salucci & Burkert 2000 scaling ~ 100-150)")
rho_crit = 3*(0.070)**2/(8*np.pi*G)
def c200_from(rhos):
    from scipy.optimize import brentq
    dc = rhos/rho_crit
    f = lambda c: (200/3.)*c**3/(np.log(1 + c) - c/(1 + c)) - dc
    try: return brentq(f, 1.0001, 1e4)
    except ValueError: return np.nan
c200 = c200_from(res_N["rho"]); r200 = c200*res_N["rs"]
M200 = 200*rho_crit*4/3*np.pi*r200**3 if np.isfinite(c200) else np.nan
print(f"  NFW implied (H0=70): c200 = {c200:.1f}, r200 = {r200:.1f} kpc, M200 = {M200:.2e} Msun")
print(f"  [LCDM expectation for a dwarf halo: c200 ~ 10-20 for M200 ~ 1e10-1e11 Msun]")
print(sep)
print(f"  weighted least squares == maximum likelihood here (independent Gaussian errors).\n")

def fit_gnfw(fs_, fg_, r=None, v=None, e=None):
    r = r_obs if r is None else r; v = v_obs if v is None else v; e = e_obs if e is None else e
    f = lambda rr, lrho, lr, gam: np.sqrt(fs_(rr)**2 + fg_(rr)**2 + V_gNFW(rr, 10**lrho, 10**lr, gam)**2)
    best = None
    for g0 in (0.1, 0.5, 1.0, 1.4):
        for lr0 in (-0.2, 0.3, 0.8):
            try:
                p, c = curve_fit(f, r, v, p0=[8.0, lr0, g0], sigma=e, absolute_sigma=True,
                                 bounds=([3, np.log10(0.05), -0.5], [12, np.log10(60), 1.9]), maxfev=40000)
            except Exception:
                continue
            ch = np.sum(((v - f(r, *p))/e)**2)
            if best is None or ch < best[0]: best = (ch, p, c)
    ch, p, c = best
    gg = np.linspace(-0.5, 1.9, 49); prof = []
    for g in gg:
        fg_fixed = lambda rr, lrho, lr: f(rr, lrho, lr, g)
        bb = np.inf
        for lr0 in (-0.2, 0.3, 0.8):
            try:
                pp, _ = curve_fit(fg_fixed, r, v, p0=[8.0, lr0], sigma=e, absolute_sigma=True,
                                  bounds=([3, np.log10(0.05)], [12, np.log10(60)]), maxfev=20000)
                bb = min(bb, np.sum(((v - fg_fixed(r, *pp))/e)**2))
            except Exception:
                pass
        prof.append(bb)
    prof = np.array(prof) - ch
    ok = gg[prof <= 1.0]; ok2 = gg[prof <= 4.0]
    return dict(chi2=ch, gam=p[2], gam_rng=(ok.min(), ok.max()), gam_rng2=(ok2.min(), ok2.max()),
                rho=10**p[0], rs=10**p[1], AIC=ch + 6)

res_G = fit_gnfw(fs, fg)
print(f"  Free inner slope (generalised NFW, 3 params, gamma = -dlnrho/dlnr at r->0; NFW = 1, core = 0):")
print(f"     gamma = {res_G['gam']:.2f}   1-sigma range [{res_G['gam_rng'][0]:.2f}, {res_G['gam_rng'][1]:.2f}],"
      f"  2-sigma range [{res_G['gam_rng2'][0]:.2f}, {res_G['gam_rng2'][1]:.2f}]")
print(f"     chi2 = {res_G['chi2']:.2f}, AIC = {res_G['AIC']:.2f}  (compare AIC: Burkert {res_B['AIC']:.2f}, NFW {res_N['AIC']:.2f})\n")

print("  Restricted-radius fits (a diagnostic of where the signal is, not a replacement for the full fit):")
for rmax_, lab in [(6.3, "r <= 6.3 kpc (drops the 4 outermost HI points)"),
                   (2.3, "r <= 2.3 kpc (optical only)")]:
    mk_ = r_obs <= rmax_
    rb_ = fit_halo("Burkert", fs, fg, r_obs[mk_], v_obs[mk_], e_obs[mk_])
    rn_ = fit_halo("NFW", fs, fg, r_obs[mk_], v_obs[mk_], e_obs[mk_])
    print(f"     {lab}: N={mk_.sum()}  chi2 Burkert = {rb_['chi2']:.2f}, NFW = {rn_['chi2']:.2f}, "
          f"dAIC = {rn_['AIC'] - rb_['AIC']:+.2f}")
print()

# =====================================================================
# 9. SENSITIVITY TESTS
# =====================================================================
print("-" * 66); print("STEP 6 - Sensitivity of the conclusion to the baryonic model & mask"); print("-" * 66)
rows = []
def one_case(label_, S0s=None, gas_prof=None, vg_override=None, hs=None):
    Vs = freeman_vcirc(r_fine, S0s if S0s is not None else Sigma0_s, hs if hs is not None else h_s)
    if vg_override is not None: Vg = vg_override
    else: Vg = thin_disc_vcirc(r_fine, r_kpc[valid], (gas_prof if gas_prof is not None else gas_face)[valid])
    a, b = make_total(Vs, Vg)
    rb = fit_halo("Burkert", a, b); rn = fit_halo("NFW", a, b)
    rows.append((label_, rb, rn))
    print(f"  {label_:44s} Burk: rho0={rb['rho']:.2e} rc={rb['rs']:.2f} chi2={rb['chi2']:.1f} | "
          f"NFW: rhos={rn['rho']:.2e} rs={rn['rs']:.2f} chi2={rn['chi2']:.1f} | dAIC={rn['AIC']-rb['AIC']:+.2f}")
one_case("baseline", )
for ml in (0.3, 0.7):
    one_case(f"M/L(W1) = {ml}", S0s=Sigma0_s*ml/ML_W1)
one_case("no stars (V*=0)", S0s=1e-9)
one_case("gas factor 1.33", gas_prof=gas_face*1.33/HELIUM)
one_case("gas x 0.8 (flux-scale -20%)", gas_prof=gas_face*0.8)
one_case("gas x 1.2 (flux-scale +20%)", gas_prof=gas_face*1.2)
one_case("gas: plain exponential", vg_override=V_gas_exp)
if V_gas_hole is not None: one_case("gas: central-suppression exponential", vg_override=V_gas_hole)
one_case("no baryons at all (pure halo)", S0s=1e-9, vg_override=np.zeros_like(r_fine))
# WISE stellar-disc systematics: sky level, profile statistic, fit range, weighting (disc S0 AND h both change)
for _lab, _fl, _er, _rm in [
        (f"W1 sky +{WISE_SKY_SYST} DN", flux_dn - WISE_SKY_SYST, wise_err_dn, WISE_FIT_RMAX),
        (f"W1 sky -{WISE_SKY_SYST} DN", flux_dn + WISE_SKY_SYST, wise_err_dn, WISE_FIT_RMAX),
        ("W1 star-masked mean profile", flux_mm, wise_err_dn, WISE_FIT_RMAX),
        ("W1 disc fit to r <= 3 kpc", flux_dn, wise_err_dn, 3.0),
        ("W1 unweighted fit (old recipe)", flux_dn, None, WISE_FIT_RMAX)]:
    try:
        _S0, _h, _, _ = fit_stellar_disc(_fl, _er, r_wise_arcs, pix_w, magzp, WISE_FIT_RMIN, _rm)
        one_case(_lab + f" (S0*={_S0:.1f}, h*={_h:.2f})", S0s=_S0, hs=_h)
    except Exception as e:
        print("  WISE variant failed:", _lab, e)
for fw, ns_, lab in [(30, 4.8, "mask 30\"/4.8 sigma (CARTA-like)"), (45, 3.0, "mask 45\"/3 sigma"), (60, 2.5, "mask 60\"/2.5 sigma")]:
    try:
        mk, _, _ = build_mask(fw, ns_); sg = mom0_to_sigma_obs(moment0(mk))
        rg = ring_profile(sg); face = HELIUM*rg[:, 1]*cosi
        one_case(lab + f" (M_HI={2.356e5*D_MPC**2*moment0(mk).sum()*(WU_MJY*1e-3)/BEAM_PIX:.2e})",
                 gas_prof=np.interp(r_kpc, rg[:, 0]*KPC_ARCS, face))
    except Exception as e:
        print("  mask variant failed:", lab, e)
rg = ring_profile(sig_obs, pa_deg=12.0, inc_deg=56.0)
one_case("HI annuli, photometric geometry (PA=12, i=56)", gas_prof=np.interp(r_kpc, rg[:, 0]*KPC_ARCS, HELIUM*rg[:, 1]*cosi))
with open(outpath("UGC731_sensitivity.txt"), "w") as fh:
    fh.write("case | rho0 rc chi2 (Burkert) | rhos rs chi2 (NFW) | dAIC=AIC_NFW-AIC_Burkert\n")
    for lab, rb, rn in rows:
        fh.write(f"{lab} | {rb['rho']:.3e} {rb['rs']:.3f} {rb['chi2']:.2f} | {rn['rho']:.3e} {rn['rs']:.3f} {rn['chi2']:.2f} | {rn['AIC']-rb['AIC']:+.3f}\n")
d_all = np.array([rn['AIC'] - rb['AIC'] for _, rb, rn in rows])
print(f"\n  Delta AIC range over all tests: {d_all.min():+.2f} to {d_all.max():+.2f}")
print("  (positive = Burkert preferred, negative = NFW preferred)\n")

# =====================================================================
# 10. PLOTS
# =====================================================================
FONT = 12
plt.rcParams.update({
    'font.family': 'serif', 'font.size': FONT, 'axes.labelsize': FONT, 'axes.titlesize': FONT + 1,
    'legend.fontsize': 9, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'axes.linewidth': 1.2,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True})
COPT, CHI, CSTAR, CGAS, CBAR, CBUR, CNFW = 'crimson', 'navy', 'darkorange', 'seagreen', 'purple', 'royalblue', 'tomato'
YL = r'$V_{\rm rot}\ (\rm km\ s^{-1})$'

def plot_data(ax):
    ax.errorbar(r_opt_raw, v_opt_raw, yerr=e_opt_raw, fmt='o', color=COPT, ecolor=COPT, ms=5, capsize=3,
                elinewidth=1.0, alpha=0.85, label=r'Optical H$\alpha$', zorder=6)
    ax.errorbar(r_hi, v_hi, yerr=[e_lo, e_up], fmt='s', color=CHI, ecolor=CHI, ms=5, capsize=3,
                elinewidth=1.0, alpha=0.85, label=f'Radio HI ($r \\geq {SPLICE_KPC}$ kpc)', zorder=5)

def finish(ax, title, loc=None):
    ax.set_xlabel('Radius (kpc)'); ax.set_ylabel(YL); ax.set_title(title)
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 90)
    ax.legend(framealpha=0.85) if loc is None else ax.legend(loc=loc, framealpha=0.85)
    ax.grid(True, alpha=0.25, ls='--')

r_all = np.concatenate([r_opt_raw, r_hi]); v_all = np.concatenate([v_opt_raw, v_hi]); oo = np.argsort(r_all)
r_low = r_all[oo]; v_low = lowess(r_all[oo], v_all[oo], frac=0.25)
def total_curve(res):
    Vh = V_Burkert if res["kind"] == "Burkert" else V_NFW
    return np.sqrt(fs(r_fine)**2 + fg(r_fine)**2 + Vh(r_fine, 10**res["popt"][0], 10**res["popt"][1])**2)
VtB = total_curve(res_B); VtN = total_curve(res_N)

fig, ax = plt.subplots(figsize=(8, 6)); plot_data(ax)
ax.plot(r_low, v_low, '-', color='black', lw=2.0, label='Hybrid Curve (central trend)')
finish(ax, 'UGC 731 \u2014 Hybrid Rotation Curve', loc='best'); plt.tight_layout()
safe_savefig(fig, 'Fig1_Hybrid_RC.png', dpi=150, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6)); plot_data(ax)
ax.plot(r_low, v_low, '-', color='black', lw=2.2, label=r'$V_{\rm obs}$')
ax.plot(r_fine, V_stars, '-.', color=CSTAR, lw=2.0, label=r'$V_{\rm stars}$')
ax.plot(r_fine, V_gas, ':', color=CGAS, lw=2.0, label=r'$V_{\rm gas}$')
ax.plot(r_fine, V_bar, '--', color=CBAR, lw=2.0, label=r'$V_{\rm baryons}$ ($\sqrt{V_\star^2 + V_g^2}$)')
finish(ax, 'UGC 731 \u2014 Baryonic Decomposition', loc='best'); plt.tight_layout()
safe_savefig(fig, 'Fig2_Baryonic_Decomposition.png', dpi=150, bbox_inches='tight'); plt.close(fig)

for fn, Vt, col, nm in [('Fig3_Burkert_Fit.png', VtB, CBUR, 'Burkert'), ('Fig4_NFW_Fit.png', VtN, CNFW, 'NFW')]:
    fig, ax = plt.subplots(figsize=(8, 6)); plot_data(ax)
    ax.plot(r_fine, Vt, '-', color=col, lw=2.5, label=rf'$V_{{\rm total}}$ ({nm})')
    finish(ax, f'UGC 731 \u2014 {nm} Dark Matter Halo Fit', loc='best'); plt.tight_layout()
    safe_savefig(fig, fn, dpi=150, bbox_inches='tight'); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 6)); plot_data(ax)
ax.plot(r_fine, VtB, '-', color=CBUR, lw=2.5, label='Burkert')
ax.plot(r_fine, VtN, '--', color=CNFW, lw=2.5, label='NFW')
finish(ax, 'UGC 731 \u2014 Burkert vs NFW Model Comparison', loc='lower right'); plt.tight_layout()
safe_savefig(fig, 'Fig5_Model_Comparison.png', dpi=150, bbox_inches='tight'); plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, res, color, label in zip(axes, [res_B, res_N], [CBUR, CNFW], ['Burkert', 'NFW']):
    ax.errorbar(r_obs, res['resid'], yerr=e_obs, fmt='o', color=color, ecolor=color, ms=5, capsize=3,
                elinewidth=1.0, alpha=0.80, label=label)
    ax.axhline(0, color='black', lw=1.0, ls='--'); ax.axhline(+5, color='gray', lw=0.7, ls=':'); ax.axhline(-5, color='gray', lw=0.7, ls=':')
    ax.set_ylabel(r'$V_{\rm obs} - V_{\rm model}\ (\rm km\ s^{-1})$'); ax.set_ylim(-30, 30)
    ax.legend(loc='upper right', framealpha=0.85); ax.grid(True, alpha=0.25, ls='--')
axes[1].set_xlabel('Radius (kpc)'); fig.suptitle('UGC 731 \u2014 Model Residuals', fontsize=13); plt.tight_layout()
safe_savefig(fig, 'Fig6_Residuals.png', dpi=150, bbox_inches='tight'); plt.close(fig)

fig, ax7 = plt.subplots(figsize=(8, 6))
ax7.errorbar(r_kpc[valid], gas_face[valid]/HELIUM, yerr=err_face[valid]/HELIUM, fmt='ko-', lw=1.5, ms=5,
             capsize=3, elinewidth=1.0, label='Extracted from cube')
if p_hole is not None:
    rr = np.linspace(0, r_kpc[valid].max(), 300)
    ax7.plot(rr, sigma_hole(rr, *p_hole)/HELIUM, '-', color='darkorange', lw=2,
             label=(r'Exp. with central suppression: '
                    fr'$h={p_hole[1]:.2f}$ kpc, $A={p_hole[2]:.2f}$, $w={p_hole[3]:.2f}$ kpc'))
ax7.axvline(0.9, color='gray', lw=0.8, ls=':')
ax7.text(0.95, 0.03, 'fit starts at 0.9 kpc\n(inner region not resolved)', transform=ax7.get_xaxis_transform(),
         fontsize=8, color='gray', va='bottom', ha='left')
ax7.set_xlabel('Radius (kpc)'); ax7.set_ylabel(r'$\Sigma_{\rm HI}\ (M_\odot\,\mathrm{pc}^{-2})$')
ax7.set_title('UGC 731 \u2014 HI Surface Density Profile'); ax7.legend(framealpha=0.85); ax7.grid(True, alpha=0.30, ls='--')
ax7.set_ylim(bottom=0)
plt.tight_layout(); safe_savefig(fig, 'Fig7_HI_Surface_Density_Profile.png', dpi=150, bbox_inches='tight'); plt.close(fig)

fig, ax8 = plt.subplots(figsize=(8, 6))
ax8.plot(r_wise_kpc, Sigma_M, 'o-', color='black', lw=2, ms=5, label=r'WISE W1 ($\Upsilon_*=0.5$)')
ax8.plot(r_fine, exp_disk(r_fine, Sigma0_s, h_s), '--', color='darkorange', lw=2,
         label=fr'Exp. fit: $\Sigma_0={Sigma0_s:.1f}\ M_\odot\,\mathrm{{pc}}^{{-2}}$, $h={h_s:.2f}\ \mathrm{{kpc}}$')
ax8.set_xlabel('Radius (kpc)'); ax8.set_ylabel(r'$\Sigma_*\ (M_\odot\,\mathrm{pc}^{-2})$')
ax8.set_title('UGC 731 \u2014 Stellar Surface Density Profile'); ax8.legend(framealpha=0.85); ax8.grid(True, alpha=0.30, ls='--')
plt.tight_layout(); safe_savefig(fig, 'Fig8_Stellar_Surface_Density_Profile.png', dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- Fig 9: WISE W1 image (sky-subtracted, stars replaced by the local median, smoothed) with the profile annuli ----------
from scipy.ndimage import binary_dilation
from matplotlib.patches import Ellipse
WISE_IMG_HALF   = 150.0                  # arcsec: half-width of the field (300" x 300")
WISE_IMG_SMOOTH = 1.5                    # Gaussian sigma [pixels], display only
WISE_IMG_CMAP   = "magma"
WISE_IMG_VMIN, WISE_IMG_VMAX = -0.2, 3.0 # DN above sky (linear stretch)
WISE_IMG_STARS  = 15.0                   # DN above the sky: pixels brighter than this (+4 px around) are treated as stars
WISE_IMG_RINGS  = (20.0, 60.0, 119.0)    # arcsec (major axis): the annuli of the stellar profile
WISE_IMG_RCOL   = ("white", "cyan", "lime")
WISE_IMG_WIN    = 10                     # pixels: half-size of the window for the local median that replaces a star

_img  = wimg - wise_sky
_star = binary_dilation(_img > WISE_IMG_STARS, iterations=4) | ~np.isfinite(_img)
_tmp  = np.where(_star, np.nan, _img)
_fill = _img.copy()
_pad  = WISE_IMG_WIN
for _y, _x in zip(*np.where(_star)):                                  # local median of the unmasked pixels around each star pixel
    _w = _tmp[max(_y - _pad, 0):_y + _pad + 1, max(_x - _pad, 0):_x + _pad + 1]
    _v = _w[np.isfinite(_w)]
    _fill[_y, _x] = np.median(_v) if _v.size else 0.0
_show = gaussian_filter(_fill, WISE_IMG_SMOOTH) if WISE_IMG_SMOOTH > 0 else _fill

_ny, _nx = wimg.shape
_El, _Nb = offsets_arcsec(wcs_w, -0.5 - WISE_CX, -0.5 - WISE_CY)
_Er, _Nt = offsets_arcsec(wcs_w, _nx - 0.5 - WISE_CX, _ny - 0.5 - WISE_CY)
_extent = [_El, _Er, _Nb, _Nt]                                         # East to the LEFT

fig, ax = plt.subplots(figsize=(7.4, 6.4))
im = ax.imshow(_show, origin="lower", extent=_extent, cmap=WISE_IMG_CMAP, vmin=WISE_IMG_VMIN, vmax=WISE_IMG_VMAX,
               interpolation="nearest", aspect="equal")
ax.set_xlim(WISE_IMG_HALF, -WISE_IMG_HALF); ax.set_ylim(-WISE_IMG_HALF, WISE_IMG_HALF)
for _r, _c in zip(WISE_IMG_RINGS, WISE_IMG_RCOL):
    ax.add_patch(Ellipse((0, 0), 2*_r, 2*_r*np.cos(np.radians(INC_DEG)), angle=90.0 - WISE_PA, fill=False, ec=_c, lw=1.3))
_L = 1.0/KPC_ARCS; _x1 = 0.86*WISE_IMG_HALF; _y1 = -0.86*WISE_IMG_HALF
ax.plot([_x1, _x1 - _L], [_y1, _y1], "-", color="white", lw=3, solid_capstyle="butt")
ax.text(_x1 - _L/2, _y1 + 0.02*WISE_IMG_HALF, "1 kpc", color="white", ha="center", va="bottom", fontsize=10)
ax.annotate("N", xy=(0.93, 0.93), xytext=(0.93, 0.80), xycoords="axes fraction", textcoords="axes fraction",
            ha="center", va="bottom", color="white", fontsize=10, arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
ax.annotate("E", xy=(0.80, 0.82), xytext=(0.93, 0.82), xycoords="axes fraction", textcoords="axes fraction",
            ha="right", va="center", color="white", fontsize=10, arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
ax.set_xlabel("East offset (arcsec)"); ax.set_ylabel("North offset (arcsec)")
ax.set_title('UGC 731 \u2014 WISE W1 (3.4 $\\mu$m)')
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("DN above sky")
plt.tight_layout(); safe_savefig(fig, "Fig9_WISE_W1_image.png", dpi=300, bbox_inches="tight"); plt.close(fig)

print("Figures and tables written to", OUTDIR)
_log.flush()
