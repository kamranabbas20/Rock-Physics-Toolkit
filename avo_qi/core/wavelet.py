"""Wavelets for synthetic seismogram generation.

All wavelets are returned as ``(t, w)`` with ``t`` in seconds, centred on
zero, and an odd number of samples so that the centre sample sits exactly at
``t = 0``.  Amplitudes are normalised to unit peak.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = [
    "ricker",
    "bandpass_ormsby",
    "load_wavelet",
    "is_zero_phase",
    "amplitude_spectrum",
    "bandwidth",
    "dominant_frequency",
]


def _centred_time_axis(length, dt):
    """Symmetric time axis of odd length covering roughly ``length`` seconds."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    n = int(round(float(length) / float(dt)))
    if n < 1:
        n = 1
    if n % 2 == 0:
        n += 1
    return (np.arange(n) - (n - 1) // 2) * float(dt)


def _unit_peak(w):
    peak = np.max(np.abs(w))
    return w / peak if peak > 0 else w


def ricker(f, dt, length=0.128):
    """Zero-phase Ricker wavelet.

    Parameters
    ----------
    f : float
        Peak frequency in Hz.
    dt : float
        Sample interval in seconds.
    length : float
        Approximate wavelet duration in seconds.

    Returns
    -------
    (t, w) : tuple of ndarray
        Time axis in seconds (centred on zero) and unit-peak amplitudes.
    """
    if f <= 0:
        raise ValueError("peak frequency must be positive")
    t = _centred_time_axis(length, dt)
    a = (np.pi * f * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return t, _unit_peak(w)


def bandpass_ormsby(f1, f2, f3, f4, dt, length=0.128):
    """Zero-phase Ormsby (trapezoidal band-pass) wavelet.

    ``f1 < f2 < f3 < f4`` are the low-cut, low-pass, high-pass and high-cut
    corner frequencies in Hz.
    """
    f1, f2, f3, f4 = (float(x) for x in (f1, f2, f3, f4))
    if not f1 < f2 < f3 < f4:
        raise ValueError("Ormsby corners must satisfy f1 < f2 < f3 < f4")

    t = _centred_time_axis(length, dt)

    # Trapezoidal amplitude spectrum, written with np.sinc (which already
    # carries the pi) for numerical safety at t = 0.
    a43 = (np.pi * f4) ** 2 / (np.pi * f4 - np.pi * f3)
    a33 = (np.pi * f3) ** 2 / (np.pi * f4 - np.pi * f3)
    a22 = (np.pi * f2) ** 2 / (np.pi * f2 - np.pi * f1)
    a11 = (np.pi * f1) ** 2 / (np.pi * f2 - np.pi * f1)

    w = (
        a43 * np.sinc(f4 * t) ** 2
        - a33 * np.sinc(f3 * t) ** 2
        - a22 * np.sinc(f2 * t) ** 2
        + a11 * np.sinc(f1 * t) ** 2
    )
    return t, _unit_peak(w)


def is_zero_phase(w, tol=1e-3):
    """True when ``w`` is (near) symmetric about its centre sample.

    A zero-phase wavelet is symmetric and peaks at its centre.  ``tol`` is the
    maximum allowed asymmetry, relative to the peak amplitude.
    """
    w = np.asarray(w, dtype=float)
    return symmetry_error(w) <= tol and int(np.argmax(np.abs(w))) == (w.size - 1) // 2


def symmetry_error(w):
    """Peak-normalised RMS difference between a wavelet and its reverse."""
    w = np.asarray(w, dtype=float)
    if w.size < 2:
        return 0.0
    peak = np.max(np.abs(w))
    if peak == 0:
        return 0.0
    # Compare about the centre sample; an even-length wavelet has no exact
    # centre, so it is compared about its midpoint and will read as slightly
    # asymmetric, which is the honest answer.
    return float(np.sqrt(np.mean((w - w[::-1]) ** 2)) / peak)


def amplitude_spectrum(w, dt, pad=8):
    """Amplitude spectrum of a wavelet — ``(frequency_hz, amplitude)``.

    The amplitude is normalised to its own peak, because what is being read
    off a wavelet spectrum is the *shape* — where the energy sits and how far
    it extends — not an absolute level that depends on how the wavelet was
    scaled.

    ``pad`` zero-pads by that factor before transforming.  A wavelet is short,
    so its raw spectrum is sampled at only a handful of frequencies and reads
    as a jagged line; padding interpolates it onto a smooth curve without
    inventing bandwidth, since zero-padding cannot add information.
    """
    w = np.asarray(w, dtype=float).ravel()
    dt = float(dt)
    if w.size < 2 or dt <= 0:
        return np.array([]), np.array([])

    n = int(max(w.size * max(int(pad), 1), w.size))
    spec = np.abs(np.fft.rfft(w, n=n))
    freqs = np.fft.rfftfreq(n, d=dt)

    peak = float(np.max(spec)) if spec.size else 0.0
    return freqs, (spec / peak if peak > 0 else spec)


def bandwidth(w, dt, level_db=-6.0):
    """Frequencies where the spectrum falls to ``level_db`` of its peak.

    Returns ``(low_hz, high_hz)``, the conventional way of quoting a
    wavelet's usable band.  Edges are linearly interpolated between spectral
    samples rather than snapped to the nearest one.
    """
    freqs, amplitude = amplitude_spectrum(w, dt)
    if freqs.size == 0:
        return float("nan"), float("nan")

    threshold = 10.0 ** (float(level_db) / 20.0)
    above = amplitude >= threshold
    if not above.any():
        return float("nan"), float("nan")

    first, last = int(np.argmax(above)), int(len(above) - 1 - np.argmax(above[::-1]))

    def crossing(inside, outside):
        """Linear interpolation of the threshold between two samples."""
        a0, a1 = amplitude[outside], amplitude[inside]
        if a1 == a0:
            return float(freqs[inside])
        t = (threshold - a0) / (a1 - a0)
        return float(freqs[outside] + t * (freqs[inside] - freqs[outside]))

    low = float(freqs[0]) if first == 0 else crossing(first, first - 1)
    high = (float(freqs[-1]) if last == len(above) - 1
            else crossing(last, last + 1))
    return low, high


def dominant_frequency(w, dt):
    """Amplitude-spectrum peak frequency of a wavelet, in Hz."""
    freqs, amplitude = amplitude_spectrum(w, dt)
    if freqs.size == 0:
        return 0.0
    return float(freqs[int(np.argmax(amplitude))])


def _read_wavelet_file(path):
    """Read a wavelet from CSV/TXT: one column of amplitudes, or time+amplitude."""
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xls"):
        import pandas as pd

        arr = pd.read_excel(path).to_numpy(dtype=float)
    else:
        try:
            arr = np.loadtxt(path, delimiter=",", ndmin=2)
        except ValueError:
            # Header row, or whitespace-delimited.
            import pandas as pd

            arr = pd.read_csv(path, sep=None, engine="python").to_numpy(dtype=float)
    return np.atleast_2d(arr)


def load_wavelet(path_or_array, dt, tol=1e-3):
    """Ingest a user wavelet and put it on the trace sample rate.

    Parameters
    ----------
    path_or_array : str, path, or array_like
        A CSV/TXT/Excel file, or an in-memory array.  One column is read as
        amplitudes (assumed already at ``dt``); two columns are read as
        ``time, amplitude`` and resampled onto ``dt``.
    dt : float
        Target trace sample interval in seconds.
    tol : float
        Asymmetry tolerance for the zero-phase report.

    Returns
    -------
    (t, w, meta) : tuple
        Time axis, unit-peak amplitudes, and a dict reporting ``zero_phase``,
        ``symmetry_error``, ``dt``, ``n_samples``, ``resampled`` and
        ``dominant_frequency``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    if isinstance(path_or_array, (str, bytes, os.PathLike)):
        arr = _read_wavelet_file(path_or_array)
    else:
        arr = np.atleast_2d(np.asarray(path_or_array, dtype=float))

    if arr.shape[0] == 1 and arr.shape[1] > 1:
        arr = arr.T  # a single row is a single column laid sideways

    resampled = False
    if arr.shape[1] >= 2:
        t_in = arr[:, 0].astype(float)
        w_in = arr[:, 1].astype(float)
        order = np.argsort(t_in)
        t_in, w_in = t_in[order], w_in[order]
        dt_in = float(np.median(np.diff(t_in))) if t_in.size > 1 else dt
        if t_in.size > 1 and abs(dt_in - dt) > 1e-12:
            n = int(round((t_in[-1] - t_in[0]) / dt)) + 1
            if n % 2 == 0:
                n += 1
            t = t_in[0] + np.arange(n) * dt
            w = np.interp(t, t_in, w_in)
            resampled = True
        else:
            t, w = t_in, w_in
    else:
        w = arr[:, 0].astype(float)
        t = _centred_time_axis((w.size - 1) * dt, dt)
        if t.size != w.size:  # even-length input: keep the samples, shift the axis
            t = (np.arange(w.size) - (w.size - 1) / 2.0) * dt

    w = _unit_peak(np.asarray(w, dtype=float))
    t = np.asarray(t, dtype=float)

    meta = {
        "dt": float(dt),
        "n_samples": int(w.size),
        "resampled": resampled,
        "symmetry_error": symmetry_error(w),
        "zero_phase": bool(is_zero_phase(w, tol=tol)),
        "dominant_frequency": dominant_frequency(w, dt),
    }
    return t, w, meta
