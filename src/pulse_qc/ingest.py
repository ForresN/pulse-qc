"""Download and read open PhysioNet WFDB records.

Only the files named in a config are fetched, straight from
``https://physionet.org/files/<db>/<version>/``. Every download is checked
against the database's published ``SHA256SUMS.txt`` before it is used, and the
raw files are cached under ``data/raw/`` (git-ignored). Nothing from PhysioNet
is ever committed.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wfdb

PHYSIONET_FILES = "https://physionet.org/files"

#: WFDB annotation symbols that mark a beat (PhysioNet convention).
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

#: Known databases: name -> (version, licence). Used for provenance and attribution.
DATABASES: dict[str, dict[str, str]] = {
    "pulse-transit-time-ppg": {"version": "1.1.0", "licence": "ODbL 1.0"},
    "bidmc": {"version": "1.0.0", "licence": "ODC-By 1.0"},
    "charisdb": {"version": "1.0.0", "licence": "ODC-By 1.0"},
}


@dataclass
class Record:
    """A single-channel signal plus optional beat annotations, in samples."""

    db: str
    name: str
    channel: str
    fs: float
    signal: np.ndarray
    ann_samples: np.ndarray | None = None
    meta: dict[str, str] = field(default_factory=dict)
    source_sha256: dict[str, str] = field(default_factory=dict)
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    units: str = ""

    @property
    def record_id(self) -> str:
        return f"{self.db}/{self.name}:{self.channel}"

    @property
    def duration_s(self) -> float:
        return len(self.signal) / self.fs


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:  # noqa: S310
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def load_checksums(db: str, raw_dir: Path) -> dict[str, str]:
    """Return ``{relative_file: sha256}`` from the database's SHA256SUMS.txt (cached)."""
    version = DATABASES[db]["version"]
    path = raw_dir / db / "SHA256SUMS.txt"
    if not path.exists():
        _fetch(f"{PHYSIONET_FILES}/{db}/{version}/SHA256SUMS.txt", path)
    sums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            sums[parts[1]] = parts[0]
    return sums


def ensure_files(db: str, names: list[str], raw_dir: Path, verify: bool = True) -> dict[str, str]:
    """Make sure each file exists locally and matches its published checksum.

    Returns ``{file_name: sha256}`` for the files requested.
    """
    version = DATABASES[db]["version"]
    sums = load_checksums(db, raw_dir) if verify else {}
    out: dict[str, str] = {}
    for name in names:
        dest = raw_dir / db / name
        if not dest.exists():
            _fetch(f"{PHYSIONET_FILES}/{db}/{version}/{name}", dest)
        digest = sha256_file(dest)
        if verify:
            expected = sums.get(name)
            if expected is None:
                raise RuntimeError(f"{db}/{name} is not listed in SHA256SUMS.txt")
            if expected != digest:
                dest.unlink()
                raise RuntimeError(
                    f"checksum mismatch for {db}/{name}: expected {expected}, got {digest}. "
                    "The file was deleted; re-run to download it again."
                )
        out[name] = digest
    return out


def read_record(
    db: str,
    name: str,
    channel: str,
    raw_dir: Path,
    annotator: str | None = None,
    verify: bool = True,
    aux_channels: list[str] | None = None,
) -> Record:
    """Read one channel of a WFDB record (downloading it first if needed), plus any
    ``aux_channels`` (same sampling rate) that rules may want, e.g. an accelerometer."""
    files = [f"{name}.hea", f"{name}.dat"]
    if annotator:
        files.append(f"{name}.{annotator}")
    digests = ensure_files(db, files, raw_dir, verify=verify)

    rec_path = str(raw_dir / db / name)
    header = wfdb.rdheader(rec_path)
    # Some headers carry stray punctuation in signal names (BIDMC has "PLETH,").
    sig_names = [n.strip(", ") for n in header.sig_name]
    if channel not in sig_names:
        raise KeyError(f"channel {channel!r} not in {db}/{name}: {sig_names}")
    wanted = [channel, *(aux_channels or [])]
    missing = [c for c in wanted if c not in sig_names]
    if missing:
        raise KeyError(f"channels {missing} not in {db}/{name}: {sig_names}")
    idx = [sig_names.index(c) for c in wanted]
    rec = wfdb.rdrecord(rec_path, channels=idx)
    signal = np.asarray(rec.p_signal[:, 0], dtype=float)
    aux = {c: np.asarray(rec.p_signal[:, i + 1], dtype=float) for i, c in enumerate(wanted[1:])}
    units = str(rec.units[0]) if getattr(rec, "units", None) else ""

    ann_samples = None
    if annotator:
        ann = wfdb.rdann(rec_path, annotator)
        # Keep only beat annotations (ignore rhythm/noise labels if present).
        beat_mask = np.array([sym in BEAT_SYMBOLS for sym in ann.symbol], dtype=bool)
        ann_samples = np.asarray(ann.sample, dtype=np.int64)[beat_mask]

    meta = _parse_header_comments(header.comments)
    return Record(
        db=db,
        name=name,
        channel=channel,
        fs=float(header.fs),
        signal=signal,
        ann_samples=ann_samples,
        meta=meta,
        source_sha256=digests,
        aux=aux,
        units=units,
    )


def read_segment(
    db: str,
    name: str,
    channel: str,
    raw_dir: Path,
    start_s: float,
    duration_s: float,
    aux_channels: list[str] | None = None,
) -> Record:
    """Stream a time segment of a long WFDB record over HTTP range requests.

    CHARIS records run for days (70 MB each), so instead of downloading whole files
    the segment is fetched with ``wfdb.rdrecord(sampfrom, sampto)`` and cached as a
    compressed ``.npz`` under ``raw_dir/<db>/segments/``. The record's provenance
    hash is the SHA-256 of the cached samples, and the segment bounds are kept in
    ``meta``. Whole-file checksum verification is not possible for a segment.
    """
    version = DATABASES[db]["version"]
    wanted = [channel, *(aux_channels or [])]
    seg_dir = raw_dir / db / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}__{int(start_s)}s+{int(duration_s)}s__{'_'.join(wanted)}"
    path = seg_dir / f"{tag}.npz"
    if not path.exists():
        header = wfdb.rdheader(name, pn_dir=f"{db}/{version}")
        sig_names = [n.strip(", ") for n in header.sig_name]
        missing = [c for c in wanted if c not in sig_names]
        if missing:
            raise KeyError(f"channels {missing} not in {db}/{name}: {sig_names}")
        fs = float(header.fs)
        sampfrom = int(round(start_s * fs))
        sampto = min(int(round((start_s + duration_s) * fs)), int(header.sig_len))
        if sampto <= sampfrom:
            raise ValueError(f"segment {start_s}s+{duration_s}s is outside {db}/{name}")
        idx = [sig_names.index(c) for c in wanted]
        rec = wfdb.rdrecord(
            name, pn_dir=f"{db}/{version}", sampfrom=sampfrom, sampto=sampto, channels=idx
        )
        arrays = {c: np.asarray(rec.p_signal[:, i], dtype=np.float32) for i, c in enumerate(wanted)}
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            fs=fs,
            sampfrom=sampfrom,
            sampto=sampto,
            units=str(rec.units[0]),
            comments="\n".join(header.comments),
            **arrays,
        )
        tmp.replace(path)
    data = np.load(path)
    fs = float(data["fs"])
    meta = _parse_header_comments(str(data["comments"]).split("\n"))
    meta.update(
        {
            "segment_start_s": str(int(data["sampfrom"]) / fs),
            "segment_end_s": str(int(data["sampto"]) / fs),
        }
    )
    return Record(
        db=db,
        name=name,
        channel=channel,
        fs=fs,
        signal=np.asarray(data[channel], dtype=float),
        ann_samples=None,
        meta=meta,
        source_sha256={path.name: sha256_file(path)},
        aux={c: np.asarray(data[c], dtype=float) for c in wanted[1:]},
        units=str(data["units"]),
    )


def _parse_header_comments(comments: list[str]) -> dict[str, str]:
    """Parse ``<key>: value`` pairs from WFDB header comment lines. Only keeps
    non-identifying descriptive fields (activity, location); demographics are dropped
    on purpose."""
    keep = {"activity", "location"}
    meta: dict[str, str] = {}
    for line in comments:
        tokens = line.replace("<", " <").split("<")
        for tok in tokens:
            if ">:" not in tok:
                continue
            key, _, val = tok.partition(">:")
            key = key.strip()
            if key in keep:
                meta[key] = val.strip().split()[0] if val.strip() else ""
    return meta
