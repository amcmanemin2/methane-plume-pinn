#!/usr/bin/env python3
"""Extract cropped pit-2 LES fields, one HDF5 file per run (see src/methane_pinn/extract.py).

    python3 scripts/extract.py --run daytime_N                     # one run
    python3 scripts/extract.py --run daytime_N --frames 3 \\
        --out-dir $SCRATCH/methane-plume-pinn/test                  # quick test
    python3 scripts/extract.py --all                               # all 24 (skips existing files)

Output goes to $PINN_DATA_ROOT/les/<run>.h5 by default. Run on a compute node.
"""

import argparse
import sys
import traceback
from pathlib import Path

from methane_pinn import runs as R
from methane_pinn.extract import extract_run
from methane_pinn.paths import data_root


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="append", help="run name, e.g. stable_NW (repeatable)")
    g.add_argument("--all", action="store_true", help="all 24 runs")
    p.add_argument("--out-dir", type=Path, default=None, help="default: $PINN_DATA_ROOT/les")
    p.add_argument("--frames", type=int, default=None, help="only the first N frames (testing)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    args = p.parse_args(argv)

    out_dir = args.out_dir or data_root() / "les"
    todo = list(R.ALL_RUNS) if args.all else [R.get(n) for n in args.run]
    times = R.snapshot_times()[: args.frames] if args.frames else None

    failed = []
    for run in todo:
        out = out_dir / f"{run.name}.h5"
        if out.exists() and not args.force:
            print(f"{run.name}: {out} exists, skipping")
            continue
        try:
            extract_run(run, out, times=times, force=args.force)
        except Exception:
            traceback.print_exc()
            failed.append(run.name)
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
