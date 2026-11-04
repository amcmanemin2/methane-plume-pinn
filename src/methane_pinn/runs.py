"""Registry of the 24 FastEddy runs and the train/test split (PLAN.md §1, §5).

Every run carries the same 30 tracers; this project uses only the 15 pit-2 sources
(AuxScalar_15..29). Runs are split by whole run so that augmented combinations of
the same flow field never cross splits.
"""

from dataclasses import dataclass

# Stability class -> FastEddy output tree under $SCRATCH/fasteddy/aus/runs/muswellbrook
TREES = {
    "daytime": "idealized_auxscbc",
    "convective": "idealized_convective_auxscbc",
    "stable": "idealized_stable_auxscbc",
}

# Bearing (near-surface wind-from label) -> run directory name
RUN_DIRS = {
    "SE": "r01_direction_sweep_se",
    "N": "r02_direction_sweep_n",
    "NE": "r03_direction_sweep_ne",
    "E": "r04_direction_sweep_e",
    "S": "r05_direction_sweep_s",
    "SW": "r06_direction_sweep_sw",
    "W": "r07_direction_sweep_w",
    "NW": "r08_direction_sweep_nw",
}

# Training bearings per class; the other four are test. They alternate between
# classes so every bearing is trained on in at least one class (PLAN.md §5).
TRAIN_BEARINGS = {
    "daytime": ("N", "E", "S", "W"),
    "convective": ("NE", "SE", "SW", "NW"),
    "stable": ("N", "SE", "S", "NW"),
}

# Pit-2 sources: tracer indices and nominal release (emission manifest)
PIT2_SCALARS = tuple(range(15, 30))
Q0_KGPH = 1000.0 / 30.0            # 33.33 kg/h per tracer
RELEASE_START_S = 3600.0
RELEASE_END_S = 7200.0
DT_MODEL = 0.02                    # s; FE.<step> with step = t / dt
OUTPUT_EVERY_S = 120.0


@dataclass(frozen=True)
class Run:
    stab: str          # daytime | convective | stable
    bearing: str       # N, NE, ...

    @property
    def tree(self) -> str:
        return TREES[self.stab]

    @property
    def run_dir(self) -> str:
        return RUN_DIRS[self.bearing]

    @property
    def name(self) -> str:
        return f"{self.stab}_{self.bearing}"

    @property
    def split(self) -> str:
        return "train" if self.bearing in TRAIN_BEARINGS[self.stab] else "test"


ALL_RUNS = tuple(Run(s, b) for s in TREES for b in RUN_DIRS)


def runs(split: str | None = None) -> list[Run]:
    """All runs, or only those in `split` ('train' or 'test')."""
    return [r for r in ALL_RUNS if split is None or r.split == split]


def get(name: str) -> Run:
    """Look up a run by name, e.g. 'stable_NW'."""
    for r in ALL_RUNS:
        if r.name == name:
            return r
    raise KeyError(f"unknown run {name!r}; expected one of {[r.name for r in ALL_RUNS]}")


def snapshot_times(t0: float = RELEASE_START_S, t1: float = RELEASE_END_S) -> list[float]:
    """Output times in [t0, t1]; 3600..7200 s gives 31 frames (the first one is tracer-free)."""
    n = int(round((t1 - t0) / OUTPUT_EVERY_S))
    return [t0 + k * OUTPUT_EVERY_S for k in range(n + 1)]


def step_of(t: float) -> int:
    return int(round(t / DT_MODEL))
