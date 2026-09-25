"""Standard GROMACS .mdp run parameter templates."""

from __future__ import annotations

from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

COMMON = """; Shared settings — amphiquantic GROMACS validation templates
pbc                 = xyz
cutoff-scheme       = Verlet
coulombtype         = PME
rcoulomb            = 1.0
rvdw                = 1.0
pme_order           = 4
fourierspacing      = 0.16
"""

EM_MDP = COMMON + """
; Energy minimization (steep)
integrator          = steep
emtol               = 100.0
emstep              = 0.01
nsteps              = 500
"""

NVT_MDP = COMMON + """
; NVT equilibration
integrator          = md
dt                  = 0.002
nsteps              = 5000
nstenergy           = 500
nstlog              = 500
continuation        = no
tcoupl              = V-rescale
tc-grps             = System
tau-t               = 0.1
ref-t               = 300
pcoupl              = no
constraints         = h-bonds
constraint-algorithm = lincs
"""

NPT_MDP = COMMON + """
; NPT equilibration
integrator          = md
dt                  = 0.002
nsteps              = 5000
nstenergy           = 500
continuation        = yes
tcoupl              = V-rescale
tc-grps             = System
tau-t               = 0.1
ref-t               = 300
pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau-p               = 2.0
ref-p               = 1.0
compressibility     = 4.5e-5
constraints         = h-bonds
constraint-algorithm = lincs
"""

MD_MDP = COMMON + """
; Production MD
integrator          = md
dt                  = 0.002
nsteps              = 10000
nstenergy           = 1000
nstlog              = 1000
continuation        = yes
tcoupl              = V-rescale
tc-grps             = System
tau-t               = 0.1
ref-t               = 300
pcoupl              = no
constraints         = h-bonds
constraint-algorithm = lincs
"""

TEMPLATES = {
    "em": EM_MDP,
    "nvt": NVT_MDP,
    "npt": NPT_MDP,
    "md": MD_MDP,
}


def write_mdp(name: str, outdir: Path) -> Path:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown MDP template: {name}")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.mdp"
    path.write_text(TEMPLATES[name])
    return path


def write_all_mdps(outdir: Path) -> dict[str, Path]:
    return {name: write_mdp(name, outdir) for name in TEMPLATES}
