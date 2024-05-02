import yaml
import os

from pdbviz import utilities as ut


ATOM_PROPERTIES_PATH = ut.get_atom_properties_path()

ATOM_PROPERTIES = None
ATOM_COLORS = {}
ATOM_RADII = {}

with open(ATOM_PROPERTIES_PATH, "r") as f:
    ATOM_PROPERTIES = yaml.safe_load(f)
    for atom_type, properties in ATOM_PROPERTIES.items():
        ATOM_COLORS[atom_type] = properties["color"]
        ATOM_RADII[atom_type] = properties["radius"]
