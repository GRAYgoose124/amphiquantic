import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from amphiquantic.utils.mappings import ATOM_COLORS, ATOM_RADII
from amphiquantic.utils.mappings_numba import BOND_DISTANCES_NUMBA


def parse_pdb(lines):
    """Parse a PDB file to extract atom coordinates and types."""
    coords = []
    atom_types = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_type = line[76:78].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coords.append((x, y, z))
            atom_types.append(atom_type)
    return np.array(coords), atom_types


@nb.njit
def determine_bonds(coords, atom_types, bond_distances):
    """Determine bonds based on distance and atom type."""
    # Bond Distances in Angstroms (Not for production use)

    bonds = []
    missing = set()
    num_atoms = len(coords)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            pair = (atom_types[i], atom_types[j])
            if pair in bond_distances:
                min_dist, max_dist = bond_distances[pair]
                if min_dist <= np.linalg.norm(coords[i] - coords[j]) <= max_dist:
                    bonds.append((i, j))
            elif pair[::-1] in bond_distances:
                min_dist, max_dist = bond_distances[pair[::-1]]
                if min_dist <= np.linalg.norm(coords[i] - coords[j]) <= max_dist:
                    bonds.append((i, j))
            else:
                if pair not in missing or pair[::-1] not in missing:
                    missing.add(pair)

    return bonds


# helpers
@nb.njit
def adjust_coordinates(
    raw_coords: list[tuple[int, int, int]],
    fill_size: tuple[int, int],
    margin: tuple[int, int] = (0, 0),
):
    """Adjusts the coordinates to fit within the display or a specified fill size.

    Args:
        raw_coords (list[tuple[int,int,int]]): The raw coordinates to adjust.
        fill_size (_type_, optional): The size to scale to if given. Defaults to None
        margin (tuple, optional): The amount to not fill to. Defaults to (0,0).
    """
    coords = np.array(raw_coords)

    min_vals = np.min(coords, axis=0)
    max_vals = np.max(coords, axis=0)

    scale = max(max_vals - min_vals)
    fill_scale = min(fill_size) / scale
    adjusted_coords = ((coords - min_vals) * fill_scale) + margin[0]

    # margin[0] applies to x and y, ensuring z remains unaffected
    return adjusted_coords


def plot_molecule(filename):
    with open(filename) as f:
        coords, atom_types = parse_pdb(f.readlines())
    bonds = determine_bonds(coords, atom_types, BOND_DISTANCES_NUMBA)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot atoms
    default_color = (0.5, 0.0, 0.5)
    default_scale = 0.5
    default_rescale = 500

    for i, (x, y, z) in enumerate(coords):
        color = ATOM_COLORS.get(atom_types[i], default_color)
        size = ATOM_RADII.get(atom_types[i], default_scale) * default_rescale
        ax.scatter(
            x, y, z, color=color, s=size, label=atom_types[i], edgecolors="black"
        )

    # Plot bonds
    for start, end in bonds:
        xs, ys, zs = zip(coords[start], coords[end])
        ax.plot(xs, ys, zs, color="black")

    # Adding a legend with unique elements
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.show()
