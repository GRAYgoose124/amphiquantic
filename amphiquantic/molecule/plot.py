import matplotlib.pyplot as plt


# from amphiquantic.molecule.bonds import determine_bonds
from rustquantic import determine_bonds, parse_pdb_file, utilities as ut

ATOM_PROPERTIES = ut.load_atom_properties()


def plot_molecule(filename, explicit_bonds=True):
    # coords, atom_types = parse_pdb_file(filename)
    ((coords, atom_types), bonds) = parse_pdb_file(filename)
    if not explicit_bonds:
        print("Ignoring explicit bonds and determining bonds automatically.")
        bonds, near, missing = determine_bonds(coords, atom_types)

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
        color = ATOM_PROPERTIES.get(atom_types[i]).get("color", default_color)
        size = (
            ATOM_PROPERTIES.get(atom_types[i]).get("radius", default_scale)
            * default_rescale
        )
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


def plot_molecule_with_py3dmol(filename):
    import py3Dmol

    (coords, atom_types), bonds = parse_pdb_file(filename)

    # bonds, _, _ = determine_bonds(coords, atom_types)

    view = py3Dmol.view(width=800, height=400)
    for i, (x, y, z) in enumerate(coords):
        color = ATOM_PROPERTIES.get(atom_types[i]).get("color", (0.5, 0.0, 0.5))
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
        )
        view.addSphere(
            {
                "center": {"x": x, "y": y, "z": z},
                "color": color_hex,
                "radius": ATOM_PROPERTIES.get(atom_types[i]).get("radius", 0.5),
            }
        )

    for start, end in bonds:
        p1 = coords[start]
        p2 = coords[end]
        view.addCylinder(
            {
                "start": {"x": p1[0], "y": p1[1], "z": p1[2]},
                "end": {"x": p2[0], "y": p2[1], "z": p2[2]},
                "radius": 0.1,
                "color": "grey",
            }
        )

    view.zoomTo()
    return view.show()
