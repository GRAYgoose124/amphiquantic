echo """
ATOM      1  N   ALA A   1      64.000  64.000  64.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      63.000  63.000  63.000  1.00  0.00           C
ATOM      3  C   ALA A   1      62.000  62.000  62.000  1.00  0.00           C
ATOM      4  O   ALA A   1      61.000  61.000  61.000  1.00  0.00           O
""" > /tmp/simple.pdb

./bin/parameterize /tmp/simple.pdb -o /tmp/simple.aqtop --smiles "N[C@@H](C)C(=O)O" || ./bin/parameterize "N[C@@H](C)C(=O)O" --from-smiles -o /tmp/simple.aqtop

./bin/equilibrate /tmp/simple.aqtop --steps 10

rm -f /tmp/simple.pdb /tmp/simple.aqtop /tmp/simple.eq.aqtop /tmp/simple.eq.aqtrj
