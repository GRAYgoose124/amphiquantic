# make simple pdb file
echo """
ATOM      1  N   ALA A   1      64.000  64.000  64.000  1.00  0.00           N
ATOM      2  CA  ALA A   1      63.000  63.000  63.000  1.00  0.00           C
ATOM      3  C   ALA A   1      62.000  62.000  62.000  1.00  0.00           C
ATOM      4  O   ALA A   1      61.000  61.000  61.000  1.00  0.00           O
""" > /tmp/simple.pdb

./bin/solvate /tmp/simple.pdb -o /tmp/simple.solv.aqtop

rm /tmp/simple.pdb /tmp/simple.solv.aqtop
