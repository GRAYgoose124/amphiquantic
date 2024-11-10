. ./rust/set_data_path.sh

PDB_ID="3NIR"

if [ ! -f "/tmp/${PDB_ID}.pdb" ]; then
    curl -o /tmp/${PDB_ID}.pdb https://files.rcsb.org/download/${PDB_ID}.pdb
fi

./bin/plot_pdb "/tmp/${PDB_ID}.pdb" #--no-explicit-bonds

#rm /tmp/${PDB_ID}.pdb