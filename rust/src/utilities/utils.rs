use std::env;

pub(crate) fn get_data_path() -> String {
    // append bond_distances.yml to the data path
     env::var("PDBVIZ_DATA_PATH").unwrap_or_else(|_| "rust/data".to_string())
}

pub(crate) fn get_bond_distances_path() -> String {
    let data_path = get_data_path();
    format!("{}/bond_distances.yml", data_path)
}

pub(crate) fn get_atom_properties_path() -> String {
    let data_path = get_data_path();
    format!("{}/atom_properties.yml", data_path)
}

