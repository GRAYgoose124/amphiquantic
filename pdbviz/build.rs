use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let data_path = Path::new(&out_dir).join("data");
    fs::create_dir_all(&data_path).unwrap();
    fs::copy("data/bond_distances.yml", data_path.join("bond_distances.yml")).unwrap();
    println!("cargo:rerun-if-changed=data/bond_distances.yml");
}
