
use crate::utilities::get_data_path;

pub fn get_shader_path() -> String {
    let data_path = get_data_path();
    format!("{}/shaders", data_path)
}

lazy_static::lazy_static! {
    pub(crate) static ref MINIMIZE_SHADER: String = {
        let path = format!("{}/minimize.wgsl", get_shader_path());
        std::fs::read_to_string(path).unwrap()
    };
    pub(crate) static ref SIMULATE_SHADER: String = {
        let path = format!("{}/simulate.wgsl", get_shader_path());
        std::fs::read_to_string(path).unwrap()
    };
}