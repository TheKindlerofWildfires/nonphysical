use std::env;

fn main() {
    let cuda_path = env::var("CUDA_PATH").expect("env CUDA_PATH is empty");
    println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cudart");
}