use std::borrow::ToOwned;
use std::format;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::process::Command;
use std::vec::Vec;
pub fn main() {
    println!("Started PTX compile");
    let result = Command::new("cargo")
        .current_dir("nonphysical_ptx")
        .args([
            "rustc",
            "--target=nvptx64-nvidia-cuda",
            "--release",
            "--features",
            "crypt",
            "--",
            "--emit=asm",
        ])
        .status()
        .expect("Failed to compile PTX");
    println!("Compiled {:?}", result);
    for entry in fs::read_dir("target/nvptx64-nvidia-cuda/release/deps/").unwrap() {
        let path = entry.unwrap().path();
        let path_str = path.to_string_lossy();
        if !path.is_dir() && path_str.ends_with("s") {
            let path_clone = path.clone();
            let name = path_clone
                .as_path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .split("-")
                .next()
                .unwrap();
            let contents = fs::read_to_string(path).expect("Failed to read file");
            let mut blocks = contents.split(".extern");
            let headers = blocks.next().unwrap().to_owned();
            let mut new_file = Vec::new();
            new_file.push(headers.to_owned());

            for block in blocks {
                // Extract the function signature
                if let Some(start_func) = block.find(".func ") {
                    let end_func =
                        block[start_func..].find('\n').unwrap_or(block.len()) + start_func;
                    let func_signature = &block[start_func..end_func].trim();

                    // Extract the parameters
                    if let Some(start_params) = block.find('(') {
                        let end_params =
                            block[start_params..].find(')').unwrap_or(block.len()) + start_params;
                        let params = &block[start_params + 1..end_params].trim();

                        // Extract the .noreturn
                        if let Some(start_noreturn) = block.find(".noreturn") {
                            let end_noreturn = start_noreturn + 9;
                            let noreturn = &block[start_noreturn..end_noreturn].trim();
                            let rest = &block[end_noreturn + 1..].trim();

                            // Construct the output
                            let result = format!(
                                ".visible {}\n(\n{}\n)\n{}{{\n	trap;\n	exit;\n}}\n{}",
                                func_signature, params, noreturn, rest
                            );

                            new_file.push(result);
                        } else {
                            new_file.push(block.to_owned());
                        }
                    }
                }
            }
            let mut file = File::create(format!("{}.ptx", name)).expect("Couldn't write file");
            for chunk in new_file {
                file.write_all(chunk.as_bytes())
                    .expect("Couldn't write to file");
            }
        }
    }
}
