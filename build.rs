use anyhow::anyhow;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

const SHADER_DIR: &str = "shaders";

fn main() -> anyhow::Result<()> {
    let profile = env::var("PROFILE")?;
    let shader_out_dir = PathBuf::from(format!("target/{}/examples/{SHADER_DIR}", profile));
    if !shader_out_dir.exists() {
        fs::create_dir_all(&shader_out_dir)?;
    }

    let shader_source_dir = PathBuf::from(SHADER_DIR);
    compile_shaders(&shader_source_dir, &shader_out_dir)?;

    println!("cargo:rerun-if-changed=src/shaders");
    Ok(())
}

fn compile_shaders(source_dir: &Path, output_dir: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(source_dir)? {
        let entry = entry?;
        let path = entry.path();

        let extension = path
            .extension()
            .expect("failed to parse the path extension");
        let file_name = entry.file_name().into_string().expect("invalid filename");
        println!("{:?}", file_name);
        if extension == "glgl"
            && (file_name.contains("vert")
                || file_name.contains("frag")
                || file_name.contains("comp"))
        {
            let shader_name = file_name;
            let spirv_name = path.file_stem().unwrap().to_str().unwrap().to_string() + ".spv";
            let spirv_path = output_dir.join(&spirv_name);

            println!("Compiling shader: {}", shader_name);

            let status = Command::new("glslang")
                .arg("-V") // Output SPIR-V binary
                .arg(&path)
                .arg("-o")
                .arg(&spirv_path)
                .status()
                .expect("Failed");

            if !status.success() {
                return Err(anyhow!("Failed to compile shader: {:?}", shader_name));
            }

            // Tell cargo to rerun if this specific shader changes
            // println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    Ok(())
}
