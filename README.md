# Forge

A modern Vulkan renderer written in Rust, focusing on simplicity and performance.

## Features

- Modern Vulkan 1.0+ support
- GLFW window management
- Validation layer support for debugging
- Cross-platform support (including macOS via MoltenVK)
- Dynamic viewport and scissor states
- Basic triangle rendering pipeline

## Prerequisites

- Rust (latest stable version)
- Vulkan SDK
- GLFW development libraries
- glslc (shader compiler)

## Building

1. Install dependencies:

```bash
# Ubuntu/Debian
sudo apt install vulkan-sdk libglfw3-dev

# macOS
brew install vulkan-sdk glfw
```

2. Clone and build the project:

```bash
git clone https://github.com/yourusername/forge.git
cd forge
cargo build --release
```

3. Compile shaders:

```bash
glslc examples/shaders/triangle.vert -o examples/shaders/triangle.vert.spv
glslc examples/shaders/triangle.frag -o examples/shaders/triangle.frag.spv
```

## Running Examples

```bash
cargo run --example testbed
```

## Project Structure

```
forge/
├── src/
│   └── lib.rs
├── examples/
│   ├── testbed.rs
│   └── shaders/
│       ├── triangle.vert
│       ├── triangle.frag
│       ├── triangle.vert.spv
│       └── triangle.frag.spv
├── Cargo.toml
└── README.md
```

## Configuration

Debug validation layers can be enabled/disabled by setting the `VALIDATION_ENABLED` constant.

## Dependencies

- `ash`: Vulkan bindings for Rust
- `glfw`: Window management
- `anyhow`: Error handling
- `log`: Logging infrastructure
- `env_logger`: Logging implementation

## License

[Your chosen license]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
