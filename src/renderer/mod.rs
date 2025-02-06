pub mod base_renderer;
pub mod buffer;
pub mod device;
pub mod image;
pub mod instance;
pub mod physical_device;
pub mod shader_object;
pub mod surface;
pub mod swapchain;
pub mod vulkan_context;

pub enum FrameState {
    PRESENT,
    SUBMIT,
    RESIZED,
}

pub trait Renderer {
    fn start_frame(&self) -> anyhow::Result<()>;
    fn end_frame(&self) -> anyhow::Result<()>;
    fn draw(&self) -> anyhow::Result<()>;

    fn resized(&mut self, dims: &[u32; 2]) -> anyhow::Result<bool>;

    fn update(&self) -> anyhow::Result<()> {
        self.start_frame()?;
        self.draw()?;
        self.end_frame()
    }
}
