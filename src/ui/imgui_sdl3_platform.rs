use imgui::{Context, Io, Key};
use sdl3::{
    event::Event,
    keyboard::{Mod, Scancode},
    mouse::MouseButton,
};

pub struct ImguiSdlPlatform {}

impl ImguiSdlPlatform {
    pub fn new(imgui_ctx: &mut Context) -> Self {
        let io = imgui_ctx.io_mut();
        io.backend_flags
            .insert(imgui::BackendFlags::HAS_MOUSE_CURSORS);
        io.backend_flags
            .insert(imgui::BackendFlags::HAS_SET_MOUSE_POS);

        io.config_flags.insert(imgui::ConfigFlags::DOCKING_ENABLE);

        Self {}
    }

    fn sdl3_key_to_imgui(sdl3_keycode: Option<Scancode>) -> Option<imgui::Key> {
        match sdl3_keycode {
            Some(Scancode::Tab) => Some(Key::Tab),
            Some(Scancode::Left) => Some(Key::LeftArrow),
            Some(Scancode::Right) => Some(Key::RightArrow),
            Some(Scancode::Up) => Some(Key::UpArrow),
            Some(Scancode::Down) => Some(Key::DownArrow),
            Some(Scancode::PageUp) => Some(Key::PageUp),
            Some(Scancode::PageDown) => Some(Key::PageDown),
            Some(Scancode::Home) => Some(Key::Home),
            Some(Scancode::End) => Some(Key::End),
            Some(Scancode::Insert) => Some(Key::Insert),
            Some(Scancode::Delete) => Some(Key::Delete),
            Some(Scancode::Backspace) => Some(Key::Backspace),
            Some(Scancode::Space) => Some(Key::Space),
            Some(Scancode::Return) => Some(Key::Enter),
            Some(Scancode::Escape) => Some(Key::Escape),
            Some(Scancode::Apostrophe) => Some(Key::Apostrophe),
            Some(Scancode::Comma) => Some(Key::Comma),
            Some(Scancode::Minus) => Some(Key::Minus),
            Some(Scancode::Period) => Some(Key::Period),
            Some(Scancode::Slash) => Some(Key::Slash),
            Some(Scancode::Semicolon) => Some(Key::Semicolon),
            Some(Scancode::Equals) => Some(Key::Equal),
            Some(Scancode::LeftBracket) => Some(Key::LeftBracket),
            Some(Scancode::Backslash) => Some(Key::Backslash),
            Some(Scancode::RightBracket) => Some(Key::RightBracket),
            Some(Scancode::Grave) => Some(Key::GraveAccent),
            Some(Scancode::CapsLock) => Some(Key::CapsLock),
            Some(Scancode::ScrollLock) => Some(Key::ScrollLock),
            Some(Scancode::NumLockClear) => Some(Key::NumLock),
            Some(Scancode::PrintScreen) => Some(Key::PrintScreen),
            Some(Scancode::Pause) => Some(Key::Pause),
            Some(Scancode::LCtrl) => Some(Key::LeftCtrl),
            Some(Scancode::LShift) => Some(Key::LeftShift),
            Some(Scancode::LAlt) => Some(Key::LeftAlt),
            Some(Scancode::LGui) => Some(Key::LeftSuper),
            Some(Scancode::RCtrl) => Some(Key::RightCtrl),
            Some(Scancode::RShift) => Some(Key::RightShift),
            Some(Scancode::RAlt) => Some(Key::RightAlt),
            Some(Scancode::RGui) => Some(Key::RightSuper),
            Some(Scancode::Application) => Some(Key::Menu),
            Some(Scancode::Kp0) => Some(Key::Keypad0),
            Some(Scancode::Kp1) => Some(Key::Keypad1),
            Some(Scancode::Kp2) => Some(Key::Keypad2),
            Some(Scancode::Kp3) => Some(Key::Keypad3),
            Some(Scancode::Kp4) => Some(Key::Keypad4),
            Some(Scancode::Kp5) => Some(Key::Keypad5),
            Some(Scancode::Kp6) => Some(Key::Keypad6),
            Some(Scancode::Kp7) => Some(Key::Keypad7),
            Some(Scancode::Kp8) => Some(Key::Keypad8),
            Some(Scancode::Kp9) => Some(Key::Keypad9),
            Some(Scancode::A) => Some(Key::A),
            Some(Scancode::B) => Some(Key::B),
            Some(Scancode::C) => Some(Key::C),
            Some(Scancode::D) => Some(Key::D),
            Some(Scancode::E) => Some(Key::E),
            Some(Scancode::F) => Some(Key::F),
            Some(Scancode::G) => Some(Key::G),
            Some(Scancode::H) => Some(Key::H),
            Some(Scancode::I) => Some(Key::I),
            Some(Scancode::J) => Some(Key::J),
            Some(Scancode::K) => Some(Key::K),
            Some(Scancode::L) => Some(Key::L),
            Some(Scancode::M) => Some(Key::M),
            Some(Scancode::N) => Some(Key::N),
            Some(Scancode::O) => Some(Key::O),
            Some(Scancode::P) => Some(Key::P),
            Some(Scancode::Q) => Some(Key::Q),
            Some(Scancode::R) => Some(Key::R),
            Some(Scancode::S) => Some(Key::S),
            Some(Scancode::T) => Some(Key::T),
            Some(Scancode::U) => Some(Key::U),
            Some(Scancode::V) => Some(Key::V),
            Some(Scancode::W) => Some(Key::W),
            Some(Scancode::X) => Some(Key::X),
            Some(Scancode::Y) => Some(Key::Y),
            Some(Scancode::Z) => Some(Key::Z),
            Some(Scancode::F1) => Some(Key::F1),
            Some(Scancode::F2) => Some(Key::F2),
            Some(Scancode::F3) => Some(Key::F3),
            Some(Scancode::F4) => Some(Key::F4),
            Some(Scancode::F5) => Some(Key::F5),
            Some(Scancode::F6) => Some(Key::F6),
            Some(Scancode::F7) => Some(Key::F7),
            Some(Scancode::F8) => Some(Key::F8),
            Some(Scancode::F9) => Some(Key::F9),
            Some(Scancode::F10) => Some(Key::F10),
            Some(Scancode::F11) => Some(Key::F11),
            Some(Scancode::F12) => Some(Key::F12),
            _ => None,
        }
    }

    fn update_key_modifier(io: &mut Io, keymod: Mod) {
        io.add_key_event(
            Key::ModCtrl,
            keymod.intersects(Mod::LCTRLMOD | Mod::RCTRLMOD),
        );
        io.add_key_event(
            Key::ModShift,
            keymod.intersects(Mod::LSHIFTMOD | Mod::RSHIFTMOD),
        );
        io.add_key_event(Key::ModAlt, keymod.intersects(Mod::LALTMOD | Mod::RALTMOD));
        io.add_key_event(
            Key::ModSuper,
            keymod.intersects(Mod::LGUIMOD | Mod::RGUIMOD),
        );
    }

    pub fn process_event(&self, imgui_ctx: &mut Context, event: &Event) {
        let io = imgui_ctx.io_mut();
        match *event {
            Event::KeyDown {
                scancode, keymod, ..
            } => {
                Self::update_key_modifier(io, keymod);
                if let Some(key) = Self::sdl3_key_to_imgui(scancode) {
                    io.add_key_event(key, true);
                }
            }
            Event::KeyUp {
                scancode, keymod, ..
            } => {
                Self::update_key_modifier(io, keymod);
                if let Some(key) = Self::sdl3_key_to_imgui(scancode) {
                    io.add_key_event(key, false);
                }
            }
            Event::MouseMotion { x, y, .. } => {
                io.add_mouse_pos_event([x, y]);
            }
            Event::MouseWheel { x, y, .. } => {
                io.add_mouse_wheel_event([x, y]);
            }
            Event::MouseButtonDown { mouse_btn, .. } => {
                if let Some(btn) = match mouse_btn {
                    MouseButton::Left => Some(imgui::MouseButton::Left),
                    MouseButton::Right => Some(imgui::MouseButton::Right),
                    MouseButton::Middle => Some(imgui::MouseButton::Middle),
                    MouseButton::X1 => Some(imgui::MouseButton::Extra1),
                    MouseButton::X2 => Some(imgui::MouseButton::Extra2),
                    _ => None,
                } {
                    io.add_mouse_button_event(btn, true);
                }
            }
            Event::MouseButtonUp { mouse_btn, .. } => {
                if let Some(btn) = match mouse_btn {
                    MouseButton::Left => Some(imgui::MouseButton::Left),
                    MouseButton::Right => Some(imgui::MouseButton::Right),
                    MouseButton::Middle => Some(imgui::MouseButton::Middle),
                    MouseButton::X1 => Some(imgui::MouseButton::Extra1),
                    MouseButton::X2 => Some(imgui::MouseButton::Extra2),
                    _ => None,
                } {
                    io.add_mouse_button_event(btn, false);
                }
            }
            Event::TextInput { ref text, .. } => {
                log::info!("{}", text);
                text.chars().for_each(|ch| io.add_input_character(ch));
            }
            _ => {}
        }
    }

    pub fn new_frame(
        &self,
        imgui_ctx: &mut Context,
        window: &sdl3::video::Window,
    ) -> anyhow::Result<()> {
        let (mut width, mut height) = window.size();
        let (d_width, d_height) = window.size_in_pixels();
        let io = imgui_ctx.io_mut();
        if window.is_minimized() {
            width = 0;
            height = 0;
        }
        if width > 0 && height > 0 {
            io.display_framebuffer_scale = [
                d_width as f32 / width as f32,
                d_height as f32 / height as f32,
            ];
        }

        imgui_ctx.io_mut().display_size = [width as f32, height as f32];

        Ok(())
    }
}
