use nalgebra_glm::{Mat4, Quat, Vec3};
use sdl3::{
    keyboard::{KeyboardState, Scancode},
    mouse::RelativeMouseState,
};

const BASE_CAMERA_SPEED: f32 = 2.5;
const BASE_ROTATION_SPEED: f32 = 30.0;

pub struct FlyCamera {
    pub camera_speed: f32,
    pub rotation_speed: f32,

    position: Vec3,
    forward: Vec3,
    rotation: Quat,

    fov: f32,
    width: f32,
    height: f32,
    z_near: f32,
    z_far: f32,

    projection: Mat4,
    view: Mat4,
}

impl FlyCamera {
    pub fn new(position: Vec3, fov: f32, width: f32, height: f32, z_near: f32, z_far: f32) -> Self {
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let rotation = Quat::identity();

        let view =
            nalgebra_glm::look_at_rh(&(position + forward), &position, &Vec3::new(0.0, 1.0, 0.0));

        Self {
            camera_speed: BASE_CAMERA_SPEED,
            rotation_speed: BASE_ROTATION_SPEED,

            position,
            forward,
            rotation,

            fov,
            width,
            height,
            z_near,
            z_far,

            view,
            projection: nalgebra_glm::perspective_fov_rh_zo(
                f32::to_radians(fov),
                width,
                height,
                z_near,
                z_far,
            ),
        }
    }

    fn translate(&mut self, keyboard_state: &KeyboardState, delta_time: f32) {
        let delta = self.camera_speed * delta_time;
        let mut updated = false;

        if keyboard_state.is_scancode_pressed(Scancode::W) {
            self.position -= self.forward * delta;
            updated = true;
        }
        if keyboard_state.is_scancode_pressed(Scancode::S) {
            self.position += self.forward * delta;
            updated = true;
        }
        if keyboard_state.is_scancode_pressed(Scancode::A) {
            let right = &nalgebra_glm::normalize(&nalgebra_glm::cross(
                &self.forward,
                &Vec3::new(0.0, 1.0, 0.0),
            ));
            self.position += right * delta;
            updated = true;
        }
        if keyboard_state.is_scancode_pressed(Scancode::D) {
            let right = &nalgebra_glm::normalize(&nalgebra_glm::cross(
                &self.forward,
                &Vec3::new(0.0, 1.0, 0.0),
            ));
            self.position -= right * delta;
            updated = true;
        }
        if updated {
            self.update_view();
        }
    }

    fn rotate(&mut self, mouse_state: &RelativeMouseState, delta_time: f32) {
        let x = f32::to_radians(-mouse_state.x() * delta_time * self.rotation_speed);
        let y = f32::to_radians(mouse_state.y() * delta_time * self.rotation_speed);
        let right = nalgebra_glm::normalize(&nalgebra_glm::cross(
            &nalgebra_glm::normalize(&self.forward),
            &Vec3::new(0.0, 1.0, 0.0),
        ));
        let up = nalgebra_glm::normalize(&nalgebra_glm::cross(
            &right,
            &nalgebra_glm::normalize(&self.forward),
        ));

        let quat_x = nalgebra_glm::quat_angle_axis(x, &up);
        let quat_y = nalgebra_glm::quat_angle_axis(y, &right);
        let rotation = quat_y * quat_x * Quat::identity();

        self.forward = nalgebra_glm::quat_rotate_vec3(&rotation, &self.forward);

        self.update_view();
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view
    }

    fn update_view(&mut self) {
        let view = nalgebra_glm::look_at_rh(
            &(self.position + self.forward),
            &self.position,
            &Vec3::new(0.0, 1.0, 0.0),
        );
        self.view = view;
    }

    fn update_projection(&mut self) {
        self.projection = nalgebra_glm::perspective_fov_rh_zo(
            f32::to_radians(self.fov),
            self.width,
            self.height,
            self.z_near,
            self.z_far,
        );
    }

    pub fn reset_settings(&mut self) {
        self.camera_speed = BASE_CAMERA_SPEED;
        self.rotation_speed = BASE_ROTATION_SPEED;
    }

    pub fn set_fov(&mut self, fov: f32) {
        self.fov = fov;
        self.update_projection();
    }

    pub fn set_aspect_ratio(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
        self.update_projection();
    }

    pub fn set_near_plane(&mut self, z_near: f32) {
        self.z_near = z_near;
        self.update_projection();
    }

    pub fn set_far_plane(&mut self, z_far: f32) {
        self.z_far = z_far;
        self.update_projection();
    }

    pub fn update(
        &mut self,
        keyboard_state: &KeyboardState,
        mouse_state: &RelativeMouseState,
        delta_time: f32,
    ) {
        self.translate(keyboard_state, delta_time);
        self.rotate(mouse_state, delta_time);
    }
}
