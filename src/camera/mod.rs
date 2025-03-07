use nalgebra_glm::{Mat4, Vec3};
pub mod fly_camera;

const ZOOM_SPEED: f32 = 5.0;
const MIN_DISTANCE: f32 = 0.5;
const ORBIT_SPEED: f32 = 30.0;

#[derive(Default, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Vec3,

    pub zoom_speed: f32,
    pub orbit_speed: f32,
    distance: f32,

    fov: f32,
    width: f32,
    height: f32,
    z_near: f32,
    z_far: f32,

    view: Mat4,
    projection: Mat4,
}

impl Camera {
    pub fn new(
        position: Vec3,
        target: Vec3,
        fov: f32,
        width: f32,
        height: f32,
        z_near: f32,
        z_far: f32,
    ) -> Self {
        let rotation = Vec3::new(0.0, 0.0, 0.0);
        let view = {
            let rot = Mat4::identity();
            let rot = nalgebra_glm::rotate_x(&rot, f32::to_radians(rotation.x));
            let rot = nalgebra_glm::rotate_y(&rot, f32::to_radians(rotation.y));
            let rot = nalgebra_glm::rotate_z(&rot, f32::to_radians(rotation.z));

            let trans = Mat4::identity();
            let trans = nalgebra_glm::translate(&trans, &position);
            trans * rot
        };

        Self {
            position,
            rotation: Vec3::zeros(),
            zoom_speed: ZOOM_SPEED,
            orbit_speed: ORBIT_SPEED,

            distance: nalgebra_glm::distance(&position, &target),
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

    pub fn eeset_settings(&mut self) {
        self.zoom_speed = ZOOM_SPEED;
        self.orbit_speed = ORBIT_SPEED;
    }

    pub fn zoom(&mut self, val: f32, delta_time: f32) {
        let zoom = val * delta_time * self.zoom_speed;
        if self.distance - zoom < MIN_DISTANCE {
            return;
        }

        let forward = nalgebra_glm::normalize(&Vec3::new(0.0, 0.0, self.position.z));
        self.distance -= zoom;
        self.position -= forward * zoom;
        self.update_view();
    }

    pub fn orbit(&mut self, x: f32, y: f32, delta_time: f32) {
        self.rotation += Vec3::new(
            y * delta_time * self.orbit_speed,
            x * delta_time * self.orbit_speed,
            0.0,
        );
        self.update_view();
    }

    pub fn set_look_at_position(&mut self, new_position: Vec3) {
        self.position = new_position;
        self.update_view();
    }

    pub fn set_fov(&mut self, new_fov: f32) {
        self.fov = new_fov;
        self.update_projection();
    }
    pub fn set_aspect(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
        self.update_projection();
    }
    pub fn set_z_near(&mut self, z_near: f32) {
        self.z_near = z_near;
        self.update_projection();
    }
    pub fn set_z_far(&mut self, z_far: f32) {
        self.z_far = z_far;
        self.update_projection();
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
    fn update_view(&mut self) {
        let rotation = Mat4::identity();
        let rotation = nalgebra_glm::rotate_x(&rotation, f32::to_radians(self.rotation.x));
        let rotation = nalgebra_glm::rotate_y(&rotation, f32::to_radians(self.rotation.y));
        let rotation = nalgebra_glm::rotate_z(&rotation, f32::to_radians(self.rotation.z));

        let translation = Mat4::identity();
        let translation = nalgebra_glm::translate(&translation, &self.position);

        self.view = translation * rotation;
    }

    pub fn view_projection(&self) -> Mat4 {
        self.projection * self.view
    }
}
