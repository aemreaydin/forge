pub trait Handle<T> {
    fn handle(&self) -> &T;
}

#[macro_export]
macro_rules! vulkan_handle {
    ($name:ident, $vk_name:ident, $vk_type:path) => {
        use $crate::utils::handle::Handle;

        impl Handle<$vk_type> for $name {
            fn handle(&self) -> &$vk_type {
                &self.$vk_name
            }
        }
    };
}
