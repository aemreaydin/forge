use anyhow::anyhow;
use ash::{ext, vk, Entry};
use std::{ffi::CStr, sync::Arc};

const VALIDATION_LAYER: &CStr =
    unsafe { CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0") };

pub struct DebugUtils {
    debug_utils_messenger: vk::DebugUtilsMessengerEXT,
    loader: ext::debug_utils::Instance,
}
impl DebugUtils {
    pub fn new(entry: &Entry, instance: &ash::Instance) -> anyhow::Result<Self> {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(Self::vulkan_debug_callback));
        let loader = ext::debug_utils::Instance::new(entry, instance);
        let debug_utils_messenger =
            unsafe { loader.create_debug_utils_messenger(&debug_info, None)? };

        Ok(Self {
            loader,
            debug_utils_messenger,
        })
    }

    pub fn destroy(&self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.debug_utils_messenger, None)
        };
    }

    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;
        let message_id_name = if callback_data.p_message_id_name.is_null() {
            std::borrow::Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };
        let message = if callback_data.p_message.is_null() {
            std::borrow::Cow::from("")
        } else {
            CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };

        let formatted_message = format!(
            "\n[{:?}] [{}] ({}): {}\n",
            message_type, message_id_name, callback_data.message_id_number, message
        );

        match message_severity {
            s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) => {
                log::error!("{}", formatted_message);
            }
            s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) => {
                log::warn!("{}", formatted_message);
            }
            s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) => {
                log::info!("{}", formatted_message);
            }
            s if s.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE) => {
                log::debug!("{}", formatted_message);
            }
            _ => log::trace!("{}", formatted_message),
        }

        vk::FALSE
    }
}

pub struct Instance {
    pub(crate) entry: Entry,
    pub instance: ash::Instance,

    debug_utils: Option<DebugUtils>,
}

impl Instance {
    pub fn new(
        required_extensions: &[&CStr],
        enable_validation: bool,
    ) -> anyhow::Result<Arc<Self>> {
        let entry = unsafe { Entry::load()? };
        let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"forge\0") };
        let layers = if enable_validation {
            Self::validate_required_layers(&entry)?
        } else {
            vec![]
        };
        let extensions = Self::validate_required_instance_extensions(
            &entry,
            required_extensions,
            enable_validation,
        )?;
        let version =
            unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);

        let appinfo = vk::ApplicationInfo::default()
            .engine_name(app_name)
            .engine_version(0)
            .api_version(version);

        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .flags(create_flags);

        let instance = unsafe { entry.create_instance(&create_info, None)? };
        let debug_utils = if enable_validation {
            Some(DebugUtils::new(&entry, &instance)?)
        } else {
            None
        };

        Ok(Arc::new(Self {
            entry,
            instance,
            debug_utils,
        }))
    }

    pub fn destroy(&self) {
        unsafe {
            if let Some(debug_utils) = &self.debug_utils {
                debug_utils.destroy();
            }
            self.instance.destroy_instance(None);
        }
    }

    fn validate_required_layers(entry: &Entry) -> anyhow::Result<Vec<*const i8>> {
        let layer_names = unsafe {
            entry
                .enumerate_instance_layer_properties()?
                .iter()
                .map(|layer| CStr::from_ptr(layer.layer_name.as_ptr()))
                .collect::<Vec<_>>()
        };
        let layers = {
            if layer_names.contains(&VALIDATION_LAYER) {
                [VALIDATION_LAYER].iter().map(|l| l.as_ptr()).collect()
            } else {
                vec![]
            }
        };
        Ok(layers)
    }

    fn validate_required_instance_extensions(
        entry: &Entry,
        requested_extensions: &[&CStr],
        enable_validation: bool,
    ) -> anyhow::Result<Vec<*const i8>> {
        let mut required_extensions = requested_extensions.to_vec();
        if enable_validation {
            required_extensions.push(vk::EXT_DEBUG_UTILS_NAME);
        }

        let system_extension_names = unsafe {
            entry
                .enumerate_instance_extension_properties(None)?
                .iter()
                .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()))
                .collect::<Vec<_>>()
        };
        for required_extension in &required_extensions {
            if !system_extension_names.contains(required_extension) {
                return Err(anyhow!(
                    "extension {} not supported by the system",
                    required_extension.to_string_lossy()
                ));
            }
        }
        Ok(required_extensions.iter().map(|ext| ext.as_ptr()).collect())
    }
}
