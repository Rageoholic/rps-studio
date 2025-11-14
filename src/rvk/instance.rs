// Reconstructed instance module: provides Instance and debug utilities
use ash::{
    Entry,
    vk::{
        self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerEXT,
    },
};
use debug_messenger::instance_debug_callback;
use std::{ffi::CStr, fmt::Debug};
use winit::raw_window_handle::DisplayHandle;

use crate::Version;

/// Represents a VkInstance and contains the Entry for proper management
/// protocols. Also bundles the debug messenger.
pub struct Instance {
    /// Represents the entry point functions as well as an owned pointer to the
    /// underlying vulkan shared library.
    entry: ash::Entry,
    /// Represents a VkInstance and the corresponding function pointers
    instance: ash::Instance,
    /// Represents a VkDebugUtilsMessengerEXT that may or may not be present
    debug_messenger: Option<DebugMessenger>,
}

impl PartialEq for Instance {
    fn eq(&self, other: &Self) -> bool {
        self.instance.handle() == other.instance.handle()
            && self.debug_messenger == other.debug_messenger
    }
}
impl Eq for Instance {}
impl Drop for Instance {
    fn drop(&mut self) {
        if let Some(dm) = self.debug_messenger.take() {
            //SAFETY: Last usage of debug_messenger. This deinitializes it.
            unsafe {
                dm.debug_utils_instance
                    .destroy_debug_utils_messenger(dm.debug_messenger, None)
            };
        };
        //SAFETY: Last use of Instance, all children destroyed
        unsafe { self.instance.destroy_instance(None) };
    }
}
impl Debug for Instance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Instance")
            .field("entry", &"{entry}")
            .field("instance", &self.instance.handle())
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
/// Possible errors returned by Instance::new
pub enum InstanceCreateError {
    #[error("Unspecified vk error: {0}")]
    /// Error has not been assigned a specific error variant. Some vk function
    /// just failed in an unforeseen way
    UnspecifiedVulkan(ash::vk::Result),
    #[error("Error loading vulkan: {0}")]
    /// Could not load our vulkan entry points from ash.
    EntryLoading(ash::LoadingError),
    #[error("Missing necessary window extensions")]
    /// This vulkan impl does not implement the necessary extensions for
    /// rendering (VK_SURFACE_KHR and the corresponding platform extension to
    /// get a VkSurface)
    MissingWindowingExtensions,
}
impl From<ash::vk::Result> for InstanceCreateError {
    fn from(value: ash::vk::Result) -> Self {
        Self::UnspecifiedVulkan(value)
    }
}

impl Instance {
    /// Get a reference to the associated debug messenger
    pub(super) fn get_debug_messenger(&self) -> Option<&DebugMessenger> {
        self.debug_messenger.as_ref()
    }
    /// Get a ref to the ash::Entry associated with this instance
    pub(super) fn parent(&self) -> &Entry {
        &self.entry
    }
    /// Get the Instance handle for comparing with other handles
    pub(super) fn raw_handle(&self) -> ash::vk::Instance {
        self.inner().handle()
    }
    /// Get the ash Instance for passing to ash
    pub(super) fn inner(&self) -> &ash::Instance {
        &self.instance
    }
    /// Construct an instance. Debug Utils validation will be initialized to debug_level if possible
    pub fn new(
        minimum_vk_version: Version,
        debug_level: VulkanDebugLevel,
        display_handle: DisplayHandle,
    ) -> Result<Self, InstanceCreateError> {
        use InstanceCreateError as Error;
        //SAFETY: Somewhat inherently unsafe due to loading a shared library
        //which can run arbitrary initialization code.
        let entry = unsafe { ash::Entry::load() }.map_err(Error::EntryLoading)?;

        //TODO: Engine version, app name, app version
        let app_info = ash::vk::ApplicationInfo::default()
            .api_version(minimum_vk_version.to_vk_version())
            .engine_name(c"RPS Studio");

        let available_extensions_vec =
            unsafe { entry.enumerate_instance_extension_properties(None) }?;

        let available_extensions = available_extensions_vec.iter().map(|ext| {
            ext.extension_name_as_c_str().unwrap_or_else(|err| {
                tracing::warn!(
                    "instance extension name invalid or missing; using empty CStr: {:#?}",
                    err
                );
                c""
            })
        });

        let available_layers_vec = unsafe { entry.enumerate_instance_layer_properties() }?;

        let available_layers = available_layers_vec.iter().map(|layer| {
            layer.layer_name_as_c_str().unwrap_or_else(|err| {
                tracing::warn!(
                    "instance layer name invalid or missing; using empty CStr: {:#?}",
                    err
                );
                c""
            })
        });

        let window_system_exts =
            ash_window::enumerate_required_extensions(display_handle.as_raw())?;

        if !window_system_exts
            .iter()
            .cloned()
            .map(|ptr| unsafe { CStr::from_ptr(ptr) })
            .all(|ext_name| available_extensions.clone().any(|ext| ext.eq(ext_name)))
        {
            return Err(Error::MissingWindowingExtensions);
        }

        const VK_KHRONOS_VALIDATION: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";

        let mut enabled_extension_names = Vec::with_capacity(8);
        let mut enabled_layer_names = Vec::with_capacity(1);

        enabled_extension_names.extend_from_slice(window_system_exts);

        let debug_enabled = debug_level != VulkanDebugLevel::None
            && available_extensions
                .clone()
                .any(|ext| ext.eq(ash::ext::debug_utils::NAME))
            && available_layers
                .clone()
                .any(|ext| ext.eq(VK_KHRONOS_VALIDATION));

        let mut debug_ci = debug_enabled.then(|| {
            enabled_extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
            enabled_layer_names.push(VK_KHRONOS_VALIDATION.as_ptr());
            ash::vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(debug_level.sev_flags())
                .message_type(
                    DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(instance_debug_callback))
        });

        let mut ci = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extension_names)
            .enabled_layer_names(&enabled_layer_names);

        if let Some(dci) = debug_ci.as_mut() {
            ci = ci.push_next(dci)
        }

        let instance = unsafe { entry.create_instance(&ci, None) }?;

        let debug_messenger = if let Some(debug_ci) = debug_ci.as_ref() {
            let debug_utils_instance = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_messenger =
                unsafe { debug_utils_instance.create_debug_utils_messenger(debug_ci, None) }?;
            Some(DebugMessenger {
                debug_messenger,
                debug_utils_instance,
            })
        } else {
            None
        };

        Ok(Self {
            entry,
            instance,
            debug_messenger,
        })
    }

    /// Get a Vec of all physical devices available to this instance
    pub(super) fn enumerate_physical_devices(
        &self,
    ) -> ash::prelude::VkResult<Vec<vk::PhysicalDevice>> {
        unsafe { self.instance.enumerate_physical_devices() }
    }
}

/// Functionality related to the debug manager
mod debug_messenger {
    use std::ffi::c_void;

    use ash::vk::{
        Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
        DebugUtilsMessengerCallbackDataEXT,
    };

    /// Debug Utils Messenger callback. Not for external calling
    pub(super) unsafe extern "system" fn instance_debug_callback(
        message_severity: DebugUtilsMessageSeverityFlagsEXT,
        message_types: DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const DebugUtilsMessengerCallbackDataEXT<'_>,
        _: *mut c_void,
    ) -> Bool32 {
        let callback_data = unsafe { *p_callback_data };

        let msg_id = callback_data.message_id_number;
        let msg_name: String = unsafe { callback_data.message_id_name_as_c_str() }
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<unknown>".into());
        let msg_data: String = unsafe { callback_data.message_as_c_str() }
            .map(|c| c.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<no-message>".into());
        use DebugUtilsMessageSeverityFlagsEXT as SevFlags;
        use DebugUtilsMessageTypeFlagsEXT as TyFlags;
        let mut ty_name = String::with_capacity(16);
        if message_types.contains(TyFlags::DEVICE_ADDRESS_BINDING) {
            ty_name.push_str("ADDRESS BINDING");
        }
        if message_types.contains(TyFlags::GENERAL) {
            ty_name.push_str("GENERAL");
        }
        if message_types.contains(TyFlags::PERFORMANCE) {
            ty_name.push_str("PERF");
        }
        if message_types.contains(TyFlags::VALIDATION) {
            ty_name.push_str("VALIDATION");
        }
        if message_severity.contains(SevFlags::ERROR) {
            tracing::error!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
        } else if message_severity.contains(SevFlags::WARNING) {
            tracing::warn!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
        } else if message_severity.contains(SevFlags::INFO) {
            tracing::info!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
        } else if message_severity.contains(SevFlags::VERBOSE) {
            tracing::trace!("{ty_name}: 0x{msg_id:08x} {msg_name}:\n{msg_data}")
        }

        ash::vk::FALSE
    }
}

/// NONOWNING implementation of a Debug Messenger. Meant only to be used in conjunction with Instance
pub(super) struct DebugMessenger {
    /// The inner debug_messenger handle
    debug_messenger: DebugUtilsMessengerEXT,
    /// The loaded extension function pointers and stuff
    debug_utils_instance: ash::ext::debug_utils::Instance,
}
impl PartialEq for DebugMessenger {
    fn eq(&self, other: &Self) -> bool {
        self.debug_messenger == other.debug_messenger
    }
}
impl Eq for DebugMessenger {}

#[derive(
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    strum_macros::EnumString,
    strum_macros::Display,
    Clone,
    Copy,
)]
/// Represents a VkDebugUtilsMessageSeverityFlags option, representing itself
/// and every option below it
pub enum VulkanDebugLevel {
    #[default]
    /// No debug layer messages
    None,
    /// Print out all messages
    Verbose,
    /// Print out useful information
    Info,
    /// Print out warnings for misuse of certain APIs
    Warn,
    /// Print out only errors like failing to destroy child objects
    Error,
}
impl VulkanDebugLevel {
    /// Convert to the corresponding VkDebugUtilsMessageSeverityFlagsEXT
    fn sev_flags(&self) -> DebugUtilsMessageSeverityFlagsEXT {
        match self {
            VulkanDebugLevel::None => DebugUtilsMessageSeverityFlagsEXT::empty(),
            VulkanDebugLevel::Verbose => {
                DebugUtilsMessageSeverityFlagsEXT::VERBOSE | Self::Info.sev_flags()
            }
            VulkanDebugLevel::Info => {
                DebugUtilsMessageSeverityFlagsEXT::INFO | Self::Warn.sev_flags()
            }
            VulkanDebugLevel::Warn => {
                DebugUtilsMessageSeverityFlagsEXT::WARNING | Self::Error.sev_flags()
            }
            VulkanDebugLevel::Error => DebugUtilsMessageSeverityFlagsEXT::ERROR,
        }
    }
}
