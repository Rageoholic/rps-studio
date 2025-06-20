//! CURRENTLY IN CONSTRUCTION

#![warn(
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks,
    missing_debug_implementations,
    clippy::allow_attributes_without_reason,
    clippy::missing_safety_doc
)]
// Uncomment to check for undocumented stuff
#![warn(clippy::missing_docs_in_private_items)]

/// Current Top Level crate for our app. Will break up in future
use core::fmt::Debug;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use app_dirs2::AppInfo;

use clap::Parser;
use naga::{
    back::spv::{self, PipelineOptions, SourceLanguage},
    front::glsl::Options,
    valid::{ShaderStages, SubgroupOperationSet},
    ShaderStage,
};
use rvk::{Device, Instance, Surface, Swapchain, VulkanDebugLevel};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop,
    raw_window_handle::HasDisplayHandle, window::Window,
};

/// Our app info for app_dirs2
const APP_INFO: AppInfo = AppInfo {
    name: "rps-studio",
    author: "Rageoholic",
};

/// State of the application while running
#[derive(Debug)]
struct RunningState {
    /// Window that we are rendering to
    win: Arc<Window>,
    /// Our vk instance
    instance: Arc<Instance>,
    /// Surface corresponding to window
    _surface: Arc<Surface>,
    /// handle to the GPU device
    device: Arc<Device>,
    /// A swapchain we are currently using
    _swapchain: Swapchain,
}

/// State of the application while suspended
#[derive(Debug)]
struct SuspendedState {
    /// Application window that is suspended
    win: Arc<Window>,
    /// Our vk instance
    instance: Arc<Instance>,
    /// handle to GPU device
    device: Arc<Device>,
}

/// State of the app before initialization
#[derive(Debug)]
struct UninitState {
    /// Our vk instance (contains pre-initialization stuff safe to make before
    /// we have a window)
    instance: Arc<Instance>,
}

/// Enumeration of the app state. Represents an FSM
#[derive(Debug)]
enum AppState {
    /// Uninitialized. Used before we enter the main loop.
    Uninit(UninitState),
    /// Running. The event loop is in polling mode and will not wait for new
    /// events
    Running(RunningState),
    /// Suspended. The app is in wait mode and once all events are dispatched,
    /// we will not resume until we recieve new events
    Suspended(SuspendedState),
    /// Exited. The app has effectively ended its run
    Exiting,
}

/// Wrapper for AppState. Wraps in an option and allows operations like taking
/// temporary Ownership of the AppState. THIS SHOULD NEVER BE NONE WHEN CONTROL
/// IS RETURNED TO THE EVENT LOOP
#[derive(Debug)]
struct AppRunner {
    /// Internal app_state. Should never be None outside of methods on
    /// AppRunner.
    app_state: Option<AppState>,
}

impl AppRunner {
    /// Construct an AppRunner from an UninitState
    fn new(us: UninitState) -> Self {
        Self {
            app_state: Some(AppState::Uninit(us)),
        }
    }

    /// Peek at an uninit state
    fn as_uninit(&self) -> Option<&UninitState> {
        if let Some(AppState::Uninit(ref state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    /// Peek at a running state
    fn as_running(&self) -> Option<&RunningState> {
        if let Some(AppState::Running(ref state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    /// Peek at a suspended state
    fn as_suspended(&self) -> Option<&SuspendedState> {
        if let Some(AppState::Suspended(ref state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }
    #[expect(dead_code, reason = "Simply unused currently but completes set")]
    /// Peek at an uninit state but mut
    fn as_uninit_mut(&mut self) -> Option<&mut UninitState> {
        if let Some(AppState::Uninit(ref mut state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    /// Peek at a running state but mut
    fn as_running_mut(&mut self) -> Option<&mut RunningState> {
        if let Some(AppState::Running(ref mut state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    #[expect(dead_code, reason = "Simply unused currently but completes set")]
    /// Peek at a suspended state but mut
    fn as_suspended_mut(&mut self) -> Option<&mut SuspendedState> {
        if let Some(AppState::Suspended(ref mut state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    /// Take ownership of an Uninit state for a state transition
    fn take_uninit(&mut self) -> Option<UninitState> {
        self.as_uninit()
            .is_some()
            .then(|| match self.app_state.take() {
                Some(AppState::Uninit(state)) => state,
                _ => {
                    unreachable!()
                }
            })
    }

    /// Take ownership of a running state for a state transition
    fn take_running(&mut self) -> Option<RunningState> {
        self.as_running()
            .is_some()
            .then(|| match self.app_state.take() {
                Some(AppState::Running(state)) => state,
                _ => {
                    unreachable!()
                }
            })
    }

    /// Take ownership of a suspended state for a state transition
    fn take_suspended(&mut self) -> Option<SuspendedState> {
        self.as_suspended()
            .is_some()
            .then(|| match self.app_state.take() {
                Some(AppState::Suspended(state)) => state,
                _ => {
                    unreachable!()
                }
            })
    }
}
///Vertex shader source
const VERT_SHADER_SOURCE: &str = include_str!("shader.vert");
///Fragment shader source
const FRAG_SHADER_SOURCE: &str = include_str!("shader.frag");
impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(UninitState { instance }) = self.take_uninit() {
            let win_attributes = Window::default_attributes()
                .with_resizable(false)
                .with_visible(false);
            let win = Arc::new(match event_loop.create_window(win_attributes) {
                Ok(win) => win,
                Err(e) => {
                    tracing::error!("Could not create window: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });

            let surface = match Surface::from_winit_window(&win, &instance) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!("Could not create surface: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };
            let device = match Device::create_compatible(&instance, &surface, Version::new(1, 3, 0))
            {
                Ok(d) => Arc::new(d),
                Err(e) => {
                    tracing::error!("Could not create device: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };

            let swapchain = match Swapchain::create(&device, &surface) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Could not create swapchain: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };

            let mut spv_writer = match spv::Writer::new(&spv::Options {
                lang_version: (1, 6),
                ..Default::default()
            }) {
                Ok(writer) => writer,
                Err(e) => {
                    tracing::error!("Couldn't create SPV backend {}", e);
                    event_loop.exit();
                    return;
                }
            };
            let vert_shader_mod = match naga::front::glsl::Frontend::default()
                .parse(&Options::from(ShaderStage::Vertex), VERT_SHADER_SOURCE)
            {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(
                        "Could not create vertex shader due to parsing error.\n{}",
                        e.emit_to_string("src/shader.vert")
                    );
                    event_loop.exit();
                    return;
                }
            };

            let vert_shader_mod_info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .subgroup_stages(ShaderStages::VERTEX)
            .subgroup_operations(SubgroupOperationSet::all())
            .validate(&vert_shader_mod)
            .unwrap();

            let mut vert_spv_bytes = Vec::new();
            match spv_writer.write(
                &vert_shader_mod,
                &vert_shader_mod_info,
                Some(&PipelineOptions {
                    shader_stage: ShaderStage::Vertex,
                    entry_point: "main".into(),
                }),
                &Some(spv::DebugInfo {
                    source_code: VERT_SHADER_SOURCE,
                    file_name: Path::new("src/shader.frag"),
                    language: SourceLanguage::GLSL,
                }),
                &mut vert_spv_bytes,
            ) {
                Ok(spv) => spv,
                Err(e) => {
                    tracing::error!("Couldn't compile shader: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            let frag_shader_mod = match naga::front::glsl::Frontend::default()
                .parse(&Options::from(ShaderStage::Fragment), FRAG_SHADER_SOURCE)
            {
                Ok(m) => m,
                Err(e) => {
                    tracing::error!(
                        "Could not create vertex shader due to parsing error.\n{}",
                        e.emit_to_string("src/shader.frag")
                    );
                    event_loop.exit();
                    return;
                }
            };

            let frag_shader_mod_info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .subgroup_stages(ShaderStages::FRAGMENT)
            .subgroup_operations(SubgroupOperationSet::all())
            .validate(&frag_shader_mod)
            .unwrap();

            let mut frag_spv_bytes = Vec::new();
            match spv_writer.write(
                &frag_shader_mod,
                &frag_shader_mod_info,
                Some(&PipelineOptions {
                    shader_stage: ShaderStage::Fragment,
                    entry_point: "main".into(),
                }),
                &Some(spv::DebugInfo {
                    source_code: VERT_SHADER_SOURCE,
                    file_name: Path::new("src/shader.frag"),
                    language: SourceLanguage::GLSL,
                }),
                &mut frag_spv_bytes,
            ) {
                Ok(spv) => spv,
                Err(e) => {
                    tracing::error!("Couldn't compile frag shader: {}", e);
                    event_loop.exit();
                    return;
                }
            };

            win.set_visible(true);

            self.app_state = Some(AppState::Running(RunningState {
                win,
                instance,
                _surface: surface,
                device,
                _swapchain: swapchain,
            }))
        } else if let Some(SuspendedState {
            win,
            instance,
            device,
        }) = self.take_suspended()
        {
            let surface = match Surface::from_winit_window(&win, &instance) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!("Could not recreate surface: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };
            let swapchain = match Swapchain::create(&device, &surface) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Could not recreate swapchain: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };
            self.app_state = Some(AppState::Running(RunningState {
                win,
                instance,
                _surface: surface,
                device,
                _swapchain: swapchain,
            }));
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    }

    fn suspended(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(RunningState {
            instance,
            win,
            _surface: _,
            device,
            _swapchain: _,
        }) = self.take_running()
        {
            event_loop.set_control_flow(event_loop::ControlFlow::Wait);
            self.app_state = Some(AppState::Suspended(SuspendedState {
                win,
                instance,
                device,
            }))
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Wait);
    }
    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(RunningState {
            win,
            instance: _,
            _surface: _,
            device: _,
            _swapchain: _,
        }) = self.as_running_mut()
        {
            if win.id() == window_id {
                match event {
                    WindowEvent::CloseRequested if window_id == win.id() => {
                        win.set_visible(false);
                        self.app_state = Some(AppState::Exiting);
                        event_loop.exit();
                    }
                    _ => {}
                }
            }
        }
    }
}

#[derive(clap::Parser, Debug)]
/// Argument parser from clap
struct Args {
    //TODO: Configure default to change based on build personality (e.g. release
    //vs internal opt vs debug)
    #[arg(short, long, default_value_t = VulkanDebugLevel::Warn)]
    /// What level to run validation layers at
    vulkan_debug_level: VulkanDebugLevel,
    #[arg(short, long, default_value_t=tracing::Level::WARN)]
    /// What level to run Rust's logging at
    log_level: tracing::Level,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    let app_data_dir = app_dirs2::app_root(app_dirs2::AppDataType::UserData, &APP_INFO)?;

    let log_target_file: PathBuf = [app_data_dir.clone(), "LOG.txt".into()].iter().collect();

    eprintln!(
        "Logging at {}",
        log_target_file.as_os_str().to_str().unwrap()
    );

    let (log_target, _guard) = tracing_appender::non_blocking(File::create(log_target_file)?);

    let main_layer = tracing_subscriber::fmt::layer()
        .with_line_number(true)
        .with_file(true)
        .with_thread_ids(true)
        .with_ansi(true)
        .with_target(true);
    let file_layer = tracing_subscriber::fmt::layer()
        .with_line_number(true)
        .with_file(true)
        .compact()
        .with_thread_ids(true)
        .with_ansi(false)
        .with_target(true)
        .with_writer(log_target);
    tracing_subscriber::registry()
        .with(main_layer)
        .with(file_layer)
        .with(tracing_subscriber::filter::LevelFilter::from_level(
            args.log_level,
        ))
        .init();

    tracing::warn!("Starting app");
    let event_loop = winit::event_loop::EventLoop::builder().build()?;
    let instance = Arc::new(Instance::new(
        Version::new(1, 3, 0),
        args.vulkan_debug_level,
        event_loop
            .display_handle()
            .expect("Couldn't get display handle"),
    )?);

    let mut app = AppRunner::new(UninitState { instance });
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// A simple version struct used for internal representation
struct Version {
    /// Major version
    major: u32,
    /// Minor version
    minor: u32,
    /// Patch Version
    patch: u32,
}

impl Version {
    /// Constructs a Version from a major version, minor version, and patch
    /// version
    fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    #[expect(dead_code, reason = "simply unused rn")]
    /// Constructs a Version from a Vulkan Version which looks approximately
    /// like this(Bits are numbered from 0-31 where 31 is the highest bit)
    ///
    /// Bits 31-29: Variant (tossed away)
    ///
    /// Bits 28-22: Major (7 bits)
    ///
    /// Bits 21-12: Minor (10 bits)
    ///
    /// Bits 11-0: Patch (12 bits)
    fn from_vk_version(vk_version: u32) -> Self {
        let major = ash::vk::api_version_major(vk_version);
        let minor = ash::vk::api_version_minor(vk_version);
        let patch = ash::vk::api_version_patch(vk_version);
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Converts from a Version to vulkan's internal version format as
    /// documented in `from_vk_version`
    fn to_vk_version(self) -> u32 {
        ash::vk::make_api_version(0, self.major, self.minor, self.patch)
    }
}

/// My personal wrapper around vulkan
mod rvk {

    pub(crate) use device::Device;
    pub(crate) use instance::{Instance, VulkanDebugLevel};
    pub(crate) use surface::Surface;
    pub(crate) use swapchain::Swapchain;

    /// Implement swapchain related functionality
    mod swapchain {
        use std::{fmt::Debug, sync::Arc};

        use ash::vk::{
            ColorSpaceKHR, ComponentMapping, CompositeAlphaFlagsKHR, Extent2D, Format, Image,
            ImageAspectFlags, ImageSubresourceRange, ImageUsageFlags, ImageView,
            ImageViewCreateInfo, ImageViewType, PresentModeKHR, SharingMode,
            SurfaceTransformFlagsKHR, SwapchainCreateInfoKHR, SwapchainKHR,
        };

        use super::{Device, Surface};

        /// Thing that we render to
        pub struct Swapchain {
            /// Device that we were made with
            parent_device: Arc<Device>,
            /// Surface that we were made with
            parent_surface: Arc<Surface>,
            /// Loaded extension function pointers
            swapchain_device: ash::khr::swapchain::Device,
            /// Actual handle to the surface
            swapchain: SwapchainKHR,
            /// Images associated with the swapchain
            _images: Vec<Image>,
            /// Image views associated with above images
            image_views: Vec<ImageView>,
        }
        #[derive(thiserror::Error, Debug)]

        /// Errors that pop up when we create a swapchain
        pub enum SwapchainCreateError {
            /// We don't know what went wrong but vulkan did it. Effectively the
            /// default shrug answer
            #[error("Unknown Vulkan error {0}")]
            UnknownVulkan(ash::vk::Result),
            /// You passed in a surface and device that are not derived from the
            /// same instance
            #[error("Incompatible parameters (device and surface are not from same instance)")]
            IncompatibleParameters,
            /// For some reason we can't pick a format. So we can't create a swapchain. Should basically
            /// never happen
            #[error("No compatible format found")]
            NoCompatibleFormat,
        }
        impl From<ash::vk::Result> for SwapchainCreateError {
            fn from(value: ash::vk::Result) -> Self {
                Self::UnknownVulkan(value)
            }
        }

        impl Swapchain {
            /// Create a swapchain from this device and surface. Might wanna
            /// make some optional parameters to configure? But rn this is
            /// wrapping how *I* set up a swapchain
            pub(crate) fn create(
                device: &Arc<Device>,
                surface: &Arc<Surface>,
            ) -> Result<Swapchain, SwapchainCreateError> {
                if device.get_parent().raw_handle() != surface.parent_instance().raw_handle() {
                    return Err(SwapchainCreateError::IncompatibleParameters);
                }
                let swapchain_device = ash::khr::swapchain::Device::new(
                    device.parent_instance().inner(),
                    device.inner(),
                );

                //SAFETY: surface and device are derived from the same instance
                let (surface_capabilities, surface_formats, present_modes) = unsafe {
                    (
                        surface.get_capabilities_unsafe(device)?,
                        surface.get_formats_unsafe(device)?,
                        surface.get_present_modes_unsafe(device)?,
                    )
                };

                let present_mode = present_modes
                    .iter()
                    .find(|pm| **pm == PresentModeKHR::MAILBOX)
                    .copied()
                    .unwrap_or(PresentModeKHR::FIFO);

                let surface_format = surface_formats
                    .iter()
                    .find(|format| {
                        format.color_space == ColorSpaceKHR::SRGB_NONLINEAR
                            && format.format == Format::B8G8R8A8_SRGB
                    })
                    .copied()
                    .or(surface_formats.first().copied())
                    .ok_or(SwapchainCreateError::NoCompatibleFormat)?;

                let swapchain_extent = if surface_capabilities.current_extent.width != u32::MAX {
                    surface_capabilities.current_extent
                } else {
                    let surface_extent = surface.get_surface_extent();

                    Extent2D {
                        width: surface_extent.width.clamp(
                            surface_capabilities.min_image_extent.width,
                            surface_capabilities.max_image_extent.height,
                        ),
                        height: surface_extent.height.clamp(
                            surface_capabilities.min_image_extent.height,
                            surface_capabilities.max_image_extent.height,
                        ),
                    }
                };
                let default_image_count = surface_capabilities.min_image_count + 1;
                let max_image_count = surface_capabilities.max_image_count;
                let requested_image_count =
                    if max_image_count != 0 && max_image_count < default_image_count {
                        max_image_count
                    } else {
                        default_image_count
                    };

                //TODO: Different image sharing mode?
                let ci = SwapchainCreateInfoKHR::default()
                    .present_mode(present_mode)
                    .image_color_space(surface_format.color_space)
                    .image_format(surface_format.format)
                    .image_extent(swapchain_extent)
                    .min_image_count(requested_image_count)
                    .image_array_layers(1)
                    .image_sharing_mode(SharingMode::EXCLUSIVE)
                    .image_usage(ImageUsageFlags::COLOR_ATTACHMENT)
                    .surface(surface.inner())
                    .pre_transform(SurfaceTransformFlagsKHR::IDENTITY)
                    .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE);

                //SAFETY: Valid ci
                let swapchain = unsafe { swapchain_device.create_swapchain(&ci, None) }?;

                //SAFETY: Swapchain was made from this swapchain_device
                let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }?;

                let mut image_views = Vec::with_capacity(images.len());

                for image in &images {
                    let ivci = ImageViewCreateInfo::default()
                        .image(*image)
                        .view_type(ImageViewType::TYPE_2D)
                        .components(ComponentMapping::default())
                        .format(surface_format.format)
                        .subresource_range(
                            ImageSubresourceRange::default()
                                .aspect_mask(ImageAspectFlags::COLOR)
                                .base_mip_level(0)
                                .level_count(1)
                                .base_array_layer(0)
                                .layer_count(1),
                        );

                    //SAFETY: ivci.image was created from the swapchain which is
                    //derived from device
                    let iv = unsafe { device.create_image_view_raw(&ivci) }?;
                    image_views.push(iv);
                }

                Ok(Self {
                    parent_device: device.clone(),
                    parent_surface: surface.clone(),
                    swapchain,
                    swapchain_device,
                    _images: images,
                    image_views,
                })
            }
        }

        impl Debug for Swapchain {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Swapchain")
                    .field("parent_device", &self.parent_device)
                    .field("parent_surface", &self.parent_surface)
                    .field("swapchain_device", &"{opaque}")
                    .field("swapchain", &self.swapchain)
                    .finish()
            }
        }

        impl Drop for Swapchain {
            fn drop(&mut self) {
                for iv in self.image_views.drain(..) {
                    //SAFETY: iv was created on this device
                    unsafe { self.parent_device.destroy_image_view_raw(iv) };
                }
                //SAFETY: Swapchain was made with this swapchain_device
                unsafe {
                    self.swapchain_device
                        .destroy_swapchain(self.swapchain, None)
                };
            }
        }
    }

    /// Functionality related to the instance. Exists for scoping reasons
    mod instance {
        use ash::{
            vk::{
                self, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
                DebugUtilsMessengerEXT,
            },
            Entry,
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
            /// just done fucked up in an unforseen way
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
                //which can run arbitrary initialization code. However, we are probably
                //fine since this is the vulkan lib which means it *shouldn't* fuck with
                //arbitrary memory or something stupid like that
                let entry = unsafe { ash::Entry::load() }.map_err(Error::EntryLoading)?;

                //TODO: Engine version, app name, app version
                let app_info = ash::vk::ApplicationInfo::default()
                    .api_version(minimum_vk_version.to_vk_version())
                    .engine_name(c"RPS Studio");

                let available_extensions_vec = {
                    //SAFETY: Basically always safe because of ash. Only marked unsafe
                    //because it's technically an FFI call wrapper but ash manages the
                    //memory management for us
                    unsafe { entry.enumerate_instance_extension_properties(None) }?
                };

                let available_extensions = available_extensions_vec
                    .iter()
                    .map(|ext| ext.extension_name_as_c_str().unwrap_or(c""));
                //SAFETY: Basically always safe
                let available_layers_vec = unsafe { entry.enumerate_instance_layer_properties() }?;

                let available_layers = available_layers_vec
                    .iter()
                    .map(|layer| layer.layer_name_as_c_str().unwrap_or(c""));

                let window_system_exts =
                    ash_window::enumerate_required_extensions(display_handle.as_raw())?;

                if !window_system_exts
                    .iter()
                    .cloned()
                    .map(|ptr|
                        //SAFETY: ash_window has to give us valid C Strings or the
                        //function is useless
                        unsafe { CStr::from_ptr(ptr) })
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

                //SAFETY: Valid InstanceCI being used, Instance is destroyed before
                //entry
                let instance = unsafe { entry.create_instance(&ci, None) }?;

                let debug_messenger = if let Some(debug_ci) = debug_ci.as_ref() {
                    let debug_utils_instance =
                        ash::ext::debug_utils::Instance::new(&entry, &instance);
                    //SAFETY: We follow all of vulkan's rules
                    let debug_messenger = unsafe {
                        debug_utils_instance.create_debug_utils_messenger(debug_ci, None)
                    }?;
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
                // SAFETY: Effectively made safe by ash managing the memory for us
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
                //SAFETY: Vulkan has to give us a valid callback data
                let callback_data = unsafe { *p_callback_data };

                let msg_id = callback_data.message_id_number;
                //SAFETY: Vulkan must give us a valid message name
                let msg_name = unsafe { callback_data.message_id_name_as_c_str() }.unwrap();
                //SAFETY: Vulkan must give us a valid message
                let msg_data = unsafe { callback_data.message_as_c_str() }.unwrap();
                let msg_name = msg_name.to_str().unwrap();
                let msg_data = msg_data.to_str().unwrap();
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
        pub(crate) enum VulkanDebugLevel {
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
    }

    /// Functionality related to the surface. Exists for scoping reasons
    mod surface {
        use std::{fmt::Debug, sync::Arc};

        use ash::{
            prelude::VkResult,
            vk::{self, Extent2D, PresentModeKHR, SurfaceFormatKHR, SurfaceKHR},
        };
        use thiserror::Error;
        use winit::{
            raw_window_handle::{HasDisplayHandle, HasWindowHandle},
            window::Window,
        };

        use super::{Device, Instance};

        /// Represents a VkSurfaceKHR.
        pub(crate) struct Surface {
            /// Loaded function pointers for VK_SURFACE_KHR
            surface_instance: ash::khr::surface::Instance,
            /// The underlying VkSurfaceKHR
            surface: SurfaceKHR,
            /// Reference counted pointer to the instance. Here for RAII purposes as
            /// Instance must be destroyed *after* surface
            parent_instance: Arc<Instance>,
            /// Reference counted pointer to the underlying window. Here for RAII
            /// purposes as Window must be destroyed *after* surface
            parent_window: Arc<Window>,
        }

        impl Debug for Surface {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Surface")
                    .field("surface_instance", &"...")
                    .field("surface", &self.surface)
                    .field("_parent_instance", &self.parent_instance)
                    .field("_parent_window", &self.parent_window)
                    .finish()
            }
        }

        impl Drop for Surface {
            fn drop(&mut self) {
                //SAFETY: Last use of surface, all children destroyed
                unsafe { self.surface_instance.destroy_surface(self.surface, None) };
            }
        }

        #[derive(Debug, Error)]
        /// The set of possible errors when creating a Surface
        pub enum SurfaceCreateError {
            #[error("Unknown Vulkan Error {0}")]
            /// An error that has not yet been defined
            UnknownVulkan(ash::vk::Result),
            /// Couldn't get a display handle
            #[error("Couldn't get display handle")]
            InvalidDisplayHandle,
            /// Couldn't get a window handle
            #[error("Couldn't get window handle")]
            InvalidWindowHandle,
        }

        impl From<ash::vk::Result> for SurfaceCreateError {
            fn from(value: ash::vk::Result) -> Self {
                SurfaceCreateError::UnknownVulkan(value)
            }
        }

        impl Surface {
            ///Get reference to the parent instance
            pub(super) fn parent_instance(&self) -> &Arc<Instance> {
                &self.parent_instance
            }

            ///Get a handle to the inner surface for whenever necessary
            pub(super) fn inner(&self) -> ash::vk::SurfaceKHR {
                self.surface
            }
            /// Get the extent of the surface in Vulkan terms
            pub fn get_surface_extent(&self) -> Extent2D {
                let inner_size = self.parent_window.inner_size();
                Extent2D {
                    width: inner_size.width,
                    height: inner_size.height,
                }
            }

            /// Get the formats associated with this surface on this device.
            ///
            /// # SAFETY
            ///
            /// Surface and Device are from the same instance
            pub unsafe fn get_formats_unsafe(
                &self,
                device: &Device,
            ) -> VkResult<Vec<SurfaceFormatKHR>> {
                //SAFETY: We pass the responsibility for the Device being from
                //the same instance as us to the caller
                unsafe {
                    self.surface_instance
                        .get_physical_device_surface_formats(device.get_phys_dev(), self.surface)
                }
            }

            /// Gets the present modes associated with this surface on this device
            ///
            /// # SAFETY
            ///
            /// Surface and Device are from the same instance
            pub unsafe fn get_present_modes_unsafe(
                &self,
                device: &Device,
            ) -> VkResult<Vec<PresentModeKHR>> {
                //SAFETY: We pass the responsibility for the Device being from
                //the same instance as us to the caller
                unsafe {
                    self.surface_instance
                        .get_physical_device_surface_present_modes(
                            device.get_phys_dev(),
                            self.surface,
                        )
                }
            }

            /// Checks if queue_family_index associated with the physical_device
            /// supports presenting to this Surface.
            ///
            /// # SAFETY
            ///
            /// Surface and Device are from the same instance
            pub unsafe fn _get_support_unsafe(
                &self,
                device: &Device,
                queue_family_index: u32,
            ) -> VkResult<bool> {
                //SAFETY: Surface and Device are from the same instance
                unsafe {
                    self.surface_instance.get_physical_device_surface_support(
                        device.get_phys_dev(),
                        queue_family_index,
                        self.surface,
                    )
                }
            }

            /// Get the capabilities associated with this surface on this device.
            ///
            /// # SAFETY
            ///
            /// Surface and Device are from the same instance
            pub unsafe fn get_capabilities_unsafe(
                &self,
                device: &Device,
            ) -> VkResult<vk::SurfaceCapabilitiesKHR> {
                //SAFETY: We pass the responsibility for the Device being from
                //the same instance as us to the caller
                unsafe {
                    self.surface_instance
                        .get_physical_device_surface_capabilities(
                            device.get_phys_dev(),
                            self.surface,
                        )
                }
            }
            /// Uses a winit Window in order to create a surface
            pub(crate) fn from_winit_window(
                win: &Arc<Window>,
                instance: &Arc<Instance>,
            ) -> Result<Self, SurfaceCreateError> {
                let surface_instance =
                    ash::khr::surface::Instance::new(instance.parent(), instance.inner());
                //SAFETY: Passing a valid window and display handle. Enforces
                //the parent/child relationship between the surface and the
                //instance/window by holding onto an Arc to the instance/window
                let surface = unsafe {
                    ash_window::create_surface(
                        instance.parent(),
                        instance.inner(),
                        win.display_handle()
                            .map_err(|_| SurfaceCreateError::InvalidDisplayHandle)?
                            .as_raw(),
                        win.window_handle()
                            .map_err(|_| SurfaceCreateError::InvalidWindowHandle)?
                            .as_raw(),
                        None,
                    )
                }?;

                Ok(Self {
                    surface_instance,
                    surface,
                    parent_instance: instance.clone(),
                    parent_window: win.clone(),
                })
            }
            /// Return a ref to the surface instance
            pub(crate) fn surface_instance(&self) -> &ash::khr::surface::Instance {
                &self.surface_instance
            }
        }
    }

    /// implementation related to device. Exists for scoping reasons
    mod device {
        use std::{fmt::Debug, sync::Arc};

        use ash::{
            prelude::VkResult,
            vk::{
                self, DeviceCreateInfo, DeviceQueueCreateInfo, ImageView, ImageViewCreateInfo,
                PhysicalDeviceFeatures2, PhysicalDeviceVulkan12Features,
                PhysicalDeviceVulkan13Features, QueueFlags,
            },
        };
        use thiserror::Error;

        use crate::Version;

        use super::{Instance, Surface};

        //TODO: Support separate present and graphics queue
        /// Represents a VkDevice
        pub(crate) struct Device {
            /// Parent for RAII purposes
            parent: Arc<Instance>,
            ///Underlying VkDevice
            device: ash::Device,
            /// A set of function pointers for interacting with the
            /// DebugMessenger in parent
            debug_utils_fps: Option<ash::ext::debug_utils::Device>,
            /// Handle to the physical device. Need to keep it around for some
            /// stuff
            phys_dev: vk::PhysicalDevice,
            /// Handle to the queue we will submit graphics to
            _graphics_queue: vk::Queue,
            /// Index of the graphics queue family
            _graphics_queue_index: u32,
        }

        impl Debug for Device {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("Device")
                    .field("parent", &self.parent)
                    .field("device", &self.device.handle())
                    .field(
                        "debug_utils_fps",
                        &if self.debug_utils_fps.is_some() {
                            "present"
                        } else {
                            "absent"
                        },
                    )
                    .finish()
            }
        }

        /// Errors that can pop up when creating devices
        #[derive(Error, Debug)]
        pub enum DeviceCreateError {
            #[error("Unknown vulkan error {0}")]
            /// An error from vulkan that has not yet been categorized
            UnknownVulkan(ash::vk::Result),
            #[error("Surface was not created from Instance")]
            /// You passed in a Surface that wasn't created from Instance
            InstanceSurfaceMismatch,
            #[error("No suitable device found")]
            /// When we went to scan for devices, we didn't find one that
            /// fulfilled our requirements
            NoSuitableDevice,
        }

        impl From<ash::vk::Result> for DeviceCreateError {
            fn from(value: ash::vk::Result) -> Self {
                Self::UnknownVulkan(value)
            }
        }

        #[derive(Debug)]
        #[expect(dead_code, reason = "We aren't using all of these members yet")]
        #[expect(clippy::missing_docs_in_private_items, reason = "Christ this is a lot")]
        /// Physical device that we've evaluated
        struct ScoredPhysDev<'a> {
            phys_dev: vk::PhysicalDevice,
            score: u32,
            features_11: vk::PhysicalDeviceVulkan11Features<'a>,
            ext_props: Vec<vk::ExtensionProperties>,
            features_10: vk::PhysicalDeviceFeatures,
            features_12: vk::PhysicalDeviceVulkan12Features<'a>,
            features_13: vk::PhysicalDeviceVulkan13Features<'a>,
            props_10: vk::PhysicalDeviceProperties,
            props_11: vk::PhysicalDeviceVulkan11Properties<'a>,
            props_12: vk::PhysicalDeviceVulkan12Properties<'a>,
            props_13: vk::PhysicalDeviceVulkan13Properties<'a>,
            surface_capabilities: Result<vk::SurfaceCapabilitiesKHR, vk::Result>,
            graphics_qfi: u32,
            present_modes: Result<Vec<vk::PresentModeKHR>, vk::Result>,
            formats: Vec<vk::SurfaceFormatKHR>,
        }
        impl Device {
            ///Get a reference to the parent instance
            pub(crate) fn get_parent(&self) -> &Instance {
                &self.parent
            }
            ///Creates a raw image view. Generally for internal use.
            ///
            /// SAFETY: ci.image must be associated with this device. Otherwise
            /// see [vkCreateImageView](https://registry.khronos.org/vulkan/specs/latest/man/html/vkCreateImageView.html)
            pub(crate) unsafe fn create_image_view_raw(
                &self,
                ci: &ImageViewCreateInfo,
            ) -> VkResult<ash::vk::ImageView> {
                //SAFETY: From preconditions of this unsafe function
                unsafe { self.device.create_image_view(ci, None) }
            }

            ///Destroys a raw image view. Generally for internal use.
            ///
            /// SAFETY: image_view must be from self. Otherwise see
            /// [vcDestroyImageView](https://registry.khronos.org/vulkan/specs/latest/man/html/vkDestroyImageView.html)
            pub(crate) unsafe fn destroy_image_view_raw(&self, image_view: ImageView) {
                //SAFETY: Inherited from preconditions of this unsafe function
                unsafe { self.device.destroy_image_view(image_view, None) };
            }
            ///Get the instance that was used to create this device
            pub(crate) fn parent_instance(&self) -> &Arc<Instance> {
                &self.parent
            }

            ///Get the inner ash device for passing into other functions
            pub(super) fn inner(&self) -> &ash::Device {
                &self.device
            }

            /// Create a device that is compatible with the surface
            pub(crate) fn create_compatible(
                instance: &Arc<Instance>,
                surface: &Surface,
                min_api_version: Version,
            ) -> Result<Self, DeviceCreateError> {
                use DeviceCreateError as Error;
                if !Arc::ptr_eq(instance, surface.parent_instance()) {
                    return Err(Error::InstanceSurfaceMismatch);
                }
                let phys_devs = instance.enumerate_physical_devices()?;
                let scored_phys_dev: ScoredPhysDev =
                    match phys_devs.into_iter().fold(None, |best_so_far, current| {
                        // SAFETY: current is a VkPhysicalDevice derived from
                        // instance
                        let ext_props = match unsafe {
                            instance
                                .inner()
                                .enumerate_device_extension_properties(current)
                        } {
                            Ok(exts) => exts,
                            Err(_) => {
                                return best_so_far;
                            }
                        };

                        let mut props_11 = vk::PhysicalDeviceVulkan11Properties::default();
                        let mut props_12 = vk::PhysicalDeviceVulkan12Properties::default();
                        let mut props_13 = vk::PhysicalDeviceVulkan13Properties::default();
                        let mut props = vk::PhysicalDeviceProperties2::default()
                            .push_next(&mut props_11)
                            .push_next(&mut props_12)
                            .push_next(&mut props_13);

                        // SAFETY: current is derived from instance. Valid
                        // current_dev_props
                        unsafe {
                            instance
                                .inner()
                                .get_physical_device_properties2(current, &mut props)
                        };

                        //Early return when this doesn't support our min api
                        //version
                        if props.properties.api_version < min_api_version.to_vk_version() {
                            return best_so_far;
                        }

                        let mut current_dev_mem_props =
                            vk::PhysicalDeviceMemoryProperties2::default();

                        // Safety: current is derived from instance
                        unsafe {
                            instance.inner().get_physical_device_memory_properties2(
                                current,
                                &mut current_dev_mem_props,
                            );
                        }

                        // SAFETY: Surface and current are both derived from
                        // instance
                        let surface_capabilities = unsafe {
                            surface
                                .surface_instance()
                                .get_physical_device_surface_capabilities(current, surface.inner())
                        };

                        // Safety: surface and current are derived from instance
                        let formats = unsafe {
                            surface
                                .surface_instance()
                                .get_physical_device_surface_formats(current, surface.inner())
                        }
                        .ok()?;

                        // SAFETY: current and surface are derived from the same
                        // instance
                        let present_modes = unsafe {
                            surface
                                .surface_instance()
                                .get_physical_device_surface_present_modes(current, surface.inner())
                        };

                        let mut features_11 = vk::PhysicalDeviceVulkan11Features::default();
                        let mut features_12 = vk::PhysicalDeviceVulkan12Features::default();
                        let mut features_13 = vk::PhysicalDeviceVulkan13Features::default();
                        let mut features = vk::PhysicalDeviceFeatures2::default()
                            .push_next(&mut features_11)
                            .push_next(&mut features_12)
                            .push_next(&mut features_13);

                        // SAFETY: current is derived from instance
                        unsafe {
                            instance
                                .inner()
                                .get_physical_device_features2(current, &mut features);
                        }

                        let props_10 = props.properties;
                        let props_11 = props_11;
                        let props_12 = props_12;
                        let props_13 = props_13;

                        let features_10 = features.features;
                        let features_11 = features_11;
                        let features_12 = features_12;
                        let features_13 = features_13;
                        //SAFETY: Basically handled by ash doing the memory
                        //management for us
                        let qfis = unsafe {
                            instance
                                .inner()
                                .get_physical_device_queue_family_properties(current)
                        };
                        let graphics_qfi = qfis.iter().cloned().enumerate().find(|(qfi, props)| {
                            // SAFETY: current and surface come from the same
                            // instance
                            let present_support = unsafe {
                                surface
                                    .surface_instance()
                                    .get_physical_device_surface_support(
                                        current,
                                        *qfi as u32,
                                        surface.inner(),
                                    )
                            }
                            .unwrap_or(false);
                            props.queue_flags.contains(QueueFlags::GRAPHICS) && present_support
                        });

                        let ext_names_iter = ext_props
                            .iter()
                            .map(|ext| ext.extension_name_as_c_str().unwrap_or(c""));

                        let swapchain_ext_present = ext_names_iter
                            .clone()
                            .any(|ext_name| ext_name.eq(ash::khr::swapchain::NAME));

                        //Check for mandatory features. This is currently a
                        //sample from vkguide.
                        if features_13.synchronization2 == vk::FALSE
                            || features_13.dynamic_rendering == vk::FALSE
                            || features_12.descriptor_indexing == vk::FALSE
                            || features_12.buffer_device_address == vk::FALSE
                            || !swapchain_ext_present
                            || graphics_qfi.is_none()
                        {
                            return best_so_far;
                        }
                        //SAFETY: We know that graphics_qfi.is_some is true. Established
                        //in the if statement for features
                        let graphics_qfi = unsafe { graphics_qfi.unwrap_unchecked() }.0 as u32;

                        //Preliminary scoring, not yet final
                        let device_type_score = match props_10.device_type {
                            vk::PhysicalDeviceType::DISCRETE_GPU => 4,
                            vk::PhysicalDeviceType::OTHER
                            | vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
                            vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                            vk::PhysicalDeviceType::CPU => 0,
                            _ => unreachable!(),
                        };
                        Some(ScoredPhysDev {
                            phys_dev: current,
                            score: device_type_score,
                            ext_props,
                            features_10,
                            features_11,
                            features_12,
                            features_13,
                            props_10,
                            props_11,
                            props_12,
                            props_13,
                            surface_capabilities,
                            formats,
                            present_modes,
                            graphics_qfi,
                        })
                    }) {
                        Some(spd) => spd,
                        None => {
                            return Err(Error::NoSuitableDevice);
                        }
                    };

                let enabled_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];

                let mut features_12 = PhysicalDeviceVulkan12Features::default()
                    .buffer_device_address(true)
                    .descriptor_indexing(true);

                let mut features_13 = PhysicalDeviceVulkan13Features::default()
                    .dynamic_rendering(true)
                    .synchronization2(true);

                let mut features = PhysicalDeviceFeatures2::default()
                    .push_next(&mut features_12)
                    .push_next(&mut features_13);

                let qcis = [DeviceQueueCreateInfo::default()
                    .queue_family_index(scored_phys_dev.graphics_qfi)
                    .queue_priorities(&[1.0])];

                let dci = DeviceCreateInfo::default()
                    .enabled_extension_names(&enabled_extensions)
                    .push_next(&mut features)
                    .queue_create_infos(&qcis);

                //SAFETY: Valid ci. phys_dev is derived from instance
                let device = unsafe {
                    instance
                        .inner()
                        .create_device(scored_phys_dev.phys_dev, &dci, None)
                }?;

                let debug_utils_fps = instance
                    .get_debug_messenger()
                    .is_some()
                    .then(|| ash::ext::debug_utils::Device::new(instance.inner(), &device));

                //SAFETY: graphics_qfi was passed in for one of the QCIs. Queue index is 0
                let graphics_queue =
                    unsafe { device.get_device_queue(scored_phys_dev.graphics_qfi, 0) };

                Ok(Self {
                    parent: instance.clone(),
                    device,
                    debug_utils_fps,
                    phys_dev: scored_phys_dev.phys_dev,
                    _graphics_queue: graphics_queue,
                    _graphics_queue_index: scored_phys_dev.graphics_qfi,
                })
            }

            /// Returns a handle to the associated physical device
            pub(crate) fn get_phys_dev(&self) -> vk::PhysicalDevice {
                self.phys_dev
            }
        }

        impl Drop for Device {
            fn drop(&mut self) {
                self.debug_utils_fps.take();

                //SAFETY: Last use of device. All children are dead
                unsafe { self.device.destroy_device(None) };
            }
        }
    }
}
