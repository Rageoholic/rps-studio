use std::{
    ffi::{c_void, CStr},
    fmt::Debug,
    fs::File,
    path::PathBuf,
    sync::Arc,
};

use app_dirs2::AppInfo;

use ash::vk::{
    Bool32, DebugUtilsMessageSeverityFlagsEXT, DebugUtilsMessageTypeFlagsEXT,
    DebugUtilsMessengerCallbackDataEXT, DebugUtilsMessengerEXT, SurfaceKHR,
};
use clap::Parser;
use thiserror::Error;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop,
    raw_window_handle::{DisplayHandle, HasDisplayHandle, HasWindowHandle},
    window::Window,
};

const APP_INFO: AppInfo = AppInfo {
    name: "rps-studio",
    author: "Rageoholic",
};

struct RunningState {
    win: Arc<Window>,
    instance: Arc<Instance>,
    _surface: Arc<Surface>,
}
struct SuspendedState {
    win: Arc<Window>,
    instance: Arc<Instance>,
}
struct UninitState {
    instance: Arc<Instance>,
}

enum AppState {
    Uninit(UninitState),
    Running(RunningState),
    Suspended(SuspendedState),
    Exiting,
}

struct AppRunner {
    app_state: Option<AppState>,
}

#[allow(dead_code)]
impl AppRunner {
    fn new(us: UninitState) -> Self {
        Self {
            app_state: Some(AppState::Uninit(us)),
        }
    }
    fn as_uninit(&self) -> Option<&UninitState> {
        if let Some(AppState::Uninit(ref state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    fn as_running(&self) -> Option<&RunningState> {
        if let Some(AppState::Running(ref state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    fn as_suspended(&self) -> Option<&SuspendedState> {
        if let Some(AppState::Suspended(ref state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    fn as_uninit_mut(&mut self) -> Option<&mut UninitState> {
        if let Some(AppState::Uninit(ref mut state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    fn as_running_mut(&mut self) -> Option<&mut RunningState> {
        if let Some(AppState::Running(ref mut state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

    fn as_suspended_mut(&mut self) -> Option<&mut SuspendedState> {
        if let Some(AppState::Suspended(ref mut state)) = self.app_state {
            Some(state)
        } else {
            None
        }
    }

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

impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(UninitState { instance }) = self.take_uninit() {
            let win_attributes = Window::default_attributes().with_resizable(false);
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

            self.app_state = Some(AppState::Running(RunningState {
                win,
                instance,
                _surface: surface,
            }))
        }
        if let Some(SuspendedState { win, instance }) = self.take_suspended() {
            let surface = match Surface::from_winit_window(&win, &instance) {
                Ok(s) => Arc::new(s),
                Err(e) => {
                    tracing::error!("Could not recreate surface: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };
            self.app_state = Some(AppState::Running(RunningState {
                win,
                instance,
                _surface: surface,
            }));
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    }

    fn suspended(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(RunningState {
            instance,
            win,
            _surface: _,
        }) = self.take_running()
        {
            self.app_state = Some(AppState::Suspended(SuspendedState { win, instance }))
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
        }) = self.as_running_mut()
        {
            match event {
                WindowEvent::CloseRequested if window_id == win.id() => {
                    self.app_state = Some(AppState::Exiting);
                    event_loop.exit();
                }
                _ => {}
            }
        }
    }
}

#[derive(clap::Parser, Debug)]
struct Args {
    //TODO: Configure default to change based on build personality (e.g. release vs internal opt vs debug)
    #[arg(short, long, default_value_t = VulkanDebugLevel::Warn)]
    vulkan_debug_level: VulkanDebugLevel,
    #[arg(short, long, default_value_t=tracing::Level::WARN)]
    log_level: tracing::Level,
}

fn main() -> eyre::Result<()> {
    let args = Args::parse();
    println!("Hello, world!");

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

struct DebugMessenger {
    debug_messenger: DebugUtilsMessengerEXT,
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
enum VulkanDebugLevel {
    #[default]
    None,
    Verbose,
    Info,
    Warn,
    Error,
}
impl VulkanDebugLevel {
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

struct Instance {
    entry: ash::Entry,
    instance: ash::Instance,
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
        unsafe { self.destroy_instance(None) };
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

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

#[derive(Debug, thiserror::Error)]
enum InstanceCreateError {
    #[error("Unspecified vk error: {0}")]
    UnspecifiedVulkan(ash::vk::Result),
    #[error("Error loading vulkan: {0}")]
    EntryLoading(ash::LoadingError),
    #[error("Missing necessary window extensions")]
    MissingWindowingExtensions,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

impl Version {
    fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    #[expect(dead_code)]
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

    fn to_vk_version(self) -> u32 {
        ash::vk::make_api_version(0, self.major, self.minor, self.patch)
    }
}

impl From<ash::vk::Result> for InstanceCreateError {
    fn from(value: ash::vk::Result) -> Self {
        Self::UnspecifiedVulkan(value)
    }
}

impl Instance {
    fn new(
        minimum_vk_version: Version,
        debug_level: VulkanDebugLevel,
        display_handle: DisplayHandle,
    ) -> Result<Self, InstanceCreateError> {
        use InstanceCreateError as Error;
        //SAFETY: Somewhat inherently unsafe due to loading a shared library which can run arbitrary initialization code. However, we are probably fine since
        let entry = unsafe { ash::Entry::load() }.map_err(Error::EntryLoading)?;

        //TODO: Engine version, app name, app version
        let app_info = ash::vk::ApplicationInfo::default()
            .api_version(minimum_vk_version.to_vk_version())
            .engine_name(c"RPS Studio");

        //SAFETY: Basically always safe
        let available_extensions_vec =
            unsafe { entry.enumerate_instance_extension_properties(None) }
                .expect("Couldn't fetch extensions");

        let available_extensions = available_extensions_vec.iter().map(|ext| {
            ext.extension_name_as_c_str()
                .expect("Vulkan extensions should be able to be converted to C Strings")
        });
        //SAFETY: Basically always safe
        let available_layers_vec = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("Couldn't fetch extensions");

        let available_layers = available_layers_vec.iter().map(|layer| {
            layer
                .layer_name_as_c_str()
                .expect("Vulkan layers should be abe to be converted to C Strings")
        });

        let window_system_exts = ash_window::enumerate_required_extensions(display_handle.as_raw())
            .expect("Unable to get needed window system extensions from the Display Handle?");

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
                    DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING
                        | DebugUtilsMessageTypeFlagsEXT::GENERAL
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

        let debug_messenger = if debug_enabled {
            let debug_utils_instance = ash::ext::debug_utils::Instance::new(&entry, &instance);
            //SAFETY: We follow all of vulkan's rules
            let debug_messenger = unsafe {
                debug_utils_instance.create_debug_utils_messenger(
                    debug_ci
                        .as_ref()
                        .expect("If debug_enabled is true, we always fill out the ci"),
                    None,
                )
            }
            .expect("How did we fail to build the debug messenger?");
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
}

unsafe extern "system" fn instance_debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_types: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT<'_>,
    _: *mut c_void,
) -> Bool32 {
    //SAFETY: Vulkan has to give us a valid callback data
    let callback_data = unsafe { *p_callback_data };

    let msg_id = callback_data.message_id_number;
    //SAFETY: Vulkan must give us a valid message name
    let msg_name = unsafe { callback_data.message_id_name_as_c_str() };
    //SAFETY: Vulkan must give us a valid message
    let msg_data = unsafe { callback_data.message_as_c_str() };
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
        tracing::error!("{ty_name}: {msg_id} {msg_name:?}: {msg_data:?}");
    } else if message_severity.contains(SevFlags::WARNING) {
        tracing::warn!("{ty_name}: {msg_id} {msg_name:?}: {msg_data:?}")
    } else if message_severity.contains(SevFlags::INFO) {
        tracing::info!("{ty_name}: {msg_id} {msg_name:?}: {msg_data:?}")
    } else if message_severity.contains(SevFlags::VERBOSE) {
        tracing::trace!("{ty_name}: {msg_id} {msg_name:?}: {msg_data:?}")
    }

    ash::vk::FALSE
}

struct Surface {
    surface_instance: ash::khr::surface::Instance,
    surface: SurfaceKHR,
    _parent_instance: Arc<Instance>,
    _parent_window: Arc<Window>,
}

impl Drop for Surface {
    fn drop(&mut self) {
        //SAFETY: Last use of surface, all children destroyed
        unsafe { self.surface_instance.destroy_surface(self.surface, None) };
    }
}

#[derive(Debug, Error)]
enum SurfaceCreateError {
    #[error("Unknown Vulkan Error {0}")]
    UnknownVulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for SurfaceCreateError {
    fn from(value: ash::vk::Result) -> Self {
        SurfaceCreateError::UnknownVulkan(value)
    }
}

impl Surface {
    fn from_winit_window(
        win: &Arc<Window>,
        instance: &Arc<Instance>,
    ) -> Result<Self, SurfaceCreateError> {
        let surface_instance = ash::khr::surface::Instance::new(&instance.entry, instance);
        let surface = unsafe {
            ash_window::create_surface(
                &instance.entry,
                instance,
                win.display_handle().unwrap().as_raw(),
                win.window_handle().unwrap().as_raw(),
                None,
            )
        }?;

        Ok(Self {
            surface_instance,
            surface,
            _parent_instance: instance.clone(),
            _parent_window: win.clone(),
        })
    }
}
