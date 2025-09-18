//! CURRENTLY IN CONSTRUCTION

#![warn(
    unsafe_op_in_unsafe_fn,
    clippy::undocumented_unsafe_blocks,
    missing_debug_implementations,
    clippy::allow_attributes_without_reason,
    clippy::missing_safety_doc
)]
// Uncomment to check for undocumented stuff
//#![warn(clippy::missing_docs_in_private_items)]

/// Current Top Level crate for our app. Will break up in future
use core::fmt::Debug;
use std::{
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

use app_dirs2::AppInfo;

use clap::Parser;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop,
    raw_window_handle::HasDisplayHandle, window::Window,
};

use rvk::{
    device::Device,
    instance::{Instance, VulkanDebugLevel},
    pipeline::{Pipeline, PipelineLayout, RenderPass},
    shader::{Shader, ShaderCompiler, ShaderDebugLevel, ShaderOptLevel, ShaderType},
    surface::Surface,
    swapchain::Swapchain,
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
    surface: Arc<Surface>,
    /// handle to the GPU device
    device: Arc<Device>,
    /// A swapchain we are currently using
    swapchain: Arc<Swapchain>,
    pipeline_layout: Arc<PipelineLayout>,
    pipeline: Pipeline,
    vert_shader: Arc<Shader>,
    frag_shader: Arc<Shader>,
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
    pipeline_layout: Arc<PipelineLayout>,
    vert_shader: Arc<Shader>,
    frag_shader: Arc<Shader>,
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
                .with_resizable(true)
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

            let src_dir = Path::new(file!()).parent().unwrap();
            let vert_shader_source_loc: PathBuf =
                [src_dir, Path::new("shader.vert")].iter().collect();
            let frag_shader_source_loc: PathBuf =
                [src_dir, Path::new("shader.frag")].iter().collect();

            let swapchain = Arc::new(match Swapchain::create(&device, &surface, None) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Could not create swapchain: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });

            let mut shader_compiler =
                match ShaderCompiler::new(&device, ShaderDebugLevel::Full, ShaderOptLevel::None) {
                    Ok(sc) => sc,
                    Err(e) => {
                        tracing::error!("Could not create shader compiler: Error {}", e);
                        event_loop.exit();
                        return;
                    }
                };

            let vert_shader = Arc::new(
                match shader_compiler.compile_shader(
                    VERT_SHADER_SOURCE,
                    ShaderType::Vertex,
                    Some(vert_shader_source_loc).as_deref(),
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Could not create vert shader: Error {}", e);
                        event_loop.exit();
                        return;
                    }
                },
            );

            let frag_shader = Arc::new(
                match shader_compiler.compile_shader(
                    FRAG_SHADER_SOURCE,
                    ShaderType::Fragment,
                    Some(frag_shader_source_loc).as_deref(),
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Could not create frag shader: Error {}", e);
                        event_loop.exit();
                        return;
                    }
                },
            );
            let pipeline_layout = Arc::new(match PipelineLayout::new(&device) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!("Could not create pipeline layout: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });

            let render_pass = Arc::new(match RenderPass::new(&device, &swapchain) {
                Ok(rp) => rp,
                Err(e) => {
                    tracing::error!("Could not create render pass: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });

            let pipeline = match Pipeline::new(
                &device,
                &pipeline_layout,
                &render_pass,
                &vert_shader,
                &frag_shader,
            ) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!("Could not create pipeline: Error {}", e);
                    event_loop.exit();
                    return;
                }
            };

            win.set_visible(true);

            self.app_state = Some(AppState::Running(RunningState {
                win,
                instance,
                surface,
                device,
                swapchain,
                pipeline_layout,
                pipeline,
                vert_shader,
                frag_shader,
            }))
        } else if let Some(SuspendedState {
            win,
            instance,
            device,
            pipeline_layout,
            vert_shader,
            frag_shader,
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
            let swapchain = Arc::new(match Swapchain::create(&device, &surface, None) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Could not recreate swapchain: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });
            let render_pass = Arc::new(match RenderPass::new(&device, &swapchain) {
                Ok(rp) => rp,
                Err(e) => {
                    tracing::error!("Could not recreate render pass: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });
            let pipeline = match Pipeline::new(
                &device,
                &pipeline_layout,
                &render_pass,
                &vert_shader,
                &frag_shader,
            ) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!("Could not recreate pipeline: Error{}", e);
                    event_loop.exit();
                    return;
                }
            };
            self.app_state = Some(AppState::Running(RunningState {
                pipeline_layout,
                win,
                instance,
                surface,
                device,
                swapchain,
                pipeline,
                vert_shader,
                frag_shader,
            }));
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    }

    fn suspended(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(RunningState {
            instance,
            win,
            surface: _,
            device,
            swapchain: _,
            pipeline_layout,
            pipeline: _,
            vert_shader,
            frag_shader,
        }) = self.take_running()
        {
            event_loop.set_control_flow(event_loop::ControlFlow::Wait);
            self.app_state = Some(AppState::Suspended(SuspendedState {
                win,
                instance,
                device,
                pipeline_layout,
                vert_shader,
                frag_shader,
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
            surface: _,
            device: _,
            swapchain: _,
            pipeline_layout: _,
            pipeline: _,
            vert_shader: _,
            frag_shader: _,
        }) = self.as_running_mut()
        {
            if win.id() == window_id {
                match event {
                    WindowEvent::CloseRequested if window_id == win.id() => {
                        win.set_visible(false);
                        self.app_state = Some(AppState::Exiting);
                        event_loop.exit();
                    }
                    WindowEvent::Resized(new_size) => {
                        let state = self.take_running().expect("We should be good");
                        assert!(state.win.inner_size() == new_size);
                        let new_swapchain = Arc::new(
                            match Swapchain::create(
                                &state.device,
                                &state.surface,
                                Some(&state.swapchain),
                            ) {
                                Ok(s) => s,
                                Err(e) => {
                                    tracing::error!(
                                        "Error while creating swapchain due to resize {}",
                                        e
                                    );
                                    return event_loop.exit();
                                }
                            },
                        );

                        let new_renderpass = match RenderPass::new(&state.device, &new_swapchain) {
                            Ok(r) => Arc::new(r),
                            Err(e) => {
                                tracing::error!(
                                    "Error while creating render pass due to resize {}",
                                    e
                                );
                                return event_loop.exit();
                            }
                        };
                        let new_pipeline = match Pipeline::new(
                            &state.device,
                            &state.pipeline_layout,
                            &new_renderpass,
                            &state.vert_shader,
                            &state.frag_shader,
                        ) {
                            Ok(p) => p,
                            Err(e) => {
                                tracing::error!(
                                    "Error while creating pipeline due to resize {}",
                                    e
                                );
                                return event_loop.exit();
                            }
                        };

                        self.app_state = Some(AppState::Running(RunningState {
                            win: state.win,
                            instance: state.instance,
                            surface: state.surface,
                            device: state.device,
                            swapchain: new_swapchain,
                            pipeline_layout: state.pipeline_layout,
                            pipeline: new_pipeline,
                            vert_shader: state.vert_shader,
                            frag_shader: state.frag_shader,
                        }));
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
mod rvk;
