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
    io::{Write, stderr},
    path::{Path, PathBuf},
    sync::Arc,
};

use app_dirs2::AppInfo;

use rps_studio::Args;

use thiserror::Error;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vek::Vec4;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop,
    raw_window_handle::HasDisplayHandle, window::Window,
};

use rps_studio::rvk::{
    command_buffers::RecordedCommandBuffer,
    device::Device,
    instance::Instance,
    pipeline::{DynamicPipeline, PipelineLayout},
    shader::{Shader, ShaderCompiler, ShaderDebugLevel, ShaderOptLevel, ShaderType},
    surface::Surface,
    swapchain::Swapchain,
};

use clap::Parser;
use rps_studio::Version;
use rps_studio::rvk;
use rps_studio::rvk::shader::ShaderCompileError;

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
    swapchain: Option<Arc<Swapchain>>,
    /// Pipeline layout used for rendering, effectively a cache
    pipeline_layout: Arc<PipelineLayout>,
    /// Pipeline used for rendering
    pipeline: Option<Arc<DynamicPipeline>>,
    /// cached vertex shader
    vert_shader: Arc<Shader>,
    /// cached fragment shader
    frag_shader: Arc<Shader>,
    /// Per-frame data for rendering
    frame_data: Vec<FrameData>,
    /// Index of the current frame in `frame_data`
    current_frame_index: usize,
    /// File to log shader compiler output to
    shader_compiler_log_file: File,
}

impl SuspendedState {
    /// Consume the `SuspendedState` and attempt to produce a `RunningState`.
    ///
    /// This will recreate the surface, swapchain, pipeline (if applicable),
    /// and per-frame resources. Any failure will be returned to the caller
    /// as an `InitializeStateError` so the caller can decide how to proceed.
    fn try_into_running(self) -> Result<RunningState, InitializeStateError> {
        use InitializeStateError as Error;
        let SuspendedState {
            win,
            instance,
            device,
            pipeline_layout,
            vert_shader,
            frag_shader,
            current_frame,
            shader_compiler_log_file,
        } = self;

        let surface = Arc::new(Surface::from_winit_window(&win, &instance)?);

        let swapchain = Swapchain::create(&device, &surface, None)?.map(Arc::new);

        let pipeline = if let Some(ref sc) = swapchain {
            Some(Arc::new(
                DynamicPipeline::new(&device, &pipeline_layout, sc, &vert_shader, &frag_shader)
                    .map_err(Error::PipelineCreation)?,
            ))
        } else {
            None
        };

        const MAX_FRAMES_IN_FLIGHT: u32 = 3;
        let mut frame_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            frame_data.push(FrameData::new(&device)?);
        }

        Ok(RunningState {
            win,
            instance,
            surface,
            device,
            swapchain,
            pipeline_layout,
            pipeline,
            vert_shader,
            frag_shader,
            frame_data,
            current_frame_index: current_frame,
            shader_compiler_log_file,
        })
    }
}

#[derive(Debug)]
/// Per-frame resources for rendering
struct FrameData {
    /// Command pool for this frame
    command_pool: std::sync::Arc<rvk::command_buffers::CommandPool>,
    /// Fence to track when previous rendering operations using these objects completed
    prev_frame_fence: rvk::sync_objects::Fence,
    /// Semaphore that signals when an image has been acquired from the swapchain
    acquired_image_semaphore: rvk::sync_objects::Semaphore,
    /// Semaphore that signals when rendering is complete
    render_complete_semaphore: rvk::sync_objects::Semaphore,
    /// place to stash previous frame's resources because they can't be
    /// destroyed according to vulkan until command buffer execution is complete
    prev_frame_stash: Option<(Arc<Swapchain>, RecordedCommandBuffer, Arc<DynamicPipeline>)>,
}

impl Drop for FrameData {
    fn drop(&mut self) {
        // Ensure any GPU work that may reference resources owned by this
        // FrameData is finished before we free the command pool or semaphores.
        // Use a blocking wait with an effectively-infinite timeout. Log on
        // failure but don't panic from Drop.
        // Wait and reset the fence as a single operation. Log failures but do
        // not panic from Drop.
        if let Err(e) = self.prev_frame_fence.wait_and_reset(u64::MAX) {
            tracing::warn!(
                "FrameData drop: failed waiting/resetting prev_frame_fence: {:?}",
                e
            );
        }
    }
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
    /// cached pipeline layout for recreating our pipeline on resume
    pipeline_layout: Arc<PipelineLayout>,
    /// cached vertex shader for recreating our pipeline on resume
    vert_shader: Arc<Shader>,
    /// cached fragment shader for recreating our pipeline on resume
    frag_shader: Arc<Shader>,
    /// Preserved current frame index while suspended
    current_frame: usize,
    /// File to log shader compiler output to
    shader_compiler_log_file: File,
}

/// State of the app before initialization
#[derive(Debug)]
struct UninitState {
    /// Our vk instance (contains pre-initialization stuff safe to make before
    /// we have a window)
    instance: Arc<Instance>,
    /// File to log shader compiler output to
    shader_compiler_log_file: File,
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

/// Errors that can occur while initializing the running `State` from an
/// uninitialized application context. These represent recoverable initialization
/// failures that should be reported to the user or cause a clean shutdown.
#[derive(Debug, Error)]
enum InitializeStateError {
    /// Failed to create a native window via the windowing backend.
    ///
    /// Source: `winit::error::OsError`.
    #[error("Error creating window: {0}")]
    WindowCreation(#[from] winit::error::OsError),

    /// Failed to create a Vulkan `Surface` for the window. This usually
    /// indicates a platform surface integration issue (e.g. missing display
    /// handle or unsupported platform).
    #[error("Error creating surface: {0}")]
    SurfaceCreation(#[from] rvk::surface::SurfaceCreateError),

    /// Failed to create or select a compatible Vulkan `Device`/physical
    /// device. This may indicate missing device support or driver issues.
    #[error("Error creating device: {0}")]
    DeviceCreation(#[from] rvk::device::DeviceCreateError),

    /// Failed to create a swapchain for the surface. May be due to an
    /// unsupported format, present mode, or out-of-date surface state.
    #[error("Error creating swapchain: {0}")]
    SwapchainCreation(#[from] rvk::swapchain::SwapchainCreateError),

    /// Failed to create the shader compiler (from Naga)
    #[error("Error creating shader compiler: {0}")]
    ShaderCompilerCreation(#[from] rvk::shader::ShaderCompilerError),

    /// Compilation of a shader source failed. The inner error contains
    /// detailed parse/compile diagnostics from the shader compiler.
    #[error("Error compiling shader {0}")]
    ShaderCompilation(#[from] Box<rvk::shader::ShaderCompileError>),

    /// Failure creating the pipeline layout required for the graphics
    /// pipeline (descriptor sets, push constant ranges, etc.).
    #[error("Error creating pipeline layout {0}")]
    PipelineLayoutCreation(rvk::pipeline::PipelineLayoutCreateError),

    /// Failure creating the graphics pipeline itself (shader stages,
    /// render state, pipeline creation errors).
    #[error("Error creating pipeline {0}")]
    PipelineCreation(rvk::pipeline::PipelineCreateError),

    /// Failure creating a command pool used for per-frame command buffer
    /// allocations.
    #[error("Error creating command pool {0}")]
    CommandPoolCreation(#[from] rvk::command_buffers::CommandPoolCreateError),

    /// Failure creating a fence object used to synchronize GPU work.
    #[error("Error creating fence: {0}")]
    FenceCreation(#[from] rvk::sync_objects::FenceCreateError),

    /// Failure creating a semaphore used for GPU/GPU or CPU/GPU signaling.
    #[error("Error creating semaphore: {0}")]
    SemaphoreCreation(#[from] rvk::sync_objects::SemaphoreCreateError),

    /// Generic I/O error (file operations, path canonicalization, etc.)
    /// encountered while preparing resources for initialization.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Errors that can occur while performing a single frame redraw.
///
/// These represent errors encountered during the acquire/record/submit/present
/// flow. The caller may handle some of these (for example, treating
/// `OutOfDateSwapchain` as a signal to recreate the swapchain) while others
/// surface as fatal Vulkan errors.
#[derive(Debug, thiserror::Error)]
enum RedrawError {
    /// An unexpected Vulkan error code was returned by an operation.
    ///
    /// The contained `ash::vk::Result` provides the exact Vulkan result.
    #[error("Unknown Vulkan error: {0}")]
    UnknownVk(#[from] ash::vk::Result),

    /// The swapchain is out of date and must be recreated before presenting.
    ///
    /// This commonly occurs when the window is resized or the surface
    /// becomes incompatible with the current swapchain configuration.
    #[error("Swapchain is out of date")]
    OutOfDateSwapchain,

    /// No per-frame data was available to perform a redraw.
    ///
    /// This indicates an internal state error where `frame_data` is empty.
    #[error("No frame data available")]
    NoFrameData,

    /// Failed to acquire the next swapchain image.
    ///
    /// The wrapped `AcquireImageError` contains more specific reasons such
    /// as timeout, out-of-date, or other Vulkan errors.
    #[error("Failed to acquire next image")]
    AcquireSwapchainImage(#[from] rvk::swapchain::AcquireImageError),

    /// Failed to allocate command buffers required for recording GPU work.
    #[error("Failed to allocate command buffer: {0}")]
    AllocateCommandBuffer(#[from] rvk::command_buffers::AllocateCommandBuffersError),

    /// Failed to submit recorded command buffer(s) to the device queue.
    #[error("Failed to submit command buffer: {0}")]
    Submit(#[from] rvk::device::SubmitError),

    /// Waiting for or resetting the per-frame fence failed.
    #[error("Failed to wait and reset fence: {0}")]
    WaitAndResetFence(#[from] rvk::sync_objects::FenceWaitError),
}

impl FrameData {
    /// Create a new FrameData from a device. Converts underlying rvk errors
    /// into `InitializeStateError` via the `#[from]` variants declared above.
    fn new(device: &std::sync::Arc<Device>) -> Result<Self, InitializeStateError> {
        let command_pool = std::sync::Arc::new(rvk::command_buffers::CommandPool::new(
            device,
            device.get_graphics_queue_family_index(),
        )?);
        let prev_frame_fence = rvk::sync_objects::Fence::new(device, true)?;
        let acquired_image_semaphore = rvk::sync_objects::Semaphore::new(device)?;
        let render_complete_semaphore = rvk::sync_objects::Semaphore::new(device)?;

        Ok(FrameData {
            command_pool,
            prev_frame_fence,
            acquired_image_semaphore,
            render_complete_semaphore,
            prev_frame_stash: None,
        })
    }
}

#[allow(clippy::result_large_err, reason = "In dev")]
/// Initialize the application `RunningState` from an `UninitState`.
///
/// This performs window creation, surface/device selection, shader
/// compilation, swapchain creation, pipeline setup, and per-frame resource
/// allocation. Errors are returned as `InitializeStateError` to allow the
/// caller to report or recover cleanly.
fn initialize_state(
    uninit_state: UninitState,
    event_loop: &event_loop::ActiveEventLoop,
) -> Result<RunningState, InitializeStateError> {
    use InitializeStateError as Error;
    let instance = uninit_state.instance;
    let win_attributes = Window::default_attributes()
        .with_resizable(true)
        .with_visible(false);
    let win = Arc::new(event_loop.create_window(win_attributes)?);

    let surface = Arc::new(Surface::from_winit_window(&win, &instance)?);
    let device = Arc::new(Device::create_compatible(
        &instance,
        &surface,
        Version::new(1, 3, 0),
    )?);

    let src_dir = Path::new(file!()).parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "source file parent not found")
    })?;
    let vert_shader_source_loc: PathBuf = [src_dir, Path::new("shader.vert")].iter().collect();
    let vert_shader_source_loc = std::fs::canonicalize(&vert_shader_source_loc)?;
    let frag_shader_source_loc: PathBuf = [src_dir, Path::new("shader.frag")].iter().collect();
    let frag_shader_source_loc = std::fs::canonicalize(&frag_shader_source_loc)?;

    let swapchain = Swapchain::create(&device, &surface, None)?.map(Arc::new);

    let mut shader_compiler =
        ShaderCompiler::new(&device, ShaderDebugLevel::Full, ShaderOptLevel::None)?;

    let vert_shader = Arc::new(shader_compiler.compile_shader(
        VERT_SHADER_SOURCE,
        ShaderType::Vertex,
        Some(vert_shader_source_loc.clone()).as_deref(),
    )?);

    let frag_shader = Arc::new(shader_compiler.compile_shader(
        FRAG_SHADER_SOURCE,
        ShaderType::Fragment,
        Some(frag_shader_source_loc.clone()).as_deref(),
    )?);
    let pipeline_layout =
        Arc::new(PipelineLayout::new(&device).map_err(Error::PipelineLayoutCreation)?);

    let pipeline = if let Some(swapchain) = swapchain.as_ref() {
        Some(Arc::new(
            DynamicPipeline::new(
                &device,
                &pipeline_layout,
                swapchain,
                &vert_shader,
                &frag_shader,
            )
            .map_err(Error::PipelineCreation)?,
        ))
    } else {
        None
    };

    const MAX_FRAMES_IN_FLIGHT: u32 = 3;
    let mut frame_data = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT as usize);
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        frame_data.push(FrameData::new(&device)?);
    }
    win.set_visible(true);

    Ok(RunningState {
        win,
        instance,
        surface,
        device,
        swapchain,
        pipeline_layout,
        pipeline,
        vert_shader,
        frag_shader,
        frame_data,
        current_frame_index: 0,
        shader_compiler_log_file: uninit_state.shader_compiler_log_file,
    })
}
impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(uninit_state) = self.take_uninit() {
            let mut shader_compiler_log_file =
                match uninit_state.shader_compiler_log_file.try_clone() {
                    Ok(f) => f,
                    Err(e) => {
                        tracing::error!("failed to clone shader compiler log file: {}", e);
                        event_loop.exit();
                        return;
                    }
                };
            self.app_state = match initialize_state(uninit_state, event_loop) {
                Ok(s) => Some(AppState::Running(s)),
                Err(InitializeStateError::ShaderCompilation(e)) => {
                    match *e {
                        ShaderCompileError::Parse { e, src, file } => {
                            if let Some(p) = file.as_ref().and_then(|p| p.to_str()) {
                                tracing::error!(
                                    "Error compiling shader {}. Outputing to stderr \
                            and to shader compiler logging file",
                                    p
                                );
                                e.emit_to_writer_with_path(
                                    &mut termcolor::Ansi::new(stderr()),
                                    &src,
                                    p,
                                );
                                _ = writeln!(
                                    shader_compiler_log_file,
                                    "Error compiling shader {} at time {}:",
                                    p,
                                    chrono::Local::now()
                                );
                                e.emit_to_writer_with_path(
                                    &mut termcolor::NoColor::new(shader_compiler_log_file),
                                    &src,
                                    p,
                                );
                            } else {
                                tracing::error!(
                                    "Error compiling unknown shader. Outputing to stderr \
                            and to shader compiler logging file"
                                );
                                e.emit_to_writer(&mut termcolor::Ansi::new(stderr()), &src);
                                _ = writeln!(
                                    shader_compiler_log_file,
                                    "Error compiling unknown shader at time {}. Printing Source",
                                    chrono::Local::now()
                                );
                                _ = writeln!(shader_compiler_log_file, "{}", &src);
                                let _ = writeln!(
                                    shader_compiler_log_file,
                                    "\nNow printing \
                             errors:"
                                );
                                e.emit_to_writer(
                                    &mut termcolor::NoColor::new(shader_compiler_log_file),
                                    &src,
                                );
                            }
                        }
                        _ => {
                            tracing::error!("Error when compiling shader: {}", e)
                        }
                    }
                    return event_loop.exit();
                }
                Err(e) => {
                    tracing::error!("{}", e);
                    return event_loop.exit();
                }
            }
        } else if let Some(suspended_state) = self.take_suspended() {
            match suspended_state.try_into_running() {
                Ok(running) => {
                    self.app_state = Some(AppState::Running(running));
                }
                Err(e) => {
                    tracing::error!("Failed to resume running state: {}", e);
                    event_loop.exit();
                    return;
                }
            }
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    }

    fn suspended(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(running_state) = self.take_running() {
            // Convert to a suspended state; `into_suspended` will attempt to
            // wait for the device to be idle and log any failures.
            let suspended = running_state.into_suspended();
            event_loop.set_control_flow(event_loop::ControlFlow::Wait);
            self.app_state = Some(AppState::Suspended(suspended));
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Wait);
    }
    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(RunningState { win, .. }) = self.as_running_mut()
            && win.id() == window_id
        {
            match event {
                WindowEvent::CloseRequested if window_id == win.id() => {
                    win.set_visible(false);
                    // If we are currently running, ensure the device is idle before exiting.
                    if let Some(running_state) = self.take_running() {
                        #[allow(
                            clippy::collapsible_if,
                            reason = "These are logically seperate steps, not a single check"
                        )]
                        if let Err(e) = running_state.device.wait_idle() {
                            tracing::warn!("device.wait_idle() failed during exit: {:?}", e);
                        }
                    }
                    self.app_state = Some(AppState::Exiting);
                    event_loop.exit();
                }
                WindowEvent::Resized(new_size) => {
                    let mut state = match self.take_running() {
                        Some(s) => s,
                        None => {
                            tracing::error!(
                                "state transition failed: expected Running state but it \
                                    was not present"
                            );
                            return;
                        }
                    };
                    assert!(state.win.inner_size() == new_size);
                    let new_swapchain = if let Some(swapchain) = state.swapchain.as_ref() {
                        match Swapchain::create(&state.device, &state.surface, Some(swapchain)) {
                            Ok(s) => s.map(Arc::new),
                            Err(e) => {
                                tracing::error!(
                                    "Error while creating swapchain due to resize {}",
                                    e
                                );
                                return event_loop.exit();
                            }
                        }
                    } else {
                        let _ = state.device.wait_idle();
                        for frame in &mut state.frame_data {
                            frame.prev_frame_stash = None;
                        }
                        match Swapchain::create(&state.device, &state.surface, None) {
                            Ok(s) => s.map(Arc::new),
                            Err(e) => {
                                tracing::error!(
                                    "Error while creating swapchain due to resize: {}",
                                    e
                                );
                                return event_loop.exit();
                            }
                        }
                    };
                    let new_pipeline = if let Some(new_swapchain) = new_swapchain.as_ref() {
                        Some(Arc::new(
                            match DynamicPipeline::new(
                                &state.device,
                                &state.pipeline_layout,
                                new_swapchain,
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
                            },
                        ))
                    } else {
                        None
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
                        frame_data: state.frame_data,
                        current_frame_index: state.current_frame_index,
                        shader_compiler_log_file: state.shader_compiler_log_file,
                    }));
                }

                WindowEvent::RedrawRequested => {
                    if let Some(state) = self.as_running_mut() {
                        match state.redraw() {
                            Ok(DrawStatus { suboptimal }) => {
                                if suboptimal {
                                    tracing::warn!(
                                        "Redraw completed but swapchain marked SUBOPTIMAL"
                                    );
                                }
                                state.win.request_redraw();
                            }
                            Err(e) => {
                                tracing::error!("Error during redraw: {}", e);
                                event_loop.exit();
                            }
                        }
                    }
                }

                _ => {}
            }
        }
    }
}

/// Result information from a successful `redraw` invocation.
///
/// `suboptimal` indicates whether the swapchain or presentation reported a
/// SUBOPTIMAL condition (a hint that the swapchain should be recreated
/// soon, though not strictly required for correctness).
struct DrawStatus {
    /// True if the swapchain or present operation was marked SUBOPTIMAL.
    suboptimal: bool,
}

impl RunningState {
    /// Redraw the current frame. Returns whether the swapchain is suboptimal
    pub fn redraw(&mut self) -> Result<DrawStatus, RedrawError> {
        if self.frame_data.is_empty() {
            //Should be an unecessary fallback path but just in case
            tracing::warn!("redraw called but no frame data is available");
            self.current_frame_index = 0;
            Err(RedrawError::NoFrameData)
        } else if let Some(swapchain) = &self.swapchain
            && let Some(pipeline) = &self.pipeline
        {
            let current_frame_index = self.current_frame_index;
            let next_frame_index = (current_frame_index + 1) % self.frame_data.len();
            let frame = &mut self.frame_data[current_frame_index];
            self.current_frame_index = next_frame_index;

            frame.prev_frame_fence.wait_and_reset(u64::MAX)?;
            frame.prev_frame_stash = None;

            // Acquire the next image from the swapchain
            let (image_index, suboptimal) =
                match swapchain.acquire_next_image(u64::MAX, &frame.acquired_image_semaphore, None)
                {
                    Ok((image_index, suboptimal)) => (image_index, suboptimal),
                    Err(rvk::swapchain::AcquireImageError::OutOfDate) => {
                        return Err(RedrawError::OutOfDateSwapchain);
                    }
                    Err(rvk::swapchain::AcquireImageError::Timeout) => {
                        return Err(RedrawError::UnknownVk(ash::vk::Result::TIMEOUT));
                    }
                    Err(rvk::swapchain::AcquireImageError::UnknownVulkan(e)) => {
                        return Err(RedrawError::UnknownVk(e));
                    }
                };

            let cb = frame.command_pool.allocate_command_buffer_primary()?;

            let mut cb_recorder = cb.begin_one_time()?;
            cb_recorder.transition_swapchain_image_for_rendering(swapchain, image_index);

            let renderer = cb_recorder.begin_rendering_clear_color(
                swapchain,
                image_index,
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            );

            renderer.set_scissor(0, 0, swapchain.width(), swapchain.height());
            renderer.set_viewport(
                0.0,
                0.0,
                swapchain.width() as f32,
                swapchain.height() as f32,
                0.0,
                1.0,
            );
            renderer.bind_pipeline(pipeline);
            renderer.draw(3, 1, 0, 0);
            drop(renderer);

            cb_recorder.transition_swapchain_image_for_presenting(swapchain, image_index);

            let cb_recorded = cb_recorder.end()?;

            // Submit the command buffer: wait on the image-acquired semaphore,
            // signal the render-complete semaphore, and use the per-frame
            // fence to be notified when the GPU work finishes.
            self.device.submit_recorded_command_buffer(
                &cb_recorded,
                &[&frame.acquired_image_semaphore],
                &[&frame.render_complete_semaphore],
                Some(&frame.prev_frame_fence),
            )?;
            frame.prev_frame_stash = Some((swapchain.clone(), cb_recorded, pipeline.clone()));

            // Present the rendered image. Treat PRESENT_OUT_OF_DATE as a
            // request to recreate the swapchain.
            match self.device.present_swapchain(
                swapchain.as_ref(),
                &[&frame.render_complete_semaphore],
                image_index,
            ) {
                Ok(present_suboptimal) => Ok(DrawStatus {
                    suboptimal: suboptimal || present_suboptimal,
                }),
                Err(rvk::swapchain::AcquireImageError::OutOfDate) => {
                    Err(RedrawError::OutOfDateSwapchain)
                }
                Err(e) => Err(RedrawError::AcquireSwapchainImage(e)),
            }
        } else {
            Ok(DrawStatus { suboptimal: false })
        }
    }

    /// Convert this `RunningState` into a `SuspendedState`, attempting to
    /// wait for the device to become idle first. Any failures while waiting
    /// are logged but do not prevent the transition.
    fn into_suspended(self) -> SuspendedState {
        if let Err(e) = self.device.wait_idle() {
            tracing::warn!("device.wait_idle() failed during suspend: {:?}", e);
        }
        SuspendedState {
            win: self.win,
            instance: self.instance,
            device: self.device,
            pipeline_layout: self.pipeline_layout,
            vert_shader: self.vert_shader,
            frag_shader: self.frag_shader,
            current_frame: self.current_frame_index,
            shader_compiler_log_file: self.shader_compiler_log_file,
        }
    }
}

// `Args` moved into the library crate (`src/lib.rs`) and re-exported as
// `rps_studio::Args` so tests and the binary can share the definition.

fn main() -> eyre::Result<()> {
    let args = Args::parse();

    let app_data_dir = app_dirs2::app_root(app_dirs2::AppDataType::UserData, &APP_INFO)?;

    let log_target_file: PathBuf = [app_data_dir.clone(), "LOG.txt".into()].iter().collect();
    let shader_compiler_error_log_file_path: PathBuf =
        [app_data_dir.clone(), "SHADER_COMPILER_ERROR.txt".into()]
            .iter()
            .collect();
    let shader_compiler_log_file = File::create(shader_compiler_error_log_file_path)?;
    eprintln!(
        "Logging at {}",
        log_target_file.as_os_str().to_string_lossy()
    );

    let (log_target, _guard) = tracing_appender::non_blocking(File::create(log_target_file)?);

    let term_layer = tracing_subscriber::fmt::layer()
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
        .with(term_layer)
        .with(file_layer)
        .with(tracing_subscriber::filter::LevelFilter::from_level(
            args.log_level,
        ))
        .init();

    tracing::warn!("Starting app");
    let event_loop = winit::event_loop::EventLoop::builder().build()?;
    let display_handle = event_loop
        .display_handle()
        .map_err(|e| eyre::eyre!("Couldn't get display handle: {e}"))?;

    let instance = Arc::new(Instance::new(
        Version::new(1, 3, 0),
        args.vulkan_debug_level,
        display_handle,
    )?);

    let mut app = AppRunner::new(UninitState {
        instance,
        shader_compiler_log_file,
    });
    event_loop.run_app(&mut app)?;
    Ok(())
}

// `rvk` is provided by the library crate `rps_studio` (see `src/lib.rs`)
