use std::{fs::File, sync::Arc};

use app_dirs2::AppInfo;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop, window::Window};

const APP_INFO: AppInfo = AppInfo {
    name: "rps-studio",
    author: "Rageoholic",
};

struct RunningState {
    win: Arc<Window>,
}
struct SuspendedState {
    win: Arc<Window>,
}
struct UninitState {
    #[expect(dead_code)]
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
        if let Some(_us) = self.take_uninit() {
            let win_attributes = Window::default_attributes().with_resizable(false);
            let win = Arc::new(match event_loop.create_window(win_attributes) {
                Ok(win) => win,
                Err(e) => {
                    tracing::error!("Could not create window: Error {}", e);
                    event_loop.exit();
                    return;
                }
            });

            self.app_state = Some(AppState::Running(RunningState { win }))
        }
        if let Some(ss) = self.take_suspended() {
            self.app_state = Some(AppState::Running(RunningState { win: ss.win }));
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Poll);
    }

    fn suspended(&mut self, event_loop: &event_loop::ActiveEventLoop) {
        if let Some(rs) = self.take_running() {
            self.app_state = Some(AppState::Suspended(SuspendedState { win: rs.win }))
        }

        event_loop.set_control_flow(event_loop::ControlFlow::Wait);
    }
    fn window_event(
        &mut self,
        event_loop: &event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(RunningState { win }) = self.as_running_mut() {
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

fn main() -> eyre::Result<()> {
    println!("Hello, world!");

    let app_data_dir = app_dirs2::app_root(app_dirs2::AppDataType::UserData, &APP_INFO)?;

    let mut log_target_file = app_data_dir.clone();
    log_target_file.push("LOG.txt");

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
        .with(tracing_subscriber::filter::LevelFilter::DEBUG)
        .init();

    tracing::error!("Test");

    let instance = Arc::new(Instance::new(Version::new(1, 3, 0))?);

    let event_loop = winit::event_loop::EventLoop::builder().build()?;
    let mut app = AppRunner::new(UninitState { instance });
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct Instance {
    _entry: ash::Entry,
    instance: ash::Instance,
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
    fn new(minimum_vk_version: Version) -> Result<Self, InstanceCreateError> {
        //SAFETY: Somewhat inherently unsafe due to loading a shared library which can run arbitrary initialization code. However, we are probably fine since
        let entry = unsafe { ash::Entry::load() }.map_err(InstanceCreateError::EntryLoading)?;

        //TODO: Engine version, app name, app version
        let app_info = ash::vk::ApplicationInfo::default()
            .api_version(minimum_vk_version.to_vk_version())
            .engine_name(c"RPS Studio");

        let enabled_extension_names = Vec::with_capacity(8);
        let enabled_layer_names = Vec::with_capacity(1);

        let ci = ash::vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&enabled_extension_names)
            .enabled_layer_names(&enabled_layer_names);

        //SAFETY: Valid InstanceCI being used, Instance is destroyed before
        //entry
        let instance = unsafe { entry.create_instance(&ci, None) }?;

        Ok(Self {
            _entry: entry,
            instance,
        })
    }
}
