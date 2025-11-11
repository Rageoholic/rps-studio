use std::{collections::HashMap, fmt::Debug, sync::Arc};

use ash::vk::{
    self, DeviceCreateInfo, DeviceQueueCreateInfo, PhysicalDeviceFeatures2,
    PhysicalDeviceVulkan12Features, PhysicalDeviceVulkan13Features, QueueFlags,
};
use thiserror::Error;

use crate::Version;

use super::{instance::Instance, surface::Surface};
use crate::rvk::command_buffers::RecordedCommandBuffer;
use crate::rvk::sync_objects::{Fence, Semaphore};

//TODO: Support separate present and graphics queue
/// Represents a VkDevice
pub struct Device {
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
    /// Map of queue family index -> mutex-wrapped queue handle.
    /// The HashMap is built once in a single-threaded context; each queue
    /// value is individually protected by a Mutex to allow concurrent
    /// submissions to different queues without a global lock.
    queues: HashMap<u32, std::sync::Mutex<vk::Queue>>,
    /// Index of the graphics queue family (used as the primary graphics key)
    graphics_queue_family_index: u32,
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        let parent_eq = self.parent == other.parent;
        let handle_eq = self.device.handle() == other.device.handle();
        let phys_dev_eq = self.phys_dev == other.phys_dev;

        let gqi_eq = self.graphics_queue_family_index == other.graphics_queue_family_index;
        parent_eq && handle_eq && phys_dev_eq && gqi_eq
    }
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
    pub fn get_graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }
    // ...existing code...
    ///Get a reference to the parent instance
    pub(super) fn get_parent(&self) -> &Instance {
        &self.parent
    }

    ///Get the instance that was used to create this device
    pub(super) fn parent_instance(&self) -> &Arc<Instance> {
        &self.parent
    }

    ///Get the inner ash device for passing into other functions
    pub(super) fn inner(&self) -> &ash::Device {
        &self.device
    }

    /// Wait for the device to be idle. This wraps `vkDeviceWaitIdle`.
    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        // SAFETY: waiting for device idle is a valid operation on a live device
        unsafe { self.device.device_wait_idle() }
    }

    /// Create a device that is compatible with the surface
    pub fn create_compatible(
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

                // SAFETY: current is a VkPhysicalDevice derived from instance. Valid
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

                let mut current_dev_mem_props = vk::PhysicalDeviceMemoryProperties2::default();

                // Safety: current is derived from instance
                unsafe {
                    instance.inner().get_physical_device_memory_properties2(
                        current,
                        &mut current_dev_mem_props,
                    );
                }

                // SAFETY: Surface and current are both derived from the same
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
                    .unwrap_or_else(|e| {
                        tracing::warn!("get_physical_device_surface_support failed: {:#?}", e);
                        false
                    });
                    props.queue_flags.contains(QueueFlags::GRAPHICS) && present_support
                });

                let ext_names_iter = ext_props.iter().map(|ext| {
                    ext.extension_name_as_c_str().unwrap_or_else(|err| {
                        tracing::warn!(
                            "physical device extension name invalid; using empty CStr: {:#?}",
                            err
                        );
                        c""
                    })
                });

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
                    vk::PhysicalDeviceType::OTHER | vk::PhysicalDeviceType::INTEGRATED_GPU => 3,
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
        let graphics_queue = unsafe { device.get_device_queue(scored_phys_dev.graphics_qfi, 0) };

        // Build the initial queue map. We currently only populate the
        // graphics queue family index -> queue entry, but this supports
        // adding separate present/transfer queues later.
        let mut queues_map: HashMap<u32, std::sync::Mutex<vk::Queue>> = HashMap::new();
        queues_map.insert(
            scored_phys_dev.graphics_qfi,
            std::sync::Mutex::new(graphics_queue),
        );

        Ok(Self {
            parent: instance.clone(),
            device,
            debug_utils_fps,
            phys_dev: scored_phys_dev.phys_dev,
            queues: queues_map,
            graphics_queue_family_index: scored_phys_dev.graphics_qfi,
        })
    }

    /// Return the raw graphics queue handle by locking the internal mutex.
    /// This returns a copy of the `vk::Queue` handle (which is `Copy`).
    pub fn graphics_queue_handle(&self) -> vk::Queue {
        if let Some(mtx) = self.queues.get(&self.graphics_queue_family_index) {
            let guard = mtx.lock().unwrap_or_else(|e| {
                tracing::warn!(
                    "graphics queue mutex poisoned for family {}; recovering: {:#?}",
                    self.graphics_queue_family_index,
                    e
                );
                e.into_inner()
            });
            *guard
        } else {
            tracing::warn!(
                "graphics queue not found for family {}; falling back to first available queue",
                self.graphics_queue_family_index
            );
            // Fall back to the first available queue in the map.
            if let Some(first_mtx) = self.queues.values().next() {
                let guard = first_mtx.lock().unwrap_or_else(|e| {
                    tracing::warn!("queue mutex poisoned during fallback recovery: {:#?}", e);
                    e.into_inner()
                });
                *guard
            } else {
                vk::Queue::null()
            }
        }
    }

    /// Submit to the device's graphics queue. `submits` are forwarded to
    /// `vkQueueSubmit` while holding the mutex to ensure the queue handle
    /// remains stable for the call.
    pub fn submit_to_graphics_queue(
        &self,
        submits: &[vk::SubmitInfo],
        fence: vk::Fence,
    ) -> Result<(), vk::Result> {
        let queue = self.graphics_queue_handle();
        // SAFETY: queue was retrieved from this device and `submits` were
        // constructed for this device.
        unsafe { self.device.queue_submit(queue, submits, fence) }
    }

    /// Submit a recorded command buffer to the device's graphics queue.
    /// `wait_semaphores` and `signal_semaphores` are arrays of rvk semaphores.
    /// The recorded command buffer is consumed by this call; the provided
    /// `fence` will be signaled on completion if present.
    /// Submit a recorded command buffer to the device's graphics queue.
    /// This function borrows the RecordedCommandBuffer; it does not take
    /// ownership. The caller remains responsible for keeping the recorded
    /// buffer alive until it is safe to free (typically after waiting on the
    /// provided fence).
    pub fn submit_recorded_command_buffer(
        &self,
        recorded: &RecordedCommandBuffer,
        wait_semaphores: &[&Semaphore],
        signal_semaphores: &[&Semaphore],
        fence: Option<&Fence>,
    ) -> Result<(), SubmitError> {
        let raw_waits: Vec<vk::Semaphore> = wait_semaphores.iter().map(|s| s.inner()).collect();
        let raw_signals: Vec<vk::Semaphore> = signal_semaphores.iter().map(|s| s.inner()).collect();

        let wait_stages: Vec<vk::PipelineStageFlags> =
            vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT; raw_waits.len()];

        let cmd_buf = [recorded.raw()];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&raw_waits)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmd_buf)
            .signal_semaphores(&raw_signals);

        let fence_handle = fence.map(|f| f.inner()).unwrap_or(vk::Fence::null());

        self.submit_to_graphics_queue(&[submit_info], fence_handle)
            .map_err(SubmitError::from)
    }

    /// Present a swapchain image on the device's graphics queue. `wait_semaphores`
    /// will be waited on before presenting. Returns `Ok(suboptimal)` where
    /// `suboptimal` indicates the swapchain is suboptimal for presentation.
    pub fn present_swapchain(
        &self,
        swapchain: &crate::rvk::swapchain::Swapchain,
        wait_semaphores: &[&Semaphore],
        image_index: u32,
    ) -> Result<bool, crate::rvk::swapchain::AcquireImageError> {
        let raw_waits: Vec<vk::Semaphore> = wait_semaphores.iter().map(|s| s.inner()).collect();
        swapchain.present(self, &raw_waits, image_index)
    }

    /// Returns a handle to the associated physical device
    pub(crate) fn get_phys_dev(&self) -> vk::PhysicalDevice {
        self.phys_dev
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SubmitError {
    #[error("Vulkan submit error: {0}")]
    Vk(ash::vk::Result),
}

impl From<ash::vk::Result> for SubmitError {
    fn from(value: ash::vk::Result) -> Self {
        SubmitError::Vk(value)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.debug_utils_fps.take();

        //SAFETY: Last use of device. All children are dead
        unsafe { self.device.destroy_device(None) };
    }
}
