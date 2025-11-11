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

use super::{device::Device, instance::Instance};

/// Represents a VkSurfaceKHR.
pub struct Surface {
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

impl PartialEq for Surface {
    fn eq(&self, other: &Self) -> bool {
        self.surface == other.surface
            && self.parent_instance == other.parent_instance
            && self.parent_window.id() == other.parent_window.id()
    }
}
impl Eq for Surface {}

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
    pub unsafe fn get_formats_unsafe(&self, device: &Device) -> VkResult<Vec<SurfaceFormatKHR>> {
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
                .get_physical_device_surface_present_modes(device.get_phys_dev(), self.surface)
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
                .get_physical_device_surface_capabilities(device.get_phys_dev(), self.surface)
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

    /// Uses a winit Window in order to create a surface
    pub fn from_winit_window(
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
