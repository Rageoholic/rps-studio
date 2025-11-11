use std::sync::Arc;

use ash::vk;
use thiserror::Error;

use crate::rvk::device::Device;

/// RAII wrapper around a Vulkan Fence
#[derive(Debug)]
pub struct Fence {
    parent_device: Arc<Device>,
    inner: vk::Fence,
}

#[derive(Debug, Error)]
pub enum FenceCreateError {
    #[error("Unknown Vulkan error: {0}")]
    UnknownVulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for FenceCreateError {
    fn from(value: ash::vk::Result) -> Self {
        Self::UnknownVulkan(value)
    }
}

#[derive(Debug, Error)]
pub enum FenceWaitError {
    #[error("Timeout waiting for fence")]
    Timeout,
    #[error("Unknown Vulkan error: {0}")]
    UnknownVulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for FenceWaitError {
    fn from(value: ash::vk::Result) -> Self {
        Self::UnknownVulkan(value)
    }
}

#[derive(Debug, Error)]
pub enum FenceResetError {
    #[error("Unknown Vulkan error: {0}")]
    UnknownVulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for FenceResetError {
    fn from(value: ash::vk::Result) -> Self {
        Self::UnknownVulkan(value)
    }
}

impl Fence {
    /// Create a new fence. If `signaled` is true the fence will be created in the signaled state
    pub fn new(device: &Arc<Device>, signaled: bool) -> Result<Self, FenceCreateError> {
        let flags = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        };
        let ci = vk::FenceCreateInfo::default().flags(flags);
        //SAFETY: Valid create info
        let f =
            unsafe { device.inner().create_fence(&ci, None) }.map_err(FenceCreateError::from)?;
        Ok(Self {
            parent_device: device.clone(),
            inner: f,
        })
    }

    /// Wait for this fence with the specified timeout (nanoseconds)
    pub fn wait(&self, timeout: u64) -> Result<(), FenceWaitError> {
        //SAFETY: waiting is safe; the fence is owned by this device
        let res = unsafe {
            self.parent_device
                .inner()
                .wait_for_fences(&[self.inner], true, timeout)
        };
        match res {
            Ok(()) => Ok(()),
            Err(vk::Result::TIMEOUT) => Err(FenceWaitError::Timeout),
            Err(e) => Err(FenceWaitError::UnknownVulkan(e)),
        }
    }

    /// Reset the fence to the unsignaled state
    pub fn reset(&self) -> Result<(), FenceResetError> {
        //SAFETY: resetting is safe
        unsafe { self.parent_device.inner().reset_fences(&[self.inner]) }
            .map_err(FenceResetError::from)
    }

    /// Wait for this fence until `timeout` (nanoseconds) and then reset it.
    /// Returns a `FenceWaitError` if waiting fails or the reset failed (the
    /// reset failure is converted into a `FenceWaitError::UnknownVulkan`).
    pub fn wait_and_reset(&self, timeout: u64) -> Result<(), FenceWaitError> {
        // First wait
        self.wait(timeout)?;
        // Then reset; map reset errors into FenceWaitError::UnknownVulkan
        match self.reset() {
            Ok(()) => Ok(()),
            Err(FenceResetError::UnknownVulkan(e)) => Err(FenceWaitError::UnknownVulkan(e)),
        }
    }

    /// Get the raw fence handle
    pub fn inner(&self) -> vk::Fence {
        self.inner
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        //SAFETY: Last use of fence
        unsafe { self.parent_device.inner().destroy_fence(self.inner, None) };
    }
}

/// RAII wrapper around a Vulkan Semaphore
#[derive(Debug)]
pub struct Semaphore {
    parent_device: Arc<Device>,
    inner: vk::Semaphore,
}

#[derive(Debug, Error)]
pub enum SemaphoreCreateError {
    #[error("Unknown Vulkan error: {0}")]
    UnknownVulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for SemaphoreCreateError {
    fn from(value: ash::vk::Result) -> Self {
        Self::UnknownVulkan(value)
    }
}

impl Semaphore {
    /// Create a new semaphore
    pub fn new(device: &Arc<Device>) -> Result<Self, SemaphoreCreateError> {
        let ci = vk::SemaphoreCreateInfo::default();
        //SAFETY: Valid create info
        let s = unsafe { device.inner().create_semaphore(&ci, None) }
            .map_err(SemaphoreCreateError::from)?;
        Ok(Self {
            parent_device: device.clone(),
            inner: s,
        })
    }

    /// Get the raw semaphore handle
    pub fn inner(&self) -> vk::Semaphore {
        self.inner
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        //SAFETY: Last use of semaphore
        unsafe {
            self.parent_device
                .inner()
                .destroy_semaphore(self.inner, None)
        };
    }
}
