use std::{fmt::Debug, sync::Arc};

use ash::vk::{
    self, ColorSpaceKHR, ComponentMapping, CompositeAlphaFlagsKHR, Extent2D, Format, Image,
    ImageAspectFlags, ImageSubresourceRange, ImageUsageFlags, ImageView, ImageViewCreateInfo,
    ImageViewType, PresentModeKHR, SharingMode, SurfaceTransformFlagsKHR, SwapchainCreateInfoKHR,
    SwapchainKHR,
};

use crate::rvk::{device::Device, surface::Surface};

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
    _surface_format: ash::vk::SurfaceFormatKHR,
    /// The chosen extent for the swapchain images (width/height)
    swapchain_extent: vk::Extent2D,
}

impl PartialEq for Swapchain {
    fn eq(&self, other: &Self) -> bool {
        self.parent_device == other.parent_device
            && self.parent_surface == other.parent_surface
            && self.swapchain == other.swapchain
    }
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

/// Error type returned by `Swapchain::acquire_next_image`.
#[derive(thiserror::Error, Debug)]
pub enum AcquireImageError {
    #[error("Swapchain is out of date")]
    OutOfDate,
    #[error("Timeout while waiting for next image")]
    Timeout,
    #[error("Unknown Vulkan error: {0}")]
    UnknownVulkan(ash::vk::Result),
}

impl From<ash::vk::Result> for AcquireImageError {
    fn from(value: ash::vk::Result) -> Self {
        match value {
            ash::vk::Result::ERROR_OUT_OF_DATE_KHR => AcquireImageError::OutOfDate,
            ash::vk::Result::TIMEOUT => AcquireImageError::Timeout,
            other => AcquireImageError::UnknownVulkan(other),
        }
    }
}

impl Swapchain {
    /// Create a swapchain from this device and surface. Might wanna
    /// make some optional parameters to configure? But rn this is
    /// wrapping how *I* set up a swapchain
    pub fn create(
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        old_swapchain: Option<&Self>,
    ) -> Result<Option<Swapchain>, SwapchainCreateError> {
        if device.get_parent().raw_handle() != surface.parent_instance().raw_handle() {
            return Err(SwapchainCreateError::IncompatibleParameters);
        }
        let swapchain_device =
            ash::khr::swapchain::Device::new(device.parent_instance().inner(), device.inner());

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
            .unwrap_or_else(|| {
                tracing::warn!(
                    "preferred present mode MAILBOX not available; falling back to FIFO"
                );
                PresentModeKHR::FIFO
            });

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
        if swapchain_extent.width == 0 || swapchain_extent.height == 0 {
            tracing::warn!("Swapchain extent has zero width or height; returning None");
            return Ok(None);
        }

        let default_image_count = surface_capabilities.min_image_count + 1;
        let max_image_count = surface_capabilities.max_image_count;
        let requested_image_count = if max_image_count != 0 && max_image_count < default_image_count
        {
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
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .old_swapchain(old_swapchain.map(|s| s.swapchain).unwrap_or_else(|| {
                tracing::trace!("no old swapchain provided; using null handle");
                vk::SwapchainKHR::null()
            }));

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
            let iv = unsafe { device.inner().create_image_view(&ivci, None) }?;
            image_views.push(iv);
        }

        Ok(Some(Self {
            parent_device: device.clone(),
            parent_surface: surface.clone(),
            swapchain,
            swapchain_device,
            _images: images,
            image_views,
            _surface_format: surface_format,
            swapchain_extent,
        }))
    }

    /// Get the width (in pixels) of swapchain images
    pub fn width(&self) -> u32 {
        self.swapchain_extent.width
    }

    /// Get the height (in pixels) of swapchain images
    pub fn height(&self) -> u32 {
        self.swapchain_extent.height
    }

    pub(crate) fn get_color_format(&self) -> vk::Format {
        self._surface_format.format
    }

    pub(crate) fn get_parent(self: &Swapchain) -> &Device {
        &self.parent_device
    }

    /// Get the raw swapchain image handle for the given index.
    pub(crate) fn image(&self, idx: usize) -> vk::Image {
        self._images[idx]
    }

    /// Get the image view associated with the given swapchain image index.
    pub(crate) fn image_view(&self, idx: usize) -> vk::ImageView {
        self.image_views[idx]
    }

    /// Acquire the next available image from the swapchain.
    ///
    /// Returns (image_index, suboptimal) on success where `suboptimal` is
    /// true if the swapchain is considered suboptimal for presentation.
    /// Returns an `AcquireImageError` on failure.
    pub fn acquire_next_image(
        &self,
        timeout: u64,
        semaphore: &crate::rvk::sync_objects::Semaphore,
        fence: Option<&crate::rvk::sync_objects::Fence>,
    ) -> Result<(u32, bool), AcquireImageError> {
        let sem = semaphore.inner();
        let f = fence.map(|f| f.inner()).unwrap_or(ash::vk::Fence::null());

        // SAFETY: Calling the swapchain acquire on the device that created the
        // swapchain with valid handles. `acquire_next_image` may return a
        // (image_index, suboptimal) tuple; we expose only the image index here
        // and leave suboptimal handling to callers.
        match unsafe {
            self.swapchain_device
                .acquire_next_image(self.swapchain, timeout, sem, f)
        } {
            Ok((idx, suboptimal)) => Ok((idx, suboptimal)),
            Err(e) => Err(AcquireImageError::from(e)),
        }
    }

    /// Present an image to the given queue. `wait_semaphores` are signaled
    /// prior to presentation (commonly the render-complete semaphores).
    /// Returns `Ok(suboptimal)` where `suboptimal` is true if the presentation
    /// was suboptimal. Returns `Err(AcquireImageError::OutOfDate)` when the
    /// swapchain is out of date.
    pub fn present(
        &self,
        device: &crate::rvk::device::Device,
        wait_semaphores: &[ash::vk::Semaphore],
        image_index: u32,
    ) -> Result<bool, AcquireImageError> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let pi = ash::vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        // Get the graphics queue from the device (locks the mutex internally).
        let queue = device.graphics_queue_handle();

        // SAFETY: presenting on the queue associated with the device that
        // created the swapchain. Ash will validate the inputs.
        match unsafe { self.swapchain_device.queue_present(queue, &pi) } {
            Ok(suboptimal) => Ok(suboptimal),
            Err(e) => Err(AcquireImageError::from(e)),
        }
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
            unsafe { self.parent_device.inner().destroy_image_view(iv, None) };
        }
        //SAFETY: Swapchain was made with this swapchain_device
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None)
        };
    }
}
