//! Library root: shared types and modules used by the binary
#![warn(missing_debug_implementations)]

use std::sync::LockResult;

/// Extension methods for [`LockResult`].
///
/// [`LockResult`]: https://doc.rust-lang.org/stable/std/sync/type.LockResult.html
pub trait LockResultExt {
    type Guard;

    /// Returns the lock guard even if the mutex is [poisoned].
    ///
    /// [poisoned]: https://doc.rust-lang.org/stable/std/sync/struct.Mutex.html#poisoning
    fn ignore_poison(self) -> Self::Guard;
}

impl<Guard> LockResultExt for LockResult<Guard> {
    type Guard = Guard;

    fn ignore_poison(self) -> Guard {
        self.unwrap_or_else(|e| {
            tracing::warn!(
                "mutex poisoned, recovering lock (ignoring poison): {:#?}",
                e
            );
            e.into_inner()
        })
    }
}

/// A simple version struct used for internal representation
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Version {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch Version
    pub patch: u32,
}

impl Version {
    /// Constructs a Version from a major version, minor version, and patch
    /// version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Constructs a Version from a Vulkan Version
    pub fn from_vk_version(vk_version: u32) -> Self {
        let major = ash::vk::api_version_major(vk_version);
        let minor = ash::vk::api_version_minor(vk_version);
        let patch = ash::vk::api_version_patch(vk_version);
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Converts from a Version to vulkan's internal version format
    pub fn to_vk_version(self) -> u32 {
        ash::vk::make_api_version(0, self.major, self.minor, self.patch)
    }
}

#[derive(clap::Parser, Debug)]
/// Argument parser exposed from the library so other crates (including the
/// binary) can parse CLI arguments using the same definition.
pub struct Args {
    // TODO: Configure default to change based on build personality
    #[arg(short, long, default_value_t = crate::rvk::instance::VulkanDebugLevel::Warn)]
    /// What level to run validation layers at
    pub vulkan_debug_level: crate::rvk::instance::VulkanDebugLevel,

    #[arg(short, long, default_value_t = tracing::Level::WARN)]
    /// What level to run Rust's logging at
    pub log_level: tracing::Level,
}

// Re-export the rvk module so callers can use `crate::rvk::...`
pub mod rvk;
