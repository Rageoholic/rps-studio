use std::str::FromStr;

use rps_studio::rvk::instance::VulkanDebugLevel;

#[test]
fn vulkan_debug_level_default_and_roundtrip() -> eyre::Result<()> {
    assert_eq!(VulkanDebugLevel::default(), VulkanDebugLevel::None);

    // Roundtrip: Display -> FromStr should parse back to same variant
    let variants = [
        VulkanDebugLevel::None,
        VulkanDebugLevel::Verbose,
        VulkanDebugLevel::Info,
        VulkanDebugLevel::Warn,
        VulkanDebugLevel::Error,
    ];

    for &v in &variants {
        let s = v.to_string();
        let parsed = VulkanDebugLevel::from_str(&s)
            .map_err(|e| eyre::eyre!("should parse display string ({s}): {e}"))?;
        assert_eq!(parsed, v, "roundtrip for {} failed", s);
    }

    // Invalid string should fail to parse
    assert!(VulkanDebugLevel::from_str("not-a-level").is_err());

    Ok(())
}
