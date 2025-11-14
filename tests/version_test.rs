use rps_studio::Version;

#[test]
fn version_vk_roundtrip() {
    let v = Version::new(1, 2, 3);
    let vk = v.to_vk_version();
    let v2 = Version::from_vk_version(vk);
    assert_eq!(v, v2);
}

#[test]
fn version_to_vk_is_deterministic() {
    let v1 = Version::new(0, 0, 0);
    let v2 = Version::from_vk_version(v1.to_vk_version());
    assert_eq!(v1, v2);
    let v3 = Version::new(1, 0, 0);
    assert_eq!(v3, Version::from_vk_version(v3.to_vk_version()));
}
