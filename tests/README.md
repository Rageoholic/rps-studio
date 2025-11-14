
This directory contains the project's test artifacts. Prefer putting
non-platform-specific tests here (they run in CI and locally with
`cargo test`). Platform-specific integration tests (GPU/windowing) should be
kept out of CI or clearly gated.

Guidelines for authors
- Keep tests small and focused. Prefer pure-Rust tests so CI can run them.
- If a test requires Vulkan or a windowing surface, mark it ignored or gate it
  behind an explicit feature/ENV opt-in and document setup steps in the test
  header.