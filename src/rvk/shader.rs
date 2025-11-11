use std::{
    fmt::{Debug, Display},
    path::{Path, PathBuf},
    sync::Arc,
};

use ash::vk;
use naga::{
    back::spv::{DebugInfo, Options, PipelineOptions, WriterFlags},
    front::glsl,
    valid::{Capabilities, SubgroupOperationSet, ValidationFlags},
};
use thiserror::Error;

use super::device::Device;

#[derive(Debug)]
/// What type of shader we're talking about. Maybe we can exclude this if we
/// have specific types for each shader? But right now this is good
pub enum ShaderType {
    ///Vertex Shader (takes input from buffers and emits points)
    Vertex,
    ///Fragment Shader (What to render into the pixels)
    Fragment,
}
impl From<ShaderType> for naga::ShaderStage {
    fn from(val: ShaderType) -> Self {
        val.to_naga_stage()
    }
}

impl From<ShaderType> for naga::valid::ShaderStages {
    fn from(val: ShaderType) -> Self {
        val.to_naga_stage_bitfield()
    }
}

impl ShaderType {
    /// Conversion method
    fn to_naga_stage(&self) -> naga::ShaderStage {
        match self {
            ShaderType::Vertex => naga::ShaderStage::Vertex,
            ShaderType::Fragment => naga::ShaderStage::Fragment,
        }
    }
    /// Conversion method
    fn to_naga_stage_bitfield(&self) -> naga::valid::ShaderStages {
        match self {
            ShaderType::Vertex => naga::valid::ShaderStages::VERTEX,
            ShaderType::Fragment => naga::valid::ShaderStages::FRAGMENT,
        }
    }
}
#[derive(Debug)]
/// Represents a shader object in vk with RAII semantics
pub struct Shader {
    #[expect(dead_code, reason = "In dev")]
    /// What type this shader is
    shader_type: ShaderType,
    /// The underlying handle to the vulkan object
    vk_shader: vk::ShaderModule,
    /// Device we came from
    parent_device: Arc<Device>,
}
impl Shader {
    pub(crate) fn inner(&self) -> vk::ShaderModule {
        self.vk_shader
    }
}

#[derive(Debug, Error)]
pub enum ShaderCompileError {
    /// Error parsing file
    Parse {
        /// The underlying naga error
        e: naga::front::glsl::ParseErrors,
        /// Source code
        src: String,
        /// Optional path to file
        file: Option<PathBuf>,
    },
    /// Error validating file
    Validation {
        /// Underlying naga error
        e: naga::WithSpan<naga::valid::ValidationError>,
        /// Source code
        src: String,
        /// Optional path to file
        path: Option<PathBuf>,
    },
    /// Some weird backend error
    Backend(naga::back::spv::Error, Option<PathBuf>),
    UnknownVulkan(vk::Result),
}

impl Display for ShaderCompileError {
    //TODO: Improve this Display impl. It sucks rn
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShaderCompileError::Parse { e, src, file } => {
                writeln!(
                    f,
                    "Could not create shader {} due to parse error.",
                    file.as_ref().and_then(|p| p.to_str()).unwrap_or_else(|| {
                        tracing::warn!(
                            "shader compile display: path missing or non-UTF8; using <unknown>"
                        );
                        "<unknown>"
                    })
                )?;
                f.write_str(&e.emit_to_string(src))?;
                Ok(())
            }
            ShaderCompileError::Validation { e, src, path } => f.write_str(&format!(
                "Could not create shader {} due to validation error\n",
                e.emit_to_string_with_path(
                    src,
                    path.as_ref().and_then(|p| p.to_str()).unwrap_or_else(|| {
                        tracing::warn!(
                            "shader validation display: path missing or non-UTF8; using <unknown>"
                        );
                        "<unknown>"
                    })
                )
            )),
            ShaderCompileError::Backend(error, path_buf) => {
                f.write_str(&format!(
                    "Could not create shader {} due to backend error\n",
                    path_buf
                        .as_ref()
                        .and_then(|p| p.to_str())
                        .unwrap_or_else(|| {
                            tracing::warn!(
                                "shader backend display: path missing or non-UTF8; using <unknown>"
                            );
                            "<unknown>"
                        })
                ))?;
                Display::fmt(error, f)?;
                Ok(())
            }
            ShaderCompileError::UnknownVulkan(e) => {
                f.write_str("Could not create shader due to unknown vulkan error: ")?;
                Display::fmt(&e, f)
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// How much do we optimize these shaders
pub enum ShaderOptLevel {
    /// Apply no optimizations
    None,
    /// Do actually optimize the shaders
    Heavy,
}

#[derive(Clone, Copy, Debug)]
/// What level of debug info do we emit
pub enum ShaderDebugLevel {
    /// Emit all debug info
    Full,
    /// Emit no debug info
    None,
}

/// Represents a compiler we can use to make shaders. Represented as an
/// object to cache any relevant state
pub struct ShaderCompiler {
    /// Responsible for taking the ir and emitting the final code
    spv_writer: naga::back::spv::Writer,
    /// How much shaders from this compiler should be optimized
    opt_level: ShaderOptLevel,
    /// What level of debug info to emit
    debug_level: ShaderDebugLevel,
    /// Parser for glsl
    glsl_parser: naga::front::glsl::Frontend,
    /// takes IR and ensures it is valid
    shader_validator: naga::valid::Validator,
    /// A device we can use to make the actual shader module. Technically
    /// not necessary but saves passing in different devices
    parent_device: Arc<Device>,
}

impl Debug for ShaderCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShaderCompiler")
            .field("spv_writer", &"{..}")
            .field("opt_level", &self.opt_level)
            .field("debug_level", &self.debug_level)
            .finish()
    }
}

#[derive(Debug, Error)]
/// Errors from compiling a shader
pub enum ShaderCompilerError {
    #[error("Error creating naga backend: {0}")]
    /// Some weird backend error
    Backend(naga::back::spv::Error),
}

impl ShaderCompiler {
    /// Create a new shader compiler
    pub fn new(
        device: &Arc<Device>,
        debug_level: ShaderDebugLevel,
        opt_level: ShaderOptLevel,
    ) -> Result<Self, ShaderCompilerError> {
        let spv_writer = naga::back::spv::Writer::new(&Options {
            flags: match debug_level {
                ShaderDebugLevel::Full => WriterFlags::DEBUG,
                ShaderDebugLevel::None => WriterFlags::empty(),
            } | WriterFlags::ADJUST_COORDINATE_SPACE
                | WriterFlags::CLAMP_FRAG_DEPTH
                | WriterFlags::LABEL_VARYINGS,
            ..Default::default()
        })
        .map_err(ShaderCompilerError::Backend)?;

        let glsl_parser = naga::front::glsl::Frontend::default();

        let shader_validator =
            naga::valid::Validator::new(ValidationFlags::all(), Capabilities::all());

        Ok(Self {
            glsl_parser,
            shader_validator,
            spv_writer,
            opt_level,
            debug_level,
            parent_device: device.clone(),
        })
    }

    /// Create a shader object from its source and optionally its path
    pub fn compile_shader(
        &mut self,
        source: &str,
        shader_type: ShaderType,
        file_path: Option<&Path>,
    ) -> Result<Shader, Box<ShaderCompileError>> {
        let shader_mod = self
            .glsl_parser
            .parse(&glsl::Options::from(shader_type.to_naga_stage()), source)
            .map_err(|e| ShaderCompileError::Parse {
                e,
                src: source.to_owned(),
                file: file_path.map(|p| p.to_owned()),
            })?;

        let shader_mod_info = self
            .shader_validator
            .subgroup_stages(shader_type.to_naga_stage_bitfield())
            .subgroup_operations(SubgroupOperationSet::all())
            .validate(&shader_mod)
            .map_err(|e| ShaderCompileError::Validation {
                e,
                src: source.to_owned(),
                path: file_path.map(Path::to_owned),
            })?;

        //Ensure the shader validator is all good for next time
        self.shader_validator.reset();

        let mut spv_bytes = Vec::new();

        let debug_info = match (self.debug_level, file_path) {
            (ShaderDebugLevel::Full, Some(file_name)) => Some(DebugInfo {
                source_code: source,
                file_name,
                language: naga::back::spv::SourceLanguage::GLSL,
            }),

            _ => None,
        };

        self.spv_writer
            .write(
                &shader_mod,
                &shader_mod_info,
                Some(&PipelineOptions {
                    shader_stage: shader_type.to_naga_stage(),
                    entry_point: "main".to_owned(),
                }),
                &debug_info,
                &mut spv_bytes,
            )
            .map_err(|e| ShaderCompileError::Backend(e, file_path.map(Path::to_owned)))?;

        let shader_mod_ci = vk::ShaderModuleCreateInfo::default().code(&spv_bytes);

        //SAFETY: We got spv_bytes from naga's backend so it *should* be valid Spir-V
        let vk_shader = unsafe {
            self.parent_device
                .inner()
                .create_shader_module(&shader_mod_ci, None)
        }
        .map_err(ShaderCompileError::UnknownVulkan)?;

        Ok(Shader {
            vk_shader,
            shader_type,
            parent_device: self.parent_device.clone(),
        })
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        //SAFETY: We know that self.vk_shader was made from
        //self.parent_device
        unsafe {
            self.parent_device
                .inner()
                .destroy_shader_module(self.vk_shader, None)
        };
    }
}
