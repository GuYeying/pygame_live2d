//! Renderer-facing helpers.
//!
//! The [`crate::render::common`] module is backend-neutral and can be used with
//! any graphics API. Enable the `wgpu` feature to use Mocari's built-in `wgpu`
//! backend.

pub mod common;

#[cfg(feature = "wgpu")]
pub mod wgpu;

/// pygamest 生态落地渲染管线（蒙版纹理方案；产物面=用户注入 Surface
/// ——pygame-rs 零渲染编排，只做宏绑定）。
#[cfg(feature = "render")]
pub mod pygst;
