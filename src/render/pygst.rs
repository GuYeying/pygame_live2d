//! pygamest 生态落地渲染管线（本 crate 自维护——pygame-rs 零渲染编排，
//! 只做宏绑定；上游 mocari 的 render/wgpu 后端不动，此为独立落地层）。
//!
//! 对象形态：用户创建 Surface 注入（[`Live2DRenderer::new`]），每对象
//! 自持渲染目标 + Mg/C 蒙版工作面池；[`Live2DRenderer::draw`] 全管线
//! 自理（自管 encoder 三段：① 蒙版白形 → ② masked 双纹理 → ③ 整模型
//! 落面）。产物 = 全透明底人物图——上屏/摆放/缩放交给 pygamest 侧
//! `gfxdraw.draw_surface`（精灵工作流）。
//!
//! 坐标定约（与 pygamest 中心系同构，全 0-1 相对注入面）：
//! - `x`/`y`：人物内容中心位置，0=面中心，±1=±半幅，y 向上
//! - `scale`：人物内容高 / 面高（0-1；>1 溢出裁边=特写）
//! - 内容包围盒 load 时算一次（静止姿态顶点联合 bbox）——"动画集中
//!   位置不一样"由内容锚定自动吸收，x/y 只做精调
//!
//! 定约沉淀（ gfxdraw 时代实测）：① mocari 顶点数组 = 索引化三角形
//! 池——必须按索引 draw；② clip 烘焙域与 plan 域须一致（现归一域直
//! 传，MVP 仿射在 VS——无 scale 混乘面）；③ 蒙版面自采样 = usage 冲突
//! （蒙版槽以白纹理代之恒等裁剪）；④ overlay/中间面提交须先于采样面
//! 的批次（此处自管 encoder 一次性 submit，天然有序）。

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use starfish_pygame::render::{gpu_info, DrawTarget, Surface as SfSurface};

use crate::assets::RuntimeModel;
use crate::moc3::Moc3DrawableMesh;
use crate::mask::{mask_clip_transform, mask_groups};

/// 渲染错误（GPU 未就绪等）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderError(pub String);

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "live2d render: {}", self.0)
    }
}

impl std::error::Error for RenderError {}

/// 渲染结果别名。
pub type RenderResult<T> = std::result::Result<T, RenderError>;

/// masked 双纹理 WGSL：色纹理 × 蒙版纹理 alpha 相乘（蒙版区域外
/// alpha=0 自动透明）。binding 0 = MVP、1/2 = 色纹理+采样、3/4 =
/// 蒙版纹理+采样。蒙版段/整模型段共用同一条管线（白纹理 = 恒等蒙版）。
const MASKED_WGSL: &str = r#"
struct Mvp {
    m: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> mvp: Mvp;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var color_samp: sampler;
@group(0) @binding(3) var mask_tex: texture_2d<f32>;
@group(0) @binding(4) var mask_samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) muv: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) p: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) muv: vec2<f32>,
) -> VsOut {
    var o: VsOut;
    o.pos = mvp.m * vec4<f32>(p, 0.0, 1.0);
    o.uv = uv;
    o.muv = muv;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(color_tex, color_samp, i.uv);
    let m = textureSample(mask_tex, mask_samp, i.muv);
    return vec4<f32>(c.rgb, c.a * m.a);
}
"#;

/// 管线三件套（进程级单例——只跟 GPU 有关、与模型/对象无关）。
struct MaskedPipeline {
    pipeline: Arc<wgpu::RenderPipeline>,
    bgl: Arc<wgpu::BindGroupLayout>,
    sampler: Arc<wgpu::Sampler>,
}

static PIPELINE: OnceLock<MaskedPipeline> = OnceLock::new();

fn pipeline() -> RenderResult<&'static MaskedPipeline> {
    if let Some(pl) = PIPELINE.get() {
        return Ok(pl);
    }
    let gpu = gpu_info().ok_or_else(|| {
        RenderError("GPU 未就绪（须先 pygame.display.set_mode 建立 GPU 会话）".into())
    })?;
    let module = gpu
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pygamest_live2d_masked"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MASKED_WGSL)),
        });
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("pygamest_live2d_masked_samp"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    // binding 0 全阶段声明（uniform 资源路径的可见性定约——与绘制侧
    // bind group 布局按位兼容校验）
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pygamest_live2d_masked_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX
                        | wgpu::ShaderStages::FRAGMENT
                        | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pygamest_live2d_masked_layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pygamest_live2d_masked_pl"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    // pos2 + uv2 + mask_uv2（顶点归一域直传——MVP 仿射在 VS）
                    array_stride: 24,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 8,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    // 离屏面恒 Rgba8Unorm（starfish 离屏画布定约）
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
    let pl = MaskedPipeline {
        pipeline: Arc::new(pipeline),
        bgl: Arc::new(bgl),
        sampler: Arc::new(sampler),
    };
    let _ = PIPELINE.set(pl);
    Ok(PIPELINE.get().expect("pipeline 刚写入"))
}

/// 一组蒙版渲染计划（**归一域**——不含 scale，与注入面尺寸解耦；
/// span/clip 每帧现算，动画驱动顶点位移）。
#[derive(Debug, Clone)]
struct MaskGroupPlan {
    group: crate::mask::MaskGroup,
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    k_x: f32,
    k_y: f32,
    c_x: f32,
    c_y: f32,
}

fn plan_mask_groups(meshes: &[Moc3DrawableMesh]) -> Vec<MaskGroupPlan> {
    mask_groups(meshes)
        .into_iter()
        .map(|group| {
            let (mut min_x, mut min_y, mut max_x, mut max_y) =
                (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for &i in group.mask_ids.iter().chain(group.masked_ids.iter()) {
                let Some(m) = meshes.get(i) else {
                    continue;
                };
                for v in m.vertices() {
                    let p = v.position();
                    min_x = min_x.min(p[0]);
                    min_y = min_y.min(p[1]);
                    max_x = max_x.max(p[0]);
                    max_y = max_y.max(p[1]);
                }
            }
            let (k_x, k_y, c_x, c_y) =
                mask_clip_transform(min_x, min_y, max_x, max_y, 0.25);
            MaskGroupPlan {
                group,
                min_x,
                min_y,
                max_x,
                max_y,
                k_x,
                k_y,
                c_x,
                c_y,
            }
        })
        .collect()
}

/// f32 → native-endian bytes（顶点/uniform 装载；零 bytemuck 依赖）。
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// u16 → native-endian bytes（索引装载）。
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_ne_bytes()).collect()
}

/// 恒等 MVP（蒙版/masked 段——clip NDC 顶点 CPU 烘焙后直绘）。
fn identity_mvp() -> [f32; 16] {
    [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

/// 内容锚定仿射 MVP（归一域 → 注入面 NDC；列主序——WGSL mat4x4 布局）。
///
/// 像素位（面左上原点，y-down）= 面中心锚 + x/y 半幅偏移 + 内容偏移：
/// px = W/2 + x·W/2 + (nx−cx)·k；py = H/2 − y·H/2 + (cy−ny)·k
/// （mocari y-up → 面像素 y-down 翻转内置；k = scale×H/内容高）。
/// NDC 化简：ndc_x = x + (nx−cx)·kx；ndc_y = y + (ny−cy)·ky。
fn content_mvp(cx: f32, cy: f32, ch: f32, x: f32, y: f32, scale: f32, w: f32, h: f32) -> [f32; 16] {
    let k = scale * h / ch.max(1e-6);
    let kx = k * 2.0 / w.max(1.0);
    let ky = k * 2.0 / h.max(1.0);
    [
        kx, 0.0, 0.0, 0.0, 0.0, ky, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x - cx * kx, y - cy * ky,
        0.0, 1.0,
    ]
}

/// clip 域顶点（蒙版/masked 段）：位置烘蒙版 NDC（ndc=(2u-1, 1-2v)——
/// 渲染到纹理 y 翻转预补偿，采样域 uv=clip 域直读）、uv 双采样。
fn clip_mesh_verts(m: &Moc3DrawableMesh, plan: &MaskGroupPlan) -> Vec<f32> {
    m.vertices()
        .iter()
        .flat_map(|v| {
            let p = v.position();
            let u = p[0] * plan.k_x + plan.c_x;
            let w = p[1] * plan.k_y + plan.c_y;
            [u * 2.0 - 1.0, 1.0 - w * 2.0, v.uv()[0], v.uv()[1], u, w]
        })
        .collect()
}

/// 归一域顶点（整模型段）：位置/uv 直传——内容仿射在 MVP。
fn plain_mesh_verts(m: &Moc3DrawableMesh) -> Vec<f32> {
    m.vertices()
        .iter()
        .flat_map(|v| {
            let p = v.position();
            [p[0], p[1], v.uv()[0], v.uv()[1], 0.0, 0.0]
        })
        .collect()
}

fn upload_mesh(device: &wgpu::Device, verts: &[f32], indices: &[u16]) -> (Arc<wgpu::Buffer>, Arc<wgpu::Buffer>) {
    use wgpu::util::DeviceExt;
    let vbo = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pygamest_l2d_vbo"),
        contents: &f32_bytes(verts),
        usage: wgpu::BufferUsages::VERTEX,
    }));
    let ibo = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("pygamest_l2d_ibo"),
        contents: &u16_bytes(indices),
        usage: wgpu::BufferUsages::INDEX,
    }));
    (vbo, ibo)
}

fn make_bind_group(
    device: &wgpu::Device,
    pl: &MaskedPipeline,
    mvp: &Arc<wgpu::Buffer>,
    color_view: &Arc<wgpu::TextureView>,
    mask_view: &Arc<wgpu::TextureView>,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &pl.bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: mvp,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(color_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&pl.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&pl.sampler),
            },
        ],
        label: Some("pygamest_l2d_bg"),
    })
}

/// 开一条透明清屏的渲染通道（三段统一——蒙版/masked 中间面与产物面
/// 都是"每帧从全透明重画"语义）。
fn begin_pass<'a>(
    enc: &'a mut wgpu::CommandEncoder,
    label: &str,
    view: &'a Arc<wgpu::TextureView>,
) -> wgpu::RenderPass<'a> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Live2D 渲染器（每对象一个——自持注入面 + 蒙版工作面池 + 内容锚定）。
pub struct Live2DRenderer {
    surface: Arc<SfSurface>,
    /// 模型纹理 Surface 化缓存（惰性一次）
    textures: RefCell<Option<Vec<Arc<SfSurface>>>>,
    mask_pool: RefCell<Vec<Arc<SfSurface>>>,
    color_pool: RefCell<Vec<Arc<SfSurface>>>,
    white: RefCell<Option<Arc<SfSurface>>>,
    /// 内容包围盒（归一域；load 时静止姿态算一次）
    content_cx: f32,
    content_cy: f32,
    content_h: f32,
}

impl Live2DRenderer {
    /// 注入面 + 模型 → 渲染器（内容包围盒此刻定锚）。
    pub fn new(surface: Arc<SfSurface>, model: &RuntimeModel) -> Self {
        let (mut min_x, mut min_y, mut max_x, mut max_y) =
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        let mut any = false;
        for m in model.runtime().meshes() {
            for v in m.vertices() {
                let p = v.position();
                any = true;
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
            }
        }
        let (cx, cy, ch) = if any {
            (
                (min_x + max_x) * 0.5,
                (min_y + max_y) * 0.5,
                (max_y - min_y).max(1e-6),
            )
        } else {
            (0.0, 0.0, 1.0)
        };
        Self {
            surface,
            textures: RefCell::new(None),
            mask_pool: RefCell::new(Vec::new()),
            color_pool: RefCell::new(Vec::new()),
            white: RefCell::new(None),
            content_cx: cx,
            content_cy: cy,
            content_h: ch,
        }
    }

    /// 注入面句柄（产物纹理本体——调试/对齐用）。
    pub fn surface(&self) -> &Arc<SfSurface> {
        &self.surface
    }

    /// 全管线自理绘制（每帧调用——先 update_meshes 后 draw）。
    ///
    /// 三段 encoder（同 submit 有序）：每组 ① 蒙版白形 → ② masked 双
    /// 纹理（组内 render_order 排序）；③ 整模型落注入面（render_order
    /// 混排，masked 组以 C 面 quad 在组首绘位合成）。产物面每帧透明
    /// 重画——"draw 一次 = 重画一次人物"。
    pub fn draw(&self, model: &RuntimeModel, x: f32, y: f32, scale: f32) -> RenderResult<()> {
        let gpu = gpu_info().ok_or_else(|| {
            RenderError("GPU 未就绪（须先 pygame.display.set_mode 建立 GPU 会话）".into())
        })?;
        let pl = pipeline()?;
        let (sw, sh) = self.surface.size();

        // 模型纹理惰性 Surface 化（一次）
        if self.textures.borrow().is_none() {
            *self.textures.borrow_mut() = Some(
                model
                    .textures()
                    .iter()
                    .map(|t| {
                        Arc::new(SfSurface::from_rgba8((t.width(), t.height()), t.rgba()))
                    })
                    .collect(),
            );
        }
        let tex_list = self
            .textures
            .borrow()
            .clone()
            .expect("纹理缓存刚确保");
        // 白 1×1（mask 白形 + 恒等蒙版槽）
        if self.white.borrow().is_none() {
            *self.white.borrow_mut() = Some(Arc::new(SfSurface::from_rgba8(
                (1, 1),
                &[255, 255, 255, 255],
            )));
        }
        let white = self.white.borrow().clone().expect("白面刚确保");
        let white_view = white.texture_view();
        let identity_view = white.texture_view();

        // 蒙版组计划（归一域）+ 面池扩容 + Arc 快照（借用 discipline）
        let plans = plan_mask_groups(model.runtime().meshes());
        {
            let mut mp = self.mask_pool.borrow_mut();
            while mp.len() < plans.len() {
                mp.push(Arc::new(SfSurface::new((512, 512))));
            }
            let mut cp = self.color_pool.borrow_mut();
            while cp.len() < plans.len() {
                cp.push(Arc::new(SfSurface::new((512, 512))));
            }
        }
        let mask_surfs: Vec<Arc<SfSurface>> = self.mask_pool.borrow().clone();
        let color_surfs: Vec<Arc<SfSurface>> = self.color_pool.borrow().clone();

        // MVP uniform（恒等 + 内容仿射）
        let idm = identity_mvp();
        let mvp = content_mvp(
            self.content_cx,
            self.content_cy,
            self.content_h,
            x,
            y,
            scale,
            sw as f32,
            sh as f32,
        );
        let idm_buf = Arc::new({
            use wgpu::util::DeviceExt;
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pygamest_l2d_idm"),
                contents: &f32_bytes(&idm),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });
        let content_buf = Arc::new({
            use wgpu::util::DeviceExt;
            gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pygamest_l2d_content_mvp"),
                contents: &f32_bytes(&mvp),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        });

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pygamest_l2d_draw"),
            });

        // ── ①② 蒙版/masked 段（每组两 pass）──
        for (gi, plan) in plans.iter().enumerate() {
            let (Some(mg), Some(cs)) = (mask_surfs.get(gi), color_surfs.get(gi)) else {
                continue;
            };
            let mask_view = mg.texture_view();
            // ① 蒙版白形（蒙版面自采样 = usage 冲突——蒙版槽以白代之）
            {
                let mut rp = begin_pass(&mut enc, "pygamest_l2d_mask", &mask_view);
                rp.set_pipeline(&pl.pipeline);
                for &di in &plan.group.mask_ids {
                    let Some(m) = model.runtime().meshes().get(di) else {
                        continue;
                    };
                    // 空索引部件（隐藏占位）——空 buffer 会 panic，跳过
                    if m.indices().is_empty() {
                        continue;
                    }
                    let (vbo, ibo) =
                        upload_mesh(&gpu.device, &clip_mesh_verts(m, plan), m.indices());
                    let bg = make_bind_group(&gpu.device, pl, &idm_buf, &white_view, &identity_view);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.set_vertex_buffer(0, vbo.slice(..));
                    rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint16);
                    rp.draw_indexed(0..m.indices().len() as u32, 0, 0..1);
                }
            }
            // ② masked 双纹理——组内 render_order 排序（索引序 ≠ 绘制序：
            // 高光 = 组内 ro 最大者，索引序画 = 高光垫底被虹膜盖死）
            {
                let cs_view = cs.texture_view();
                let mut rp = begin_pass(&mut enc, "pygamest_l2d_masked", &cs_view);
                rp.set_pipeline(&pl.pipeline);
                let mut sorted = plan.group.masked_ids.clone();
                sorted.sort_by_key(|&di| {
                    model.runtime().meshes().get(di).map(|m| m.render_order()).unwrap_or(i32::MAX)
                });
                for &di in &sorted {
                    let Some(m) = model.runtime().meshes().get(di) else {
                        continue;
                    };
                    if m.indices().is_empty() {
                        continue;
                    }
                    let Some(tex) = tex_list.get(m.texture_index() as usize) else {
                        continue;
                    };
                    let (vbo, ibo) =
                        upload_mesh(&gpu.device, &clip_mesh_verts(m, plan), m.indices());
                    let color_view = tex.texture_view();
                    let bg = make_bind_group(&gpu.device, pl, &idm_buf, &color_view, &mask_view);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.set_vertex_buffer(0, vbo.slice(..));
                    rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint16);
                    rp.draw_indexed(0..m.indices().len() as u32, 0, 0..1);
                }
            }
        }

        // ── ③ 整模型段（产物面透明重画；masked 组以 C quad 组首绘位合成）──
        {
            let out_view = self.surface.texture_view();
            let mut rp = begin_pass(&mut enc, "pygamest_l2d_model", &out_view);
            rp.set_pipeline(&pl.pipeline);
            let mut ordered: Vec<_> = model.runtime().meshes().iter().enumerate().collect();
            ordered.sort_by_key(|(_, m)| m.render_order());
            let mut composited: Vec<usize> = Vec::new();
            for (di, m) in ordered {
                if !m.masks().is_empty() {
                    let Some((gi, plan)) = plans
                        .iter()
                        .enumerate()
                        .find(|(_, p)| p.group.masked_ids.contains(&di))
                    else {
                        continue;
                    };
                    if composited.contains(&gi) {
                        continue;
                    }
                    composited.push(gi);
                    let Some(cs) = color_surfs.get(gi) else {
                        continue;
                    };
                    // C 面 quad：span 角点（归一域，MVP 映射）+ clip uv
                    //（margin 域直读——mask 段 NDC 烘焙时已预补偿 y 翻转）
                    let quad = [
                        plan.min_x, plan.min_y, plan.max_x, plan.min_y, plan.max_x, plan.max_y,
                        plan.min_x, plan.max_y,
                    ];
                    let quv = [
                        plan.min_x * plan.k_x + plan.c_x, plan.min_y * plan.k_y + plan.c_y,
                        plan.max_x * plan.k_x + plan.c_x, plan.min_y * plan.k_y + plan.c_y,
                        plan.max_x * plan.k_x + plan.c_x, plan.max_y * plan.k_y + plan.c_y,
                        plan.min_x * plan.k_x + plan.c_x, plan.max_y * plan.k_y + plan.c_y,
                    ];
                    let (vbo, ibo) = upload_mesh(
                        &gpu.device,
                        &[
                            quad[0], quad[1], quv[0], quv[1], 0.0, 0.0, quad[2], quad[3],
                            quv[2], quv[3], 0.0, 0.0, quad[4], quad[5], quv[4], quv[5], 0.0,
                            0.0, quad[6], quad[7], quv[6], quv[7], 0.0, 0.0,
                        ],
                        &[0u16, 1, 2, 0, 2, 3],
                    );
                    let color_view = cs.texture_view();
                    let bg =
                        make_bind_group(&gpu.device, pl, &content_buf, &color_view, &identity_view);
                    rp.set_bind_group(0, &bg, &[]);
                    rp.set_vertex_buffer(0, vbo.slice(..));
                    rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint16);
                    rp.draw_indexed(0..6, 0, 0..1);
                    continue;
                }
                if m.indices().is_empty() {
                    continue;
                }
                let Some(tex) = tex_list.get(m.texture_index() as usize) else {
                    continue;
                };
                let (vbo, ibo) = upload_mesh(&gpu.device, &plain_mesh_verts(m), m.indices());
                let color_view = tex.texture_view();
                let bg = make_bind_group(&gpu.device, pl, &content_buf, &color_view, &identity_view);
                rp.set_bind_group(0, &bg, &[]);
                rp.set_vertex_buffer(0, vbo.slice(..));
                rp.set_index_buffer(ibo.slice(..), wgpu::IndexFormat::Uint16);
                rp.draw_indexed(0..m.indices().len() as u32, 0, 0..1);
            }
        }

        gpu.queue.submit([enc.finish()]);
        Ok(())
    }
}
