# pygame_live2d

pygamest 生态 Live2D/Cubism 运行时适配层——**mocari v0.4.0 vendored**
（MIT，Copyright 2026 Eatgrapes）。

## 溯源

- 上游 = [Eatgrapes/Mocari](https://github.com/Eatgrapes/Mocari)（MIT，
  纯 Rust Live2D/Cubism 兼容运行时；**上游仓已删除**）
- 本仓源 = LunaMate（TR0MXI）内嵌副本 `crates/lunamate-mocari`（MIT 同
  许可再分发），恢复其 upstream 形态的 `wgpu 30` 可选依赖（LunaMate
  fork 曾替换为 gpui_wgpu——zed 生态专属，此处还原）
- `LUNAMATE.md` / `Cargo.toml.upstream` = 溯源留档

## 面

- 加载：`assets::load_model`（静态：meshes/textures/canvas）+
  `assets::load_model_runtime`（参数驱动：set_parameter_normalized →
  update_meshes → 网格重建）
- 动作/表情：`motion::MotionPlayer` / `expression::ExpressionManager`
- 渲染：`render::common`（后端无关 draw order/裁剪/顶点辅助）+
  `wgpu` feature（可选，原生 wgpu 30）
- 自带测试 54 + doctest 1（零资产——错误路径内联 JSON）

## pygame-rs 绑定

`pygamest` crate feature `live2d` → Python 面
`import pygame_live2d`（load_model / load_runtime / Live2DModel /
Live2DRuntime——mesh 数据流出口，效仿 pygame_spine 范式）。
