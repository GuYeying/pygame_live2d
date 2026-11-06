# pygame_live2d API（随库维护文档）

> **归档声明**（2026-11-06）：Live2D/Cubism 因法律风险归档为**独立库**
> ——本仓（pygame_live2d）自持实现 + 渲染管线 + 文档；pygame-rs 侧仅存
> feature 门控的薄绑定（`features = ["live2d"]`），随库演进、不随
> pygame-rs 主线维护。本文档为 API 权威来源。

上游 = [mocari](https://github.com/Eatgrapes/Mocari) v0.4.0（MIT，纯
Rust Cubism 兼容运行时；上游仓已删，vendored 溯源见本仓 README）。
**独立顶层 Python 模块**（`import pygame_live2d` 直达）。

```{note}
[接口风格] 上游词汇原生化——canvas / drawable / parameter（Live2D
惯例即文档，不 pygame 化）。

[坐标系] 归一化顶点域（y 向上）；脚本侧用 rt.draw 的 x/y/scale 定位。

[sprite 工作流] 用户创建 Surface → `rt.draw(surface, x, y, scale)`
全管线自理（蒙版纹理方案 GPU 化——masked 双纹理、体素化外的完整
Cubism 蒙版语义）→ 产物 = 全透明底人物图 → gfxdraw.draw_surface 上屏。

[双形态] `Live2DModel` = 静态数据面；`Live2DRuntime` = 参数驱动面。
```

## 函数

### load_model(path) -> Live2DModel

`.model3.json` 入口（moc3/纹理相对路径按其目录解析）。缺文件/坏内容
= ValueError（不崩溃——严格模式同理，Live2D 面无静默兜底定约）。

### load_runtime(path) -> Live2DRuntime

参数驱动形态加载。

### load_motion(path) -> Live2DMotion

`.motion3.json` 加载（缺文件/坏内容 = ValueError）。

### motion_player(motion, looping=None) -> Live2DMotionPlayer

播放器构造（looping 缺省跟随 motion 自带 Loop 元数据）。

## 类

### Live2DModel —— 静态数据面

- `canvas_size() -> (w, h)`：模型画布尺寸（Live2D 原生单位）
- `mesh_count() -> int`：drawable 网格数
- `mesh(i) -> (texture_index, opacity, positions, uvs, indices)`：
  网格数据流（positions/uvs 扁平 f32；indices u16）
- `texture_count() -> int`
- `texture_info(i) -> (w, h)`
- `texture_rgba(i) -> bytes`：纹理原始 RGBA 字节（大对象——调试用）

### Live2DRuntime —— 参数驱动面

- `canvas_size() -> (w, h)`
- `parameter_ids() -> list[str]`：可驱动参数 id 表
- `set_parameter_normalized(id, v) -> bool`：归一参数驱动（0..1；id
  不存在返回 False）
- `update_meshes()`：参数 → 网格重建（每帧参数更新后调用）
- `mesh_count() -> int` / `mesh(i)`：数据面同 Live2DModel.mesh
- **`draw(surface, x=0.0, y=0.0, scale=0.9)`**：全管线自理渲染——
  蒙版/masked/整模型三段落 surface（产物=全透明底人物图）。
  坐标定约（与 pygamest 中心系同构，全 0-1 相对产物面）：x/y=内容
  中心位置（0=面中心，±1=±半幅，y 向上）；scale=内容高/面高（>1
  溢出裁边=特写）。

### Live2DMotion

- `duration() -> float`：motion 时长（秒）

### Live2DMotionPlayer

- `tick(dt)`：推进时间轴（秒）
- `apply(model)`：motion 参数写进 runtime（写后 update_meshes）
- `is_finished() -> bool`：非循环 motion 是否播完
- `restart()`：回零
- `time() -> float`：当前时间轴
- `set_weight(w)`：混合权重（0..1）

## panic 注意事项

- `draw` 内 GPU 会话缺失（未 set_mode）= RuntimeError（不 panic）
- 坏模型/坏 motion = ValueError（加载期，严格模式无静默兜底）
- 蒙版渲染路径的 GPU 资源（Mg/C 面池、双纹理管线）惰性创建于首次
  draw——无 panic 面

## 上游限制

- rapier（物理侧）无关；本模块无已知 panic 面
- 体素化的物理侧限制见 pygame_physics 文档（与 Live2D 无关，此处无）

## 资产

- Hiyori 官方样例模型 vendor 至本仓 `assets/models/Hiyori/`
  （Live2D Free Material License，经 neocari 分发副本）
- demo：`pygame-rs-dev/examples/live2d_render_demo.py`、
  `live2d_full_test.py`（22 断言）、`live2d_shot.py`、`live2d_closeup.py`
