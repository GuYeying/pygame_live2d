//! Live2D 蒙版组解析（Cubism 蒙版纹理方案的纯逻辑面）。
//!
//! Cubism 蒙版语义：masked drawable 的 `masks` 列表 = 用作裁剪形状的
//! drawable **索引**表（引用同一模型的其它 drawable）。相同 mask 集合
//! 的 masked drawable 归入同一**蒙版组**——组内共享一张蒙版纹理
//! （官方 MaskBuffer 惯例：组包围盒 → 0..1 蒙版 UV 空间的正交映射）。
//!
//! 渲染流程（绑定层编排）：
//! 1. 蒙版 pass：组内 mask 来源 drawable 以纯白画入组蒙版缓冲
//!    （顶点 CPU 预乘 [`mask_clip_transform`]）
//! 2. masked pass：masked drawable 以双纹理（颜色 × 蒙版 alpha）绘制
//!    （每顶点 uv_mask = [`mask_clip_transform`] 预乘）

use crate::moc3::drawable::Moc3DrawableMesh;

/// 一个蒙版组：共享同一蒙版缓冲的一批 drawable。
#[derive(Debug, Clone, PartialEq)]
pub struct MaskGroup {
    /// 组内 mask 来源 drawable 索引（升序、去重）。
    pub mask_ids: Vec<usize>,
    /// 使用该组的 masked drawable 索引（按引用顺序）。
    pub masked_ids: Vec<usize>,
}

/// 按 mask 集合把 drawable 分组（无 mask 的 drawable 不入组）。
///
/// 组 key = masks 列表排序去重后的集合（Cubism 允许一 drawable 多 mask
/// ——多个形状取并集裁剪）。
pub fn mask_groups(meshes: &[Moc3DrawableMesh]) -> Vec<MaskGroup> {
    let mut groups: Vec<MaskGroup> = Vec::new();
    for (i, m) in meshes.iter().enumerate() {
        let masks = m.masks();
        if masks.is_empty() {
            continue;
        }
        let mut key: Vec<usize> = masks.iter().map(|&v| v as usize).collect();
        key.sort_unstable();
        key.dedup();
        if let Some(g) = groups
            .iter_mut()
            .find(|g| g.mask_ids == key)
        {
            g.masked_ids.push(i);
        } else {
            groups.push(MaskGroup {
                mask_ids: key,
                masked_ids: vec![i],
            });
        }
    }
    groups
}

/// 蒙版 clip 仿射（模型空间 → 蒙版 UV 空间 0..1）：
/// 返回 `(sx, sy, tx, ty)`——`uv = (x*sx + tx, y*sy + ty)`。
///
/// `span` = 组联合包围盒（模型空间），外扩 `margin`（比例，0.25 =
/// 官方惯例量级——蒙版边缘留白防采样裁切）。恒等兜底：空 span 时
/// 返回全屏映射（退化安全）。
pub fn mask_clip_transform(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    margin: f32,
) -> (f32, f32, f32, f32) {
    // 外扩 margin 后的盒子宽度
    let span_x = (max_x - min_x).max(1e-6) * (1.0 + margin * 2.0);
    let span_y = (max_y - min_y).max(1e-6) * (1.0 + margin * 2.0);
    // u(x) = (x - min) * k + margin/(1+2m) —— 斜率 + 截距形态
    let k_x = 1.0 / span_x;
    let k_y = 1.0 / span_y;
    let m_frac = margin / (1.0 + margin * 2.0);
    let c_x = -min_x * k_x + m_frac;
    let c_y = -min_y * k_y + m_frac;
    (k_x, k_y, c_x, c_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moc3::drawable::{Moc3DrawableMesh, Moc3DrawableVertex};

    fn mesh(render_order: i32, masks: Vec<i32>) -> Moc3DrawableMesh {
        Moc3DrawableMesh::from_parts_with_render_order(
            0,
            0,
            1.0,
            render_order as f32,
            render_order,
            vec![
                Moc3DrawableVertex::new([-1.0, -1.0], [0.0, 0.0]),
                Moc3DrawableVertex::new([1.0, -1.0], [1.0, 0.0]),
                Moc3DrawableVertex::new([1.0, 1.0], [1.0, 1.0]),
                Moc3DrawableVertex::new([-1.0, 1.0], [0.0, 1.0]),
            ],
            vec![0, 1, 2, 2, 3, 0],
            masks,
        )
    }

    #[test]
    fn groups_share_mask_sets() {
        let meshes = vec![mesh(0, vec![]), mesh(1, vec![0]), mesh(2, vec![0]), mesh(3, vec![])];
        let groups = mask_groups(&meshes);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].mask_ids, vec![0]);
        assert_eq!(groups[0].masked_ids, vec![1, 2]);
    }

    #[test]
    fn distinct_masks_split_groups() {
        let meshes = vec![mesh(0, vec![]), mesh(1, vec![0]), mesh(2, vec![1])];
        let groups = mask_groups(&meshes);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn clip_transform_maps_bounds_to_unit() {
        let (sx, sy, tx, ty) = mask_clip_transform(10.0, -20.0, 30.0, 40.0, 0.25);
        let u = 10.0 * sx + tx;
        let v = -20.0 * sy + ty;
        assert!((u - 0.1667).abs() < 1e-3, "min 角映射到 margin={}", u);
        assert!((v - 0.1667).abs() < 1e-3);
        let u2 = 30.0 * sx + tx;
        let v2 = 40.0 * sy + ty;
        assert!((u2 - 0.8333).abs() < 1e-3);
        assert!((v2 - 0.8333).abs() < 1e-3);
    }
}
