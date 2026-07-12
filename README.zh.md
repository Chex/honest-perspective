# honest-perspective

> 不虚构画面内容的透视矫正——宁可不动，也不瞎猜。

[English →](README.md)

一个用于矫正照片透视畸变的 prototype。和 Snapseed 的"透视"功能定位类似，但有两条刻意"诚实"的约束：

- **只做裁剪、不做填充**——warp 结果只保留原图有效覆盖范围。几何变换会正常重采样，但不会用黑边、内容感知合成或 AI 生成的"假背景"填补缺失区域。
- **错误的矫正不如不矫正**——自动模式把"无需调整"当作一等候选，证据不够充分时就选它。

特别适合处理建筑物照片：拍摄时仰拍/俯拍带来的"楼往里收"可以扶正。

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

推荐 Python **3.11**。Windows 下请改用 `.venv\Scripts\activate` 激活环境。

如果跑起来报 `cv2.createLineSegmentDetector` 不存在，换装 `opencv-contrib-python` 即可。

## 基本用法

```bash
python fix.py input.jpg output.jpg
```

默认只矫正**竖直方向**（像 Lightroom 的 Auto Vertical），保留水平线的自然透视。这对建筑物照片通常是最好的效果。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode {vertical,horizontal,both}` | `vertical` | 矫正方向 |
| `--strength FLOAT` | `1.0` | 矫正强度，`0.0`=完全不动，`1.0`=完全矫正，中间值=插值 |
| `--keep-aspect / --no-keep-aspect` | 开 | 是否按原图比例居中再裁一刀（保持画幅比例） |

### 三种 mode 的区别

- **`vertical`**（默认）：只让竖向线变成真正的竖直，水平线维持透视。**最稳**，适合建筑物。
- **`horizontal`**：反过来，只让水平线变水平。适合从侧面拍的物体、桌面摆拍。
- **`both`**：两个方向都矫正，变成完全的"正交投影"风格。需要画面里两组消失点都很明显，否则容易过矫（典型症状：画面被强行拉成奇怪的梯形，最后裁剪损失很大）。

### `--strength` 的用法

如果觉得默认 `1.0` 矫正得"过头了"（楼变得太规整反而不自然），可以试 `0.6` 或 `0.7`。插值发生在旋转流形上，中间值仍然是物理合法的相机旋转。

## Web app

本机 LAN 上跑一个 web 编辑器：

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

- 桌面：http://127.0.0.1:8000/
- 同 WLAN 的 iPhone / iPad：替换成本机的局域网 IP，例如 `http://192.168.x.x:8000/`

支持 JPG / PNG / **HEIC**（iPhone "高效率"格式，服务端通过 `pillow-heif` 解码；Safari 17+ 能原生预览，其他浏览器预览失败但服务端处理无问题）。

### 交互

- 拖入照片 / 点击选择 → 服务端一次提取线段和 VP，生成无需调整/垂直/水平/全面四个候选；只有线条真的改善且源像素保留足够多时才自动应用
- 默认显示自动选择结果；"无需调整"是一等候选，也可以从模式菜单主动恢复
- **拖拽**：水平位移只调整相机偏航（yaw），垂直位移只调整俯仰（pitch）；单指拖拽不会主动加入绕光轴的滚转（roll）增量，同向同距离的手势在画面任意位置含义相同
- **按住"原图"按钮**：临时显示未校正源图，抬手回到校正后；和拖拽互斥
- **保存**：桌面端下载 JPG。iOS 在 HTTPS 下可使用 Web Share API 并选择"存储到照片"；上面普通 HTTP 的局域网地址会退化成下载，Safari 通常会把文件存入"文件"。EXIF（拍摄时间、相机、GPS、镜头）、ICC profile 全保留
- 预览图在浏览器里 downscale 到 ~`max(viewport) × DPR`（封顶 2500px），保证大图 matrix3d 60fps；保存仍是原图全分辨率

### 手动校正背后的数学（参考）

Web app 的手动模式是个"严格相机旋转"模型。权威状态不再是四个可任意变形的角点，而是：

1. `intrinsics`：优先从 EXIF `FocalLengthIn35mmFilm` 计算 K，缺失时才 fallback 到 `max(w,h)`
2. `rotation`：一个满足 `RᵀR=I, det(R)=1` 的 3×3 SO(3) 矩阵
3. `crop`：在旋转后源图四边形内的同画幅裁剪，只允许统一缩放
4. 最终 warp = `S_crop · K · R · K⁻¹`；模型没有独立的任意剪切或非均匀缩放自由度，所有投影变化都来自物理合法的相机旋转

手动拖拽从按下时姿态计算 `yaw = atan(dx/fx)`、`pitch = -atan(dy/fy)`，再在 SO(3) 上合成。两维手势只控制两维相机轴，不再为了让某个角点严格跟手而混入滚转；从画面任意位置开始的同向、同距离手势因此具有相同含义。

前端 `webapp/geometry.js` 和后端 `geometry.py` 是同一份几何合同；随机旋转 contract tests（`tests/test_geometry.py`）会逐项比较两边的 matrix、crop 和角度。保存接口直接发送 correction state，旧 corners API 只作为兼容路径保留。

**手动路径的两道安全阀**：

| 常数 | 当前值 | 触发时 |
|---|---|---|
| `MAX_ROT_RAD` | 35° | correction state 的真实轴角超过 35° → 拒绝该次 drag 更新 |
| `MIN_PROJECTIVE_W_RATIO` | 0.22 | 投影 w 分量 min/max 比 < 0.22（接近退化） → 拒绝 |

`window.__rejectStats` 实时统计每个阈值触发次数，控制台可读：用一阵子之后看分布能反过来 calibrate 这几个数字。

## 自动模式的工作原理

1. **找线段**：`cv2.createLineSegmentDetector` 提取所有显著线段
2. **按角度聚类**：把线段分成"接近竖直"和"接近水平"两组（±25° 容差）
3. **RANSAC 找消失点**：每组算出对应的消失点
4. **构造单应矩阵**：让消失点跑到无穷远的方向
5. **warp 图像**：此时画面边缘是个不规则四边形
6. **找最大内接矩形**——这就是"只裁剪、不填充"的关键
7. **（可选）按原图比例再居中裁一刀**

候选选择跟踪同一批**源**线段经过候选旋转后的方向（不对 warp 结果重新检测，避免重采样噪声）；改善少于 0.25° 时"无需调整"胜出。

### 算法的能力边界（这个工具对什么场景不工作）

当前的自动检测本质上是一个**线段对齐器**，隐含假设是 **"画面里的主要线段方向 = 世界的正交轴"**。按"出错的危险程度"排：

- **建筑、文档、屏幕、招牌、桌面**（人造强直线 + 正交结构）：✅ 算法的甜区
- **金字塔 / 帐篷 / 圆顶**（无竖线，但人眼能感知"指向天的方向"）：算法看不到"隐含的竖直"，要么 VP 找不到，要么只矫正横向。失败模式相对**安全**
- **杯子 / 雕塑 / 曲面物体**（基本没直线）：LSD 几乎找不到线段，算法返回原图。**最安全**的失败
- **树林 / 草丛 / 人群**（多直线但弱平行）：**最危险**——RANSAC 找到某个勉强成立的 VP，算法**自信地施加方向可能完全错误的矫正**。这正是"无需调整"必须是一等候选的原因

### Gravity 与视觉估计

iPhone 拍照时陀螺仪和加速度计是开着的——**HEIC 的 Apple Maker Notes 区里保存了拍摄瞬间的重力向量**。这意味着：

> **iPhone 已经知道"哪边是下"了，跟图像内容无关。**

项目已经能解析这个向量并按设备姿态映射到图像坐标，同时用 acceleration norm 排除明显运动污染。但它是单次总加速度，不是无条件准确的重力真值。

| 维度 | 当前 LSD+RANSAC | 陀螺仪先验 |
|---|---|---|
| 建筑 | ✅ 可工作 | ✅ 提供独立的方向估计 |
| 金字塔 / 杯子 / 雕塑 | ❌ 无解 | ✅ 无需可见直线也能估计倾斜 |
| 树林 / 无明显结构 | ⚠️ 自信乱矫正 | ✅ 不依赖线段 |
| 镜头畸变 | 不补 | 不补 |
| 截图 / 非相机来源（无 EXIF）| 能跑 | 没数据 → fallback 到 LSD |

GeoCalib spike 对 9 张样本做了独立交叉验证：多数样本与 Apple gravity 相差不超过 1.7°，但有两张即使 norm 正常仍分别冲突 5.6° 和 9.4°。因此下一步不是"gravity 覆盖视觉 VP"，而是让 Apple、视觉 VP、GeoCalib 各自产生带置信度的提案，冲突时默认不动。详见 `spike_geocalib/FINDINGS.md`。

各 `spike_*/FINDINGS.md` 记录了这些决定背后的研究过程，包括死胡同。

## 测试

```bash
python -m unittest discover -s tests -v
```

浏览器与后端的几何合同测试需要安装 Node.js。

## 关于色彩

iPhone 拍的 JPG 通常是 **Display P3** 色域，内嵌 ICC profile。本工具用 Pillow 做 IO，**原样保留 ICC profile 和 EXIF**，避免色彩在处理后变"褪色"。

## 已知限制

- `--mode both` 在画面元素复杂时容易把杂线当成水平线，结果不稳；遇到问题先 fallback 到 `vertical`
- 如果消失点完全找不到（比如纯自然风景没有直线），工具会直接保存原图
- **EXIF Orientation 仍只完整覆盖 Orientation=1**：其他取值需要补 pixel transpose、gravity 坐标变换和保存时清理 orientation tag 的完整测试
- **HDR 不保留**：iPhone 的 HDR 照片在 HEIC 容器里是"主图 + gainmap 辅助层"。当前 pipeline 解码时只取主图，输出 JPEG 也没地方放 gainmap，所以保存后 iOS Photos 把它当 SDR 渲染。
  - 上游卡点：libheif / pillow-heif / libvips 现在都只能**读** Apple gainmap，**写**还没支持。Google libultrahdr 路线图上 2026 加 HEIC gainmap 支持，估计 libvips 跟进后这块能解。
  - Apple 官方的 HDR 编辑 API（`CIContext.writeHEIFRepresentation` + `kCGImageAuxiliaryDataTypeISOGainMap`）只在 Swift/Obj-C 里能调。要做"保留 HDR"，路径基本只剩 iOS 原生 Photo Editing Extension。

## TODO / 后续

- [ ] **Apple / visual VP / GeoCalib 三方提案与置信度融合**
- [ ] **补 EXIF Orientation 非 1 的完整像素与 metadata 测试**
- [ ] HDR gainmap 保留：等 libvips / libultrahdr 加 HEIC gainmap 写入（2026 路线图）
- [ ] 检测多组消失点（提升复杂场景鲁棒性）
- [ ] 用 M-LSD 或 DeepLSD 替代经典 LSD
- [ ] Mac Photos / iOS Shortcut 集成
- [ ] iOS 原生 Photo Editing Extension（"在相册里原地处理" + HDR 自动保留的唯一官方路径）

## 许可

[MIT](LICENSE)
