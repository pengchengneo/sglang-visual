# SGLang Optimization Visual Learning Platform — Design Spec

## Overview

将现有的 SGLang 推理配置可视化工具升级为面向 AI/ML 工程师的**优化学习平台**。通过交互式探索 + 关键流程动画，帮助已具备 Transformer 和推理基础的工程师深入理解 SGLang 的核心优化实现。

## Target Audience

AI/ML 工程师：已了解 Transformer 架构和推理基础，想深入理解 SGLang 具体优化实现细节。不需要科普基础概念，直接聚焦优化原理和实现。

## Priority

1. **P0（最高优先）**：KV Cache（RadixAttention）、请求调度（Continuous Batching）
2. **P1**：架构总览、并行策略
3. **P2**：算子优化、投机解码

---

## 1. Navigation & Page Structure

### 整体方案：顶部主题导航 + 侧边参数控制

- 顶部水平导航栏列出所有主题页面
- 侧边栏保留为参数控制面板，根据当前 Tab 动态展示相关控制
- 引入 `react-router` 支持 URL 路由（方便分享特定主题链接）

### 导航 Tabs

| Tab | 内容 | 状态 |
|-----|------|------|
| 架构总览 | SGLang 运行时全景 + 请求生命周期 | 基于现有 Control Plane 扩展 |
| 并行策略 | TP/DP/PP/EP/DP Attention 对比 | 基于现有 Compute Plane 扩展 |
| KV Cache | RadixAttention 树 + 内存分配 + 淘汰策略 | **新建 (P0)** |
| 请求调度 | Continuous Batching + Chunked Prefill 动画 | **新建 (P0)** |
| 算子优化 | FlashInfer / CUDA Graph / 量化 | 新建 (P2) |
| 投机解码 | EAGLE 推测-验证流程动画 | 新建 (P2) |
| Compute | 现有计算面（模型架构 + GPU 内存） | 保留原样 |
| Control | 现有控制面（运行时数据流） | 保留原样 |

### 技术决策

- 引入 `react-router-dom` 作为唯一新运行时依赖
- 路由结构：`/`, `/architecture`, `/parallelism`, `/kv-cache`, `/scheduling`, `/operators`, `/speculative`, `/compute`, `/control`
- 侧边栏组件按页面动态渲染不同的参数控制集合
- 现有 Compute/Control 页面代码保持不变，仅包装为路由组件

---

## 2. KV Cache Page (P0)

### 2.1 RadixAttention Prefix Tree

**交互式树形结构：**
- 以 trie 树形式展示 prompt token 的共享前缀
- 节点表示 token 序列片段，边表示 KV cache block
- 颜色编码：
  - 绿色 = 活跃引用（正在被请求使用）
  - 蓝色 = 缓存中（可复用）
  - 灰色 = 可淘汰（引用计数为 0）
- 悬停节点显示：token 内容、引用计数、内存占用

**动画演示：**
- "新请求到达"按钮：动画展示请求沿树查找最长前缀匹配
  - 匹配部分节点逐个高亮（绿色脉冲）
  - 未匹配部分动画生长出新节点
  - 引用计数 +1 动画
- "请求完成"按钮：展示引用计数减少、LRU 淘汰过程
  - 引用计数降为 0 的节点变灰
  - 当 cache 容量不足时，灰色节点从叶子开始淘汰（淡出动画）

**预设场景（一键切换）：**
1. 多轮对话共享 system prompt — 展示 system prompt 前缀被多个对话复用
2. Few-shot 共享 examples — 展示不同请求共享相同的 few-shot 前缀
3. 无共享前缀 — 展示完全独立的请求，无 cache 命中
4. 混合场景 — 部分请求共享前缀，部分不共享

**侧边栏控制：**
- Cache 容量（block 数量）
- 淘汰策略（LRU）
- 请求序列文本编辑器（用户自定义请求内容观察匹配）

### 2.2 Token-Level Memory Allocation

**GPU 内存池可视化：**
- 矩形网格表示 KV cache block 池
- 每个 block 着色对应其在 trie 树中的节点
- 悬停 block 高亮对应的树节点（双向关联）
- 展示 page-based 内存管理：分配/释放/引用计数

**参数对比：**
- MHA vs MLA 模式下 per-token KV bytes 对比
- 拖动 cache 容量滑块实时观察可用 slot 数变化

---

## 3. Request Scheduling Page (P0)

### 3.1 Continuous Batching Animation

**双队列可视化：**
- 左侧 Waiting Queue：请求按到达时间排列
- 右侧 Running Batch：当前正在执行的请求批次
- 请求用带颜色的条形表示，长度代表 token 数（prefill 长度 + 已生成 decode 长度）

**核心动画流程：**
- 请求持续进入 Waiting Queue（可调到达速率）
- 每个 iteration：Scheduler 从 Waiting Queue 取请求加入 Running Batch
- Running Batch 中每个请求逐 token 生成（条形逐步增长）
- 请求完成后立即移出 Running Batch，空位被新请求填入
- 支持播放/暂停/步进/调速控制

**关键对比（一键切换）：**
- **Static Batching**：整批请求必须全部完成后才能处理下一批，短请求等待长请求
- **Continuous Batching**：完成即替换，GPU 利用率持续饱满

**实时指标：**
- GPU 利用率百分比
- 平均请求等待时间
- 吞吐量（tokens/s）
- 已完成请求数

### 3.2 Chunked Prefill

**分块执行动画：**
- 长 prefill 请求被切分为固定大小 chunk
- 时序图展示 chunk 与 decode token 交替执行
- 每个时间步高亮当前执行的内容

**对比模式：**
- **无 Chunked Prefill**：长 prefill 独占 GPU，decode 请求被阻塞（延迟抖动大）
- **有 Chunked Prefill**：prefill chunk 与 decode 交替，decode 延迟稳定

**侧边栏控制：**
- Chunk size（2K / 4K / 8K / 16K）
- Prefill 请求长度
- Decode 请求数量

### 3.3 Scheduling Policy Comparison

**三列并排可视化：**
- FCFS / LPM / DFS-Weight 三种策略同时运行
- 同一组请求序列输入
- 每列展示各自的调度顺序和 batch 组成
- 底部对比指标：吞吐量、平均延迟、尾延迟

---

## 4. Architecture Overview Page (P1)

### 4.1 Request Lifecycle

**端到端数据流动画：**
- 路径：`Client → TokenizerManager → Scheduler → TpModelWorker → ModelRunner → GPU → Detokenizer → Client`
- 基于现有 Control Plane 扩展

**步进模式：**
- "下一步"按钮：请求逐阶段前进
- 每个阶段高亮并展示：
  - 组件职责说明
  - 关键数据变化（如 Tokenizer：string → token ids，Scheduler：单请求 → batch）
- 支持自动播放模式（可调速）

**DP 模式切换：**
- DP Attention / 传统 DP 模式复用现有逻辑
- 增加请求流动动画展示 DataParallelController 分发过程

### 4.2 Component Relationship

- 悬停任意组件高亮其上下游依赖
- 点击组件展开详细说明面板（功能、关键参数、优化点）

---

## 5. Parallelism Strategies Page (P1)

### 顶部切换栏：TP / DP / PP / EP / DP Attention

**TP（张量并行）：**
- 复用现有矩阵分区可视化
- 增加通信动画：AllReduce 时数据在 GPU 间流动
- 展示完整流程：Column Parallel → compute → AllReduce → Row Parallel

**PP（流水线并行）：**
- 多 stage 流水线时序图（甘特图风格）
- 展示 micro-batch 如何填充 pipeline bubble
- 动画对比：1 micro-batch（大 bubble）vs 多 micro-batch（小 bubble）

**EP（专家并行）：**
- MoE 层 token 经 Router 分发到不同 GPU 上的 expert
- 动画：AllToAll dispatch → expert compute → AllToAll combine

**DP / DP Attention：**
- 数据分发到多组 worker 的过程
- DP Attention 模式下 attention 子分组 vs 传统 DP 独立副本对比

**通信开销对比（底部）：**
- 统一条形图：不同并行策略在当前模型配置下的通信字节数

---

## 6. Operator Optimization Page (P2)

### 6.1 FlashInfer Attention Kernel

- 对比动画：标准 attention（生成完整 n×n attention matrix）vs FlashAttention（分块计算，不物化完整矩阵）
- 内存曲线：O(n²) vs O(n) 显存随序列长度变化
- 交互：拖动序列长度滑块实时观察差异

### 6.2 CUDA Graph

- 时序图对比：无 CUDA Graph（每次 iteration 逐个 launch kernel）vs 有 CUDA Graph（一次录制，replay 执行）
- 展示 kernel launch overhead 减少量

### 6.3 Quantization

- 权重内存对比柱状图：FP16 / FP8 / INT4
- 复用现有 GPU Memory Panel 计算逻辑
- 动态展示同一模型不同量化下的显存占用

---

## 7. Speculative Decoding Page (P2)

### 7.1 Speculate-Verify Flow Animation

- Draft Model 快速生成 N 个候选 token（绿色标记）
- Target Model 一次 forward pass 并行验证
- 动画：匹配 token 保留（亮绿），不匹配处截断，从 target 取正确 token（黄色标记）
- Token Tree 可视化：EAGLE 的树状推测结构，多分支并行验证

### 7.2 Performance Gain

- 交互对比：无投机（逐 token 生成，每步一个 forward）vs 有投机（跳跃式前进）
- 参数控制：draft token 数量、acceptance rate 滑块
- 实时展示加速比变化曲线

---

## 8. Technical Architecture

### 新增依赖
- `react-router-dom` — URL 路由（唯一新运行时依赖）

### 文件组织
```
src/
  pages/                        # 新增：页面级组件
    ArchitecturePage.tsx
    ParallelismPage.tsx
    KVCachePage.tsx              # P0
    SchedulingPage.tsx           # P0
    OperatorsPage.tsx
    SpeculativePage.tsx
    ComputePlanePage.tsx         # 包装现有 Compute Plane
    ControlPlanePage.tsx         # 包装现有 Control Plane
  components/
    navigation/                 # 新增：导航组件
      TopNav.tsx
      RouteConfig.tsx
    kv-cache/                   # 新增 P0
      RadixTree.tsx             # 前缀树 SVG 可视化
      RadixTreeAnimation.ts     # 动画控制逻辑
      MemoryPool.tsx            # 内存池网格
      ScenarioPresets.ts        # 预设场景数据
    scheduling/                 # 新增 P0
      ContinuousBatching.tsx    # 双队列动画
      ChunkedPrefill.tsx        # Chunked Prefill 时序图
      PolicyComparison.tsx      # 三策略并排对比
      SchedulingEngine.ts       # 模拟引擎（调度逻辑）
      AnimationControls.tsx     # 播放/暂停/步进/调速
    parallelism/                # 新增
      TPAnimation.tsx
      PPTimeline.tsx
      EPDispatch.tsx
      CommOverhead.tsx
    operators/                  # 新增
      FlashAttentionCompare.tsx
      CUDAGraphTimeline.tsx
      QuantizationCompare.tsx
    speculative/                # 新增
      SpecVerifyFlow.tsx
      TokenTree.tsx
      PerformanceGain.tsx
    shared/                     # 新增：跨页面复用
      AnimationPlayer.tsx       # 通用动画播放控制器
      ComparisonToggle.tsx      # A/B 对比切换组件
      MetricsPanel.tsx          # 实时指标展示面板
```

### 动画技术方案
- 继续使用纯 SVG + CSS transitions（与现有项目一致）
- 动画状态通过 `requestAnimationFrame` + React state 驱动
- 复杂时序动画使用自定义 `useAnimation` hook 管理帧、播放/暂停/步进
- 不引入额外动画库，保持零运行时依赖的极简风格

### 数据模型
- 动画场景数据以 TypeScript 常量定义在各模块内
- 不需要后端支持，所有模拟逻辑在前端完成
- 调度模拟引擎 `SchedulingEngine.ts` 为纯函数，接收配置输出帧序列

### 状态管理
- 继续使用 React useState/useMemo（与现有项目一致）
- 全局参数（模型选择、TP/PP/EP 等）通过 React Context 在路由间共享
- 页面级动画状态保持在各页面组件内部
