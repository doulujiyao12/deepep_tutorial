# DeepEP V1 Normal 跨节点通信：SM、Warp、Thread 与 Channel 分工详解

> 本文只聚焦 DeepEP V1 legacy **非独立 low-latency API** 的 normal/high-throughput 多节点 Dispatch 路径，即 `csrc/kernels/legacy/internode.cu` 中的 `dispatch` kernel。
>
> 目标是回答：channel 怎样对应 SM；每个“SM”上的 warp 怎样分工；warp 内 32 个 thread/lane 怎样分工；节点 0 的 GPU5 发送后，节点 1 GPU5 的 Forward SM 如何与节点 1 各 GPU 的 Send/Receive SM 交互。
>
> 对应代码版本：`f99f06868616c6fa96f83ff1caa5f0231f9ee3bc`。

## 0. 先纠正术语：代码中的 `sm_id` 其实是 `blockIdx.x`

源码写：

```cpp
const auto num_sms = static_cast<int>(gridDim.x);
const auto sm_id   = static_cast<int>(blockIdx.x);
```

因此，代码把一个 CUDA block 当作一个“逻辑通信 SM”来编号。本文沿用社区常说的：

- Forward SM；
- Send/Receive SM；
- 一个 channel 占两个 SM。

但必须知道更精确的含义：

> `sm_id` 是 block id，不是硬件 SM 的物理编号。block 最终调度到哪个物理 SM 由 GPU 决定。

Dispatch kernel 使用 512 threads、较大的 shared memory/TMA 资源，并按“一块通信工作占据一个 SM”的设计调优，所以把 block 口语化称为 SM 有助于理解资源预算；但不要拿 `blockIdx.x=6` 理解成“物理 SM 6”。

后文统一使用：

- **Forward block（逻辑 Forward SM）**：偶数 block；
- **S/R block（逻辑 Send/Receive SM）**：奇数 block；
- **物理 SM**：GPU 真正的 Streaming Multiprocessor。

---

## 1. Grid、Channel 和两类 block 的总关系

### 1.1 Launch 配置

Dispatch launch：

```cpp
constexpr int kNumDispatchRDMASenderWarps = 7;

SETUP_LAUNCH_CONFIG(
    num_channels * 2,
    (7 + 1 + 8) * 32,
    stream
);
```

所以：

```text
gridDim.x  = num_channels * 2
blockDim.x = 16 warps * 32 lanes = 512 threads
```

kernel 内：

```cpp
num_channels = num_sms / 2;
channel_id   = sm_id / 2;
is_forwarder = sm_id % 2 == 0;
```

### 1.2 每个 channel 对应两个 block

对于 channel `c`：

```text
block 2*c     → Forward block
block 2*c + 1 → Send/Receive block
```

若 `num_sms=20`：

| Channel | Forward block | S/R block |
|---:|---:|---:|
| 0 | 0 | 1 |
| 1 | 2 | 3 |
| 2 | 4 | 5 |
| 3 | 6 | 7 |
| 4 | 8 | 9 |
| 5 | 10 | 11 |
| 6 | 12 | 13 |
| 7 | 14 | 15 |
| 8 | 16 | 17 |
| 9 | 18 | 19 |

### 1.3 每张 GPU 都有同样一套 channel

假设有两个节点，每节点 8 GPU，配置 10 channel。每一张 GPU 都 launch：

```text
channel 0: Forward block 0 + S/R block 1
...
channel 9: Forward block 18 + S/R block 19
```

“节点 0 是发送节点、节点 1 是接收节点”只是针对某一条 token 流的说法。实际每张 GPU 的同一个 kernel 同时具备：

- 把本 GPU token 发往其他节点的 Sender；
- 接收其他节点 RDMA 数据的 Forwarder；
- 接收节点内其他 Forwarder 转发数据的 NVLReceiver。

所以没有永久的“发送 GPU”和“接收 GPU”。通信是全双向、角色对称的。

### 1.4 channel 是端到端的独立流水线

一个 token 在源 GPU 被分给 channel `c` 后，其路径保持 channel id：

```text
源 GPU channel c S/R block
 → channel c RDMA send/meta/data buffer
 → 远端同号 GPU channel c Forward block
 → 目标 GPU channel c NVLink queue
 → 目标 GPU channel c S/R block 的 NVLReceiver warp
```

它不会在中间从 channel 3 跳到 channel 5。

每个 channel 有独立的：

- token 任务区间；
- RDMA data/meta/head/tail；
- NVLink data/prefix/head/tail；
- shared-memory 协调状态；
- QP 映射。

### 1.5 channel、QP、CUDA stream 不是一回事

| 概念 | 含义 |
|---|---|
| channel | 一套 GPU 工作切分、buffer、queue、warp 角色和 QP 映射 |
| QP | NIC/RDMA queue pair |
| CUDA stream | kernel launch 和依赖顺序 |
| block/逻辑 SM | 执行某一 channel 中一半角色的 CUDA thread block |
| warp | block 内 32 threads 的协作组 |

normal Dispatch 常用 `qp_id=channel_id`，但 channel 还包含远多于 QP 的状态。

### 1.6 一张 GPU 上的不同 channel 怎样划分 token

这是理解后续 SM/warp 分工的前提。先给出结论：

> 同一张 GPU 上，不同 channel 主要负责不同的一段本地 source token。所有 channel 的功能相同，区别在于 token 分片以及各自使用的 buffer、queue、QP 和通信 block。

#### 1.6.1 channel 按连续 token 区间划分

假设当前 GPU/rank 有：

```text
num_tokens   = T
num_channels = C
```

`get_channel_task_range` 使用：

```cpp
num_tokens_per_channel = ceil_div(num_tokens, num_channels);

token_start_idx =
    min(num_tokens_per_channel * channel_id, num_tokens);

token_end_idx =
    min(token_start_idx + num_tokens_per_channel, num_tokens);
```

所以：

```text
channel 0 → [start0, end0)
channel 1 → [start1, end1)
...
channel C-1 → [startC-1, endC-1)
```

这些区间：

- 互不重叠；
- 合起来覆盖当前 GPU 的全部本地 token；
- 一个 source token 在本轮 Dispatch 中只属于一个 channel；
- 每张 GPU 根据自己的本地 `num_tokens` 独立划分。

#### 1.6.2 一个简单例子

若一张 GPU 有 10 个 token、4 个 channel：

```text
ceil_div(10,4) = 3
```

| Channel | Token index |
|---:|---|
| channel 0 | `[0,3)`：0、1、2 |
| channel 1 | `[3,6)`：3、4、5 |
| channel 2 | `[6,9)`：6、7、8 |
| channel 3 | `[9,10)`：9 |

token 7 只由 channel 2 处理，不会再由 channel 0、1、3 发送。

若 token 数小于 channel 数，靠后的 channel 可能为空：

```text
T=3, C=10
```

此时仍 launch 完整 channel/block 结构，只是部分 channel 没有实际 token payload。

#### 1.6.3 “不同 channel 发不同 token”不等于“负责不同目的地”

错误理解：

```text
channel0 专门发往 Node0
channel1 专门发往 Node1
channel2 专门发往 GPU2
```

正确理解：

```text
channel 决定：处理哪一段源 token
top-k 决定：这些 token 发往哪些节点、GPU、expert
```

因此每个 channel 都可能：

- 发往任意远端节点；
- 发往目标节点内任意 GPU；
- 同时发往多个节点；
- 在目标节点内 fan-out 给多张 GPU；
- 因自己的 token 没有某目标而不向该目标发送。

例如 channel 2 负责 token 6、7、8：

```text
token6 → Node1 GPU3
token7 → Node1 GPU3、GPU7
token8 → Node2 GPU1
```

那么 channel 2 同时产生发往 Node1、Node2 的通信。channel 划分的是源端工作量，不是固定目的地。

#### 1.6.4 一个 token 有多个 top-k 时仍只属于一个 channel

假设 token 7 属于 channel 2，top-k expert 位于：

```text
Node1 GPU3
Node1 GPU7
Node2 GPU1
```

完整逻辑为：

```text
token7 / channel2
├─ 一份 RDMA payload 发往 Node1
│   ├─ Node1 GPU3
│   └─ Node1 GPU7
└─ 一份 RDMA payload 发往 Node2
    └─ Node2 GPU1
```

关键点：

- 不同目标节点可产生不同 RDMA payload；
- 同一目标节点内多个目标 GPU，共享一份节点级去重 RDMA payload；
- `SourceMeta` 的 GPU bitmask 表示节点内 fan-out 目标；
- 所有分支仍属于原来的 channel 2。

#### 1.6.5 channel 内再由 7 个 Sender warp 分片

channel 先取得连续区间，S/R block 的 7 个 Sender warp 再按模 7 划分：

```cpp
sender_warp =
    (token_idx - token_start_idx) % 7;
```

假设某 channel 负责 token 100～119：

| Sender warp | Token |
|---:|---|
| warp 0 | 100、107、114 |
| warp 1 | 101、108、115 |
| warp 2 | 102、109、116 |
| warp 3 | 103、110、117 |
| warp 4 | 104、111、118 |
| warp 5 | 105、112、119 |
| warp 6 | 106、113 |

所以 token 工作有两级划分：

```text
第一级：本 GPU 全部 token 按连续区间划给 channel
第二级：channel 内 token 按模 7 划给 Sender warp
```

### 1.7 一个节点下每张 GPU 是否都有很多 channel

是。每张 GPU/rank 都独立 launch 相同数量的 channel。

假设：

```text
节点内 GPU 数 = 8
num_sms        = 20
num_channels   = num_sms / 2 = 10
```

每张 GPU：

```text
GPU0: channel0～9
GPU1: channel0～9
...
GPU7: channel0～9
```

从节点总量看：

```text
8 GPUs × 10 channel instances = 80 个 channel 实例

每 channel 2 blocks：
80 × 2 = 160 个通信 block
```

但这些 block 分散在 8 张 GPU 上。每张 GPU 自己只 launch：

```text
10 channels × 2 blocks = 20 blocks
```

#### 1.7.1 相同编号 channel 不处理同一批 token

例如：

```text
Node0 GPU0 channel3
Node0 GPU5 channel3
Node1 GPU3 channel3
```

它们是三个独立 channel 实例：

```text
GPU0 channel3 → GPU0 本地 token 的第3段
GPU5 channel3 → GPU5 本地 token 的第3段
GPU3 channel3 → GPU3 本地 token 的第3段
```

如果各 rank 的 `num_tokens` 不同，相同 channel id 的 token 数也可能不同。

相同编号的意义是通信布局对齐：

```text
Node0 GPU5 channel3 发出
  → Node1 GPU5 channel3 RDMA recv
  → Node1 GPU5 channel3 Forward
  → Node1 GPU3 channel3 NVLReceiver
```

#### 1.7.2 每张 GPU 的每个 channel 都有完整功能

每个 channel 不是“只有发送”或“只有接收”，而是一条重复的完整流水线：

```text
channel c
├─ Forward block 2*c
│   ├─ 消费落在本 GPU 的 channel c RDMA queue
│   ├─ 按 SourceMeta 判断本节点目标 GPU
│   ├─ 写目标 GPU 的 channel c NVLink queue
│   └─ 汇总并归还 RDMA credit
│
└─ S/R block 2*c+1
    ├─ Sender：发送本 GPU channel c 的本地 token
    ├─ SenderCoordinator：批量发 RDMA并发布 tail
    └─ NVLReceiver：接收本节点其他 Forward GPU 的 channel c 数据
```

因此一张 GPU 在同一轮中可以同时：

- 发送自己的本地 token；
- 接收远端同号 GPU 发来的 RDMA；
- 把这些远端 token 转发给本节点其他 GPU；
- 接收本节点其他 Forward GPU 转发给自己的 token。

#### 1.7.3 不同 channel 如何共同写最终 `recv_x`

多个 channel 最终都写当前 GPU 的 `recv_x`，但 notify/prefix 阶段已为它们分配不重叠的输出区间：

```text
channel0 → recv_x [0,50)
channel1 → recv_x [50,120)
channel2 → recv_x [120,200)
...
```

所以多个 channel 可并行写同一个输出 tensor，而不会互相覆盖。

#### 1.7.4 为什么需要很多 channel

多个 channel 提供：

- 更多通信 block 和 warp；
- 多套独立 RDMA/NVLink ring；
- 多个 QP；
- 更多在途 chunk；
- 更高 NIC、NVLink 并行度；
- 更好的 producer/consumer 流水覆盖。

但 channel 不是越多越好：

```text
channel 太少
  → 推进并行度低，NIC/NVLink 可能吃不满

channel 太多
  → 占更多 SM、QP、buffer、HBM/L2
  → 与 Attention/GEMM 竞争
```

因此 `num_channels=num_sms/2` 是通信吞吐与计算资源之间的调优点。

### 1.8 用一个 token 串起 channel 分片与目的地 fan-out

假设：

```text
Node0 GPU5 有 100 个 token
num_channels = 10
```

近似划分：

```text
channel0 → token 0～9
channel1 → token 10～19
channel2 → token 20～29
channel3 → token 30～39
...
```

token37 属于 channel3，top-k 包含 Node1 GPU3、GPU7：

```text
Node0 GPU5 channel3
  S/R block7
  Sender warp = (37-30)%7 = 0
      │
      │ 一份发往 Node1 的 RDMA payload
      │ SourceMeta: bit3=1, bit7=1
      ▼
Node1 GPU5 channel3
  Forward block6
      ├─ 对应 GPU3 的 Forwarder → Node1 GPU3 channel3
      └─ 对应 GPU7 的 Forwarder → Node1 GPU7 channel3
             │
             ▼
Node1 GPU3/GPU7 channel3
  对应 source Forward GPU5 的 NVLReceiver
      │
      ▼
  最终 recv_x 的 channel3 prefix 区间
```

整个过程：

- token37 不会进入其他 channel；
- 目的地由 top-k 决定；
- channel3 身份从源端保持到最终目标 GPU；
- 同节点多个目标 GPU 共享节点级 RDMA payload，再由 SourceMeta fan-out。

最简记忆：

```text
Channel 决定：
  “处理哪一段 token、使用哪一套通信资源”

Top-k 决定：
  “这些 token 最终发到哪里”
```

---

## 2. Dispatch 的五种 WarpRole

源码：

```cpp
enum class WarpRole {
    kRDMASender,
    kRDMASenderCoordinator,
    kRDMAAndNVLForwarder,
    kForwarderCoordinator,
    kNVLReceivers
};
```

### 2.1 两类 block 的总分工

```text
偶数 Forward block
├─ warp 0..7   : 8 个 RDMAAndNVLForwarder
├─ warp 8      : 1 个有效 ForwarderCoordinator
└─ warp 9..15  : 进入 Coordinator 分支后直接退出

奇数 S/R block
├─ warp 0..6   : 7 个 RDMASender
├─ warp 7      : 1 个 RDMASenderCoordinator
└─ warp 8..15  : 8 个 NVLReceiver
```

S/R block 之所以叫 Send/Receive，是因为同一个 block 里：

- 前 8 个 warp 负责**向其他节点发送**；
- 后 8 个 warp 负责**接收本节点 Forwarder 经 NVLink 发来的数据**。

它不是“Sender SM”和“Receiver SM”各一块。

### 2.2 Forward block 的 target GPU 映射

Forwarder warp：

```cpp
dst_nvl_rank = (warp_id + channel_id) % 8;
```

同一 Forward block 的 warp 0～7 恰好覆盖目标 GPU 0～7，只是按 channel 旋转起点。

反推目标 GPU `d` 对应的 forward warp：

```text
forward_warp(c,d) = (d - c) mod 8
```

旋转的目的不是改变语义，而是让不同 channel 的 warp 编号与目标 GPU 映射错开，减少所有 channel 在同一 warp 编号上形成完全相同的访问模式。

### 2.3 S/R block 的 NVLReceiver 来源 GPU 映射

warp 8～15：

```cpp
src_nvl_rank = (warp_id + channel_id - 7) % 8;
```

令 `q=warp_id-8`，则：

```text
src_nvl_rank = (q + channel_id + 1) mod 8
receiver_warp(c,s) = 8 + (s - c - 1) mod 8
```

所以每张目标 GPU 的一个 S/R block 中，8 个 receiver warp 分别接收来自节点内 Forward GPU 0～7 的队列。

### 2.4 Channel 3 的完整 warp 表

#### Forward block 6，channel 3

| Warp | 角色 | 目标 NVL GPU |
|---:|---|---:|
| 0 | Forwarder | 3 |
| 1 | Forwarder | 4 |
| 2 | Forwarder | 5 |
| 3 | Forwarder | 6 |
| 4 | Forwarder | 7 |
| 5 | Forwarder | 0 |
| 6 | Forwarder | 1 |
| 7 | Forwarder | 2 |
| 8 | ForwarderCoordinator | 全部目标 GPU 的消费进度 |
| 9～15 | 退出 | — |

#### S/R block 7，channel 3

| Warp | 角色 | 负责对象 |
|---:|---|---|
| 0～6 | RDMASender | token 按模 7 分片 |
| 7 | RDMASenderCoordinator | 所有目标 RDMA rank |
| 8 | NVLReceiver | 来源 NVL GPU4 |
| 9 | NVLReceiver | 来源 NVL GPU5 |
| 10 | NVLReceiver | 来源 NVL GPU6 |
| 11 | NVLReceiver | 来源 NVL GPU7 |
| 12 | NVLReceiver | 来源 NVL GPU0 |
| 13 | NVLReceiver | 来源 NVL GPU1 |
| 14 | NVLReceiver | 来源 NVL GPU2 |
| 15 | NVLReceiver | 来源 NVL GPU3 |

因此在 **任意目标 GPU** 上，channel 3 中负责接收“来自 Forward GPU5”的都是 warp 9。

---

## 3. 为什么要用 7 个 Sender Warp

一个 channel 已经先获得连续 token 区间：

```cpp
get_channel_task_range(
    num_tokens,
    num_channels,
    channel_id,
    token_start_idx,
    token_end_idx
);
```

channel 内再由 7 个 Sender warp 分：

```cpp
if ((token_idx - token_start_idx) % 7 != warp_id)
    continue;
```

所以：

```text
Sender warp 0: channel 内 token 0, 7, 14, ...
Sender warp 1: channel 内 token 1, 8, 15, ...
...
Sender warp 6: channel 内 token 6, 13, 20, ...
```

为什么不是 8 个？

因为一个 512-thread block 一共 16 warp，后 8 warp 已预留给 NVLReceiver，发送侧剩余 8 warp；其中一个必须做 SenderCoordinator，所以留下 7 个并行消息生产 warp。

---

## 4. RDMASender Warp 内 32 个 Lane 的分工

RDMASender 不是“lane 0 做全部工作”。同一 warp 会在不同阶段切换 lane 的解释。

### 4.1 阶段一：写 18 个 meta

对于某个目标 RDMA rank：

| Lane | 写入 | 含义 |
|---:|---|---|
| 0～7 | `meta[0..7]` | 8 个目标 NVL GPU 的 channel 起点 |
| 8～15 | `meta[8..15]` | 8 个目标 NVL GPU 的 channel 终点 |
| 16 | `meta[16]` | 本 channel 节点级 RDMA payload 起点 |
| 17 | `meta[17]` | 本 channel 节点级 RDMA payload 终点 |
| 18～31 | 不写 meta | 等待 `__syncwarp()` |

如果目标是远端节点，整个 warp 协作调用 `put_nbi_warp` 把 72 B meta 发到远端同号 GPU 的同 channel recv buffer。

### 4.2 阶段二：lane 对应目标 RDMA rank

代码要求 `kNumRDMARanks<=32`。对一个 token：

```text
lane 0 → 目标 RDMA rank 0
lane 1 → 目标 RDMA rank 1
lane 2 → 目标 RDMA rank 2
...
```

每个有效 lane 读取一个 64-bit 值：

```cpp
is_token_in_rank[
  token_idx,
  dst_rdma_rank * 8 : dst_rdma_rank * 8 + 8
]
```

这 8 个 bool 表示 token 是否需要目标节点的 GPU0～7。

若 64-bit 值不为 0：

- 这个 token 至少要去该目标节点一张 GPU；
- lane 为该节点计算 RDMA 逻辑 tail；
- 检查 ring head/credit；
- 保存 `send_rdma_head` 供 combine；
- 后续为该目标节点准备一份去重 payload。

### 4.3 阶段三：warp shuffle 收集所有目标节点

各 lane 先独立得到自己目标节点的 `rdma_tail_idx`，然后通过 `__shfl_sync`/broadcast 把有效目标收集成：

```text
topk_ranks[]
slot_indices[]
dst_send_buffers[]
```

同一个 token 可以发往多个节点。hidden 只从输入读取一次，再 broadcast 写入多个目标节点的 send slot。

### 4.4 阶段四：32 lane 合作搬 payload

hidden：

```text
lane 0 复制第 0、32、64... 组向量
lane 1 复制第 1、33、65... 组向量
...
```

实际使用展开的 `UNROLLED_WARP_COPY` 和 `int4` 向量化。

FP8 scales：

```cpp
for (i = lane_id; i < num_scales; i += 32)
```

SourceMeta：

- 前 `num_topk_ranks` 个 lane 各负责一个目标节点的 SourceMeta；
- SourceMeta 含 `src_rdma_rank` 与目标节点内 8-bit GPU mask。

top-k metadata：

```cpp
for (i = lane_id;
     i < num_topk * num_topk_ranks;
     i += 32)
```

各 lane 分担 index/weight 复制。

### 4.5 阶段五：completion window

7 个 Sender warp 可能乱序完成 token。每个目标节点有：

```cpp
rdma_send_channel_lock[dst]
rdma_send_channel_tail[dst]
rdma_send_channel_window[dst]
```

window 是 32-bit 完成位图：

```text
逻辑 token: 100 101 102 103 104
完成位:       1   1   0   1   0
```

只能把“连续完成前缀”发布给 Coordinator。103 虽完成，也不能跨过 102。

在这一步，lane 仍对应目标 RDMA rank：

- lane `r` 更新目标节点 `r` 的 lock/window/tail；
- `__syncwarp()` 让 warp 内保持一致推进。

---

## 5. RDMASenderCoordinator Warp 内 Lane 分工

### 5.1 lane 对应目标 RDMA rank

```text
lane r 保存目标 RDMA rank r 的：
  num_tokens_to_send
  last_issued_tail
  progress
```

初始 token 数来自 `rdma_channel_prefix_matrix` 当前/前一 channel 的差。

### 5.2 为什么看 Sender 的连续 tail

Coordinator 读取 `rdma_send_channel_tail[dst]`。只有：

- 已连续准备数量达到 send chunk；或
- 已经是最后不足一个 chunk 的部分，

才提交 RDMA。

这避免每个 token 一个 WQE，也避免把中间有空洞的 buffer 当作连续消息发送。

### 5.3 整个 warp 协作发 IBGDA

Coordinator 选择一个 `dst_rdma_rank` 后：

```cpp
dst = remote recv_buffer(src_rdma_rank) + slot;
src = local  send_buffer(dst_rdma_rank) + slot;

nvshmemi_ibgda_put_nbi_warp(
    dst, src, bytes,
    dst_pe,
    channel_id,
    lane_id,
    ...
);
```

此时不是只有“负责 dst 的 lane”在搬全部数据：

- 目标 lane 保存该 peer 的计数状态；
- warp shuffle 把状态广播；
- lane 0 预留 WQE slots/提交；
- 需要多个 WQE 时多个 lane 分别填 WQE；
- 所有 lane 在 put 前后同步。

### 5.4 tail 发布

payload put 发起后，`lane_id==dst_rdma_rank` 的 lane：

- 增加 `last_issued_tail`；
- 减少剩余 token；
- 对远端 `rdma_channel_tail.buffer(src_rdma_rank)` 做 atomic add。

远端 Forwarder 随后用 acquire/system-scope 读取 tail，才把对应 ring slots 视为可消费。

---

## 6. Forwarder Warp 内 Lane 分工

一个 Forwarder warp 固定负责一个目标 NVL GPU，但必须处理所有来源 RDMA rank。

### 6.1 Warp 级固定目标

```text
整个 warp 的 dst_nvl_rank 固定
lane 的 src_rdma_rank 可不同
```

例如 Node1 GPU5、channel3、warp0：

```text
dst_nvl_rank = (0+3)%8 = 3
```

整个 warp 只负责把需要 GPU3 的 token写入 GPU3 的 NVLink queue。

### 6.2 meta 阶段：lane 对应来源节点

```text
lane 0 → 来源 RDMA rank 0
lane 1 → 来源 RDMA rank 1
...
```

每个 lane 从：

```cpp
rdma_channel_meta.recv_buffer(lane_id)
```

读取当前目标 GPU 的四个 meta：

- GPU 级 start/end；
- 节点级 RDMA payload start/end。

解码后 lane：

1. 得到该来源本 channel 的 payload 数；
2. 把 GPU 级 prefix start/end 写到目标 GPU 的 NVLink prefix buffer；
3. 保存本来源的 expected count；
4. `__syncwarp()` 等所有来源 lane 完成 meta 阶段。

### 6.3 数据阶段：lane 保存一个来源的私有状态

每个来源 lane 保留：

```text
num_tokens_to_recv_from_rdma
cached_rdma_channel_head
cached_rdma_channel_tail
src_rdma_channel_prefix
```

warp 通过 round-robin 选择下一个有数据的 `src_rdma_rank`。

选中来源 `r` 后：

- lane `r` 必要时从 `rdma_channel_tail.buffer(r)` 刷新 tail；
- `__shfl_sync(..., r)` 把 head/tail 广播给全 warp；
- 全 warp 共同处理来源 `r` 的一段 token。

这是一种常见 CUDA 模式：

> lane 持有 per-peer 状态，warp shuffle 选择并广播当前 peer，全 warp 再合作处理大 payload。

### 6.4 SourceMeta 过滤

每个 Forwarder warp都会扫描来源 ring 中的 token：

```cpp
is_in_dst_nvl_rank =
    src_meta.is_token_in_nvl_rank(dst_nvl_rank);
```

- bit 为 0：该 token 不发给本 warp 的目标 GPU，但 RDMA head 仍前进；
- bit 为 1：写入目标 GPU NVLink queue。

因此同一个 RDMA token 可被 8 个 Forwarder warp 分别扫描，并由其中一个或多个 warp实际转发。

### 6.5 TMA 搬运时的 lane 分工

payload 较大，使用 warp 级 TMA：

1. `elect_one_sync()` 选一个 leader lane 发起 TMA load；
2. 全 warp `__syncwarp()`；
3. 全 warp 等 mbarrier；
4. leader 发起 TMA store 到目标 GPU IPC buffer；
5. 全 warp等待 store 完成。

所以 TMA 指令由一个 elected lane 发出，但这个 warp 的其他 lane 负责同步、per-source 状态和控制流，不能删掉。

### 6.6 Forwarder 进度

来源 `r` 的一段处理后：

```cpp
if (lane_id == r)
    forward_channel_head[dst_nvl_rank][r] = new_head;
```

注意这是 block 内 shared memory，记录：

```text
目标 GPU d 的 Forwarder warp
对来源节点 r
扫描到了哪个 RDMA head
```

---

## 7. ForwarderCoordinator Warp 内 Lane 分工

只有 Forward block 的 warp8 有效；warp9～15 因 `target_rank>0` 直接返回。

### 7.1 lane 对应来源 RDMA rank

```text
lane r → 汇总来源节点 r 的消费进度
```

lane `r` 遍历 8 个目标 GPU 的 Forwarder：

```text
forward_channel_head[0][r]
forward_channel_head[1][r]
...
forward_channel_head[7][r]
```

对尚未 retired 的 Forwarder 取最小值：

```text
min_head(r) =
  min(active target GPU forwarders' head for source r)
```

### 7.2 为什么必须取最小值

同一 RDMA payload 可能要转发给 GPU3 和 GPU7：

```text
GPU3 Forwarder 已扫到 head=120
GPU7 Forwarder 只扫到 head=100
```

只能归还到 100。若告诉源端可复用到 120，ring slot 100～119 可能在 GPU7 Forwarder 读取前被覆盖。

### 7.3 归还 RDMA credit

当最小 head 至少前进一个 send chunk：

```cpp
nvshmemi_ibgda_amo_nonfetch_add(
    source_node_head_slot,
    min_head - last_head,
    source_pe,
    channel_id + num_channels,
    ...
);
```

源节点 Sender warp 轮询本地 head slot，获得可复用空间。

ForwarderCoordinator 不是直接调用或唤醒 SenderCoordinator；二者通过远端 symmetric head 间接协作。

---

## 8. NVLReceiver Warp 内 Lane 分工

一个 NVLReceiver warp 固定负责一个来源 NVL GPU：

```text
整个 warp 的 src_nvl_rank 固定
```

例如 channel3 的 warp9 固定接收来源 GPU5。

### 8.1 prefix 阶段：lane 对应来源 RDMA rank

Node1 GPU5 Forwarder 转来的 token，原始来源可能是：

- Node0；
- Node2；
- Node3；
- ……

所以 NVLReceiver 仍要区分原始 `src_rdma_rank`。

```text
lane r → 原始来源 RDMA rank r
```

每个 lane 读取 Forwarder 写来的：

```text
nvl_channel_prefix_start[src_rdma_rank]
nvl_channel_prefix_end[src_rdma_rank]
```

并保存各来源在最终 `recv_x` 中的 offset。随后 `warp_reduce_sum` 得到该来源 NVL GPU 总共会送多少 token。

### 8.2 queue 阶段：lane0/leader 控制 tail

当本地 cached head 等于 tail 时：

```cpp
cached_tail = __shfl_sync(
    full_mask,
    ld_acquire_sys_global(nvl_channel_tail.buffer()),
    0
);
```

即 lane0 读取 tail，再广播给全 warp。

### 8.3 每个 token 的最终输出位置

token 中的 `SourceMeta.src_rdma_rank` 指出原始来源节点。

```cpp
recv_token_idx =
    __shfl_sync(full_mask, total_offset, meta.src_rdma_rank);

if (lane_id == meta.src_rdma_rank)
    total_offset += 1;
```

也就是说：

- lane `r` 持有原始来源 `r` 的当前输出 offset；
- SourceMeta 选择对应 lane；
- warp shuffle 得到最终 `recv_x` index；
- 只有该 lane 增加自己的 offset。

### 8.4 搬 hidden/scale

hidden 和对齐 scale 使用 TMA：

- elected leader 发 load；
- 全 warp 等 barrier；
- elected leader发 store到最终输出。

不对齐 scale 使用：

```cpp
for (i=lane_id; i<num_scales; i+=32)
```

由 32 lane 分摊。

### 8.5 搬 top-k

```cpp
if (lane_id < num_topk) {
    idx    = topk_idx[lane_id];
    weight = topk_weights[lane_id];
}
```

每个 lane 负责一个 top-k 项，并把全局 expert id 转换为当前 rank 的 local expert id：

```text
属于本 rank expert → global id - local_expert_begin
不属于本 rank       → -1，weight=0
```

### 8.6 归还 NVLink credit

消费完一批后，receiver 更新 `nvl_channel_head`。

这个 head **物理上放在 Forwarder GPU 的 buffer 中**，目标 GPU 的 receiver 通过 CUDA IPC/NVLink 写回；Forwarder 在自己的 GPU 上轮询它。

---

## 9. Forward GPU 与目标 GPU 的 NVLink Queue 怎样对应

这是理解“节点 1 GPU5 Forward 怎么和各卡 S/R SM 交互”的核心。

设：

```text
Forward GPU 的 nvl_rank = s
目标 GPU 的 nvl_rank   = d
channel                 = c
```

### 9.1 数据、prefix、tail 放在目标 GPU

Forwarder 构造：

```text
target GPU d buffer
  [channel c]
  [source Forward GPU s]
  ├─ nvl_channel_x
  ├─ prefix_start/end
  └─ tail
```

Forwarder通过 IPC 映射/NVLink写这些位置；目标 GPU `d` 的 channel `c` NVLReceiver 本地读取。

### 9.2 head 放在 Forward GPU

```text
Forward GPU s buffer
  [channel c]
  [destination GPU d]
  └─ head
```

目标 GPU `d` 的 receiver 消费后，通过 IPC/NVLink 把 head 写回 Forward GPU `s`。

### 9.3 Queue 的 producer/consumer 表

| 对象 | 物理位置 | Producer | Consumer |
|---|---|---|---|
| NVL data | 目标 GPU d | Forward GPU s 的 Forwarder | 目标 GPU d 的 NVLReceiver |
| prefix start/end | 目标 GPU d | Forwarder | NVLReceiver |
| tail | 目标 GPU d | Forwarder | NVLReceiver |
| head | Forward GPU s | 目标 GPU d 的 NVLReceiver | Forwarder |

所以两块 GPU 并不是通过“直接调用对方 warp”交互，而是通过四类全局内存队列状态交互。

### 9.4 地址为什么天然对齐

Forwarder：

```text
ws_rr_buffer_ptr = buffer_ptrs[d]
rs_wr_rank        = s
channel_id        = c
```

Receiver：

```text
ws_rr_buffer_ptr = buffer_ptrs[本地 d]
rs_wr_rank        = source s
channel_id        = c
```

两边用相同 `AsymBuffer` 公式：

```text
base
+ per_channel_bytes * c
+ per_peer_bytes * s
```

因此 Forwarder 写的正是 Receiver 读的位置。

---

## 10. 完整案例：Node0 GPU5 → Node1 GPU5 → Node1 GPU3

### 10.1 拓扑与假设

两个节点，每节点 8 GPU：

```text
Node0 global rank 0..7
Node1 global rank 8..15

Node0 GPU5 global rank = 5
Node1 GPU5 global rank = 13
Node1 GPU3 global rank = 11
```

假设：

- `num_channels=10`；
- token 被分到 channel 3；
- token 的一个 expert 位于 Node1 GPU3；
- 为展示 fan-out，再假设另一个 expert 位于 Node1 GPU7。

SourceMeta 的目标 GPU bits：

```text
bit3 = 1
bit7 = 1
其他 bit = 0
src_rdma_rank = 0
```

### 10.2 第一步：Node0 GPU5 的 channel3 S/R block 发送

channel3：

```text
S/R block = 2*3+1 = block7
```

token 在 channel3 区间内的局部序号决定 Sender warp：

```text
sender_warp = local_token_index % 7
```

该 Sender warp 中：

- lane1 对应目标 RDMA rank1（Node1）；
- lane1 读取 Node1 GPU0～7 的 8-bit/8-bool mask；
- 发现 bit3、bit7 有效；
- 为 Node1 分配一个 RDMA ring slot；
- 所有 lane 合作把 hidden/scales/top-k/SourceMeta 写到：

```text
Node0 GPU5
channel3
rdma_data.send_buffer(dst_rdma_rank=1)
```

虽然有两个目标 GPU，跨节点 payload 只写一份，因为它们同属 Node1。

### 10.3 第二步：Node0 GPU5 的 SenderCoordinator 发 RDMA

Node0 GPU5 block7 warp7 等该 slot 及其前面的 slot 连续完成，凑成 chunk 后：

```text
local source:
  Node0 GPU5
  channel3 send_buffer(dst_node=1)

remote target:
  Node1 GPU5
  channel3 recv_buffer(src_node=0)
```

为什么落到 Node1 GPU5，而不是直接 Node1 GPU3？

因为 normal V1 的 RDMA 域按相同 `nvl_rank` 连接：

```text
Node0 GPU5 ↔ Node1 GPU5
```

payload 后，SenderCoordinator 更新 Node1 GPU5 channel3 的 tail[src_node0]。

### 10.4 第三步：Node1 GPU5 channel3 Forward block 接手

Node1 GPU5：

```text
Forward block = 2*3 = block6
```

这个 block 的 Forwarder warp：

| Warp | 目标 GPU |
|---:|---:|
| 0 | GPU3 |
| 1 | GPU4 |
| 2 | GPU5 |
| 3 | GPU6 |
| 4 | GPU7 |
| 5 | GPU0 |
| 6 | GPU1 |
| 7 | GPU2 |

该 token 的 bit3、bit7 有效，因此：

- warp0 实际转发到 GPU3；
- warp4 实际转发到 GPU7；
- 其他 warp 扫描 token、推进自己的 RDMA head，但不写数据。

### 10.5 第四步：Forward warp0 把数据写入 Node1 GPU3

Node1 GPU5 block6 warp0：

```text
dst_nvl_rank = 3
```

lane0 代表原始来源 Node0。它：

1. 从 `recv_buffer(src_node0)` 的 meta 得到期望数；
2. 轮询 `tail[src_node0]`；
3. 选中 Node0 来源；
4. 全 warp 读取 RDMA ring token；
5. 检查 SourceMeta bit3；
6. 等 Node1 GPU3 的 NVLink queue 有 credit；
7. TMA copy 到：

```text
Node1 GPU3
channel3
source Forward GPU5
nvl_channel_x[slot]
```

然后更新 Node1 GPU3 本地的 queue tail。

### 10.6 第五步：Node1 GPU3 channel3 S/R block 接收

Node1 GPU3 channel3 S/R block 仍是 block7。

负责来源 Forward GPU5 的 receiver warp：

```text
receiver_warp(3,5)
  = 8 + (5-3-1) mod 8
  = 9
```

所以是 **Node1 GPU3 block7 warp9**。

warp9：

1. 读取本地 channel3/sourceGPU5 的 prefix；
2. 轮询本地 tail；
3. 从 `nvl_channel_x` 取 token；
4. SourceMeta 指出原始来源 Node0；
5. lane0 提供 Node0 对应的最终 output offset；
6. TMA 把 hidden 写入 GPU3 `recv_x`；
7. lanes 0～`num_topk-1` 写 local top-k/weights；
8. 更新位于 Node1 GPU5 上的 head[destinationGPU3]。

### 10.7 第六步：NVLink credit 回到 Node1 GPU5

Node1 GPU3 warp9 更新：

```text
Node1 GPU5
channel3
destination GPU3
nvl_channel_head
```

Node1 GPU5 block6 warp0 轮询该 head，知道 GPU3 已消费哪些 NVLink queue slots。

GPU7 的路径相同，只是：

- Forwarder 是 Node1 GPU5 block6 warp4；
- 目标是 Node1 GPU7；
- Node1 GPU7 channel3 负责 sourceGPU5 的仍是 block7 warp9。

### 10.8 第七步：RDMA credit 回到 Node0 GPU5

Node1 GPU5 block6 的 8 个 Forwarder warp都对 Node0 来源推进 `forward_channel_head[d][0]`。

block6 warp8（ForwarderCoordinator）的 lane0：

1. 读取 8 个目标 GPU Forwarder 对 Node0 的 head；
2. 对 active Forwarder 取最小值；
3. 达到 chunk 后，对 Node0 GPU5 的 channel3 RDMA head slot做 atomic add。

Node0 GPU5 Sender warp 读到 head 前进，才可复用原 RDMA ring slot。

### 10.9 完整链路图

```text
Node0 GPU5
channel3 S/R block7
  Sender warp (0..6)
      │ 准备 payload
      ▼
  SenderCoordinator warp7
      │ IBGDA RDMA，channel3
      ▼
Node1 GPU5
channel3 Forward block6
  ├─ warp0: 目标 GPU3 ──NVLink──► Node1 GPU3
  │                                  channel3 S/R block7
  │                                  receiver warp9(source GPU5)
  │                                      │
  │                                      └── head 写回 Node1 GPU5
  │
  ├─ warp4: 目标 GPU7 ──NVLink──► Node1 GPU7
  │                                  channel3 S/R block7
  │                                  receiver warp9(source GPU5)
  │
  └─ warp8: ForwarderCoordinator
           └── RDMA head/credit 写回 Node0 GPU5
```

---

## 11. Node1 GPU5 Forward 如何和“各卡不同 channel”交互

最重要的答案：

> Node1 GPU5 的 channel `c` Forward block，只和各目标 GPU 的 channel `c` S/R block交互；不会和 channel `c'` 的 receiver 交互。

### 11.1 对单个 channel

Node1 GPU5 channel3：

```text
Forward block6
  → GPU0 channel3 S/R block7 的 sourceGPU5 receiver
  → GPU1 channel3 S/R block7 的 sourceGPU5 receiver
  ...
  → GPU7 channel3 S/R block7 的 sourceGPU5 receiver
```

### 11.2 对全部 channel

Node1 GPU5 同时有：

```text
channel0 Forward block0
  ↔ 各 GPU channel0 S/R block1

channel1 Forward block2
  ↔ 各 GPU channel1 S/R block3

...

channel9 Forward block18
  ↔ 各 GPU channel9 S/R block19
```

所以看上去 GPU5 的 Forwarder 与“各张卡的很多 S/R SM”交互，实际是严格按 channel 一一分层：

```text
Forward(c,s,d) ↔ Receiver(c,d,s)
```

其中：

- `c`：相同 channel；
- `s`：Forward GPU/source NVL GPU；
- `d`：目标 GPU。

### 11.3 不同 channel 为什么不会撞 buffer

`AsymBuffer`：

```text
ptr =
  base
  + per_channel_bytes * channel_id
  + per_peer_bytes * peer_offset
```

channel0、channel1、channel3 的 data/head/tail/prefix 地址区间互不重叠。

### 11.4 一个 token 是否会出现在多个 channel

同一轮 Dispatch 中，一个源 token由 `get_channel_task_range` 归属一个 channel，不会同时被两个 channel 发送。

但不同 token、不同 channel 可并行发往同一目标 GPU，于是目标 GPU 上多个 S/R block同时向最终 `recv_x` 的不同 prefix 区间写入。

prefix matrix 确保不同 channel 的最终输出区间不重叠。

---

## 12. 三种同步边界

### 12.1 同一 warp 内

- `__syncwarp()`；
- `__shfl_sync()`；
- `warp_reduce_sum`；
- `elect_one_sync()`；
- TMA mbarrier。

### 12.2 同一 block 的多个 warp

Sender 侧：

```cpp
barrier.sync 0, (7 + 1) * 32
```

只同步 7 Sender + 1 SenderCoordinator。

Forwarder 侧：

```cpp
barrier.sync 1, (8 + 1) * 32
```

只同步 8 Forwarder + 1 有效 ForwarderCoordinator。

### 12.3 跨 block / 跨 GPU

Forward block 与同 channel S/R block 是两个不同 CUDA block，不能共享 `__shared__` memory，也没有“直接调用”。

跨边界全靠：

- global HBM；
- NVSHMEM symmetric RDMA buffer；
- CUDA IPC/NVLink buffer；
- release/acquire/system-scope load/store；
- RDMA atomic head/tail；
- 负编码 prefix/meta；
- polling 和 timeout。

这同样适用于同一 GPU 上的 Forward block 与 S/R block：block 之间仍通过 global memory 协议，不通过 block shared memory。

---

## 13. Backpressure 如何逐级传播

假设 Node1 GPU3 的 NVLReceiver 变慢：

```text
GPU3 Receiver 慢
  → NVL head 不前进
  → GPU5→GPU3 NVLink queue 变满
  → GPU5 Forwarder warp0 等待 credit
  → forward_channel_head[GPU3] 落后
  → ForwarderCoordinator 的 min_head 落后
  → 不向 Node0 归还 RDMA head
  → Node0 RDMA ring 逐渐变满
  → Node0 Sender 等待 head
```

这是端到端有界队列的自然背压。

取最小 Forwarder head 会让最慢目标 GPU限制该来源的 RDMA credit。代价是可能出现 head-of-line blocking；好处是不会覆盖任何仍被某目标 Forwarder 使用的数据。

---

## 14. 为什么 SourceMeta 必不可少

```cpp
struct SourceMeta {
    int src_rdma_rank;
    int is_token_in_nvl_rank_bits;
};
```

它解决两个问题。

### 14.1 节点内 fan-out

一个 token 在目标节点可能命中 GPU3 和 GPU7：

```text
RDMA 只传一份
bit3=1, bit7=1
Forwarder warp3/7 对应的实际 warp各自转发
```

注意“warp3/7”这里只是目标概念；真实 warp id还要按 channel 旋转。

### 14.2 最终输出按原来源分段

NVLReceiver 使用 `src_rdma_rank` 选择哪个 lane 的 `total_offset`，把 token写到最终 `recv_x` 的正确来源段。

所以 `SourceMeta` 同时连接：

```text
RDMA 节点级去重
 → NVLink GPU级 fan-out
 → 最终 recv_x 来源排序
```

---

## 15. 常见误解逐条回答

### Q1：每个 channel 只有一个 SM 吗？

不是。normal Dispatch 中每 channel 两个 block/逻辑 SM：偶数 Forward、奇数 S/R。

### Q2：S/R block 是一半时间 send、一半时间 receive 吗？

不是。不同 warp 分工，可并行推进：

- warp0～6 Sender；
- warp7 SenderCoordinator；
- warp8～15 NVLReceiver。

### Q3：Forward block 的 8 个 warp 分别接收 8 个来源节点吗？

不是。8 个 Forwarder warp分别对应 8 个**目标 NVL GPU**。warp 内 lane 0～`kNumRDMARanks-1` 才对应来源节点。

### Q4：NVLReceiver 的 8 个 warp 分别对应什么？

分别对应 8 个来源 Forward GPU/NVL rank。warp 内 lane 对应原始来源 RDMA rank。

### Q5：Node1 GPU5 Forwarder 如何通知 Node1 GPU3 Receiver？

写 GPU3 的 NVLink data/prefix/tail。GPU3 Receiver 轮询本地 tail；没有直接 warp 消息。

### Q6：GPU3 Receiver 如何通知 GPU5 Forwarder？

把 head 写回 GPU5 的 IPC buffer slot。GPU5 Forwarder 轮询本地 head。

### Q7：Node1 GPU5 Forward channel3 会把数据给 GPU3 channel4 吗？

不会。它写 GPU3 channel3 的 queue，由 GPU3 channel3 S/R block 消费。

### Q8：Node1 GPU5 的 Forward block 与本机 GPU5 的 S/R block直接共享 shared memory吗？

不能。它们是不同 block。即使落在同一物理 GPU，也通过 global queue 交互。

### Q9：一个 Forwarder warp只看会发给自己的 token吗？

它扫描本来源 ring 的所有 token，用 SourceMeta 过滤。不属于其目标 GPU 的 token不复制，但 head仍推进。

### Q10：为什么 ForwarderCoordinator 取 8 个 warp的最小 head？

因为 RDMA ring slot只有在所有仍可能读取它的目标 GPU Forwarder都扫过后才能复用。

### Q11：为什么 channel id 不写进 token message？

channel 已编码在：

- 执行该 token 的 block；
- RDMA/NVL buffer offset；
- prefix matrix；
- QP id。

数据从 channel `c` 的专属 queue 到 channel `c` 的专属 queue，无需每条 message 再存 channel。

### Q12：lane 的意义为什么一直变化？

CUDA warp 程序会分阶段复用 32 个 lane：

- 某阶段 lane=peer rank；
- 某阶段 lane=meta offset；
- 某阶段 lane=top-k index；
- 某阶段所有 lane=向量搬运协作者；
- 某阶段只有 elected leader 发 TMA/WQE。

不能给 lane0 一个贯穿整个 kernel 的固定职业。

---

## 16. Dispatch 与 Combine 的 SM 分工不要混淆

本文主体是 Dispatch，因为“Forward SM + SendReceive SM”问题对应它。

normal Combine 同样每 channel 两个 block，但奇偶定义和 WarpRole 不同：

```cpp
const bool is_forwarder_sm = sm_id % 2 == 1;

enum class WarpRole {
    kNVLSender,
    kNVLAndRDMAForwarder,
    kRDMAReceiver,
    kCoordinator
};
```

Combine：

- 偶数 block是非-forwarder，含 NVLSender、部分 RDMAReceiver、Coordinator；
- 奇数 block是 `kNVLAndRDMAForwarder` 与 Coordinator；
- block warp 数依赖 `kNumForwarders`，不固定为 Dispatch 的 16 warp；
- 数据沿 Dispatch handle 保存的路径反向返回并规约。

因此：

> “Dispatch 偶数 block 是 Forward、奇数 block 是 S/R”的结论，不能原样套到 Combine。

---

## 17. 调试时怎样定位一个 timeout

### 17.1 Sender timeout

日志里的：

```text
channel
本地 RDMA rank/nvl rank
目标 RDMA lane
head/tail
```

可定位是哪条 `(source GPU, channel, destination node)` ring 没获得 credit。

### 17.2 Forwarder meta timeout

看：

```text
channel
本地 Forward GPU
src RDMA lane
dst NVL GPU
四个 meta
```

这直接映射到一个 Forwarder warp和一个来源 lane。

### 17.3 Forwarder NVL timeout

表示目标 GPU 的 receiver 没推进 head。用公式：

```text
forward_warp = (dst_gpu - channel) mod 8
receiver_warp = 8 + (src_forward_gpu - channel - 1) mod 8
```

可同时找到两端 warp。

### 17.4 Receiver timeout

日志给出 source NVL rank。根据 channel 与 source rank 算 receiver warp，再检查：

- 目标 GPU 本地 tail；
- Forward GPU 对应 warp；
- prefix start/end；
- SourceMeta；
- 是否某 rank 未进入同一轮。

---

## 18. 最简记忆模型

### 层级一：Channel

```text
一个 channel = 一对 block + 一套 RDMA queue + 一套 NVLink queue
```

### 层级二：Block

```text
Dispatch:
  偶数 block → 8 Forwarder + 1 Coordinator
  奇数 block → 7 Sender + 1 Coordinator + 8 Receiver
```

### 层级三：Warp

```text
Sender warp      → token 子集
Forwarder warp   → 目标 NVL GPU
Receiver warp    → 来源 Forward GPU
Coordinator warp → 汇总/批量发布
```

### 层级四：Lane

```text
多数控制阶段：
  lane r → RDMA rank r

数据阶段：
  32 lane → 合作向量搬运

特殊阶段：
  lane → meta offset / top-k index / elected leader
```

### 层级五：跨界交互

```text
跨节点:
  payload + tail 走 IBGDA
  head/credit 走远端 atomic

节点内:
  data/prefix/tail 写目标 GPU
  head 写回 Forward GPU
```

---

## 19. 最终总结

用 Node0 GPU5 → Node1 GPU3 的 token 来看，V1 normal Dispatch 是一条严格保持 channel 的三级生产者/消费者流水线：

```text
第一级：
Node0 GPU5 channel c S/R block
  7 Sender warps 生产消息
  1 Coordinator warp 批量发 RDMA

第二级：
Node1 GPU5 channel c Forward block
  8 Forwarder warps分别面向目标 GPU0..7
  1 Coordinator 汇总最小 RDMA head

第三级：
Node1 GPU3 channel c S/R block
  对应 sourceGPU5 的 NVLReceiver warp消费 queue
  写最终 recv_x 并归还 NVLink head
```

其中：

- channel 决定端到端 buffer 分区；
- Sender warp按 token 模 7；
- Forwarder warp按目标 GPU；
- Receiver warp按来源 Forward GPU；
- warp 内 lane常按 RDMA rank 保存 per-peer 状态；
- 大 payload由全 warp合作搬运；
- block之间、GPU之间没有直接函数调用，只通过 data/prefix/head/tail 和内存顺序协议交互；
- 最慢目标 Receiver 会通过 NVLink head、Forwarder min-head、RDMA head 把背压逐级传回源 Sender。

掌握下面两个公式，就能从日志或代码快速定位具体 warp：

```text
Forward block 中，目标 GPU d:
  warp = (d - channel_id) mod 8

目标 GPU S/R block 中，来源 Forward GPU s:
  warp = 8 + (s - channel_id - 1) mod 8
```

## 附录：相关源码

- [normal internode kernel](csrc/kernels/legacy/internode.cu)
- [Buffer/AsymBuffer/SymBuffer](csrc/kernels/legacy/buffer.cuh)
- [normal Config](csrc/legacy/config.hpp)
- [V1 总体深度解析](DeepEP_LowLatency_IBGDA_DeepDive.md)
- [V1 SM 方案说明](docs/implementation-v1-sm.md)
