# DeepEP V1 通信机制深度解析：Normal、Low-Latency、NVSHMEM 与 IBGDA

> 面向第一次阅读 DeepEP 的学习者。本文从 MoE 的 dispatch/combine 语义开始，逐层解释 V1 legacy normal/high-throughput 路径、V1 low-latency 路径、NVSHMEM 对称内存、IBGDA、channel、五种 warp 角色、四个 meta、RDMA/NVLink 两级转发、环形队列以及 send/recv hook。
>
> 本文依据工作区提交 `f99f06868616c6fa96f83ff1caa5f0231f9ee3bc` 整理。代码继续演进后行号可能变化，应以函数名、变量名和数据结构为准。

## 0. 文档来源、恢复原则与重要纠正

这是一份从旧聊天碎片恢复，并结合当前代码重新校验、扩充的学习文档。碎片中包含旧文档的局部段落、问答、修改记录和相互矛盾的中间结论，因此本文遵循：

1. 能从碎片与代码同时确认的内容，按完整教程重新组织；
2. 只有碎片、无法由当前代码确认的表述，不作为确定事实；
3. 聊天中后来纠正的错误，以最终代码结论为准；
4. 示例数字只用于解释索引和前缀和，不代表固定运行配置；
5. “共享内存”若未特别说明，指 NVSHMEM symmetric heap，而不是 CUDA block 的 `__shared__` memory。

先记住五条纠正：

- **IBGDA 是传输技术，不等于 low-latency 模式。** `internode.cu` 的 normal 路径与 `internode_ll.cu` 的 low-latency 路径都调用 IBGDA。
- **五种 warp 角色、四个 meta、channel ring buffer、同号 GPU 收 RDMA 后再 NVLink 转发，属于 normal/high-throughput 路径。**
- **V1 low-latency 的“0 SM”是阶段性描述。** SEND issue 和 RECV/poll 仍会运行 CUDA kernel；WQE 已提交、数据在 NIC/DMA 中飞行的中间窗口才不需通信 kernel 常驻占用 SM。
- **`send_buffer` 和 `recv_buffer` 是同一对称堆中的不同逻辑分区，不是同一个地址。**
- **18 个 int 的 meta 是一次逻辑 put（72 B），但不应笼统称为“72 字节原子写”。** IBGDA 可能按注册内存 chunk 拆 WQE；协议不依赖任意 72 字节观察具有通用原子性。

---

## 1. 一页建立全局心智模型

### 1.1 DeepEP 解决什么问题

MoE 层中，gate/top-k 为每个 token 选择少量 expert：

```text
源 token
  └─ Dispatch：按 top-k 把 token 送到 expert 所在 rank
       └─ 本地 expert GEMM / FFN
            └─ Combine：把 expert 输出送回源 token 并合并
```

若 EP rank 数为 `R`、全局 expert 数为 `E`，则：

```text
num_local_experts = E / R
owner_rank(e)      = e / num_local_experts
local_expert(e)    = e % num_local_experts
```

Dispatch 不是普通固定切片 All-to-All，因为：

- 一个 token 可复制给多个 expert；
- 多个目标 expert 可能在同一 rank，hidden payload 应按 rank 去重；
- expert 负载动态且可能严重偏斜；
- FP8 payload 还带 scale；
- combine 必须恢复原 source rank/token；
- 通信应与 Attention、GEMM 重叠。

### 1.2 V1 两条路径

| 维度 | V1 normal / high-throughput | V1 low-latency |
|---|---|---|
| 主要 CUDA 文件 | `internode.cu` | `internode_ll.cu` |
| 典型负载 | 训练、prefill、较大 token 数 | decode、小 batch、高频调用 |
| 数据布局 | 动态计数、分 channel 有界队列 | expert × source 固定上界 slot |
| 通信推进 | 通信 block/warp 持续推进 | 短 SEND；NIC 后台传输；短 RECV |
| SM 控制 | `Config.num_sms`，两个 block 一个 channel | 不使用 normal 的通信 channel/五角色结构 |
| 跨节点 | 同号 GPU RDMA，再 NVLink 转发 | 对目标全局 rank/expert 的固定对称地址 |
| 到达协议 | meta + ring head/tail + NVLink queue | count/flag |
| 输出形状 | 首次可按真实接收量分配 | 固定 expert-major 上界 + 有效 count |
| IBGDA | 使用 | 也使用 |

一句话：

> normal 用“常驻通信工人 + 多级有界流水线”换吞吐；low-latency 用“固定地址 + 简单信号 + 分离 issue/receive”换小消息时延和可隐藏的网络飞行时间。

---

## 2. 从硬件到源码

### 2.1 硬件角色

- **SM**：执行 CUDA kernel；normal 的 sender/forwarder/receiver 都消耗 SM 周期。
- **HBM/L2**：保存 tensor、IPC buffer、对称堆、workspace。
- **NVLink**：同节点 GPU 间传输。
- **NIC/HCA**：执行 RDMA WQE，从本地 GPU memory DMA 读、写远端 GPU memory。
- **QP**：RDMA queue pair，上层用 QP id 分流请求。

### 2.2 软件分层

```text
Python deep_ep.Buffer
    ↓ 参数检查、tensor/handle/event
C++ legacy::Buffer
    ↓ 内存、stream、IPC、NVSHMEM、launch
CUDA kernels
    ├─ layout.cu
    ├─ intranode.cu
    ├─ internode.cu
    └─ internode_ll.cu
        ↓
ibgda_device.cuh
    ↓ lkey/rkey、WQE、QP、doorbell
NVSHMEM symmetric heap + NIC
```

### 2.3 代码地图

| 文件 | 作用 |
|---|---|
| `deep_ep/buffers/legacy.py` | Python Buffer、normal/LL API、hook、event |
| `csrc/legacy/buffer.hpp` | C++ 编排、输出分配、stream、ping-pong |
| `csrc/legacy/config.hpp` | Config、size hint、LowLatencyLayout |
| `csrc/kernels/legacy/layout.cu` | top-k 路由统计 |
| `csrc/kernels/legacy/intranode.cu` | 单节点 NVLink normal |
| `csrc/kernels/legacy/internode.cu` | 多节点 normal RDMA + NVLink |
| `csrc/kernels/legacy/internode_ll.cu` | LL dispatch/combine、count/flag |
| `csrc/kernels/legacy/buffer.cuh` | Buffer/AsymBuffer/SymBuffer |
| `csrc/kernels/legacy/ibgda_device.cuh` | GPU 侧 IBGDA put/atomic |
| `tests/legacy/test_internode.py` | normal 测试 |
| `tests/legacy/test_low_latency.py` | LL、FP8、hook、LogFMT、zero-copy |

---

## 3. Rank 与两级拓扑

### 3.1 normal 的 rank 分解

V1 normal 把最多 8 张 GPU 视为一个 NVLink 域：

```cpp
rdma_rank = rank / LEGACY_NUM_MAX_NVL_PEERS; // 8
nvl_rank  = rank % LEGACY_NUM_MAX_NVL_PEERS;
```

```text
global_rank = rdma_rank * 8 + nvl_rank
```

- global rank：EP group 的 GPU 编号；
- RDMA rank：节点/scale-out 域；
- NVL rank：节点内 GPU 编号；
- NVSHMEM PE：NVSHMEM 通信实体，normal 与 LL 的映射不同。

### 3.2 normal 的“同号 GPU 中转”

源 `(node2, GPU5)` 的 token 要到 `(node0, GPU3)`：

```text
节点 2 / GPU 5
  └─ IBGDA RDMA
       └─ 节点 0 / GPU 5   ← 远端同号 GPU
            └─ NVLink
                 └─ 节点 0 / GPU 3
```

这规整了 scale-out 路径，但多了 Forwarder 和 NVLink 二跳。

### 3.3 LL 的区别

LL 以全局 rank 作为 NVSHMEM PE，按目标 expert 推导目标 rank，对固定 recv slot 发 put；它没有 normal 的五种 warp 角色和同号 GPU Forwarder 流水线。

判断模式应看调用者是 `internode.cu` 还是 `internode_ll.cu`，不能只看是否出现 `nvshmemi_ibgda_put_nbi_warp`。

---

## 4. NVSHMEM 对称内存

### 4.1 “对称”的含义

每个 PE 从 symmetric heap 分配相同布局。相同逻辑对象拥有相同 heap offset，底层结合目标 PE 的 remote heap base/rkey 定位：

```text
PE0: heap_base_0 + offset_X
PE1: heap_base_1 + offset_X
PE2: heap_base_2 + offset_X
```

最稳妥的理解是“相同 offset 可在任意 PE 定位”，不要把正确性简单归因于所有物理内存地址完全相同。

### 4.2 `SymBuffer` 的 send/recv 切分

`buffer.cuh` 核心公式：

```cpp
num_bytes = num_elems * sizeof(dtype_t);
per_channel_bytes = num_bytes * num_ranks;
total_bytes = per_channel_bytes * num_sms * (kDecoupled + 1);
send_ptr = base + per_channel_bytes * sm_id;
recv_ptr = base + per_channel_bytes * (sm_id + num_sms);
```

normal 中传入：

```text
sm_id   参数 ← channel_id
num_sms 参数 ← num_channels
```

`kDecoupled=true` 时：

```text
[ch0 send]...[chN-1 send]
[ch0 recv]...[chN-1 recv]
```

每个 channel 内再按 peer：

```text
send_buffer(dst_rank)  = 发往谁
recv_buffer(src_rank)  = 来自谁
```

### 4.3 send/recv 都属于对称堆

整个 `rdma_buffer_ptr` 来自 NVSHMEM 分配，send/recv 只是逻辑角色：

| 分区 | 写者 | 读者 |
|---|---|---|
| 本 PE send 区 | 本地 Sender | 本地 NIC 作为 RDMA 源读取 |
| 本 PE recv 区 | 远端 RDMA write；本地特例 | 本地 Forwarder/Receiver |

### 4.4 为什么 send 也必须在当前实现的注册区域

RDMA write 的 NIC 要 DMA 读本地源，WQE 需要 `local address + lkey`。代码从 symmetric heap key table 查 lkey：

```cpp
heap_start = nvshmemi_device_state_d.heap_base;
idx = ((laddr - heap_start) >> log2_cumem_granularity)
      * num_devices_initialized + dev_idx;
device_key = state->constmem.lkeys[idx];
```

远端目标以 heap offset 和 `dst_pe` 查 rkey：

```cpp
roffset = raddr - heap_start;
out_raddr = peer_heap_base_remote[dst_pe] + roffset;
```

WQE data segment：

```cpp
data_seg.byte_count = bytes;
data_seg.lkey = lkey;
data_seg.addr = laddr;
```

所以 send 必须位于这条 IBGDA 快路径能查到 lkey 的注册区域；原因是 NIC 要读它，不是远端 GPU 会直接读 send 区。理论上可以单独注册其他 GPU memory，但当前路径没有为普通 send buffer 使用另一套注册/key 管理。

### 4.5 data/meta 为什么分两段，head/tail 为什么不分

data/meta 同时有 outgoing 与 incoming 写入，若同 slot 会覆盖，故 `kDecoupled=true`。

head/tail 是按方向与 peer slot 管理的单调计数器：

- head：consumer 更新、producer 读取；
- tail：producer 更新、consumer 读取。

单值所有权已明确，故使用 `SymBuffer<..., false>`，不再复制 send/recv 两套。

---

## 5. IBGDA 基础

### 5.1 名词

- GPUDirect RDMA：NIC 直接 DMA GPU memory；
- NVSHMEM：对称堆、PE、put/atomic/barrier；
- IBGDA：GPU 直接构造/提交 NIC 工作请求；
- QP：RDMA 上下文；
- WQE：一次 write/atomic 的描述；
- lkey/rkey：NIC 访问本地/远端注册内存的 key。

### 5.2 `put_nbi_warp` 的过程

```text
1. 选 (dst_pe, qp_id) 的 RC QP
2. 源地址查 lkey
3. 目标 offset + dst_pe 查 remote address/rkey
4. 跨注册 chunk 时计算多个 WQE
5. warp 预留 WQE slots
6. lanes 填 ctrl/raddr/data segments
7. lane 0 提交/doorbell
8. 返回；网络可能仍在传输
```

`nbi` 表示非阻塞发起，不表示 remote consumer 已完成。上层仍需 tail、count、flag 或 barrier。

### 5.3 QP id 的两种映射

normal 常见：

```cpp
qp_id = channel_id;
```

LL dispatch 常见：

```cpp
qp_id = dst_expert_local_idx;
```

QP 不是 CUDA stream。channel 还包含 GPU 工作切分、buffer 分区、队列状态，范围比 QP 更大。

### 5.4 内存顺序

读通信代码必须同时问：

1. payload 写到哪里；
2. 哪个 signal 表示可见；
3. payload 与 signal 走何种 QP/顺序；
4. consumer 用何种 acquire/轮询。

不能只凭 C++ 调用先后，宣称远端对多字节写入有任意原子可见性。

---

## 6. normal dispatch：layout、channel 与五种角色

### 6.1 调用链

```text
topk_idx
 → get_dispatch_layout
 → rank/expert counts + is_token_in_rank
 → notify_dispatch
 → prefix matrices
 → internode dispatch
 → RDMA Sender
 → IBGDA 到目标同号 GPU
 → Forwarder
 → NVLink Receiver
 → recv_x + handle
```

### 6.2 layout 统计

| 输出 | 语义 |
|---|---|
| `num_tokens_per_rank` | 发往每个 global rank 的去重 token 数 |
| `num_tokens_per_rdma_rank` | 发往每个节点的去重 token 数 |
| `num_tokens_per_expert` | 每 expert 选择数 |
| `is_token_in_rank[T,R]` | token 是否命中该 rank 任一 expert |

若同一 token 的两个 expert 在同一 rank，hidden 只发一次，local top-k metadata 再表达它属于哪些本地 expert。

### 6.3 notify 与 prefix matrix

对每个目标 RDMA rank、channel：

```text
total_count:
  本 channel 有多少 token 至少发往目标节点任一 GPU

per_nvl_rank_count[i]:
  本 channel 有多少 token 发往目标节点 GPU i
```

写矩阵后沿 channel 做前缀和：

```text
channel:     0    1    2    3
count:      50   80  120  250
prefix:     50  130  250  500
```

channel 3 区间为 `[250,500)`。

### 6.4 channel 切 token

`get_channel_task_range`：

```cpp
per = ceil_div(num_tokens, num_channels);
start = min(per * channel_id, num_tokens);
end = min(start + per, num_tokens);
```

notify 与 dispatch 用同一函数，所以统计与发送不会错位。

### 6.5 两个 block 一个 channel

```cpp
num_channels = num_sms / 2;
channel_id = sm_id / 2;
is_forwarder = (sm_id % 2 == 0);
```

`num_sms=20` 时有 10 个 channel：

```text
block 0,1   → channel 0
block 2,3   → channel 1
...
block 18,19 → channel 9
```

口语常说“20 个通信 SM”，严格说 launch 的是 20 个 block，`__launch_bounds__` 与资源占用使其按一个 block/SM 的目标运行。

### 6.6 五种 warp 角色

| 角色 | 作用 |
|---|---|
| `kRDMASender` | 准备各目标的 token message |
| `kRDMASenderCoordinator` | 等连续事务，按 chunk 发 put，更新 tail |
| `kRDMAAndNVLForwarder` | 消费 RDMA recv queue，生产 NVLink queue |
| `kForwarderCoordinator` | 汇总消费进度，归还 head/credit |
| `kNVLReceivers` | 消费 NVLink queue，写最终输出 |

dispatch 中 `kNumDispatchRDMASenderWarps=7`，每 block：

```text
7 Sender + 1 Coordinator + 8 NVL-related = 16 warps
```

偶数 block 的前 8 warp 是 Forwarder；奇数 block 的前 7 warp 是 Sender，第 8 个是 SenderCoordinator，后 8 个是 NVLReceiver。

---

## 7. 四个 meta 深度解析

### 7.1 18-int 布局

```cpp
SymBuffer<int>(..., 8 * 2 + 2, kNumRDMARanks,
               channel_id, num_channels)
```

每 `(channel,peer)` 18 int，共 72 B：

| offset | 含义 |
|---:|---|
| `0..7` | 各目标 NVL GPU 的本 channel 起点（前一 channel prefix） |
| `8..15` | 各目标 NVL GPU 的本 channel 终点（当前 prefix） |
| `16` | 目标节点全部 RDMA payload 的 channel 起点 |
| `17` | 目标节点全部 RDMA payload 的 channel 终点 |

Forwarder 对一个目标 GPU 读：

```cpp
meta_0 = recv_buffer(src)[dst_nvl_rank];
meta_1 = recv_buffer(src)[8 + dst_nvl_rank];
meta_2 = recv_buffer(src)[16];
meta_3 = recv_buffer(src)[17];
```

### 7.2 负数就绪编码

```text
encoded = -value - 1
decoded = -encoded - 1

原值 0   → -1
原值 100 → -101
```

初始 0 表示未到；所有 meta `<0` 表示相应目录值已发布。

### 7.3 Sender 写哪里

远端目标时：

1. 本地写 `send_buffer(dst_rdma_rank)`；
2. put 的远端目标 offset 使用 `recv_buffer(src_rdma_rank)`；
3. `dst_pe` 决定实际目标节点；
4. `qp_id=channel_id`。

三个视角：

```text
本地 send slot 下标：目标是谁
远端 recv slot offset：来源是谁
dst_pe：写到哪个远端 PE
```

send 与 recv 不是同地址；对应关系来自明确的 offset 公式和目标 PE。

### 7.4 Receiver 怎么读

Forwarder 的 `lane_id < kNumRDMARanks` 时，一个 lane 负责一个来源：

```text
lane 0 → recv_buffer(0)
lane 1 → recv_buffer(1)
lane 2 → recv_buffer(2)
...
```

各 lane 独立轮询，最后 `__syncwarp()`。快来源可先满足条件，但整个 warp 在同步点仍受最慢必要来源限制。

### 7.5 `recv_buffer(2)[5]` 的 channel 在哪

假设：

```text
channel_id=3
src_rdma_rank=2
dst_nvl_rank=5
```

`recv_buffer(2)[5]` 表示“来源节点 2、目标为本节点 GPU5 的起点 meta”。channel 3 不在两个下标里，而在 `recv_buffer` 指针本身：

```text
recv_ptr = base + per_channel_bytes * (channel_id + num_channels)
```

Sender 不是运行时额外“选择写 channel 3”；它所在 block 已有 `channel_id=sm_id/2`，这个 id 同时进入 SymBuffer 指针和 QP id。

### 7.6 数字例子

来源节点 2 发往目标 GPU5：

```text
channel count: 40  60  30  70
prefix:        40 100 130 200
```

发往目标节点全部 GPU 的去重 payload：

```text
channel count: 200 300 150 250
prefix:         200 500 650 900
```

channel 3：

```text
meta_0 = -131   → GPU5 start=130
meta_1 = -201   → GPU5 end=200
meta_2 = -651   → node payload start=650
meta_3 = -901   → node payload end=900

发往 GPU5 数量 = 70
节点级去重 payload 数 = 250
```

同一 payload 可通过 `SourceMeta` 转发给多个目标 GPU，所以 per-GPU 数之和可大于节点级去重数。

### 7.7 meta 不等于 data 已全部到齐

meta 像目录，给出期望范围；真正可消费进度由 tail 发布：

```text
meta 到达 → 知道预期数量/区间
tail 推进 → 知道哪些 ring slots 可读
```

不能把“meta 先发”理解为“meta 一到所有 data 已同步到齐”。

---

## 8. normal ring buffer、head/tail 与两级转发

### 8.1 message

```text
hidden
+ FP8 scales（可选）
+ SourceMeta
+ topk_idx
+ topk_weights
```

`SourceMeta` 表达来源与目标节点内哪些 GPU 需要 token，使 RDMA 层按节点去重、NVLink 层再 fan-out。

### 8.2 多 Sender 与 Coordinator

多个 Sender warp 可能乱序完成 message，shared memory 维护 lock、连续 tail 和 32-bit completion window。只有从当前 tail 起连续完成的事务能发布。SenderCoordinator：

1. 等连续已准备区间；
2. 凑 chunk 或处理最后一批；
3. 从 `send_buffer(dst)` 发到远端 `recv_buffer(src)`；
4. 更新远端 tail。

这把“乱序生产”与“连续 RDMA chunk”解耦。

### 8.3 环形队列

容量 `C`：

```text
physical_slot = logical_index % C
0 <= tail - head <= C
```

- tail：producer 已发布；
- head：consumer 已消费可复用；
- tail-head：占用槽；
- C-(tail-head)：credit。

Sender 容量不足时轮询 head。`rdma_send_chunk <= rdma_recv_capacity/2` 是 lazy head update 安全推进的重要约束。

### 8.4 tail 与 head 的方向

```text
源 SenderCoordinator
  └─ atomic 更新目标节点 tail
       └─ 目标 Forwarder 轮询：“数据到哪了”

目标 ForwarderCoordinator
  └─ atomic 更新源节点 head
       └─ 源 Sender 轮询：“哪些槽可复用”
```

两个 coordinator 不直接发消息，而通过远端对称 head/tail slot 间接协作。

一条 RDMA payload 可能被多个目标 NVL Forwarder 使用，所以 ForwarderCoordinator 对 active Forwarder 的 head 取最小值后才归还 credit，防止过早复用。

### 8.5 Forwarder 与 NVLReceiver

Forwarder：

1. 解 meta，得到期望范围；
2. 轮询来源 tail；
3. 读 `rdma_channel_data.recv_buffer(src)`；
4. 依据 `SourceMeta` 判断当前目标 GPU 是否需要；
5. 等 NVLink queue credit；
6. TMA/向量复制到目标 GPU IPC buffer；
7. 更新 NVLink tail 和自身 head。

`kNVLReceivers` 读的是 `nvl_channel_x`，不是 RDMA recv buffer；它最终写 `recv_x`、top-k、weights、scales、source metadata。

```text
源 GPU send buffer
 → IBGDA
 → 目标节点同号 GPU RDMA recv buffer
 → Forwarder
 → 目标 GPU NVLink queue
 → NVLReceiver
 → recv_x
```

### 8.6 dispatch handle 与 combine

handle 保存 prefix、`recv_src_meta`、`is_token_in_rank`、`send_rdma_head`、`send_nvl_head` 等逆向路由。combine 沿 dispatch 逆路径返回并累加。

normal combine 的源码语义强调向量 “addition without weights”；不要与 LL combine 内部明确执行 `result * topk_weights` 混淆。上层需确认权重在哪一层应用。

cached handle 只能在路由布局、容量、rank/expert/hidden/config 匹配且旧异步操作完成时复用。

---

## 9. V1 low-latency 的固定布局

### 9.1 为什么另做协议

decode token 少、调用频繁，normal 的 layout、notify、CPU exact-count、ring 流控、Forwarder 二跳可能成为固定成本。LL 用更大固定显存换固定地址和简单信号。

### 9.2 expert-major 预留

设：

```text
M = max tokens per source rank
R = num_ranks
L = local experts
```

```text
expert0: [src0 M slots][src1 M slots]...[srcR-1 M slots]
expert1: [src0 M slots][src1 M slots]...[srcR-1 M slots]
...
```

容量为 `L*R*M`；真实数量由 `packed_recv_count[e]` 给出，RECV 把有效 token 压到每 expert 前部。

### 9.3 `LowLatencyLayout` 与 ping-pong

对称 RDMA buffer：

```text
[signal0][signal1]
[send0][send1]
[recv0][recv1]
```

dispatch/combine 共享物理区域，取二者 message 大小的最大值。当前轮使用一套，同时清下一套 signal，index 0/1 交替。

限制：

- 只有两套，不能无限增加在途请求；
- hook、返回 view、zero-copy 必须与 index 对齐；
- 不应长期保留并继续消费超过两轮的旧 view。

### 9.4 message 大小

dispatch：

```text
int4 control/source
+ max(BF16 hidden, FP8 hidden + per-128 scales)
```

combine：

```text
per-128 metadata + BF16 hidden 上界
```

总 size 约为两套 send + 两套 recv + 两套对齐 signal。`M`、hidden、rank、expert 数增大会明显增加显存。

---

## 10. LL dispatch

### 10.1 输入输出

```text
x: BF16 [T,H], T<=M
topk_idx: [T,K]

packed_recv_x:
  [local_experts, R*M, H]
packed_recv_x_scales:
  逻辑 [local_experts, R*M, H/128]
packed_recv_count:
  [local_experts]
```

handle 保存 source info、每 `(expert,src_rank)` 的 packed range，以及布局参数。

### 10.2 expert 映射

```cpp
responsible_expert_idx =
    sm_id * num_warp_groups + warp_group_id;
```

数据 warp 负责 hidden/FP8/put；计数 warp 扫 top-k；RECV sub-warps 等 count 并压紧。SEND/RECV 仍占 SM，只是中间网络飞行不需常驻通信 kernel。

### 10.3 SEND phase

```text
读 BF16
 → 可选每 128 channel 求 amax/scale并转 FP8
 → 读 top-k expert
 → atomicAdd(target expert counter) 分配 slot
 → 算 dst rank/local expert/固定地址
 → 本地可 P2P ? warp copy : IBGDA put
 → release finish counter
```

目标地址：

```text
recv_base
+ dst_local_expert * R * M * msg_bytes
+ src_rank * M * msg_bytes
+ slot_idx * msg_bytes
```

唯一定位 `(目标 local expert,来源 rank,该来源 slot)`，无需动态 offset 协商。

### 10.4 payload 后发布 count

固定地址只解决“写哪”，receiver 还需知道数量。计数 warp 统计 expert count，并等待相应 payload 请求已发起，再发布 count：

```text
payload → count
```

count 为 0 也可能是有效结果，所以 signal 的初始化和清理是协议一部分。

### 10.5 RECV phase

对每个 local expert、source rank：

1. 等 recv count；
2. atomic add 分配 packed 连续区间；
3. 记录 `layout_range=(offset,count)`；
4. 复制 hidden/scale/source index；
5. 更新 `packed_recv_count`；
6. 可选累计 expert 负载与等待周期。

grouped GEMM 仅处理 `packed_recv_x[e,:count[e],:]`。

### 10.6 FP8 scale stride

逻辑 shape 为 `[expert,token,H/128]`，物理布局可能为 TMA/GEMM 使用列主序 stride，再以 view/transpose 暴露。因此调试同时检查 `shape` 与 `stride`。`use_ue8m0`、`round_scale`、hidden 对齐要遵守当前断言。

---

## 11. LL combine

### 11.1 SEND

输入 expert-major BF16 输出、top-k、weights 和 dispatch handle。对 `(local_expert,source rank)`：

1. 从 `layout_range` 解 offset/count；
2. 从 `src_info` 恢复原 token index；
3. 读 expert output；
4. 可选复制/LogFMT 编码；
5. 写源 rank 的固定 recv slot；
6. payload 后写 flag。

### 11.2 RECV 与加权

```text
for token:
  accum = 0
  for top-k:
    invalid -1 → skip
    wait flag
    accum += expert_result * topk_weight
  store BF16
```

### 11.3 LogFMT

LogFMT 是 combine 内部 payload 压缩，不是用户输入 dtype。它减少网络字节但增加编解码；收益取决于网络瓶颈与 GPU 余量，和 zero-copy 的组合按测试支持使用。

### 11.4 zero-copy

```python
rdma_out = buffer.get_next_low_latency_combine_buffer(handle)
# GEMM 直接写 rdma_out
combined_x, event, hook = buffer.low_latency_combine(
    rdma_out, topk_idx, topk_weights, handle,
    zero_copy=True,
)
```

必须保证 shape/dtype/stride、ping-pong index、GEMM→SEND event 和旧通信生命周期正确。

---

## 12. Hook 与“0 SM”

### 12.1 phase

```text
LEGACY_LOW_LATENCY_SEND_PHASE
LEGACY_LOW_LATENCY_RECV_PHASE
```

`return_recv_hook=True`：

1. 首次只执行 SEND；
2. 返回 callable；
3. 安排独立计算；
4. 调 hook，执行 RECV。

### 12.2 准确时间线

```text
GPU SM: [SEND issue]                    [RECV/poll/pack]
NIC:                 [RDMA flying.....]
Compute:             [independent work]
```

“0 SM”只指 NIC 飞行窗口。SEND、RECV、本地 P2P、pack、规约仍可占 SM/HBM/L2。

### 12.3 双 micro-batch

```text
Batch A: Attn A | D-SEND A |··RDMA··| D-RECV A | MoE A | C-SEND A |··RDMA··| C-RECV A
Batch B:                           Attn B | D-SEND B |··RDMA··| D-RECV B | MoE B
```

若独立计算覆盖网络时间，hook 时等待短；否则仍等待剩余网络。hook 只提供调度机会。

### 12.4 生命周期

- hook 必须调用；
- 调用前不能释放/覆盖捕获的 tensor、handle、buffer；
- 与 `async_finish` 的组合按当前接口断言；
- 各 rank 调用顺序要兼容；
- ping-pong 只有有限在途深度；
- P2P bypass 可能与计算争资源。

---

## 13. Python 使用框架

### 13.1 LL Buffer

```python
max_tokens = 128
num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
    max_tokens, hidden, group.size(), num_experts
)

buffer = deep_ep.Buffer(
    group,
    num_nvl_bytes=0,
    num_rdma_bytes=num_rdma_bytes,
    low_latency_mode=True,
    num_qps_per_rank=num_experts // group.size(),
)
```

### 13.2 Dispatch + Combine hook

```python
recv_x, recv_count, handle, event, recv_hook = \
    buffer.low_latency_dispatch(
        hidden_states,
        topk_idx,
        num_max_dispatch_tokens_per_rank=max_tokens,
        num_experts=num_experts,
        use_fp8=True,
        async_finish=False,
        return_recv_hook=True,
    )

run_independent_work()
recv_hook()

expert_output = run_grouped_gemm(recv_x, recv_count)

combined_x, event, combine_hook = buffer.low_latency_combine(
    expert_output,
    topk_idx,
    topk_weights,
    handle,
    async_finish=False,
    return_recv_hook=True,
)

run_other_independent_work()
combine_hook()
```

具体 tuple 和参数以当前 `legacy.py` 为准，示例强调依赖边界。

### 13.3 stream/event

```text
compute stream 生产输入
 → previous_event
 → comm_stream 等待并启动通信
 → EventOverlap
 → consumer 真正读结果前 wait
```

异步不是“不等待”，而是把等待移动到真实数据依赖边界。缺 wait 常表现为偶发错误。

---

## 14. 初始化、容量与清理

### 14.1 normal Config

```text
nvl_send_chunk < nvl_recv_capacity
rdma_send_chunk < rdma_recv_capacity
rdma_send_chunk <= rdma_recv_capacity / 2
```

RDMA recv capacity 还会向 send chunk 倍数对齐，以避免跨 ring 尾、支持 lazy head update。

### 14.2 LL 约束

- `num_experts % num_ranks == 0`；
- QP 数最好覆盖 local expert 数；
- `T <= max_tokens`；
- hidden/top-k/scale pack 满足模板和对齐；
- `R*max_tokens` 满足接收/TMA 对齐；
- QP depth 覆盖可能在途 WQE。

显存近似随 `num_experts * max_tokens * message_bytes * 2` 增长，`max_tokens` 不应仅为保险设得极大。

### 14.3 清理

异常中断、hook 未完成、Graph 状态不确定、布局切换、buffer 被其他路径写过时，应调用 low-latency clean。清理涉及全局同步，不能在旧 WQE 仍在途时单 rank 清零 signal。

### 14.4 shrink/mask

启用 shrink 后可 mask 异常 rank，通信与 barrier 跳过它，并用诊断状态定位超时。但被 mask rank 的 expert 结果会缺失；这是降级机制，不是透明一致性恢复。

---

## 15. 性能分析与调优

### 15.1 normal 为什么快

- 多 channel/QP；
- 多 Sender + Coordinator 合并 chunk；
- 节点级 RDMA 去重 + NVLink fan-out；
- TMA/向量化；
- producer/consumer ring 流水；
- 通信 SM 与计算并发。

成本是 layout/notify、通信 SM、二跳、atomic/polling 和 HBM/L2/NVLink 竞争。

### 15.2 LL 为什么适合 decode

- 固定地址，消除动态 offset/CPU exact-count；
- FP8 cast + route + put 融合；
- expert-major 直接 grouped GEMM；
- count/flag 协议简单；
- 网络飞行可被独立计算覆盖；
- combine 融合权重/压缩/规约；
- zero-copy 可少一次 HBM copy。

### 15.3 LL 可能更慢

- token 多、持续带宽成为瓶颈；
- gate 偏斜导致 expert/QP 热点；
- 无独立计算可覆盖网络；
- hook 调太早；
- 编解码成本大于节省字节；
- P2P 与主计算争 SM/HBM；
- max_tokens 太大导致显存和清理成本高。

### 15.4 采集指标

端到端：

- SEND issue、network gap、RECV、GEMM、combine 各阶段；
- overlap 覆盖率；
- P50/P95/P99。

负载：

- per-expert/per-rank token；
- rank 去重前后复制比；
- 最大/平均 expert 偏斜。

硬件：

- SM、HBM、L2、NVLink、NIC；
- QP/WQE depth、拥塞/重传；
- dispatch/combine wait stats 的慢 rank。

### 15.5 `num_sms`

```text
太少 → 推进慢，NIC/NVLink 可能吃不满
太多 → 计算可用 SM 减少，内存层次竞争加剧
```

应以完整模型 step 调优，而不是只看 dispatch microbenchmark。

---

## 16. 新手 FAQ：碎片中反复出现的问题

### Q1：出现 IBGDA API 就是 low-latency 吗？

不是。normal 和 LL 都用 IBGDA。看 `internode.cu` 还是 `internode_ll.cu`，以及是否有 normal 的 channel/Forwarder/meta。

### Q2：`recv_buffer(2)[5]` 中 2、5、channel 分别是什么？

- 2：source RDMA rank slot；
- 5：18-int meta 中目标 NVL GPU 的 offset；
- channel：已编码进 `recv_ptr`，由 `channel_id=sm_id/2` 决定。

### Q3：Sender 怎么知道写 channel 3？

运行它的 block 已决定 `channel_id`；该 id 同时用于 token 范围、SymBuffer offset 和 QP id，不需消息里再放 channel 字段。

### Q4：send/recv 是靠地址相同对应吗？

不是。send 与 recv 是不同分区。Sender 从本地 `send_buffer(dst)` 取源，显式指定远端 `recv_buffer(src)` offset 与 `dst_pe`。

### Q5：为什么 put 代码在发送端却写 `recv_buffer`？

传给 put 的第一个地址是**远端目标的对称 offset 表达**。在发送端算这个 offset，不等于写发送端本地 recv 区；`dst_pe` 把它翻译到目标 PE。

### Q6：send 区只有本地写，为什么还在对称堆？

NIC 要 DMA 读源，需要 lkey。当前 IBGDA 通过 `laddr-heap_base` 查 symmetric heap 的 key table，所以 send 源也放在已注册对称区域。

### Q7：为什么 data/meta 有 send/recv 两段？

防止 outgoing 与 incoming 双向写同 slot 覆盖。head/tail 所有权单一，所以不用两段。

### Q8：四个 meta 分别是什么？

```text
meta0/1：当前来源、当前 channel、目标 NVL GPU 的起止 prefix
meta2/3：当前来源、当前 channel、目标节点去重 RDMA payload 的起止 prefix
```

### Q9：为什么 meta 编成负数？

初始 0 表示未到，prefix 也可能为 0。`-value-1` 让所有有效值都小于 0。

### Q10：meta 到了就能读 token 吗？

不能仅凭 meta。meta 给期望范围，tail 给已发布 data 进度。

### Q11：72 B meta 是否原子到达？

代码以一次逻辑 put 请求 72 B，但实现可能跨注册 chunk 拆 WQE。不要把它推广成硬件保证的 72 B 原子可见性。

### Q12：ForwarderCoordinator 如何通知 SenderCoordinator？

不直接通知。目标侧更新源端 symmetric head；源侧 Sender 轮询 head。反方向 Sender 更新目标 tail；Forwarder 轮询 tail。

### Q13：`kNVLReceivers` 读 RDMA recv buffer 吗？

不读。Forwarder 读 RDMA recv；NVLReceiver 读节点内 IPC/NVLink queue。

### Q14：“0 SM”是否从 API 开始到结束都不用 SM？

不是。只指 SEND 与 RECV 之间 NIC 后台传输窗口不需常驻通信 SM。

### Q15：hook 返回后 output 可立刻用于 GEMM 吗？

不可。先调用 hook，或等待对应完成事件。

### Q16：为什么 LL 更耗显存？

它为每个 `(local expert,source rank)` 预留 max_tokens，并有 ping-pong send/recv/signal。

### Q17：为什么 LL 不先交换真实 count 再分配？

小 batch 下 count 交换、CPU round trip、动态分配的固定开销可能主导时延；固定上界正是它的取舍。

### Q18：normal 一定比 LL 吞吐高吗？

不是绝对。它们是针对不同 workload 的设计点；应在真实 token、topology、overlap 和端到端模型中测量。

---

## 17. 常见故障定位

### 17.1 meta timeout

检查：

- 所有 rank 是否进入同一轮；
- num_channels/num_sms/QP 数是否一致；
- prefix matrix 是否越界或旧 handle 错配；
- symmetric buffer 大小、布局、初始化是否一致；
- 某来源 rank 是否早退或崩溃。

### 17.2 RDMA sender timeout

常见是 head 不推进：

- 目标 Forwarder 未消费；
- 某 NVLink destination 阻塞，导致最小 head 不前进；
- recv capacity/chunk 配错；
- target rank 调用顺序错；
- tail/head slot 或 QP 映射不一致。

### 17.3 Forwarder NVL timeout

- 目标 NVLReceiver 未运行；
- NVLink queue head 不更新；
- IPC handle/peer mapping 错；
- 目标输出或 prefix 不匹配；
- 某 channel 的目标 count 计算错误。

### 17.4 LL count/flag timeout

- 某 rank 漏调 SEND/hook；
- ping-pong index 失步；
- signal 未清理或被提前清理；
- max_tokens/QP depth 不足；
- 被 mask rank 的语义与等待集合不一致；
- payload 发起/finish counter 未达到协议值。

### 17.5 偶发数值错

- event/wait 缺失；
- handle 与另一轮 top-k 混用；
- 旧 tensor view 已被 ping-pong 复用；
- FP8 scale stride 误解；
- normal/LL combine 权重语义混淆；
- zero-copy buffer 写入未完成就 SEND。

---

## 18. 建议阅读顺序

### 第一轮：只看语义

1. `docs/legacy.md` 的 API；
2. 本文第 1～3 章；
3. `deep_ep/buffers/legacy.py` 的公开方法。

### 第二轮：normal

1. `layout.cu`；
2. `internode.cu::notify_dispatch`；
3. `buffer.cuh::SymBuffer`；
4. `internode.cu::dispatch` 的五角色；
5. meta 写入 597～623 附近；
6. data put/tail 815～844 附近；
7. Forwarder meta/tail/data 与 coordinator head；
8. combine 与 tests。

### 第三轮：LL

1. `config.hpp::LowLatencyLayout`；
2. Python size hint/dispatch；
3. C++ ping-pong/phase/hook；
4. `internode_ll.cu::dispatch`；
5. `internode_ll.cu::combine`；
6. `test_low_latency.py` 的 FP8、hook、LogFMT、zero-copy、shrink。

### 第四轮：IBGDA

1. `ibgda_get_lkey_and_rkey`；
2. `ibgda_write_rdma_write_wqe`；
3. `nvshmemi_ibgda_put_nbi_warp`；
4. atomic add 与 QP id；
5. 回到上层逐个标记 payload、signal、producer、consumer。

---

## 19. 术语表

| 术语 | 含义 |
|---|---|
| EP | Expert Parallelism |
| Dispatch | token 按 top-k 发往 expert owner |
| Combine | expert 结果返回源 token 并合并 |
| NVL rank | 节点内 GPU 编号 |
| RDMA rank | normal 中节点/scale-out 编号 |
| PE | NVSHMEM processing element |
| Symmetric heap | 各 PE 具有相同逻辑布局的注册内存 |
| IBGDA | GPU 直接发起 InfiniBand 工作请求 |
| QP | Queue Pair |
| WQE | Work Queue Element |
| lkey/rkey | NIC 本地/远端内存访问 key |
| channel | GPU 工作、buffer、queue、QP 分流的独立通信流水线 |
| head/tail | 有界 ring 的消费/发布逻辑位置 |
| credit | 可复用空槽数量 |
| meta | normal Forwarder 的前缀目录 |
| SourceMeta | token 来源及目标 NVL GPU 位图 |
| hook | 将 LL RECV 延后启动的 callable |
| ping-pong | 两套 buffer 交替使用 |
| P2P bypass | 本地 peer 可达时不用 NIC，直接 GPU copy |
| TMA | Hopper Tensor Memory Accelerator |
| LogFMT | LL combine 内部低比特 payload 编码路径 |

---

## 20. 最终总结

DeepEP V1 normal 的本质：

> 用多个通信 channel 和五种 warp 角色驱动分层有界队列；跨节点先由同号 GPU 通过 IBGDA 收取，再由 Forwarder 经 NVLink fan-out；四个负编码 meta 描述期望区间，head/tail 完成数据发布和 credit 回收，dispatch handle 保存 combine 的逆路由。

DeepEP V1 low-latency 的本质：

> 为每个 expert/source 预留固定对称 RDMA slot，用 GPU 融合路由、FP8 cast 与 IBGDA issue，以 count/flag 表达到达，再用 ping-pong 和 SEND/RECV hook 把 NIC 网络飞行窗口隐藏在其他计算后面。

理解两者的关键，不是背函数名，而是每次都回答：

```text
数据由谁生产？
写在哪个 PE 的哪段 buffer？
谁读取？
用哪个 signal 表示可见？
容量不足时谁归还 credit？
网络飞行时是否仍有 kernel 占 SM？
```

只要能沿这六个问题追踪一条 token，`send_buffer`、`recv_buffer`、channel、meta、head/tail、Forwarder 与 hook 就会从零散变量变成一条完整的数据生命线。

## 附录：仓库内进一步阅读

- [V1 SM 通信方案](docs/implementation-v1-sm.md)
- [V1 Low-Latency 方案](docs/implementation-v1-low-latency.md)
- [Legacy API 与使用说明](docs/legacy.md)
- [NVSHMEM 配置](docs/nvshmem.md)
- [normal 多节点内核](csrc/kernels/legacy/internode.cu)
- [low-latency 内核](csrc/kernels/legacy/internode_ll.cu)
- [IBGDA 设备接口](csrc/kernels/legacy/ibgda_device.cuh)
- [对称/非对称 buffer](csrc/kernels/legacy/buffer.cuh)
