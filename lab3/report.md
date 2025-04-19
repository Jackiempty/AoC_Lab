# Lab 3 Report

## Lab Design Description
Scores will be assigned based on the level of detail and the logical soundness of your description.
Each section should ideally include a block diagram to explain the module.

### Explain how you implement PE
> Explain how you handle the computation of the zero point of dequantization.

透過 `PE_config` 解出本次任務的設定（F, q），並進行 `F × q` 次的 MAC 計算，最後輸出一個結果 opsum，搭配有效的 opsum_valid。

此外整個資料交換過程透過 valid/ready handshaking 實現資料同步與控制流

#### PE_config 解碼區塊
```verilog
is_fc     = i_config[9];
p         = i_config[8:7] + 1;
F         = i_config[6:2] + 1;
q         = i_config[1:0] + 1;
total_ops = F * q;
```
* 這段邏輯解析輸入的 `i_config` 並計算總共要做幾次 MAC。
* 其中 p 雖未被使用，但保留以後擴展多 PE 協作。
* `total_ops` 是整個 1D convolution primitive 的迭代長度。

#### FSM 狀態控制區塊
```
IDLE → COMPUTE → WAIT_OUT → IDLE ...
```
* DLE：等 `PE_en` 為 1，開始新的 primitive
* COMPUTE：等待資料三方 valid，進行一次 MAC 並累加至 `accum`
* WAIT_OUT：計算結束，等待 `opsum_ready` 才送出結果、回到 idle

#### 資料運算與 Zero-Point Subtraction
```verilog
ifmap_s  = $signed(ifmap[7:0]) + 8'sd128;
filter_s = $signed(filter);
ipsum_s  = $signed(ipsum);
```
* `ifmap` 是 `uint8`，經過減去 128 轉換成 signed int（symmetric quantization）
* `filter` 與 `ipsum` 皆視為 signed 32-bit
* `mac_result = ifmap_s * filter_s` 是單次乘法
* `accum = ipsum_s + mac_result` 是累加加法

#### Handshake 控制與流控
```verilog
always_comb begin
opsum        = accum;
ifmap_ready  = (state == COMPUTE);
filter_ready = (state == COMPUTE);
ipsum_ready  = (state == COMPUTE);
end
```
* 保持資料在 COMPUTE 狀態中才能送進來
* 搭配 `*_valid` 檢查三路資料是否同時有效
* `opsum_valid` 在 `WAIT_OUT` 階段拉高，等待下游 `opsum_ready`

### Explain how you implement PE array
> Including network (GIN/GON/LN) and multicast controller (MC)

### Explain how you implement PPU
> Explain how you handle the computation of the zero point of requantization \
> Explain the order of 3 component(PostQuant, MaxPool, ReLU) and the reason why you design like that.

#### Zero Point 的 Requantization 實作
```verilog
shifted = data_in >>> scaling_factor;
added   = shifted + 128;
```
原理：  
假設你在 PE 輸出端得到的是 32-bit 的 signed accumulator，為了符合後續 uint8 格式，你需要將這些結果：  
1. 重新縮放（rescale） — 因為量化過的 ifmap 與 weight 在原始 domain（例如 float）中經歷了 s_x * s_w，現在要換成 s_y
2. 加回 zero-point — symmetric quantization 的 zero-point 是 128，需再加回
3. clamp 到 0~255 — 最終結果需要對應到 uint8 範圍

```verilog
shifted = data_in >>> scaling_factor;  // ≈ divide by scale
added   = shifted + 128;               // add zero point (轉回 uint8)
```
對應到：  
$$
\bar y = \text{clamp} \left( \left\lfloor \frac{accum}{scale} \right\rceil + 128, 0, 255 \right)
$$

#### 三個 submodule 的順序 & 為什麼要這樣設計
模組順序是：  
```
ReLU_Qint8 -> post_quant -> max_pool
```
1. `ReLU_Qint8`:
* 作用：對 PE 輸出的 signed 32-bit 做 ReLU：若 < 0 則變成 0
* 為什麼在最前面
    * ReLU 是對 float/signed domain 下的結果運作的
    * 若等到 quantize 成 uint8 後再做 ReLU，0 的位置會變在 128，所以判斷就不是 `data < 0` 了
    * 把它放在 quantize 前面，可以更準確地在 signed domain 做裁切

2. `post_quant`:
* 作用：將 32-bit signed 結果縮放 + 加 zero-point，轉成 uint8，範圍 0~255
* 為什麼在第二
    * 因為需要把 ReLU 裁切過的結果轉成 uint8 給後面 maxpool 用
    * 縮放與 zero-point 是為了 align 到同一 scale domain，通常對應輸出 feature map 的量化參數

3. `Comparator_Qint8`:
* 作用：執行 2x2 區域內的 max pooling，比較數值保留最大值
* 為什麼在最後
    * MaxPooling 是基於空間資料比較的操作，通常是在 quantized 的 uint8 domain 中進行
    * 對已量化過的 feature map 做 max 比較最符合 CNN 的實務推論流程
    * 若對未 quantize 的結果做 max 比較，會造成 scale 不一致而不準確

4. 輸出選擇器 `relu_sel`:
這個設計可以動態決定：
* 要輸出量化後的結果（`post_quant`），或
* 要輸出 max pooling 後的結果（`Comparator_Qint8`）
讓 `PPU` 保有彈性，在不同 layer (e.g. ReLU only, 或 ReLU + MaxPool) 間共用模組。

### Result

| Component | Pass (Y/N) |
|:---------:|:----------:|
|    PE     |      N     |
| PE array  |      N     |
|    PPU    |      Y     |


## Question

### Question 1
Explain how data reuse is achieved in the design presented in the Eyeriss paper.  

Data reuse in Eyeriss is achieved through the Row Stationary (RS) dataflow, which maximizes:  
* Weight reuse within a processing element (PE)
* Input feature map (ifmap) reuse across neighboring PEs
* Partial sum reuse within a PE

Eyeriss also uses:  
* NoC (Network-on-Chip) with multicast & broadcast for efficient distribution
* Spatial mapping of loops (F, Q, P, etc.) across PE array

### Question 2

Compute a `16×16` Conv2D operation, given the following configuration
- Kernel size: `3×3`
- Stride: `1`
- Padding: `1`
- Global Buffer (GLB) size: `128 KB`
- Mapping parameters:
    - `p = 4`
    - `q = 4`
    - `r = 1`
    - `t = 2`
    - `e = 8`

Determine the value of the mapping parameter `m`

---------------------

Given:  
* Input feature map size: 16×16
* Kernel size: 3×3
* Stride: 1
* Padding: 1

So the output feature map will be:  
$$
O_H = O_W = \frac{16 + 2 * 1 -3 }{1} + 1 = 16
$$

Other mappings:
* `p = 4` : # output channels per PE row
* `q = 4` : # input channels per PE column
* `r = 1` : # vertical filter elements per PE
* `t = 2` : # horizontal filter elements per PE
* `e = 8` : # output width tile per PE
 
Let’s recall the mapping parameter m:
> `m = number of output pixels processed over time per PE`

Total output spatial positions:  
$$\text{Total output pixels} =  O_H × O_W = 16 × 16 = 256$$

Each PE computes for `p × e × m` output values over time.  
Let’s compute `m` based on total number of output pixels:  

$$
\text{Total \# of PEs} = p \times q \times r \times t = 4 × 4 × 1 × 2 = 32 \\
\text{Total outputs per PE} = m × e \\
\text{Total outputs from all PEs} = 32 × m × e = 32 × m × 8 = 256 \\
\Rightarrow m = \frac{256}{32 × 8} = 1 
$$

* **Answer: m = 1**

### Question3
In the testbench, explain how to compute the multicast controller’s ID (`X_ID`, `Y_ID`) and the data tags (`tag_X`, `tag_Y`) based on the mapping parameters and shape parameters.  

#### X_ID and Y_ID in multicast controller:  
These are spatial addresses (in PE array) that determine where data is sent.  

For Ifmap (typically multicast across filters):  
* X_ID → fixed for a row (since same ifmap used across filters)
* Y_ID → varies for spatial positions

For Weight:  
* Y_ID → fixed for column (since same filter reused across spatial positions)
* X_ID → varies for output position

#### tag_X, tag_Y:  
These are logical identifiers used to:  
* Track which data belongs to which PE's computation
* Help match data between memory controller and computation schedule

To compute them:  
Usually based on mapping loop index orders (e.g. F, Q, R, T, E, M):  
For example:  
* tag_X = floor(i / e) — the horizontal tile position
* tag_Y = floor(j / m) — the temporal loop index

> Actual expressions depend on loop unrolling and PE array layout.

### Question 4

For the test case where e = 4,
why are the configurations `(r, t)` = `(1, 4)` and `(2, 2)` present, but `(4, 1)` is not?
What do you think is the reason behind this?

|Config | Vertical (r) | Horizontal (t) | PE Utilization | Memory Access
|--|--|--|--|--|
|(1,4) | low | high | good | better ifmap reuse
|(2,2) | balanced | balanced | good | balanced reuse
|(4,1) ❌ | high | low | poor | ifmap reuse poor

Problem with (4,1):  
* You only process 1 horizontal position at a time (1 pixel column)
* But need 4 rows of data from ifmap → poor spatial locality
* Harder to reuse ifmap and more pressure on memory bandwidth

## Lab Feedback and Suggestions