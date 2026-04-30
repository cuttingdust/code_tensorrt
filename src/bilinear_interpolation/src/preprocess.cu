#include "../../nsight_system_and_compute/include/utils.hpp"
/// \brief 最近邻插值 + BGR→RGB 转换 Kernel
/// \param tar   目标图像指针 (RGB格式，3通道，输出)
/// \param src   源图像指针 (BGR格式，3通道，输入)
/// \param tarW  目标图像宽度 (像素)
/// \param tarH  目标图像高度 (像素)
/// \param srcW  源图像宽度 (像素)
/// \param srcH  源图像高度 (像素)
/// \param scaled_w  宽度缩放比例 = srcW / tarW (统一缩放时使用较大值)
/// \param scaled_h  高度缩放比例 = srcH / tarH (统一缩放时使用较大值)
__global__ void resize_nearest_BGR2RGB_kernel(uint8_t* tar, uint8_t* src, int tarW, int tarH, int srcW, int srcH,
                                              float scaled_w, float scaled_h)
{
    /// ==================== 第一部分：计算目标像素坐标 ====================
    /// 每个线程负责处理目标图像中的一个像素
    /// blockIdx: 线程块在网格中的索引，例如 (2, 3) 表示第2行第3列的线程块
    /// blockDim: 线程块的维度，例如 (16, 16) 表示每个线程块有16x16=256个线程
    /// threadIdx: 线程在线程块中的索引，例如 (5, 7) 表示线程块内第5行第7列的线程
    /// 最终坐标计算公式： 全局索引 = 块索引 × 块维度 + 线程块内索引
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 目标列坐标 [0, tarW-1]
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 目标行坐标 [0, tarH-1]

    /// ==================== 第二部分：最近邻映射 ====================
    /// 将目标像素坐标映射到源图像坐标
    /// 例如：目标坐标 x=100，缩放比例 scaled_w=3.84，则源坐标 = 100×3.84 = 384
    /// 使用 round() 四舍五入找到最近的整数像素坐标
    int src_y = round((float)y * scaled_h); // 源图像行坐标
    int src_x = round((float)x * scaled_w); // 源图像列坐标

    /// ==================== 第三部分：边界检查 ====================
    /// 防止访问越界内存，只处理源图像范围内的像素
    /// src_x 有效范围: [0, srcW-1]，src_y 有效范围: [0, srcH-1]
    if (src_x < 0 || src_y < 0 || src_x >= srcW || src_y >= srcH)
    {
        /// 越界的像素不进行计算，目标图像保持原值（初始为0/黑色）
        /// 这种情况通常发生在目标图像边缘，由于向上取整导致映射超出源图范围
    }
    else
    {
        /// ==================== 第四部分：计算内存索引 ====================
        /// 图像在内存中是连续存储的一维数组
        /// 计算公式：索引 = (行 × 宽度 + 列) × 通道数
        /// 乘以3是因为每个像素有3个通道（B、G、R）
        int tarIdx = (y * tarW + x) * 3;         /// 目标像素的起始索引（R通道位置）
        int srcIdx = (src_y * srcW + src_x) * 3; /// 源像素的起始索引（B通道位置）

        /// ==================== 第五部分：BGR → RGB 通道转换 ====================
        /// 源图像内存布局（BGR格式）：
        ///   src[srcIdx+0] = B (蓝色通道)
        ///   src[srcIdx+1] = G (绿色通道)
        ///   src[srcIdx+2] = R (红色通道)
        /// 目标图像内存布局（RGB格式）：
        ///   tar[tarIdx+0] = R (红色通道)
        ///   tar[tarIdx+1] = G (绿色通道)
        ///   tar[tarIdx+2] = B (蓝色通道)
        /// 所以需要交换通道顺序：R←R, G←G, B←B（源R在+2，源B在+0）
        tar[tarIdx + 0] = src[srcIdx + 2]; /// 目标R ← 源R（偏移+2）
        tar[tarIdx + 1] = src[srcIdx + 1]; /// 目标G ← 源G（偏移+1）
        tar[tarIdx + 2] = src[srcIdx + 0]; /// 目标B ← 源B（偏移+0）
    }
}

/// \brief 双线性插值 + BGR→RGB 转换 Kernel
/// \param tar   目标图像指针 (RGB格式，3通道，输出)
/// \param src   源图像指针 (BGR格式，3通道，输入)
/// \param tarW  目标图像宽度 (像素)
/// \param tarH  目标图像高度 (像素)
/// \param srcW  源图像宽度 (像素)
/// \param srcH  源图像高度 (像素)
/// \param scaled_w  宽度缩放比例 (拉伸模式=srcW/tarW，Letterbox模式=统一较大值)
/// \param scaled_h  高度缩放比例 (拉伸模式=srcH/tarH，Letterbox模式=统一较大值)
__global__ void resize_bilinear_BGR2RGB_kernel(uint8_t* tar, uint8_t* src, int tarW, int tarH, int srcW, int srcH,
                                               float scaled_w, float scaled_h)
{
    /// ==================== 第一部分：计算目标像素坐标 ====================
    /// 每个线程负责处理目标图像中的一个像素
    /// blockIdx: 线程块在网格中的索引
    /// blockDim: 线程块的维度
    /// threadIdx: 线程在线程块中的索引
    int x = blockIdx.x * blockDim.x + threadIdx.x; /// 目标列坐标 [0, tarW-1]
    int y = blockIdx.y * blockDim.y + threadIdx.y; /// 目标行坐标 [0, tarH-1]

    /// ==================== 第二部分：映射到源图像4个邻近像素 ====================
    /// 使用像素中心对齐公式，与 OpenCV 的 resize 函数保持一致
    /// 公式推导：src = (dst + 0.5) * scale - 0.5
    /// 其中 scale = srcW / tarW (源宽/目标宽)
    /// +0.5 和 -0.5 是为了将像素索引转换为像素中心坐标
    /// floor() 向下取整，得到左上角像素的坐标
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5); /// 左上角行坐标
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5); /// 左上角列坐标
    int src_y2 = src_y1 + 1;                        /// 右下角行坐标（左上+1）
    int src_x2 = src_x1 + 1;                        /// 右下角列坐标（左上+1）

    /// ==================== 第三部分：边界检查 ====================
    /// 确保左上角坐标在源图像有效范围内
    if (src_y1 < 0 || src_x1 < 0 || src_y1 >= srcH || src_x1 >= srcW)
    {
        /// 越界的像素不进行计算，保持黑色
    }
    else
    {
        /// ==================== 第四部分：计算偏移比例 ====================
        /// th, tw 范围 [0, 1)，表示浮点映射点距离左上角像素的距离
        /// 例如：src_y1=2.3，则 th=0.3 表示距离左上角向下30%的位置
        float th = ((y + 0.5) * scaled_h - 0.5) - src_y1; /// 垂直偏移（小数部分）
        float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1; /// 水平偏移（小数部分）

        /// ==================== 第五部分：双线性插值权重（面积法） ====================
        /// 双线性插值的核心：目标像素值 = Σ(权重 × 源像素值)
        /// 权重由映射点与对角像素围成的矩形面积决定
        ///
        /// 权重分布图（映射点 P 将正方形分成4个小矩形）：
        ///
        ///        ←── tw ──→ ←── (1-tw) ───→
        ///        ┌───────┬──────────┐ ↑
        ///        │    D         │      B             │ │
        ///        │ tw×th       │ tw×(1-th)         │ th
        ///        │    ↓        │                    │ ↓
        ///        │    P         │                    │
        ///        │  (tw,th)     │                    │
        ///        ├───────┼──────────┤ ↑
        ///        │    C         │      A             │ │
        ///        │(1-tw)×th    │ (1-tw)×(1-th)     │ 1-th
        ///        │              │                    │ ↓
        ///        └───────┴──────────┘
        ///
        /// 每个角点的权重 = 对角矩形的面积
        /// 距映射点越近的角点，其对角矩形面积越大，权重越高
        float a1_1 = (1.0f - tw) * (1.0f - th); /// 左上角权重（矩形A的面积）
        float a1_2 = tw * (1.0f - th);          /// 右上角权重（矩形B的面积）
        float a2_1 = (1.0f - tw) * th;          /// 左下角权重（矩形C的面积）
        float a2_2 = tw * th;                   /// 右下角权重（矩形D的面积）
        ///  验证：所有权重之和 = (1-tw)(1-th) + tw(1-th) + (1-tw)th + tw·th = 1

        /// ==================== 第六部分：计算内存索引 ====================
        /// 计算4个源像素的一维内存索引
        /// 每个像素占3个字节（B、G、R通道连续存储）
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; /// 左上角像素的B通道索引
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; /// 右上角像素的B通道索引
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; /// 左下角像素的B通道索引
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; /// 右下角像素的B通道索引

        /// 计算目标像素的一维内存索引
        int tarIdx = (y * tarW + x) * 3; /// 目标像素的R通道索引（RGB格式）

        /// ==================== 第七部分：双线性插值 + BGR→RGB ====================
        /// 注意：源图像是BGR格式，所以：
        ///   src[+2] = R（红色通道）
        ///   src[+1] = G（绿色通道）
        ///   src[+0] = B（蓝色通道）
        /// 目标图像是RGB格式，所以：
        ///   tar[+0] = R，tar[+1] = G，tar[+2] = B
        /// 计算红色通道（目标R ← 源R）
        tar[tarIdx + 0] = round(a1_1 * src[srcIdx1_1 + 2] + /// 左上角R
                                a1_2 * src[srcIdx1_2 + 2] + /// 右上角R
                                a2_1 * src[srcIdx2_1 + 2] + /// 左下角R
                                a2_2 * src[srcIdx2_2 + 2]); /// 右下角R

        /// 计算绿色通道（目标G ← 源G）
        tar[tarIdx + 1] = round(a1_1 * src[srcIdx1_1 + 1] + /// 左上角G
                                a1_2 * src[srcIdx1_2 + 1] + /// 右上角G
                                a2_1 * src[srcIdx2_1 + 1] + /// 左下角G
                                a2_2 * src[srcIdx2_2 + 1]); /// 右下角G
        /// 计算蓝色通道（目标B ← 源B）
        tar[tarIdx + 2] = round(a1_1 * src[srcIdx1_1 + 0] + /// 左上角B
                                a1_2 * src[srcIdx1_2 + 0] + /// 右上角B
                                a2_1 * src[srcIdx2_1 + 0] + /// 左下角B
                                a2_2 * src[srcIdx2_2 + 0]); /// 右下角B
    }
}

/// \brief 双线性插值 + 平移居中 + BGR→RGB 转换 Kernel
/// \details 在双线性插值的基础上，增加图像的居中平移功能
///          实现 Letterbox + Center 效果：等比例缩放后，将图像放置在画布正中央
/// \param tar   目标图像指针 (RGB格式，3通道，输出)
/// \param src   源图像指针 (BGR格式，3通道，输入)
/// \param tarW  目标图像宽度 (像素)
/// \param tarH  目标图像高度 (像素)
/// \param srcW  源图像宽度 (像素)
/// \param srcH  源图像高度 (像素)
/// \param scaled_w  宽度缩放比例 (Letterbox模式=统一较大值)
/// \param scaled_h  高度缩放比例 (Letterbox模式=统一较大值)
__global__ void resize_bilinear_BGR2RGB_shift_kernel(uint8_t* tar, uint8_t* src, int tarW, int tarH, int srcW, int srcH,
                                                     float scaled_w, float scaled_h)
{
    /// ==================== 第一部分：计算目标像素坐标 ====================
    /// 每个线程负责处理目标图像中的一个像素
    /// blockIdx: 线程块在网格中的索引
    /// blockDim: 线程块的维度
    /// threadIdx: 线程在线程块中的索引
    int x = blockIdx.x * blockDim.x + threadIdx.x; /// 目标列坐标 [0, tarW-1]
    int y = blockIdx.y * blockDim.y + threadIdx.y; /// 目标行坐标 [0, tarH-1]

    /// ==================== 第二部分：映射到源图像4个邻近像素 ====================
    /// 使用像素中心对齐公式，与 OpenCV 的 resize 函数保持一致
    int src_y1 = floor((y + 0.5) * scaled_h - 0.5); /// 左上角行坐标
    int src_x1 = floor((x + 0.5) * scaled_w - 0.5); /// 左上角列坐标
    int src_y2 = src_y1 + 1;                        /// 右下角行坐标
    int src_x2 = src_x1 + 1;                        /// 右下角列坐标

    /// ==================== 第三部分：边界检查 ====================
    if (src_y1 < 0 || src_x1 < 0 || src_y1 >= srcH || src_x1 >= srcW)
    {
        /// 越界的像素不进行计算
    }
    else
    {
        /// ==================== 第四部分：计算偏移比例 ====================
        float th = ((y + 0.5) * scaled_h - 0.5) - src_y1; /// 垂直偏移
        float tw = ((x + 0.5) * scaled_w - 0.5) - src_x1; /// 水平偏移
        /// ==================== 第五部分：双线性插值权重 ====================
        float a1_1 = (1.0f - tw) * (1.0f - th); /// 左上角权重
        float a1_2 = tw * (1.0f - th);          /// 右上角权重
        float a2_1 = (1.0f - tw) * th;          /// 左下角权重
        float a2_2 = tw * th;                   /// 右下角权重
        /// ==================== 第六部分：计算源像素内存索引 ====================
        int srcIdx1_1 = (src_y1 * srcW + src_x1) * 3; /// 左上角
        int srcIdx1_2 = (src_y1 * srcW + src_x2) * 3; /// 右上角
        int srcIdx2_1 = (src_y2 * srcW + src_x1) * 3; /// 左下角
        int srcIdx2_2 = (src_y2 * srcW + src_x2) * 3; /// 右下角
        /// ==================== 第七部分：平移居中（关键！） ====================
        /// 原理：将目标像素坐标平移到以图像区域为中心的虚拟坐标系
        ///
        /// 实际图像宽度 = srcW / scaled_w
        /// 实际图像高度 = srcH / scaled_h
        ///
        /// 图像区域中心在目标画布中的位置 = (tarW/2, tarH/2)
        /// 图像区域左上角偏移量 = (tarW/2 - 实际宽度/2, tarH/2 - 实际高度/2)
        ///
        /// 平移公式：平移后坐标 = 原坐标 - 偏移量
        /// 即：new_x = x - (tarW/2 - 实际宽度/2) = x - tarW/2 + 实际宽度/2
        /// 实际宽度/2 = (srcW / scaled_w) / 2 = srcW / (scaled_w × 2)
        y = y - int(srcH / (scaled_h * 2)) + int(tarH / 2);
        x = x - int(srcW / (scaled_w * 2)) + int(tarW / 2);

        /// ==================== 第八部分：计算目标像素内存索引 ====================
        int tarIdx = (y * tarW + x) * 3;

        /// ==================== 第九部分：双线性插值 + BGR→RGB ====================
        /// 红色通道
        tar[tarIdx + 0] = round(a1_1 * src[srcIdx1_1 + 2] + a1_2 * src[srcIdx1_2 + 2] + a2_1 * src[srcIdx2_1 + 2] +
                                a2_2 * src[srcIdx2_2 + 2]);

        /// 绿色通道
        tar[tarIdx + 1] = round(a1_1 * src[srcIdx1_1 + 1] + a1_2 * src[srcIdx1_2 + 1] + a2_1 * src[srcIdx2_1 + 1] +
                                a2_2 * src[srcIdx2_2 + 1]);

        /// 蓝色通道
        tar[tarIdx + 2] = round(a1_1 * src[srcIdx1_1 + 0] + a1_2 * src[srcIdx1_2 + 0] + a2_1 * src[srcIdx2_1 + 0] +
                                a2_2 * src[srcIdx2_2 + 0]);
    }
}


//////////////////////////////////////////////////////////////////
// 主机端入口函数
//////////////////////////////////////////////////////////////////

/*
    这里面的所有函数都实现了kernel fusion。这样可以减少kernel launch所产生的overhead
    如果使用了shared memory的话，就可以减少分配shared memory所产生的overhead以及内部线程同步的overhead。(这个案例没有使用shared memory)
    CUDA编程中有一些cuda runtime api是implicit synchronize(隐式同步)的，比如cudaMalloc, cudaMallocHost，以及shared memory的分配。
    高效的CUDA编程需要意识这些implicit synchronize以及其他会产生overhead的地方。比如使用内存复用的方法，让cuda分配完一次memory就一直使用它

    这里建议大家把我写的每一个kernel都拆开成不同的kernel来分别计算
    e.g. resize kernel + BGR2RGB kernel + shift kernel 
    之后用nsight去比较融合与不融合的差别在哪里。去体会一下fusion的好处
*/

/// \brief GPU 图像缩放入口函数（策略模式）
/// \details 根据不同的战术类型，计算缩放比例并调用对应的 CUDA Kernel
///          实现了 kernel fusion（融合了 resize + BGR2RGB + shift）
/// \param d_tar   目标图像设备指针 (RGB格式，3通道，输出)
/// \param d_src   源图像设备指针 (BGR格式，3通道，输入)
/// \param tarW    目标图像宽度 (像素)
/// \param tarH    目标图像高度 (像素)
/// \param srcW    源图像宽度 (像素)
/// \param srcH    源图像高度 (像素)
/// \param tactis  战术类型 (0:最近邻拉伸, 1:双线性拉伸, 2:双线性Letterbox, 3:双线性Letterbox+居中)
void resize_bilinear_gpu(uint8_t* d_tar, uint8_t* d_src, int tarW, int tarH, int srcW, int srcH, int tactis)
{
    /// ==================== 第一部分：配置线程网格 ====================
    /// 每个线程块包含 16×16 = 256 个线程
    /// 为什么选择 16×16？
    ///   - 16 是 warp 大小（32）的因数，可以充分利用 GPU 调度
    ///   - 16×16 在共享内存使用和线程调度之间取得平衡
    ///   - 2D 布局方便处理 2D 图像（一个线程处理一个像素）
    dim3 dimBlock(16, 16, 1);

    /// 计算需要的线程块数量
    /// 公式：向上取整 = (宽度 + 块大小 - 1) / 块大小
    /// 这里使用 tarW/16 + 1 是一种简单写法，可能会多启动一个块
    /// 更精确的写法应该是 (tarW + 15) / 16
    dim3 dimGrid(tarW / 16 + 1, tarH / 16 + 1, 1);

    /// ==================== 第二部分：计算缩放比例 ====================
    /// 拉伸模式（tactis = 0 或 1）：
    ///   scaled_w = srcW / tarW  (宽度方向的缩放因子)
    ///   scaled_h = srcH / tarH  (高度方向的缩放因子)
    ///   == 两个方向缩放比例不同 → 图像会被拉伸填满画布 ==
    ///
    /// Letterbox 模式（tactis = 2 或 3）：
    ///   scale = max(scaled_w, scaled_h)  (取较大的缩放比例)
    ///   scaled_w = scale
    ///   scaled_h = scale
    ///   == 两个方向缩放比例相同 → 图像保持宽高比，四周出现黑边 ==
    float scaled_h = (float)srcH / tarH;                          /// 高度方向缩放因子
    float scaled_w = (float)srcW / tarW;                          /// 宽度方向缩放因子
    float scale    = (scaled_h > scaled_w ? scaled_h : scaled_w); /// 取较大值

    /// tactis > 1 时（case 2 或 case 3）使用统一缩放比例（Letterbox 模式）
    if (tactis > 1)
    {
        scaled_h = scale;
        scaled_w = scale;
    }

    /// ==================== 第三部分：根据战术调度不同的 Kernel ====================
    /// tactis 战术说明：
    ///   0: 最近邻插值 + 拉伸填充（速度快，有锯齿）
    ///   1: 双线性插值 + 拉伸填充（平滑，会变形）
    ///   2: 双线性插值 + Letterbox 填充（平滑，保持宽高比，黑边填充）
    ///   3: 双线性插值 + Letterbox + 平移居中（平滑，保持宽高比，图像居中）
    switch (tactis)
    {
        case 0:
            /// 最近邻插值 + BGR→RGB
            /// 优点：速度最快，无浮点运算
            /// 缺点：图像有锯齿，边缘不平滑
            resize_nearest_BGR2RGB_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w,
                                                                 scaled_h);
            break;

        case 1:
            /// 双线性插值 + BGR→RGB（拉伸模式）
            /// 优点：图像平滑，无锯齿
            /// 缺点：图像会被拉伸变形（因为 scaled_w ≠ scaled_h）
            resize_bilinear_BGR2RGB_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w,
                                                                  scaled_h);
            break;

        case 2:
            /// 双线性插值 + BGR→RGB（Letterbox 模式）
            /// 与 case 1 使用相同的 kernel，但 scaled_w = scaled_h（统一缩放）
            /// 效果：图像保持宽高比，四周出现黑边
            /// 注意：这里和 case 1 调用的是同一个 kernel！区别在于传入的缩放比例参数不同
            resize_bilinear_BGR2RGB_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w,
                                                                  scaled_h);
            break;

        case 3:
            /// 双线性插值 + BGR→RGB + 平移居中（Letterbox + Center）
            /// 在 case 2 的基础上，增加坐标平移，将图像放置在画布正中央
            /// 这是最完整的预处理：平滑 + 保持比例 + 居中
            resize_bilinear_BGR2RGB_shift_kernel<<<dimGrid, dimBlock>>>(d_tar, d_src, tarW, tarH, srcW, srcH, scaled_w,
                                                                        scaled_h);
            break;

        default:
            /// 未知战术类型，什么都不做
            break;
    }

    /// ==================== 第四部分：错误检查 ====================
    LAST_KERNEL_CHECK();
}
