import gradio as gr
import psutil


# 尝试加载 NVIDIA 管理库，缺失或无显卡（如 macOS）时自动降级为不可用
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_READY = True
except Exception:
    _NVML_READY = False

psutil.cpu_percent()  # 预热一次 CPU 采样，使后续读数反映两次调用之间的真实占用


def to_gb(num_bytes):
    return f"{num_bytes / (1024 ** 3):.1f} GB"


def get_gpu_stats():
    """读取 0 号显卡的利用率与显存；任何异常都降级为「读不到」。

    nvmlInit 成功不代表之后每次调用都成功：双显卡笔记本的 Optimus 关掉独显、
    驱动发生 TDR 重置、或显卡被独占时，这几个调用都会抛异常。状态栏每 2 秒
    调用一次，不兜住就会变成刷屏。
    """
    if not _NVML_READY:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    except Exception:
        return None
    return {
        "gpu_pct": util.gpu,
        "vram_used": mem.used,
        "vram_total": mem.total,
    }


def get_cuda_version():
    # 驱动支持的最高 CUDA 版本，整数 12040 表示 12.4
    if not _NVML_READY:
        return "N/A"
    try:
        v = pynvml.nvmlSystemGetCudaDriverVersion()
    except Exception:
        return "N/A"
    return f"{v // 1000}.{(v % 1000) // 10}"


def is_cuda_available():
    # 优先用 torch 判断，未安装则回退为“存在可用的 NVIDIA 设备”
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return _NVML_READY and pynvml.nvmlDeviceGetCount() > 0


def collect_stats():
    vm = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "mem_used": vm.used,
        "mem_total": vm.total,
        "gpu": get_gpu_stats(),
        "cuda_version": get_cuda_version(),
        "cuda_available": is_cuda_available(),
    }


def build_items(d):
    # CPU：占用百分比 / 总核数；内存、显存：实际占用 / 总量；GPU：利用率
    items = [
        ("CPU", f"{d['cpu_pct']:.0f}% / {d['cpu_cores']} 核"),
        ("内存", f"{to_gb(d['mem_used'])} / {to_gb(d['mem_total'])}"),
    ]
    gpu = d["gpu"]
    if gpu:
        items.append(("GPU", f"{gpu['gpu_pct']}%"))
        items.append(("显存", f"{to_gb(gpu['vram_used'])} / {to_gb(gpu['vram_total'])}"))
    else:
        items.append(("GPU", "N/A"))
        items.append(("显存", "N/A"))
    items.append(("最高 CUDA", d["cuda_version"]))
    items.append(("CUDA", "可用" if d["cuda_available"] else "不可用"))
    return items


def brand_html(brand_name, brand_url):
    return f"""
    <div class='yzystatus-brand'>
        <a href='{brand_url}' target='_blank' rel='noopener'>{brand_name}</a>
        <span class='yzystatus-by'>Packaged by</span>
        <a href='https://space.bilibili.com/3493266750179909' target='_blank' rel='noopener'>余子越Talk</a>
    </div>
    """


def theme_toggle_html(place):
    # 主题切换按钮。place='bar' 宽屏时贴在状态栏最右；place='panel' 收纳进“详情”面板。
    # 三个图标 + 文字标签都靠 CSS 根据 #yzystatus-bar[data-yzy-theme] 来显隐，
    # 当前模式由 STATUS_JS 从 URL 的 __theme 参数读出后写到该属性上。
    extra = "yzystatus-theme-bar" if place == "bar" else "yzystatus-theme-panel"
    return (
        f"<button type='button' class='yzystatus-theme {extra}' "
        f"onclick='yzystatusCycleTheme(event)' "
        f"title='切换页面模式：白天 / 暗黑 / 跟随系统'>"
        f"<span class='yzystatus-theme-ico yzystatus-ico-light'>☀️</span>"
        f"<span class='yzystatus-theme-ico yzystatus-ico-dark'>🌙</span>"
        f"<span class='yzystatus-theme-ico yzystatus-ico-system'>🖥️</span>"
        f"<span class='yzystatus-theme-label'></span>"
        f"</button>"
    )


def build_html(brand_name, brand_url, d):
    def item_html(label, value):
        return (
            f"<span class='yzystatus-item'>"
            f"<span class='yzystatus-k'>{label}</span>"
            f"<span class='yzystatus-v'>{value}</span>"
            f"</span>"
        )
    cells = "".join(item_html(k, v) for k, v in build_items(d))
    edition = "<span class='yzystatus-edition'>此版本为Lemon-X修改版</span>"
    metrics = f"<div class='yzystatus-metrics'>{cells}{edition}</div>"

    # “详情”面板：原指标 + 一行主题切换（窄屏时这里是唯一的切换入口）
    more = (
        f"<div class='yzystatus-more' tabindex='0'>"
        f"<span class='yzystatus-badge'>⋯ 详情</span>"
        f"<div class='yzystatus-panel'>"
        f"{cells}"
        f"<div class='yzystatus-panel-row'>{theme_toggle_html('panel')}</div>"
        f"</div>"
        f"</div>"
    )

    # 右侧控件区（绝对定位贴右，不参与指标溢出测量）：详情按钮 + 状态栏内主题按钮
    controls = (
        f"<div class='yzystatus-controls'>"
        f"{more}{theme_toggle_html('bar')}"
        f"</div>"
    )

    return (
        f"<div class='yzystatus-inner'>"
        f"{brand_html(brand_name, brand_url)}{metrics}{controls}"
        f"</div>"
    )


# 上一次成功渲染的状态栏 HTML。采集失败时沿用它，避免顶栏忽然消失。
_LAST_STATUS_HTML: str | None = None


def refresh_status(brand_name, brand_url):
    """每 2 秒刷新一次顶部状态栏。

    这里绝对不能抛 gr.Error：原来任何一次采集失败都会升级成 duration=None
    的红色错误条（永不自动消失），而它每 2 秒触发一次——一分钟就能铺满界面
    把按钮挡住。状态栏是纯展示，读不到就降级显示，不该打断用户干活。
    """
    global _LAST_STATUS_HTML
    try:
        _LAST_STATUS_HTML = build_html(brand_name, brand_url, collect_stats())
        return _LAST_STATUS_HTML
    except Exception:
        if _LAST_STATUS_HTML is not None:
            return _LAST_STATUS_HTML
        return (
            f"<div class='yzystatus-inner'>{brand_html(brand_name, brand_url)}"
            f"<div class='yzystatus-metrics'>系统状态暂时读取不到</div>"
            f"</div>"
        )


def initial_status():
    try:
        return build_html("", "", collect_stats())
    except Exception as exc:
        return (
            f"<div class='yzystatus-inner'>{brand_html("", "")}"
            f"<div class='yzystatus-metrics'>初始化失败：{exc}</div>"
            f"</div>"
        )


# 状态栏专属类名 / 元素 ID 统一加 yzystatus 前缀；以下三项是 Gradio 外壳所需的全局规则
STATUS_CSS = """
.gradio-container {
    padding-top: 46px !important;
}

.fillable {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 10px !important;
}

footer {
    display: none !important;
}

#yzystatus-bar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    width: 100%;
    z-index: 1000;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    background: transparent !important;

    /* 配色色板：此处为浅色默认值，深色模式只在 .dark 分支整体覆盖这些变量 */
    --yzy-bg: #f6f3f3;
    --yzy-bg-hover: #fbfcfe;
    --yzy-fg: #1f2329;
    --yzy-border: rgba(0, 0, 0, 0.10);
    --yzy-divider: rgba(0, 0, 0, 0.12);
    --yzy-accent: #2f6fed;
    --yzy-accent-fg: #ffffff;
    --yzy-accent-shadow: rgba(47, 111, 237, 0.4);
    --yzy-panel-bg: #ffffff;
    --yzy-panel-border: rgba(0, 0, 0, 0.12);
    --yzy-link: #2f6fed;
}

/* 去掉外层包裹的内外边距 */
#yzystatus-bar .html-container,
#yzystatus-bar > * {
    padding: 0 !important;
    margin: 0 !important;
}

/* 关闭 Gradio 事件处理期间的变暗 / 闪烁 / 脉冲描边 */
#yzystatus-bar.pending,
#yzystatus-bar .pending {
    opacity: 1 !important;
    animation: none !important;
}

#yzystatus-bar.generating,
#yzystatus-bar .generating {
    border: none !important;
    animation: none !important;
}

.yzystatus-inner {
    position: relative;            /* 作为右侧控件区的定位参照 */
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 18px;
    box-sizing: border-box;
    width: 100%;
    padding: 7px 96px 7px 16px;    /* 右侧预留出控件区的位置，避免居中内容被压到按钮下面 */
    font-size: 13px;
    line-height: 1.4;
    background: var(--yzy-bg);
    color: var(--yzy-fg);
    border-bottom: 1px solid var(--yzy-border);
    box-shadow: 0 1px 6px rgba(0, 0, 0, 0.08);
    transition: box-shadow .15s ease, background .15s ease;
    height: 42px;
    user-select: none;
}

/* 鼠标悬停在状态栏上的浮动（上浮）效果 */
.yzystatus-inner:hover {
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.22);
    background: var(--yzy-bg-hover);
}

.yzystatus-brand {
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    padding-right: 16px;
    border-right: 1px solid var(--yzy-divider);
}

.yzystatus-by {
    font-weight: 400;
    opacity: 0.6;
    margin: 0 6px;
}

.yzystatus-brand a {
    color: var(--yzy-link);
    text-decoration: none;
}

.yzystatus-brand a:hover {
    text-decoration: underline;
}

/* 指标区单独裁剪，溢出时只截断这里，不影响下拉面板 */
.yzystatus-metrics {
    flex: 0 1 auto;
    min-width: 0;
    display: flex;
    gap: 18px;
    overflow: hidden;
    white-space: nowrap;
}

.yzystatus-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: 0 0 auto;
}

.yzystatus-k {
    opacity: 0.6;
}

.yzystatus-v {
    font-variant-numeric: tabular-nums;
    font-weight: 600;
}

.yzystatus-edition {
    display: inline-flex;
    align-items: center;
    flex: 0 0 auto;
    padding-left: 18px;
    border-left: 1px solid var(--yzy-divider);
    font-weight: 600;
}

/* 右侧控件区：绝对定位贴右、垂直居中。脱离文档流，因此其显隐不会改变指标区宽度，
   从而避免窄屏临界点上“显示按钮→腾不下→隐藏按钮→又腾得下”的反复抖动。 */
.yzystatus-controls {
    position: absolute;
    top: 50%;
    right: 16px;
    transform: translateY(-50%);
    display: flex;
    align-items: center;
    gap: 10px;
    z-index: 1102;
}

.yzystatus-more {
    position: relative;
    display: none;          /* 默认隐藏，溢出时才出现 */
    cursor: pointer;
    outline: none;
}

#yzystatus-bar.yzystatus-has-overflow .yzystatus-more {
    display: inline-flex;
}

/* 徽标与面板之间补一段透明桥接区，避免鼠标穿过间隙时 hover 断开导致面板消失 */
.yzystatus-more::after {
    content: '';
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    height: 12px;
}

.yzystatus-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 10px;
    border-radius: 999px;
    font-weight: 700;
    background: var(--yzy-accent);
    color: var(--yzy-accent-fg);
    box-shadow: 0 1px 4px var(--yzy-accent-shadow);
    height: 30px;
}

.yzystatus-panel {
    display: none;
    position: absolute;
    top: calc(100% + 10px);
    right: 0;
    z-index: 1100;
    min-width: 240px;
    padding: 10px 14px;
    white-space: normal;
    background: var(--yzy-panel-bg);
    color: var(--yzy-fg);
    border: 1px solid var(--yzy-panel-border);
    border-radius: 10px;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.18);
}

.yzystatus-more:hover .yzystatus-panel,
.yzystatus-more:focus-within .yzystatus-panel {
    display: block;
}

.yzystatus-panel .yzystatus-item {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    line-height: 2;
}

/* 面板里的主题切换行：与上方指标用一条分隔线隔开 */
.yzystatus-panel-row {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid var(--yzy-panel-border);
    display: flex;
    justify-content: center;
}

/* ============ 主题切换按钮 ============ */
/* 复用状态栏的色板变量，因此深色模式下自动跟随，无需额外写深色规则 */
.yzystatus-theme {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    height: 30px;
    padding: 0 12px;
    border: 1px solid var(--yzy-border);
    border-radius: 999px;
    background: var(--yzy-bg-hover);
    color: var(--yzy-fg);
    font: inherit;
    font-size: 13px;
    line-height: 1;
    cursor: pointer;
    transition: background .15s ease, border-color .15s ease, box-shadow .15s ease;
}

.yzystatus-theme:hover {
    border-color: var(--yzy-accent);
    box-shadow: 0 1px 6px var(--yzy-accent-shadow);
}

.yzystatus-theme-ico {
    display: none;
    font-size: 14px;
}

.yzystatus-theme-label {
    font-weight: 600;
}

/* 当前模式由 JS 写到 #yzystatus-bar 的 data-yzy-theme 上，对应显示图标与文字 */
#yzystatus-bar[data-yzy-theme="light"]  .yzystatus-ico-light  { display: inline; }
#yzystatus-bar[data-yzy-theme="dark"]   .yzystatus-ico-dark   { display: inline; }
#yzystatus-bar[data-yzy-theme="system"] .yzystatus-ico-system { display: inline; }
#yzystatus-bar[data-yzy-theme="light"]  .yzystatus-theme-label::after { content: "Light"; }
#yzystatus-bar[data-yzy-theme="dark"]   .yzystatus-theme-label::after { content: "Dark"; }
#yzystatus-bar[data-yzy-theme="system"] .yzystatus-theme-label::after { content: "System"; }

/* 宽屏：状态栏内显示主题按钮；溢出（窄屏）时隐藏，改由“详情”面板里那个承担 */
.yzystatus-theme-bar {
    display: inline-flex;
}

#yzystatus-bar.yzystatus-has-overflow .yzystatus-theme-bar {
    display: none;
}

.yzystatus-panel .yzystatus-theme {
    background: transparent;
}

/* 深色模式：跟随 Gradio 的 .dark 主题，只覆盖色板变量，其余样式自动适配 */
.dark #yzystatus-bar {
    --yzy-bg: rgba(28, 30, 36, 0.96);
    --yzy-bg-hover: rgba(40, 43, 51, 0.98);
    --yzy-fg: #e8eaed;
    --yzy-border: rgba(255, 255, 255, 0.10);
    --yzy-divider: rgba(255, 255, 255, 0.14);
    --yzy-accent: #6ea8ff;
    --yzy-accent-fg: #ffffff;
    --yzy-accent-shadow: rgba(110, 168, 255, 0.35);
    --yzy-panel-bg: #1c1e24;
    --yzy-panel-border: rgba(255, 255, 255, 0.12);
    --yzy-link: #6ea8ff;
}

/* Ocean 的主题button颜色问题 */
button {
    color: mediumslateblue;
}

.dark button {
    color: #ffffff;
}

"""

STATUS_JS = """
() => {
    // ---------- 主题切换 ----------
    // 走 Gradio 官方的 __theme 查询参数：light / dark / 删除参数=跟随系统。
    // 由 Gradio 在页面加载阶段套用主题，和状态栏的 .dark 配色保持一致。
    const THEME_ORDER = ['dark', 'light', 'system'];

    const getMode = () =>
        new URL(window.location.href).searchParams.get('__theme') || 'system';

    // 把当前模式写到状态栏元素上，驱动按钮图标 / 文字（只改属性，不触发下面的观察器）
    const applyMode = () => {
        const bar = document.getElementById('yzystatus-bar');
        if (bar) bar.setAttribute('data-yzy-theme', getMode());
    };

    window.yzystatusSetTheme = (mode) => {
        const url = new URL(window.location.href);
        if (mode === 'system') {
            url.searchParams.delete('__theme');   // 删除即回到“跟随系统”
        } else {
            url.searchParams.set('__theme', mode);
        }
        window.location.href = url.href;          // 刷新，让 Gradio 在加载时套用主题
    };

    // 按一下循环：白天 → 暗黑 → 跟随系统 → 白天
    window.yzystatusCycleTheme = (e) => {
        if (e) e.stopPropagation();
        const cur = getMode();
        const idx = THEME_ORDER.indexOf(cur);
        const next = THEME_ORDER[(idx + 1) % THEME_ORDER.length];
        window.yzystatusSetTheme(next);
    };

    applyMode();

    // ---------- 溢出布局 ----------
    const layout = () => {
        const bar = document.getElementById('yzystatus-bar');
        if (!bar) return;
        applyMode();   // 状态栏每 2s 重绘，这里持续保证模式标记存在
        const metrics = bar.querySelector('.yzystatus-metrics');
        if (metrics) {
            const overflow = metrics.scrollWidth > metrics.clientWidth + 1;
            bar.classList.toggle('yzystatus-has-overflow', overflow);
        }
    };
    const run = () => requestAnimationFrame(layout);
    window.addEventListener('resize', run);
    new MutationObserver(run).observe(
        document.body, { subtree: true, childList: true, characterData: true }
    );
    run();
    setTimeout(run, 300);
}
"""
