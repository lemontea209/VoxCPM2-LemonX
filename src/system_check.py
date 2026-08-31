#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 整合包 · 系统环境检测 (极简版)
用法:  python_embeded\\python.exe system_check.py
检测不到 GPU/CUDA 信息时, 请安装:  pip install nvidia-ml-py
"""

import sys
import platform
import subprocess
import unicodedata

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"
USE_COLOR = "--no-color" not in sys.argv


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"
    RED = "\033[91m"; GREEN = "\033[92m"; YELLOW = "\033[93m"; GRAY = "\033[90m"


def _enable_win_ansi():
    if IS_WINDOWS:
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass


def col(t, c):
    return f"{c}{t}{C.RESET}" if USE_COLOR else t


def dwidth(s):
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def pad(s, w):
    return s + " " * max(0, w - dwidth(s))


def kv(label, value, w=14):
    """无标记行 (系统/Python/CPU/内存)。"""
    line = f"{pad(label + ':', w)} {value}"
    print(line)


def item(label, value, status="info", w=16):
    """带 [✓]/[ ]/[!]/[x] 标记的行。"""
    mark = {"info": " ", "ok": "✓", "warn": "!", "error": "x"}[status]
    cc = {"info": C.RESET, "ok": C.GREEN, "warn": C.YELLOW, "error": C.RED}[status]
    lbl = pad(label + ":", w)
    cval = col(value, cc) if status in ("ok", "warn", "error") else value
    print(f"  [{col(mark, cc)}] {lbl} {cval}")


def run(args, timeout=10):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout, text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return None, "", ""
    except Exception as e:
        return -1, "", str(e)


def gb(n):
    try:
        return f"{n / 1024 ** 3:.2f} GB"
    except Exception:
        return str(n)


def cpu_name():
    if IS_WINDOWS:
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    if IS_LINUX:
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "(未知)"


def memory():
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.total, m.available
    except Exception:
        pass
    if IS_WINDOWS:
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("t", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong),
                            ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                            ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong),
                            ("f", ctypes.c_ulonglong)]
            s = MS(); s.dwLength = ctypes.sizeof(s)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.t, s.a
        except Exception:
            pass
    if IS_LINUX:
        try:
            info = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    a, b = line.split(":", 1)
                    info[a.strip()] = int(b.strip().split()[0]) * 1024
            return info.get("MemTotal"), info.get("MemAvailable")
        except Exception:
            pass
    return None, None


def gpu_via_smi():
    """返回 (gpus, max_cuda)。gpus=[(name, mem_bytes, driver), ...]"""
    code, out, _ = run(["nvidia-smi",
                        "--query-gpu=name,memory.total,driver_version",
                        "--format=csv,noheader,nounits"])
    if code != 0 or not out.strip():
        return None, None
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            try:
                mem = int(float(p[1])) * 1024 * 1024
            except Exception:
                mem = None
            gpus.append((p[0], mem, p[2]))
    # 头部最高支持 CUDA
    maxc = None
    c2, o2, _ = run(["nvidia-smi"])
    if c2 == 0 and o2:
        for l in o2.splitlines():
            if "CUDA Version" in l:
                tk = l.split("CUDA Version")[-1].replace(":", " ").replace("|", " ").split()
                if tk:
                    maxc = tk[0]
                break
    return gpus, maxc


def max_cuda_via_nvml():
    """用 nvidia-ml-py(pynvml) 获取驱动支持的最高 CUDA 版本。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        except Exception:
            v = pynvml.nvmlSystemGetCudaDriverVersion()
        pynvml.nvmlShutdown()
        return f"{v // 1000}.{(v % 1000) // 10}"
    except Exception:
        return None


def gpu_via_nvml():
    """nvidia-smi 失败时, 用 nvidia-ml-py(pynvml) 兜底。"""
    try:
        import pynvml
        pynvml.nvmlInit()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            drv = pynvml.nvmlSystemGetDriverVersion()
            name = name.decode() if isinstance(name, bytes) else name
            drv = drv.decode() if isinstance(drv, bytes) else drv
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).total
            gpus.append((name, mem, drv))
        try:
            v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        except Exception:
            v = pynvml.nvmlSystemGetCudaDriverVersion()
        maxc = f"{v // 1000}.{(v % 1000) // 10}"
        pynvml.nvmlShutdown()
        return gpus, maxc
    except Exception:
        return None, None


def main():
    _enable_win_ansi()

    kv("操作系统", platform.platform())
    kv("Python 版本", platform.python_version())
    kv("CPU 型号", cpu_name())
    total, avail = memory()
    kv("内存", f"总 {gb(total) if total else '?'}  /  可用 {gb(avail) if avail else '?'}")

    # GPU: 先 nvidia-smi, 失败用 pynvml 兜底
    gpus, maxc = gpu_via_smi()
    if gpus is None:
        gpus, maxc = gpu_via_nvml()
    # 最高支持 CUDA 优先用 nvidia-ml-py 获取 (nvidia-smi 文本解析在新驱动上常失败)
    nvml_maxc = max_cuda_via_nvml()
    if nvml_maxc:
        maxc = nvml_maxc
    if gpus:
        for i, (name, mem, drv) in enumerate(gpus):
            item(f"GPU {i} 型号", name, "ok")
            item(f"GPU {i} 显存", gb(mem) if mem else "?", "ok")
            item(f"GPU {i} 驱动版本", drv)
        if maxc:
            item("最高支持 CUDA", maxc, "ok")
        else:
            item("最高支持 CUDA", "检测不到, 请: pip install nvidia-ml-py", "warn")
    else:
        item("GPU", "检测不到, 请: pip install nvidia-ml-py", "warn")

    # PyTorch
    try:
        import torch
        item("torch 版本", torch.__version__, "ok")
        ok = False
        try:
            ok = torch.cuda.is_available()
        except Exception:
            pass
        item("CUDA 可用", "是" if ok else "否", "ok" if ok else "error")
        try:
            cd = torch.backends.cudnn.version()
            item("cuDNN 版本", str(cd) if cd else "(无)")
        except Exception:
            item("cuDNN 版本", "(无法获取)")
        n = torch.cuda.device_count() if ok else 0
        item("可用 GPU 数量", str(n), "ok" if ok else "info")
        if ok:
            for i in range(n):
                pr = torch.cuda.get_device_properties(i)
                item(f"torch GPU {i}", f"{pr.name} | {gb(pr.total_memory)} | 算力 {pr.major}.{pr.minor}", "ok")
            try:
                item("支持 bf16", "是" if torch.cuda.is_bf16_supported() else "否")
            except Exception:
                pass
    except ImportError:
        item("torch", "未安装", "error")
    except Exception as e:
        item("torch", f"导入异常: {e}", "error")

    # nvcc
    code, out, _ = run(["nvcc", "--version"])
    if code == 0:
        ver = "(未解析)"
        for l in out.splitlines():
            if "release" in l:
                ver = l.split("release")[-1].strip().rstrip(",")
                break
        item("nvcc 版本", ver, "ok")
    elif code is None:
        item("nvcc 版本", "未安装 (整合包多自带运行库, 不影响使用)", "info")
    else:
        item("nvcc 版本", "调用失败", "warn")

    # VC++ 运行库
    if not IS_WINDOWS:
        item("运行库VC_redist.x64 (2015-2022)", "非 Windows, 跳过", "info", w=30)
    else:
        found = None
        try:
            import winreg
            for p in (r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
                      r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"):
                try:
                    k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, p)
                    if winreg.QueryValueEx(k, "Installed")[0] == 1:
                        found = winreg.QueryValueEx(k, "Version")[0]
                        break
                except FileNotFoundError:
                    continue
        except Exception:
            pass
        if found:
            item("运行库VC_redist.x64 (2015-2022)", f"已安装 ({found})", "ok", w=30)
        else:
            item("运行库VC_redist.x64 (2015-2022)",
                 "未检测到! 装 https://aka.ms/vs/17/release/vc_redist.x64.exe", "error", w=30)

    if IS_WINDOWS:
        try:
            input(col("\n按回车键退出...", C.GRAY))
        except Exception:
            pass


if __name__ == "__main__":
    main()