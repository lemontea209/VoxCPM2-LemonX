import os


# VoxCPM uses a local Gradio server and China-region cloud speech services.
# Keep the entire bundled Python process on a direct connection so system proxy
# settings cannot intercept localhost startup checks or MiniMax API requests.
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# gradio 6.18 自己的 routes.py 会反复抛 StarletteDeprecationWarning
# （HTTP_422_UNPROCESSABLE_ENTITY），在启动器的「日志」标签页里显示成红色 [ERR]。
# 那是第三方包内部的事，与本整合包无关，但发布版每次使用都会刷几条像报错的东西。
# 在这里按来源过滤掉，不去改动 site-packages —— 改了第三方包，谁重装依赖谁踩坑。
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*HTTP_422_UNPROCESSABLE_ENTITY.*",
    module=r"gradio\.routes",
)


# ---------------------------------------------------------------------------
# 运行日志落盘
#
# 启动器「日志」标签页的内容来自本进程的 stdout / stderr，只存在于界面里，
# 磁盘上没有对应文件——出问题时只能靠人一屏一屏截图，既不现实也容易漏。
# 这里把两个流各分流一份到 logs\runtime.log，界面照常显示，不改变任何既有行为。
#
# 三条硬性约束：
#   1) 真实的 stdout 必须原样写入。webui.py 会往 stdout 打印端口号的 JSON，
#      Electron 靠解析那一行拿到端口，写坏了应用就起不来。
#   2) 落盘失败（磁盘满、只读目录、杀软拦截）绝不能影响启动，一律降级为不落盘。
#   3) 不引入第三方依赖，不改动 site-packages。
# ---------------------------------------------------------------------------
import datetime
import sys
import threading


class _TeeStream:
    """把写入同时送往原始流和日志文件；原始流永远优先且不受文件写入影响。"""

    def __init__(self, stream, handle, lock, tag):
        self._stream = stream
        self._handle = handle
        self._lock = lock
        self._tag = tag
        self._at_line_start = True

    def write(self, data):
        # 先写真实流：即使随后落盘失败，界面与 Electron 的解析也不受影响。
        written = self._stream.write(data)
        if data:
            try:
                self._write_to_file(data)
            except Exception:
                pass          # 落盘是附加能力，任何失败都静默降级
        return written

    def _write_to_file(self, data):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            for index, part in enumerate(data.split("\n")):
                if index:
                    self._handle.write("\n")
                    self._at_line_start = True
                if not part:
                    continue
                if self._at_line_start:
                    self._handle.write(f"{stamp} [{self._tag}] ")
                    self._at_line_start = False
                self._handle.write(part)
            self._handle.flush()

    def flush(self):
        self._stream.flush()
        try:
            with self._lock:
                self._handle.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._stream.isatty()
        except Exception:
            return False

    def fileno(self):
        # 有些库会直接要 fileno（例如子进程重定向），必须给真实流的。
        return self._stream.fileno()

    def __getattr__(self, item):
        return getattr(self._stream, item)


def _install_runtime_log():
    """把 stdout / stderr 分流到 logs\\runtime.log；失败时安静地什么都不做。"""
    try:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(base, exist_ok=True)
        current = os.path.join(base, "runtime.log")
        previous = os.path.join(base, "runtime.prev.log")
        # 只保留上一次运行的日志：每次启动轮转一次，文件不会无限增长。
        if os.path.exists(current):
            if os.path.exists(previous):
                os.remove(previous)
            os.replace(current, previous)
        handle = open(current, "w", encoding="utf-8", errors="replace")
    except Exception:
        return

    lock = threading.Lock()
    handle.write(
        f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} [SYS] "
        f"VoxCPM 2 运行日志；上一次运行见 runtime.prev.log\n"
    )
    handle.flush()
    sys.stdout = _TeeStream(sys.stdout, handle, lock, "OUT")
    sys.stderr = _TeeStream(sys.stderr, handle, lock, "ERR")


_install_runtime_log()


# ---------------------------------------------------------------------------
# 单实例保护
#
# 启动器界面上有「启动」按钮，点一次 Electron 就再 spawn 一个 Python 进程，
# 并覆盖掉自己持有的进程引用——旧进程再也杀不掉，占着显存和端口。
# 开机时 createWindow() 已经自动起过一个了，所以点一次必然出现两个。
#
# 这里在 import webui 之前拦截：发现已有实例就复用它的端口并退出。
# 必须赶在 import webui 之前——webui 模块一被导入就会去探测并占用端口。
#
# 任何环节出错都降级为「照常启动」：这是附加保护，不能反过来挡住启动。
# ---------------------------------------------------------------------------
import json
import socket
import time

_INSTANCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "logs", "instance.json"
)


def _pid_alive(pid):
    """进程是否还活着。

    Windows 上不能用 os.kill(pid, 0)：Python 在 Windows 的 os.kill 会调用
    TerminateProcess，那是真的去杀进程，不是探测。改用 OpenProcess 查询。
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return bool(ok) and code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return False


def _port_open(port, timeout=1.0):
    """本机该端口上是否有人在监听。"""
    if not isinstance(port, int) or not (0 < port < 65536):
        return False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _running_instance():
    """已有实例的端口；没有则返回 None。"""
    try:
        with open(_INSTANCE_FILE, "r", encoding="utf-8") as handle:
            info = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict):
        return None
    pid, port = info.get("pid"), info.get("port")
    # 三条都要满足；少一条就当陈旧锁（多半是上一次被 taskkill 强杀留下的）。
    if _pid_alive(pid) and _port_open(port):
        return port
    return None


def _claim_instance(port):
    """记下本进程的 pid 与端口，并在正常退出时清掉。"""
    try:
        os.makedirs(os.path.dirname(_INSTANCE_FILE), exist_ok=True)
        tmp = _INSTANCE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "port": port}, handle)
        os.replace(tmp, _INSTANCE_FILE)      # 原子替换，避免读到半截文件
    except OSError:
        return

    import atexit

    def _release():
        try:
            with open(_INSTANCE_FILE, "r", encoding="utf-8") as handle:
                info = json.load(handle)
            if info.get("pid") == os.getpid():   # 只删自己写的，不误删后来者的
                os.remove(_INSTANCE_FILE)
        except (OSError, ValueError):
            pass

    atexit.register(_release)


try:
    _existing_port = _running_instance()
except Exception:
    _existing_port = None

if _existing_port:
    # 按 webui.py 相同的格式输出端口，Electron 的 findGradioPort 才认得。
    # 它对整块 stdout 直接 JSON.parse，所以这一行必须独占一个数据块：
    # 先 flush 再等一秒，之后才允许打印别的东西。
    print(json.dumps({"server_port": _existing_port}))
    sys.stdout.flush()
    time.sleep(1)
    print(
        f"检测到 VoxCPM 已在运行（端口 {_existing_port}），"
        f"本次不重复启动，界面将连接到已在运行的那一个。"
    )
    sys.stdout.flush()
    raise SystemExit(0)

import webui

try:
    _claim_instance(getattr(webui, "server_port", 0))
except Exception:
    pass          # 记不上锁不影响运行，只是下次挡不住重复启动

webui.main()
