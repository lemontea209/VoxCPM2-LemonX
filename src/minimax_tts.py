"""MiniMax 云端 TTS 的最小本地适配层。

API Key 只用于请求头，绝不写入日志或历史记录。用户主动选择保存时，
密钥仅以当前 Windows 用户范围的 DPAPI 加密二进制形式保存。
"""

from __future__ import annotations

import ctypes
import datetime as _dt
import json
import logging
import os
import re
import secrets
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from ctypes import POINTER, Structure, byref, c_byte, cast, create_string_buffer
from ctypes import wintypes
from pathlib import Path
from typing import Tuple

import requests


logger = logging.getLogger(__name__)


MINIMAX_T2A_URL = "https://api.minimaxi.com/v1/t2a_v2"
MINIMAX_VOICE_DESIGN_URL = "https://api.minimaxi.com/v1/voice_design"
MINIMAX_GET_VOICE_URL = "https://api.minimaxi.com/v1/get_voice"
MINIMAX_MODEL_LIST_URL = "https://api.minimaxi.com/v1/models"
MINIMAX_FILES_UPLOAD_URL = "https://api.minimaxi.com/v1/files/upload"
MINIMAX_VOICE_CLONE_URL = "https://api.minimaxi.com/v1/voice_clone"
MINIMAX_DELETE_VOICE_URL = "https://api.minimaxi.com/v1/delete_voice"
SUPPORTED_MODELS = (
    "speech-2.8-turbo", "speech-2.8-hd", "speech-2.6-turbo", "speech-2.6-hd",
    "speech-02-turbo", "speech-02-hd", "speech-01-turbo", "speech-01-hd",
)
SUPPORTED_FORMATS = ("mp3", "wav", "flac")
SUPPORTED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100)
SUPPORTED_MP3_BITRATES = (32000, 64000, 128000, 256000)
SUPPORTED_CHANNELS = (1, 2)
_TTS_MODEL_ID_PATTERN = re.compile(r"^speech-[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUPPORTED_EMOTIONS = ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm")
_EXPRESSION_TAG_PATTERN = re.compile(
    r"\((?:laughs|chuckle|coughs|clear-throat|groans|breath|pant|inhale|exhale|gasps|sniffs|sighs|snorts|burps|lip-smacking|humming|hissing|emm|sneezes)\)"
)
_DPAPI_UI_FORBIDDEN = 0x1
_DPAPI_ENTROPY = b"VoxCPM2 MiniMax API key v1"


class _DataBlob(Structure):
    """Windows DATA_BLOB 的最小 ctypes 定义。"""

    _fields_ = [("cbData", wintypes.DWORD), ("pbData", POINTER(c_byte))]


def _secure_key_path() -> Path:
    """返回当前 Windows 用户专属的 DPAPI 密钥文件位置。"""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("无法确定当前 Windows 用户的本地配置目录。")
    return Path(local_app_data) / "YZYLauncher" / "VoxCPM2" / "minimax_tts_key.dpapi"


def _voice_library_path() -> Path:
    """返回本机音色库位置；仅保存 voice_id 和本地显示名，不保存密钥或音色描述。"""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("无法确定当前 Windows 用户的本地配置目录。")
    return Path(local_app_data) / "YZYLauncher" / "VoxCPM2" / "minimax_voice_library.json"


def _load_voice_library() -> list[dict[str, str]]:
    """读取本机已确认的云端音色；异常时安全地返回空库。"""
    try:
        raw_entries = json.loads(_voice_library_path().read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, TypeError):
        return []
    if not isinstance(raw_entries, list):
        return []
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        voice_id = item.get("voice_id")
        label = item.get("label")
        if not isinstance(voice_id, str) or not voice_id.strip() or voice_id in seen:
            continue
        if not isinstance(label, str) or not label.strip():
            label = "已保存云端音色"
        seen.add(voice_id)
        entries.append({"voice_id": voice_id, "label": label})
    return entries


def saved_voice_choices() -> list[tuple[str, str]]:
    """返回本机音色选项；同名项显示完整 voice_id，避免选错真实音色。"""
    entries = _load_voice_library()
    label_counts = Counter(entry["label"].casefold() for entry in entries)
    return [
        (
            entry["label"]
            if label_counts[entry["label"].casefold()] == 1
            else f"{entry['label']} · {entry['voice_id']}",
            entry["voice_id"],
        )
        for entry in entries
    ]


# 女声的判定线索：voice_id 里的 female / 中文标签里的性别字。
# 系统音色的 id 形如 female-shaonv / male-qn-daxuesheng，复刻与设计音色
# 的 id 是随机串，只能靠标签里的字判断，判不出来的一律归到后面。
_FEMALE_ID_PATTERN = re.compile(r"(^|[-_])female([-_]|$)", re.IGNORECASE)
_FEMALE_LABEL_HINTS = ("女", "少女", "御姐", "萌妹", "姐姐", "妹妹", "奶奶", "妈妈", "girl", "lady")


def _is_female_voice(choice: tuple[str, str]) -> bool:
    """判断一条音色是否为女声，用于「已获取音色」的排序。"""
    label, voice_id = choice[0], choice[1]
    if _FEMALE_ID_PATTERN.search(voice_id or ""):
        return True
    lowered = (label or "").lower()
    return any(hint.lower() in lowered for hint in _FEMALE_LABEL_HINTS)


def fetch_available_voice_choices(api_key: str) -> tuple[list[tuple[str, str]], str]:
    """按用户点击查询 MiniMax 可调用音色；查询结果仅留在当前页面会话。"""
    effective_key = (api_key or "").strip() or _saved_api_key()
    if not effective_key:
        raise RuntimeError("请填写 MiniMax API Key，或先点击“安全保存 API Key（DPAPI）”。")

    headers = {
        "Authorization": f"Bearer {effective_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            MINIMAX_GET_VOICE_URL,
            json={"voice_type": "all"},
            headers=headers,
            timeout=(10, 60),
        )
        response_data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        raise _safe_error() from None

    base_resp = response_data.get("base_resp") if isinstance(response_data, dict) else None
    base_status = base_resp.get("status_code") if isinstance(base_resp, dict) else None
    if response.status_code != 200 or (base_status is not None and base_status != 0):
        raise _safe_error()
    if not isinstance(response_data, dict):
        raise _safe_error()

    choices = saved_voice_choices()
    seen = {voice_id for _, voice_id in choices}
    remote_count = 0

    def add_choices(category: str, raw_items) -> None:
        nonlocal remote_count
        if not isinstance(raw_items, list):
            return
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            voice_id = item.get("voice_id")
            if not isinstance(voice_id, str) or not voice_id.strip() or voice_id in seen:
                continue
            voice_name = item.get("voice_name")
            label = f"{category} · {voice_name} · {voice_id}" if isinstance(voice_name, str) and voice_name.strip() else f"{category} · {voice_id}"
            seen.add(voice_id)
            choices.append((label, voice_id))
            remote_count += 1

    add_choices("系统音色", response_data.get("system_voice"))
    add_choices("我的设计音色", response_data.get("voice_generation"))
    add_choices("我的复刻音色", response_data.get("voice_cloning"))

    # 三百多条平铺很难找。已保存音色（列表最前面那一段）保持原顺序不动，
    # 只对「已获取」的部分重排：女声优先，同组内按标签字母序。
    # 只改顺序、不改标签文本——加前缀会让本来就长的标签更长，也会破坏
    # 用户已经熟悉的显示格式。
    saved_count = len(saved_voice_choices())
    head, tail = choices[:saved_count], choices[saved_count:]
    tail.sort(key=lambda item: (0 if _is_female_voice(item) else 1, item[0]))
    return (
        head + tail,
        f"已读取 {remote_count} 个 MiniMax 可用音色（女声排在前面）；"
        "结果仅保留在当前页面会话，不会写入配置文件。",
    )


def fetch_available_tts_model_ids(api_key: str) -> tuple[list[str], str]:
    """读取账户模型列表；仅把 speech-* 语音模型交给界面使用。"""
    effective_key = (api_key or "").strip() or _saved_api_key()
    if not effective_key:
        raise RuntimeError("请填写 MiniMax API Key，或先点击“安全保存 API Key（DPAPI）”。")

    try:
        response = requests.get(
            MINIMAX_MODEL_LIST_URL,
            headers={"Authorization": f"Bearer {effective_key}"},
            timeout=(10, 30),
        )
        response_data = response.json()
    except (requests.RequestException, ValueError, TypeError):
        raise _safe_error() from None

    if response.status_code != 200 or not isinstance(response_data, dict):
        raise _safe_error()

    # 密钥错误时服务端会返回 HTTP 200 + base_resp.status_code=1004。
    # 不校验这一层就会被当成「账户没有 speech-* 模型」走进兜底分支，
    # 状态栏用成功的口吻提示，用户会误以为密钥是有效的。
    base_resp = response_data.get("base_resp")
    base_status = base_resp.get("status_code") if isinstance(base_resp, dict) else None
    if base_status is not None and base_status != 0:
        raise _safe_error()

    remote_models: list[str] = []
    seen: set[str] = set()
    for item in response_data.get("data", []):
        model_id = item.get("id") if isinstance(item, dict) else None
        if not isinstance(model_id, str) or not _TTS_MODEL_ID_PATTERN.fullmatch(model_id):
            continue
        if model_id not in seen:
            seen.add(model_id)
            remote_models.append(model_id)

    if remote_models:
        return remote_models, f"已从 MiniMax 账户模型接口读取 {len(remote_models)} 个语音模型。"

    # MiniMax 的 /v1/models 在部分账户仅返回 OpenAI 兼容文本模型。
    # 此时仍使用官方同步语音合成接口当前声明的 speech-* 列表，且不混入文本模型。
    return list(SUPPORTED_MODELS), (
        "账户模型接口未返回 speech-*；已载入 MiniMax 官方同步语音合成支持的语音模型列表。"
    )


def clone_voice_with_upload(
    api_key: str,
    audio_file_path: str,
    voice_name: str = "",
    preview_text: str = "",
    model: str = "speech-2.8-turbo",
    text_validation: str = "",
    output_dir: str | Path | None = None,
) -> tuple[str, list[tuple[str, str]], str | None, str]:
    """上传本地音频到 MiniMax 云端克隆音色。

    官方接口为两段式：
      1) POST /v1/files/upload  上传音频，换取 file_id（integer）；
      2) POST /v1/voice_clone   用 file_id 创建复刻音色，返回 voice_id。
    若提供 preview_text（试听文本），voice_clone 会附带生成试听音频
    （返回 demo_audio 链接），本函数会下载到本地供播放器试听。
    创建成功后按用户填写的名称记入本机音色库。

    Returns:
        (voice_id, saved_choices, demo_audio_path_or_None, status_message)
    """
    resolved_key = (api_key or "").strip() or _saved_api_key()
    if not resolved_key:
        raise RuntimeError("请填写 MiniMax API Key，或先点击“安全保存 API Key（DPAPI）”。")
    audio_file_path = (audio_file_path or "").strip()
    if not audio_file_path or not Path(audio_file_path).is_file():
        raise RuntimeError("请先上传用于克隆的参考音频（或直接录音）。")

    # 实测 MiniMax 复刻对音频时长有下限（5.5s 被拒 "voice duration too short"，
    # 22.1s 通过）；上传前预检，避免浪费一次上传调用。
    try:
        import librosa
        audio_dur = librosa.get_duration(path=audio_file_path)
    except Exception:
        audio_dur = None
    if audio_dur is not None and audio_dur < 10.0:
        raise RuntimeError(
            f"克隆音频太短（{audio_dur:.1f} 秒）：MiniMax 云端复刻要求较长音频，建议 10 秒以上再试。"
        )

    headers = {"Authorization": f"Bearer {resolved_key}"}

    # 1) 上传音频拿 file_id
    try:
        with open(audio_file_path, "rb") as audio_fh:
            response = requests.post(
                MINIMAX_FILES_UPLOAD_URL,
                headers=headers,
                files={"file": (Path(audio_file_path).name, audio_fh)},
                data={"purpose": "voice_clone"},
                timeout=(10, 180),
            )
        response_data = response.json()
    except (requests.RequestException, ValueError) as error:
        # 上传本身不计费，只需区分「没送达」和「送达了」。
        raise _safe_error(error) from None

    if response.status_code != 200:
        raise _safe_error()
    if not isinstance(response_data, dict):
        raise _safe_error()
    base_resp = response_data.get("base_resp") if isinstance(response_data, dict) else None
    base_status = base_resp.get("status_code") if isinstance(base_resp, dict) else None
    if base_status not in (0, None):
        raise RuntimeError(
            f"音频上传失败（{base_resp.get('status_msg', '未知错误')}），请检查音频格式与大小后重试。"
        )
    file_obj = response_data.get("file") or {}
    file_id = None
    if isinstance(file_obj, dict):
        file_id = file_obj.get("file_id") or file_obj.get("id")
    # 官方 OpenAPI 定义 file_id 为 integer(int64)；保持原始数字类型回传，
    # 避免服务端严格类型校验时报 2013 invalid params。
    if file_id is None or str(file_id).strip() == "":
        raise RuntimeError("音频上传完成但未返回 file_id，请稍后重试。")

    # 2) 创建克隆音色（voice_id 要求字母开头且至少 8 字符）
    voice_id = "vc" + secrets.token_hex(5)
    payload: dict = {"voice_id": voice_id, "file_id": file_id}
    # 提供试听文本才返回 demo_audio（官方要求 text 与 model 成对出现）
    preview_text = (preview_text or "").strip()
    if preview_text:
        payload["text"] = preview_text[:1000]
        payload["model"] = model
    # 参考音频文本：作为 text_validation 供服务端 ASR 比对，提升克隆质量（上限 200 字符）
    text_validation = (text_validation or "").strip()
    if text_validation:
        payload["text_validation"] = text_validation[:200]
    try:
        response = requests.post(
            MINIMAX_VOICE_CLONE_URL,
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=(10, 180),
        )
        response_data = response.json()
    except (requests.RequestException, ValueError) as error:
        # billable=True：付费接口，读取超时不能笼统地叫用户「重试」。
        raise _safe_error(error, billable=True) from None

    if response.status_code != 200:
        raise _safe_error()
    if not isinstance(response_data, dict):
        raise _safe_error()
    base_resp = response_data.get("base_resp") if isinstance(response_data, dict) else None
    base_status = base_resp.get("status_code") if isinstance(base_resp, dict) else None
    if base_status not in (0, None):
        status_msg = base_resp.get("status_msg", "未知错误") if isinstance(base_resp, dict) else "未知错误"
        # 服务端对未认证账户笼统返回 invalid params；按官方 FAQ，声音复刻
        # 需要先完成个人/企业实名认证，把这一最常见原因直接提示给用户。
        if str(status_msg).strip() == "invalid params":
            raise RuntimeError(
                "云端克隆失败（invalid params）。请确认：① 账户已完成个人实名认证或企业认证"
                "（MiniMax 平台 → 账户信息 → 认证信息，声音复刻按法规要求必须认证）；"
                "② 音频时长 10 秒以上且不超过 5 分钟、大小不超过 20MB、格式为 mp3/m4a/wav。"
            )
        raise RuntimeError(f"云端克隆失败（{status_msg}），请稍后重试。")
    returned_id = response_data.get("voice_id")
    if isinstance(returned_id, str) and returned_id.strip():
        voice_id = returned_id.strip()

    # 下载试听音频（若有）：demo_audio 为 OSS 链接，落到本地 outputs 供播放器使用
    demo_path: str | None = None
    demo_url = response_data.get("demo_audio")
    if isinstance(demo_url, str) and demo_url.strip().startswith("http"):
        try:
            demo_resp = requests.get(demo_url.strip(), timeout=(10, 90))
            if demo_resp.status_code == 200 and demo_resp.content:
                target_dir = Path(output_dir).resolve() if output_dir else Path("outputs").resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                ext = _detect_audio_extension(demo_resp.content, fallback="mp3")
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                demo_path = str(target_dir / f"minimax_clone_demo_{ts}.{ext}")
                Path(demo_path).write_bytes(demo_resp.content)
        except Exception:
            demo_path = None

    # 3) 记入本机音色库
    label = (voice_name or "").strip() or f"克隆音色 {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
    entries = [entry for entry in _load_voice_library() if entry["voice_id"] != voice_id]
    entries.append({"voice_id": voice_id, "label": label})
    try:
        library_path = _voice_library_path()
        library_path.parent.mkdir(parents=True, exist_ok=True)
        library_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, RuntimeError):
        # 云端克隆是付过钱且不可撤销的。写库失败时必须把 voice_id 交给用户，
        # 否则这条音色在云端存在、本地却无从引用，只能重新克隆再付一次钱。
        raise RuntimeError(
            f"云端克隆已创建（voice_id `{voice_id}`，已产生计费），"
            "但无法写入本机音色库。请记下这个 voice_id，"
            "可在「云端配音」的「自定义 voice_id」里直接填写使用。"
        ) from None

    if demo_path:
        status = (
            f"云端克隆完成：已创建音色「{label}」（voice_id `{voice_id}`），"
            "已加入“已保存音色”下拉；试听音频已生成，可在右侧播放器试听。"
        )
    else:
        status = (
            f"云端克隆完成：已创建音色「{label}」（voice_id `{voice_id}`），"
            "已加入“已保存音色”下拉，可直接用于云端配音。"
        )
    return voice_id, saved_voice_choices(), demo_path, status


def save_designed_voice_to_library(voice_id: str) -> tuple[list[tuple[str, str]], str]:
    """由用户确认后把设计音色记入本机库；这不等同于 MiniMax 云端激活。"""
    voice_id = (voice_id or "").strip()
    if not voice_id:
        raise RuntimeError("请先完成音色设计并获得 voice_id，再确认保存。")
    entries = [entry for entry in _load_voice_library() if entry["voice_id"] != voice_id]
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    entries.append({"voice_id": voice_id, "label": f"设计音色 {timestamp}"})
    try:
        library_path = _voice_library_path()
        library_path.parent.mkdir(parents=True, exist_ok=True)
        library_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, RuntimeError):
        raise RuntimeError("无法保存本机音色库。") from None
    return saved_voice_choices(), "已保存到本机音色库，并已带入“云端配音”。首次正式配音会激活该临时音色。"


def _delete_cloud_voice(voice_id: str, api_key: str = "") -> tuple[bool, str]:
    """调用 MiniMax 官方接口删除云端音色；返回 (是否成功, 可直接展示的说明)。

    官方接口要求显式给出 voice_type，而本机音色库没有记录该条是克隆产生还是
    设计产生，因此按 voice_cloning -> voice_generation 依次尝试。删除不可逆：
    官方文档明确“删除后该 voice_id 将无法再次使用”。
    """
    voice_id = (voice_id or "").strip()
    if not voice_id:
        return False, "未指定 voice_id，云端音色未删除。"

    effective_key = (api_key or "").strip() or _saved_api_key()
    if not effective_key:
        return False, "未找到可用的 MiniMax API Key，云端音色未删除。"

    last_message = ""
    for voice_type in ("voice_cloning", "voice_generation"):
        try:
            response = requests.post(
                MINIMAX_DELETE_VOICE_URL,
                headers={
                    "Authorization": f"Bearer {effective_key}",
                    "Content-Type": "application/json",
                },
                json={"voice_type": voice_type, "voice_id": voice_id},
                timeout=(10, 30),
            )
            payload = response.json()
        except (requests.RequestException, ValueError, TypeError):
            return False, "云端删除请求失败（网络或响应异常），云端音色可能仍然存在。"

        if not isinstance(payload, dict):
            return False, "云端删除返回内容无法解析，云端音色可能仍然存在。"

        base_resp = payload.get("base_resp")
        base_resp = base_resp if isinstance(base_resp, dict) else {}
        status_code = base_resp.get("status_code", payload.get("status_code"))
        succeeded = response.status_code == 200 and (
            status_code == 0 or (status_code is None and payload.get("voice_id"))
        )
        if succeeded:
            return True, f"云端音色已删除（类别 {voice_type}）。"

        last_message = (
            str(base_resp.get("status_msg") or payload.get("status_msg") or "").strip()
            or f"HTTP {response.status_code}"
        )

    return False, (
        f"云端删除未成功：{last_message[:120]}"
        "（该音色可能已被 MiniMax 自动回收，或不属于可删除的两个类别）。"
    )


def delete_saved_voice(
    selection: str, api_key: str = "", delete_cloud: bool = True
) -> tuple[list[tuple[str, str]], str]:
    """删除一条已保存音色：先删 MiniMax 云端音色，再移除本机索引。

    selection 允许传入下拉框显示文本或 voice_id，两者都按精确匹配解析；
    解析不到唯一目标时直接报错，避免误删相邻条目。

    云端删除不可逆。云端这一步失败时（例如没有密钥、断网、音色已被
    MiniMax 自动回收）本机索引仍然照常移除——用户点了删除就该删掉，
    不能让一条已经无效的索引永远卡在下拉里；失败原因会在状态里如实说明。
    """
    selection = (selection or "").strip()
    if not selection:
        raise RuntimeError("未指定要删除的音色。")

    entries = _load_voice_library()
    label_by_voice_id = {voice_id: label for label, voice_id in saved_voice_choices()}

    def _normalized(value: str) -> str:
        # 前端从下拉项取文本时会把换行折叠成空格，这里用同样的规则比对，
        # 避免显示名里本来就带多个空格的音色匹配不上。
        return re.sub(r"\s+", " ", value or "").strip()

    target_id = selection if selection in label_by_voice_id else ""
    if not target_id:
        matched = [
            voice_id for voice_id, label in label_by_voice_id.items()
            if label == selection
        ]
        if len(matched) != 1:
            wanted = _normalized(selection)
            matched = [
                voice_id for voice_id, label in label_by_voice_id.items()
                if _normalized(label) == wanted
            ]
        if len(matched) == 1:
            target_id = matched[0]
    if not target_id:
        raise RuntimeError(f"本机音色库中未找到「{selection}」，未执行任何删除。")

    removed_label = label_by_voice_id.get(target_id, target_id)
    remaining = [entry for entry in entries if entry["voice_id"] != target_id]
    if len(remaining) == len(entries):
        raise RuntimeError(f"本机音色库中未找到「{selection}」，未执行任何删除。")

    # 先动云端：本机索引是找回云端音色的唯一线索，索引一旦先删就没法重试了。
    cloud_note = "未请求云端删除；"
    cloud_ok = True
    if delete_cloud:
        cloud_ok, cloud_note = _delete_cloud_voice(target_id, api_key)

    try:
        library_path = _voice_library_path()
        library_path.parent.mkdir(parents=True, exist_ok=True)
        # 沿用整合包既有约定：删除前先留一份带时间戳的快照，便于随时还原。
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = library_path.with_name(
            f"{library_path.stem}.before-delete-{stamp}-{secrets.token_hex(3)}.json"
        )
        if library_path.exists():
            backup_path.write_text(
                library_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
        # 先写临时文件再原子替换，避免写入中断把音色库截断成空文件。
        temp_path = library_path.with_name(library_path.name + ".tmp")
        temp_path.write_text(
            json.dumps(remaining, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temp_path, library_path)
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"无法写回本机音色库：{exc}") from None

    # 写回后重新读取校验，确保界面刷新前该条目确实已经不在库里。
    if any(entry["voice_id"] == target_id for entry in _load_voice_library()):
        raise RuntimeError("音色库写回后校验失败，删除未生效。")

    # 云端没删掉时前置一个醒目标记：本机索引一旦移除，voice_id 就是找回
    # 那条云端音色的唯一线索，用户必须当场知道云端还留着东西。
    prefix = "" if cloud_ok else "⚠️ 云端未删除 —— "
    return (
        saved_voice_choices(),
        f"{prefix}已删除「{removed_label}」：{cloud_note}本机索引已移除。",
    )


def _as_blob(data: bytes) -> tuple[_DataBlob, object]:
    """保留 ctypes 缓冲区引用，确保 Windows API 调用期间内存有效。"""
    buffer = create_string_buffer(data)
    return _DataBlob(len(data), cast(buffer, POINTER(c_byte))), buffer


def _dpapi_protect(data: bytes) -> bytes:
    """使用当前 Windows 用户范围的 DPAPI 加密，不显示系统 UI。"""
    if os.name != "nt":
        raise RuntimeError("安全保存仅支持 Windows DPAPI。")
    source, source_buffer = _as_blob(data)
    entropy, entropy_buffer = _as_blob(_DPAPI_ENTROPY)
    protected = _DataBlob()
    _ = (source_buffer, entropy_buffer)
    success = ctypes.windll.crypt32.CryptProtectData(
        byref(source),
        "VoxCPM2 MiniMax API Key",
        byref(entropy),
        None,
        None,
        _DPAPI_UI_FORBIDDEN,
        byref(protected),
    )
    if not success:
        raise RuntimeError("Windows DPAPI 无法保护密钥。")
    try:
        return ctypes.string_at(protected.pbData, protected.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(protected.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    """仅为当前 Windows 用户解密此前由本应用保护的密钥。"""
    if os.name != "nt":
        raise RuntimeError("安全保存仅支持 Windows DPAPI。")
    source, source_buffer = _as_blob(data)
    entropy, entropy_buffer = _as_blob(_DPAPI_ENTROPY)
    unprotected = _DataBlob()
    _ = (source_buffer, entropy_buffer)
    success = ctypes.windll.crypt32.CryptUnprotectData(
        byref(source),
        None,
        byref(entropy),
        None,
        None,
        _DPAPI_UI_FORBIDDEN,
        byref(unprotected),
    )
    if not success:
        raise RuntimeError("Windows DPAPI 无法读取已保存的密钥。")
    try:
        return ctypes.string_at(unprotected.pbData, unprotected.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(unprotected.pbData)


def has_saved_api_key() -> bool:
    """只报告是否存在保护文件，不解密，也不向前端发送密钥。"""
    try:
        return _secure_key_path().is_file()
    except RuntimeError:
        return False


def save_api_key(api_key: str) -> str:
    """由用户明确点击后保存密钥；落盘内容始终为 DPAPI 密文。"""
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("请先粘贴 MiniMax API Key，再点击安全保存。")
    try:
        encrypted = _dpapi_protect(api_key.encode("utf-8"))
        key_path = _secure_key_path()
        key_path.parent.mkdir(parents=True, exist_ok=True)
        key_path.write_bytes(encrypted)
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError("无法写入 Windows DPAPI 密钥存储。") from None
    return "已通过 Windows DPAPI 安全保存。页面不会回显密钥；之后可将输入框留空后直接生成。"


def _saved_api_key() -> str:
    """在 Python 进程内读取密钥；失败时不暴露路径、密文或底层错误。"""
    try:
        key_path = _secure_key_path()
        if not key_path.is_file():
            return ""
        return _dpapi_unprotect(key_path.read_bytes()).decode("utf-8").strip()
    except (OSError, RuntimeError, UnicodeDecodeError):
        return ""


def resolve_tts_model(selected_model: str, custom_model: str = "") -> str:
    """只接受 MiniMax TTS 使用的 speech-* 模型 ID，避免误用文本模型。"""
    candidate = (custom_model or "").strip() or (selected_model or "").strip()
    if candidate in SUPPORTED_MODELS:
        return candidate
    if _TTS_MODEL_ID_PATTERN.fullmatch(candidate):
        return candidate
    raise ValueError("自定义模型 ID 必须是 MiniMax 官方发布的 speech-* 语音模型，例如 speech-2.8-turbo。")


def _voice_design_candidate_count(candidate_count: int | str) -> int:
    """仅接受界面提供的 1、3、5 个候选槽位，避免意外扩大付费请求数量。"""
    try:
        count = int(candidate_count)
    except (TypeError, ValueError):
        raise RuntimeError("音色槽数量只能选择 1、3 或 5。") from None
    if count not in (1, 3, 5):
        raise RuntimeError("音色槽数量只能选择 1、3 或 5。")
    return count


def character_counter_markdown(value: str, maximum: int, label: str) -> str:
    """给 UI 显示字符数；这是接口字符数，不等同于计费字符数。"""
    count = len(value or "")
    remaining = max(0, maximum - count)
    return f"**{label}：** 已输入 {count} / {maximum} 个字符，剩余 {remaining} 个字符。"


def apply_pause_between_lines(text: str, pause_seconds: float) -> str:
    """把非空换行段落以官方 `<#x#>` 停顿标记连接，避免首尾或连续停顿。"""
    try:
        duration = float(pause_seconds)
    except (TypeError, ValueError):
        raise RuntimeError("断句停顿时长必须是 0.01 到 99.99 秒之间的数字。") from None
    if not 0.01 <= duration <= 99.99:
        raise RuntimeError("断句停顿时长必须在 0.01 到 99.99 秒之间。")
    parts = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if len(parts) < 2:
        raise RuntimeError("请至少输入两行文本，再按换行插入停顿。")
    return f"<#{duration:.2f}#>".join(parts)


def _as_dict(value: object) -> dict:
    """把响应里的字段安全地当字典用。

    服务端返回 {"base_resp": null} 时，dict.get("base_resp", {}) 拿到的是 None
    而不是 {}（键存在，只是值为 null），紧接着的 .get() 会抛 AttributeError——
    它不在捕获范围内，配合 show_error=True 会把原始 Python 报错甩给用户。
    """
    return value if isinstance(value, dict) else {}


def _safe_error(error: Exception | None = None, *, billable: bool = False) -> RuntimeError:
    """不透传远端错误，避免敏感请求上下文进入 UI 或日志。

    付费接口必须区分两种超时，否则「请重试」这句话会让用户重复付费：
      - 连接超时：请求根本没送达，没有产生计费，重试是安全的；
      - 读取超时：请求已经送达，服务端很可能已经完成合成并计费，
        此时重试等于再付一次钱。
    """
    if isinstance(error, requests.exceptions.ConnectTimeout):
        return RuntimeError(
            "连接 MiniMax 超时，请求未送达服务端（未产生计费），请检查网络后重试。"
        )
    if billable and isinstance(error, requests.exceptions.ReadTimeout):
        return RuntimeError(
            "请求已送达 MiniMax，但本地等待响应超时——服务端很可能已经完成合成并计费。"
            "直接重试会再计一次费；请先到 MiniMax 官网核对本次用量，再决定是否重试。"
        )
    return RuntimeError("MiniMax 请求未完成，请检查密钥、音色 ID、网络和账户余额后重试。")


def synthesize(
    api_key: str,
    text: str,
    voice_id: str,
    model: str,
    output_format: str,
    speed: float,
    language_boost: str,
    output_dir: str | Path,
    emotion: str = "",
    volume: float = 1.0,
    pitch: int = 0,
    bitrate: int = 128000,
    sample_rate: int = 44100,
    channel: int = 2,
) -> Tuple[str, str]:
    """调用 MiniMax 同步 T2A，并把结果直接写入本地 outputs 目录。"""
    api_key = (api_key or "").strip()
    text = (text or "").strip()
    voice_id = (voice_id or "").strip()
    try:
        model = resolve_tts_model(model)
    except ValueError as error:
        raise RuntimeError(str(error)) from None
    output_format = output_format if output_format in SUPPORTED_FORMATS else "mp3"
    emotion = (emotion or "").strip()
    try:
        volume = float(volume)
        pitch = int(pitch)
        bitrate = int(bitrate) if bitrate is not None else 128000
        sample_rate = int(sample_rate)
        channel = int(channel)
    except (TypeError, ValueError):
        raise RuntimeError("MiniMax 音频参数格式无效。") from None
    if not -12 <= pitch <= 12:
        raise RuntimeError("音高范围必须在 -12 到 12 之间。")
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise RuntimeError("采样率必须使用 MiniMax 官方支持的选项。")
    if channel not in SUPPORTED_CHANNELS:
        raise RuntimeError("声道只能选择单声道或双声道。")
    if output_format == "mp3" and bitrate not in SUPPORTED_MP3_BITRATES:
        raise RuntimeError("MP3 码率必须使用 MiniMax 官方支持的选项。")

    if not api_key:
        raise RuntimeError("请填写 MiniMax API Key。")
    if not voice_id:
        raise RuntimeError("请填写 MiniMax voice_id；可使用系统音色或你已有的自定义音色。")
    if not text:
        raise RuntimeError("请输入需要生成的文本。")
    if len(text) > 10_000:
        raise RuntimeError("单次同步生成最多支持 10,000 个输入字符，请拆分后重试。")
    if emotion and emotion not in _SUPPORTED_EMOTIONS:
        raise RuntimeError("情绪参数无效，请从页面提供的情绪选项中选择。")
    if not model.startswith("speech-2.8-") and _EXPRESSION_TAG_PATTERN.search(text):
        raise RuntimeError("语气词标签仅支持 speech-2.8-hd 和 speech-2.8-turbo；请切回 2.8 模型或删除语气词。")

    voice_setting = {
        "voice_id": voice_id,
        "speed": float(speed),
        "vol": volume,
        "pitch": pitch,
    }
    if emotion:
        voice_setting["emotion"] = emotion

    payload = {
        "model": model,
        "text": text,
        "stream": False,
        "language_boost": language_boost,
        "output_format": "hex",
        "voice_setting": voice_setting,
        "audio_setting": {
            "sample_rate": sample_rate,
            "format": output_format,
            "channel": channel,
        },
    }
    if output_format == "mp3":
        payload["audio_setting"]["bitrate"] = bitrate
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(MINIMAX_T2A_URL, json=payload, headers=headers, timeout=(10, 120))
        response_data = response.json()
    except (requests.RequestException, ValueError) as error:
        # billable=True：付费接口，读取超时不能笼统地叫用户「重试」。
        raise _safe_error(error, billable=True) from None

    if response.status_code != 200:
        raise _safe_error()
    response_data = _as_dict(response_data)
    if _as_dict(response_data.get("base_resp")).get("status_code") not in (0, None):
        raise _safe_error()

    audio_hex = _as_dict(response_data.get("data")).get("audio")
    if not isinstance(audio_hex, str) or not audio_hex.strip():
        raise _safe_error()

    try:
        audio_bytes = bytes.fromhex(audio_hex.strip())
    except ValueError:
        raise _safe_error() from None
    if not audio_bytes:
        raise _safe_error()

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = target_dir / f"minimax_{timestamp}_{secrets.token_hex(3)}.{output_format}"
    target.write_bytes(audio_bytes)

    return str(target), "生成完成，音频已保存到本地 整合包\\win-unpacked\\python\\outputs 目录。"


def synthesize_with_saved_key(
    api_key: str,
    text: str,
    voice_id: str,
    model: str,
    output_format: str,
    speed: float,
    language_boost: str,
    output_dir: str | Path,
    custom_model: str = "",
    emotion: str = "",
    volume: float = 1.0,
    pitch: int = 0,
    bitrate: int = 128000,
    sample_rate: int = 44100,
    channel: int = 2,
) -> Tuple[str, str]:
    """优先使用本次输入；留空时仅在本机 Python 进程内读取 DPAPI 密钥。"""
    resolved_key = (api_key or "").strip() or _saved_api_key()
    if not resolved_key:
        raise RuntimeError("请填写 MiniMax API Key，或先点击“安全保存 API Key（DPAPI）”。")
    try:
        resolved_model = resolve_tts_model(model, custom_model)
    except ValueError as error:
        raise RuntimeError(str(error)) from None
    return synthesize(
        resolved_key,
        text,
        voice_id,
        resolved_model,
        output_format,
        speed,
        language_boost,
        output_dir,
        emotion,
        volume,
        pitch,
        bitrate,
        sample_rate,
        channel,
    )


def _detect_audio_extension(audio_bytes: bytes, fallback: str = "mp3") -> str:
    """依据常见文件头保存试听，避免依赖远端错误文本或不可靠扩展名。"""
    if audio_bytes.startswith(b"RIFF") and audio_bytes[8:12] == b"WAVE":
        return "wav"
    if audio_bytes.startswith(b"fLaC"):
        return "flac"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    return fallback


def design_voice_with_saved_key(
    api_key: str,
    prompt: str,
    preview_text: str,
    aigc_watermark: bool,
    output_dir: str | Path,
) -> Tuple[str, str, str, str]:
    """设计 MiniMax 音色并保存试听；返回值会把新 voice_id 回填至配音区。"""
    resolved_key = (api_key or "").strip() or _saved_api_key()
    prompt = (prompt or "").strip()
    preview_text = (preview_text or "").strip()
    if not resolved_key:
        raise RuntimeError("请填写 MiniMax API Key，或先点击“安全保存 API Key（DPAPI）”。")
    if not prompt:
        raise RuntimeError("请填写音色描述，例如性别、年龄感、音质、情绪与语速倾向。")
    if len(prompt) > 1000:
        raise RuntimeError("音色描述最多 1000 个字符，请缩短后重试。")
    if not preview_text:
        raise RuntimeError("请填写试听文本。")
    if len(preview_text) > 500:
        raise RuntimeError("MiniMax 音色设计的试听文本最多 500 个字符，请缩短后重试。")

    payload = {
        "prompt": prompt,
        "preview_text": preview_text,
        "aigc_watermark": bool(aigc_watermark),
    }
    headers = {
        "Authorization": f"Bearer {resolved_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            MINIMAX_VOICE_DESIGN_URL,
            json=payload,
            headers=headers,
            timeout=(10, 120),
        )
        response_data = response.json()
    except (requests.RequestException, ValueError) as error:
        # billable=True：付费接口，读取超时不能笼统地叫用户「重试」。
        raise _safe_error(error, billable=True) from None

    if response.status_code != 200:
        raise _safe_error()
    if response_data.get("base_resp", {}).get("status_code") not in (0, None):
        raise _safe_error()

    voice_id = response_data.get("voice_id")
    trial_audio_hex = response_data.get("trial_audio")
    if not isinstance(voice_id, str) or not voice_id.strip():
        raise _safe_error()
    if not isinstance(trial_audio_hex, str) or not trial_audio_hex.strip():
        raise _safe_error()
    try:
        audio_bytes = bytes.fromhex(trial_audio_hex.strip())
    except ValueError:
        raise _safe_error() from None
    if not audio_bytes:
        raise _safe_error()

    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    extension = _detect_audio_extension(audio_bytes)
    target = target_dir / f"minimax_voice_design_{timestamp}_{secrets.token_hex(3)}.{extension}"
    target.write_bytes(audio_bytes)
    status = (
        "音色设计试听已保存；已返回候选 voice_id，尚未写入本机音色库。"
        "请确认保存后，在 7 天内用该音色完成一次正式合成，以免临时音色失效。"
    )
    return str(target), voice_id, voice_id, status


def design_voice_candidates_with_saved_key(
    api_key: str,
    prompt: str,
    preview_text: str,
    aigc_watermark: bool,
    output_dir: str | Path,
    candidate_count: int | str = 1,
) -> Tuple[list[dict[str, str]], str]:
    """抽取 1、3 或 5 个独立音色候选；不保存候选 ID，也不记录密钥。"""
    count = _voice_design_candidate_count(candidate_count)
    resolved_key = (api_key or "").strip() or _saved_api_key()
    if not resolved_key:
        raise RuntimeError("请填写 MiniMax API Key，或先点击“安全保存 API Key（DPAPI）”。")

    # 本地参数校验必须放在提交线程池之前：放在被 submit 的函数里，异常会被
    # future.result() 的 except 吞成「未完成计数」，最终报成「请检查密钥、
    # 网络和账户余额」——用户会跑去平台充值，而真正的原因只是有个输入框空着。
    if not (prompt or "").strip():
        raise RuntimeError("请填写音色描述，例如性别、年龄感、音质、情绪与语速倾向。")
    if len((prompt or "").strip()) > 1000:
        raise RuntimeError("音色描述最多 1000 个字符，请缩短后重试。")

    def _design_one() -> Tuple[str, str, str, str]:
        return design_voice_with_saved_key(
            resolved_key, prompt, preview_text, aigc_watermark, output_dir
        )

    candidates: list[dict[str, str] | None] = [None] * count
    failed = 0
    # 5 个候选仍是 5 次独立请求；最多并发 3 次，减少限流风险并缩短等待时间。
    with ThreadPoolExecutor(max_workers=min(3, count)) as executor:
        pending = {executor.submit(_design_one): index for index in range(count)}
        for future in as_completed(pending):
            index = pending[future]
            try:
                audio_path, voice_id, _, _ = future.result()
                candidates[index] = {"audio_path": audio_path, "voice_id": voice_id}
            except Exception as error:
                failed += 1
                # 原来这里整个吞掉，事后完全查不到失败原因。
                # 只记异常类型与文案，不记密钥、不记请求体。
                logger.warning(
                    "音色槽 %s 抽取失败：%s: %s",
                    index + 1, type(error).__name__, error,
                )

    completed = [candidate for candidate in candidates if candidate is not None]
    if not completed:
        raise _safe_error()
    status = (
        f"已完成 {len(completed)} / {count} 个音色槽试听，均已保存到本地 整合包\\win-unpacked\\python\\outputs 目录。"
        "当前默认选中音色槽 1；点击其他音色槽可切换右侧播放器和 voice_id。"
        "确认保存前不会写入本机音色库。"
    )
    if failed:
        # 失败的那几次很可能已经在云端合成并计费，无脑邀请「重新抽取」
        # 会让用户按提示重复付费。这里如实说明，把决定权交回给用户。
        status += (
            f"另有 {failed} 个音色槽未完成——这些请求可能已经送达 MiniMax 并产生计费，"
            "重新抽取会再计一次费。建议先到 MiniMax 官网核对用量再决定。"
        )
    return completed, status
