import os

# 显存优化：让 CUDA 分配器使用可扩展段，减少多段生成时的显存碎片化与持续增长。
# 必须在导入 torch（funasr / voxcpm 会触发）之前设置才生效；尊重用户已有的设置。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

current_dir = os.path.dirname(os.path.abspath(__file__))
ffmpeg = os.path.join(current_dir, "ffmpeg", "bin")
rubberband = os.path.join(current_dir, "rubberband")
os.environ["PATH"] = ffmpeg + os.pathsep + rubberband + os.pathsep + os.environ.get("PATH", "")

import re
import sys
import io
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "src"))

import time
import uuid
import wave
import shutil
import logging
import datetime
import threading
import atexit
import contextlib
import inspect
import numpy as np
import gradio as gr
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from system_status import initial_status, refresh_status, STATUS_CSS, STATUS_JS
from about import create_about_tab
from minimax_tts import (
    character_counter_markdown,
    clone_voice_with_upload,
    delete_saved_voice as minimax_delete_saved_voice,
    design_voice_candidates_with_saved_key,
    fetch_available_tts_model_ids,
    fetch_available_voice_choices,
    has_saved_api_key,
    save_api_key as minimax_save_api_key,
    save_designed_voice_to_library,
    saved_voice_choices,
    synthesize_with_saved_key as minimax_synthesize,
)

try:
    import torch
except Exception:  # pragma: no cover - torch 一定存在，仅作兜底
    torch = None

import voxcpm
from voxcpm.model.utils import resolve_runtime_device

# Gradio 6 把 css 从 Blocks 构造器挪到了 launch()。仍然传给构造器不会出错，
# 但会打印一条 UserWarning——在启动器的「日志」标签页里显示成红色 [ERR]，
# 发布版每次启动都让用户看到一条像报错的东西。
# 这里不写死传给谁，按当前 Gradio 的实际签名决定：新旧版本都不会告警，
# 也不会因为某个版本不认这个参数而导致启动失败。
_LAUNCH_SUPPORTS_CSS = "css" in inspect.signature(gr.Blocks.launch).parameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False  # 避免重复打印
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

################################ YZY启动器配置专用 开始 ##########################################
import socket, json
YZY_CONFIG_PATH = os.path.join(current_dir, "..", "yzy_config.json")

def load_config():
    """ 加载启动器的配置文件 """
    if os.path.exists(YZY_CONFIG_PATH):
        with open(YZY_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def find_available_port(start_port=9000):
    """ 找一个可用端口 """
    port = start_port
    cnt = 0
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except Exception:
            cnt += 1
            if cnt >= 20:
                print("请检查网络是否正常.")
                sys.exit(1)
            print(f"端口 {port} 已被占用，尝试下一个...")
            sys.stdout.flush()
            time.sleep(0.2)  # 必须强制立即打印
            port += 1

    # 必须是第一个打印的：把端口号打印到 stdout（electron 会捕获）
server_port = find_available_port()
print(json.dumps({"server_port": server_port}))
sys.stdout.flush()
time.sleep(1)

yzy_config = load_config()
fp16 = True if yzy_config.get("fp16") is None else yzy_config.get("fp16")

################################ YZY启动器配置专用 结束 ##########################################


# ---------- 配置 ----------

VOICES = Path(current_dir) / "voices"  # 预设参考音频目录；不依赖进程工作目录
# 与 VOICES 保持一致，改用绝对路径：原来是相对路径，输出目录会随进程工作目录
# 漂移，界面提示的路径与实际落盘位置可能对不上，历史记录也会时有时无。
# 当前启动方式下 cwd 恰好就是本目录，因此这次改动不会移动任何已有文件。
OUTPUTPUTS = Path(current_dir) / "outputs"   # 历史记录 / 生成结果保存目录
PERSONAL_VOICE_META = VOICES / ".personal_voices.json"

# 预设下拉框首项占位符（不是真实音色，仅作提示）
# 云端「待生成文本」的标题在三处出现（初始、SRT 预览、取消 SRT 后重置），
# 统一由这个常量提供，避免改了一处漏两处。
MINIMAX_TEXT_LABEL = "待生成文本（云端在线生成音频）"

# 本地声音设计的试听文件保留数量。这些文件不进历史列表也没有清理入口，
# 不修剪就会在 outputs 里无限堆积。
VOICE_DESIGN_PREVIEW_KEEP = 20

PRESET_PLACEHOLDER = "--- 请选择预设音频 ---"

# 参考音频支持的扩展名
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

# 多角色配音最多支持的角色数
MAX_ROLES = 8

# 长文本分段生成：单段目标字符数上限（超过则自动从合理位置切分后逐段生成再合并）
MAX_TTS_CHARS = 70

# 本地配音的长段生成档位：默认保守兼容 8GB 显存，也允许高显存用户减少分段次数。
SEGMENT_LIMIT_CHOICES = (70, 100, 120, 150)

# 计数必须在浏览器端完成：浏览器会恢复上次输入，但不会因此触发 Python 回调。
MINIMAX_DESIGN_COUNTER_JS = """
() => {
  if (window.__minimaxDesignCounterInstalled) {
    window.__minimaxRefreshDesignCounters?.();
    return;
  }
  window.__minimaxDesignCounterInstalled = true;
  const configs = [
    { inputId: "minimax-dubbing-text", counterId: "minimax-dubbing-text-counter", maximum: 10000, label: "待生成文本" },
    { inputId: "minimax-design-prompt", counterId: "minimax-design-prompt-counter", maximum: 1000, label: "音色描述" },
    { inputId: "minimax-design-preview", counterId: "minimax-design-preview-counter", maximum: 500, label: "试听文本" },
  ];
  const renderCounter = (config, value) => {
    const counter = document.getElementById(config.counterId);
    if (!counter) return;
    const target = counter.querySelector(".prose") || counter;
    const count = Array.from(value || "").length;
    const remaining = Math.max(0, config.maximum - count);
    target.textContent = `${config.label}：已输入 ${count} / ${config.maximum} 个字符，剩余 ${remaining} 个字符。`;
  };
  const refreshCounters = () => {
    for (const config of configs) {
      const root = document.getElementById(config.inputId);
      const input = root && root.querySelector("textarea, input");
      if (input) renderCounter(config, input.value);
    }
  };
  window.__minimaxRefreshDesignCounters = refreshCounters;
  document.addEventListener("input", (event) => {
    for (const config of configs) {
      const root = document.getElementById(config.inputId);
      if (root && root.contains(event.target)) {
        renderCounter(config, event.target.value || "");
        break;
      }
    }
  }, true);
  document.addEventListener("change", (event) => {
    for (const config of configs) {
      const root = document.getElementById(config.inputId);
      if (root && root.contains(event.target)) {
        renderCounter(config, event.target.value || "");
        break;
      }
    }
  }, true);
  // 所有输入控件在首次加载时均已挂载；避免监听整个页面的音频/状态刷新，
  // 以免处理动画和标签切换时反复扫描 DOM，影响浏览器端的流畅度。
  requestAnimationFrame(refreshCounters);
  setTimeout(refreshCounters, 250);
  setTimeout(refreshCounters, 1000);
}
"""

# 分段之间插入的静音时长（秒），让合并后的语音衔接更自然
LOCAL_TEXTAREA_AUTOGROW_JS = """
() => {
  // The first local dubbing textarea keeps its existing auto-growth behavior.
  const configs = {
    "local-dubbing-text": { minLines: 8, maxLines: 8 },
    "local-reference-text": { minLines: 8, maxLines: 8 },
    "minimax-design-prompt": { minLines: 8, maxLines: 8 },
    "minimax-design-preview": { minLines: 8, maxLines: 8 },
  };
  const ids = Object.keys(configs);
  const resize = (input, config) => {
    input.style.height = "auto";
    const style = window.getComputedStyle(input);
    const lineHeight = Number.parseFloat(style.lineHeight) || 24;
    const padding = (Number.parseFloat(style.paddingTop) || 0) + (Number.parseFloat(style.paddingBottom) || 0);
    const minHeight = lineHeight * config.minLines + padding;
    const wantedHeight = Math.max(input.scrollHeight, minHeight);
    const maxHeight = config.maxLines == null ? null : lineHeight * config.maxLines + padding;
    const targetHeight = maxHeight == null ? wantedHeight : Math.min(wantedHeight, maxHeight);
    input.style.height = `${targetHeight}px`;
    input.style.overflowY = maxHeight != null && wantedHeight > maxHeight ? "auto" : "hidden";
  };
  const refresh = () => {
    for (const id of ids) {
      const root = document.getElementById(id);
      const input = root && root.querySelector("textarea");
      if (!input) continue;
      resize(input, configs[id]);
    }
  };
  if (window.__localTextareaAutogrowInstalled) {
    window.__localRefreshTextareas?.();
    return;
  }
  window.__localTextareaAutogrowInstalled = true;
  window.__localRefreshTextareas = refresh;
  document.addEventListener("input", (event) => {
    for (const id of ids) {
      const root = document.getElementById(id);
      if (root && root.contains(event.target)) {
        resize(event.target, configs[id]);
        break;
      }
    }
  }, true);
  // 输入事件和以下首次加载刷新足以覆盖自动增高；不监听整页 DOM 变化，
  // 防止音频处理状态刷新时重复测量全部文本框。
  requestAnimationFrame(refresh);
  setTimeout(refresh, 250);
  setTimeout(refresh, 1000);
}
"""

# 参考音频底部两个来源按钮的原生悬停提示，不改变点击行为。
LOCAL_REFERENCE_AUDIO_TOOLTIP_JS = """
() => {
  if (window.__localReferenceAudioTooltipInstalled) return;
  window.__localReferenceAudioTooltipInstalled = true;
  const decorate = () => {
    const root = document.getElementById("local-reference-audio");
    if (!root) return;
    const buttons = Array.from(root.querySelectorAll("button"));
    let fallbackIndex = 0;
    for (const button of buttons) {
      const marker = `${button.getAttribute("aria-label") || ""} ${button.title || ""} ${button.textContent || ""}`.toLowerCase();
      let label = "";
      if (/record|microphone|mic|录音/.test(marker)) label = "录音";
      else if (/upload|file|上传/.test(marker)) label = "上传";
      else if (fallbackIndex < 2) label = fallbackIndex++ === 0 ? "上传" : "录音";
      if (!label) continue;
      button.title = label;
      button.setAttribute("aria-label", label);
    }
  };
  const observer = new MutationObserver(decorate);
  observer.observe(document.body, { childList: true, subtree: true });
  decorate();
  setTimeout(decorate, 250);
  setTimeout(decorate, 1000);
}
"""

# 在预设音色下拉菜单的个人音色选项右侧注入“删除”动作；不改变普通选择行为。
LOCAL_RESET_TOOLTIP_JS = """
() => {
  if (window.__localResetTooltipInstalled) return;
  window.__localResetTooltipInstalled = true;
  const label = "\u91cd\u7f6e";
  const decorate = () => {
    const roots = document.querySelectorAll(
      "#page-local-dubbing, #page-local-voice-design, #page-minimax-dubbing, #page-minimax-voice-design"
    );
    for (const root of roots) {
      const icons = root.querySelectorAll(
        'svg[aria-label="undo"], svg.feather-rotate-ccw'
      );
      for (const icon of icons) {
        const button = icon.closest("button");
        if (!button) continue;
        button.title = label;
        button.setAttribute("aria-label", label);
      }
    }
  };
  const observer = new MutationObserver(decorate);
  observer.observe(document.body, { childList: true, subtree: true });
  decorate();
  setTimeout(decorate, 250);
  setTimeout(decorate, 1000);
}
"""

LOCAL_MULTIROLE_COPY_JS = """
() => {
  if (window.__localMultiroleCopyInstalled) return;
  window.__localMultiroleCopyInstalled = true;
  const text = "[\u89d2\u8272\u540d]:";
  const copyText = async () => {
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch (_) {
        // Fall through to the legacy copy path when clipboard permissions are unavailable.
      }
    }
    const fallback = document.createElement("textarea");
    fallback.value = text;
    fallback.setAttribute("readonly", "");
    fallback.style.position = "fixed";
    fallback.style.opacity = "0";
    document.body.appendChild(fallback);
    fallback.select();
    document.execCommand("copy");
    fallback.remove();
  };
  const decorate = () => {
    const root = document.getElementById("local-multirole-panel");
    if (!root) return;
    const buttons = root.querySelectorAll("button.copy_code_button");
    for (const button of buttons) {
      if (button.dataset.localMultiroleCopy === "1") continue;
      button.dataset.localMultiroleCopy = "1";
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopImmediatePropagation();
        copyText();
      }, true);
    }
  };
  const observer = new MutationObserver(decorate);
  observer.observe(document.body, { childList: true, subtree: true });
  decorate();
  setTimeout(decorate, 250);
  setTimeout(decorate, 1000);
}
"""

LOCAL_PRESET_DELETE_JS = """
() => {
  if (window.__localPresetDeleteInstalled) return;
  window.__localPresetDeleteInstalled = true;

  const decorate = () => {
    const listboxes = Array.from(document.querySelectorAll('[role="listbox"]'));
    for (const listbox of listboxes) {
      const rect = listbox.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;
      for (const option of listbox.querySelectorAll('[role="option"]')) {
        if (option.querySelector('.local-preset-delete-action')) continue;
        // Gradio 的选项节点会额外渲染一个选中标记（✓）。
        // aria-label 保留的是后端 choices 提供的原始显示文本，避免把该标记
        // 一并当成音色名称传回删除事件。
        // 注意：不能先 trim() —— U+2060 不属于 JS 认定的空白字符，
        // 但先 trim 会让后面的判断顺序变得难读，这里直接在原串上判定。
        const raw = option.getAttribute('aria-label') || option.textContent || '';
        if (raw.indexOf('\\u2060') < 0) continue;
        const name = raw.replace(/\\u2060/g, '').trim();
        if (!name) continue;

        option.style.position = 'relative';
        // 选项标签只保留音色名称；“删除”由右侧动作按钮单独显示，避免出现两次。
        option.textContent = name;
        const action = document.createElement('button');
        action.type = 'button';
        action.className = 'local-preset-delete-action';
        action.textContent = '\\u00d7';
        action.setAttribute('aria-label', `删除音色 ${name}`);
        action.addEventListener('mousedown', (event) => {
          event.preventDefault();
          event.stopPropagation();
        });
        action.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();
          if (!window.confirm(`是否删除个人音色“${name}”？`)) return;
          const refreshDropdownAfterDelete = () => {
            for (const candidate of document.querySelectorAll('[role="option"]')) {
              const rawCandidate = candidate.getAttribute('aria-label') || candidate.textContent || '';
              const candidateName = rawCandidate.replace(/\\u2060/g, '').trim();
              if (candidateName === name) candidate.remove();
            }
            const selectInput = document.querySelector('#local-preset-voice-dropdown input');
            if (selectInput) {
              const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
              if (setter) setter.call(selectInput, '--- 请选择预设音频 ---');
              else selectInput.value = '--- 请选择预设音频 ---';
              selectInput.dispatchEvent(new Event('input', { bubbles: true }));
              selectInput.dispatchEvent(new Event('change', { bubbles: true }));
            }
            document.dispatchEvent(new KeyboardEvent('keydown', {
              key: 'Escape',
              code: 'Escape',
              bubbles: true,
            }));
          };
          // Electron 内嵌页面中，移到屏幕外的 Gradio 按钮可能收不到
          // 合成 click。直接调用同一 Gradio API，仍由后端执行回收站删除。
          // 使用当前 Gradio 页面来源，避免 Electron 内嵌页面把绝对回环地址
          // 视为跨来源请求而拦截；页面与 API 同源时仍保持原有接口路径。
          const gradioBase = window.location.origin;
          fetch(`${gradioBase}/gradio_api/call/_delete_voice`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: [name, name] }),
          }).then(async (response) => {
            if (!response.ok) throw new Error(`delete request failed: ${response.status}`);
            const payload = await response.json();
            if (!payload.event_id) throw new Error('delete event id missing');
            const result = await fetch(`${gradioBase}/gradio_api/call/_delete_voice/${encodeURIComponent(payload.event_id)}`);
            const stream = await result.text();
            // Gradio SSE 返回 complete 即表示后端函数已完成；后端函数本身
            // 会校验文件并在成功后返回刷新后的 choices。
            if (!result.ok) {
              throw new Error(`delete result failed: ${result.status}`);
            }
            // 后端拒绝或回收站操作失败时会发送 error；此时 Gradio 已显示
            // 具体错误提示，必须保留当前选项，不能再走乐观刷新或重复调用。
            if (stream.includes('event: error')) return;
            if (!stream.includes('event: complete')) {
              throw new Error('delete event did not complete');
            }
            refreshDropdownAfterDelete();
          }).catch(() => {
            // API 失败时保留原有隐藏控件路径作为兼容回退。
            const field = document.querySelector('#local-delete-voice-name textarea, #local-delete-voice-name input');
            const button = document.querySelector('#local-delete-voice-button button');
            if (!field || !button) return;
            const valueSetter = Object.getOwnPropertyDescriptor(
              field instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
              'value',
            )?.set;
            if (valueSetter) valueSetter.call(field, name);
            else field.value = name;
            field.dispatchEvent(new Event('input', { bubbles: true }));
            field.dispatchEvent(new Event('change', { bubbles: true }));
            field.blur();
          });
          // 到这里说明回退路径已经派发完毕；下面原有 36 行「legacy hidden-control path」在 return 之后永远不可达，
          // 且 .catch() 分支已用同样的逻辑覆盖，已于 2026-08-28 删除。
          return;
        });
        option.appendChild(action);
      }
    }
  };

  const observer = new MutationObserver(decorate);
  observer.observe(document.body, { childList: true, subtree: true });
  document.addEventListener('click', () => setTimeout(decorate, 0), true);
  document.addEventListener('focusin', () => setTimeout(decorate, 0), true);
  decorate();
}
"""

# 在“已保存音色”下拉的每一行右侧注入“×”删除动作；确认后真实写回本机音色库。
MINIMAX_SAVED_VOICE_DELETE_JS = """
() => {
  if (window.__minimaxSavedVoiceDeleteInstalled) return;
  window.__minimaxSavedVoiceDeleteInstalled = true;

  const WRAP_ID = 'minimax-saved-voice-id';
  // 已确认删除的显示名。下拉每次重新渲染时按此集合剔除，避免后端已经删掉、
  // 而当前页面会话仍持有旧 choices 时把它重新画回列表里。
  const deleted = new Set();

  // Gradio 6 的下拉选项层既可能渲染在组件内部，也可能被挪到别处。
  // 依次用「组件内部」「aria-controls/aria-owns 指向」「组件展开或聚焦时的可见列表」
  // 三种方式定位，确保只处理“已保存音色”这一个下拉，不误伤“已获取音色”。
  const savedVoiceListboxes = () => {
    const wrap = document.getElementById(WRAP_ID);
    if (!wrap) return [];
    const found = new Set();
    for (const box of wrap.querySelectorAll('[role="listbox"]')) found.add(box);
    for (const input of wrap.querySelectorAll('input')) {
      const owned = input.getAttribute('aria-controls') || input.getAttribute('aria-owns');
      if (!owned) continue;
      const box = document.getElementById(owned);
      if (box) found.add(box);
    }
    if (!found.size) {
      const expanded = wrap.querySelector('[aria-expanded="true"]');
      const focused = document.activeElement && wrap.contains(document.activeElement);
      if (expanded || focused) {
        for (const box of document.querySelectorAll('[role="listbox"]')) {
          const rect = box.getBoundingClientRect();
          if (rect.width && rect.height) found.add(box);
        }
      }
    }
    return Array.from(found);
  };

  const optionName = (option) => {
    // aria-label 保留后端 choices 的原始显示文本；回退到 textContent 时要剔除
    // Gradio 的选中标记和本脚本注入的“×”，并把换行折叠回单个空格。
    const raw = option.getAttribute('aria-label') || option.textContent || '';
    return raw.replace(/[\\u2713\\u2714\\u00d7]/g, '').replace(/\\s+/g, ' ').trim();
  };

  const closeDropdown = () => {
    document.dispatchEvent(new KeyboardEvent('keydown', {
      key: 'Escape', code: 'Escape', bubbles: true,
    }));
  };

  // 被删掉的音色如果正好是当前选中项，清掉输入框里的残留显示。
  const clearSelectionIfDeleted = (name) => {
    const input = document.querySelector('#' + WRAP_ID + ' input');
    if (!input) return;
    if ((input.value || '').replace(/\\s+/g, ' ').trim() !== name) return;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
    if (setter) setter.call(input, '');
    else input.value = '';
    input.dispatchEvent(new Event('input', { bubbles: true }));
  };

  // 主路径：直接调用同一个 Gradio 事件接口。移到屏幕外的 Gradio 控件在
  // Electron 内嵌页面里收不到合成事件，这条 fetch 路径才是可靠的。
  const requestDelete = async (name) => {
    const base = window.location.origin;
    const post = await fetch(`${base}/gradio_api/call/_delete_minimax_saved_voice`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data: [name] }),
    });
    if (!post.ok) throw new Error(`提交失败 HTTP ${post.status}`);
    const payload = await post.json();
    if (!payload.event_id) throw new Error('未拿到 event_id');
    const res = await fetch(
      `${base}/gradio_api/call/_delete_minimax_saved_voice/${encodeURIComponent(payload.event_id)}`
    );
    if (!res.ok) throw new Error(`读取结果失败 HTTP ${res.status}`);
    const stream = await res.text();
    if (stream.includes('event: error')) {
      // 后端已用 gr.Error 拒绝并在界面提示；把原因带出来便于排查。
      const line = stream.split('\\n').find((item) => item.startsWith('data:')) || '';
      throw new Error(`后端拒绝：${line.replace(/^data:\\s*/, '').slice(0, 200)}`);
    }
    if (!stream.includes('event: complete')) throw new Error('事件未完成');
    // 取回后端返回的状态文案（outputs 的第 4 项是 minimax_status）。
    const dataLine = stream.split('\\n').filter((l) => l.startsWith('data:')).pop() || '';
    try {
      const outputs = JSON.parse(dataLine.replace(/^data:\\s*/, ''));
      return Array.isArray(outputs) ? String(outputs[3] || '') : '';
    } catch (parseError) {
      return '';
    }
  };

  // 回退路径：隐藏输入框的 input 事件。后端成功后会把该输入框清空，
  // 以此判断这条路是否真的走通。
  const fallbackDelete = (name, reason) => {
    const field = document.querySelector(
      '#minimax-delete-voice-name textarea, #minimax-delete-voice-name input'
    );
    if (!field) {
      window.alert(`删除失败：${reason}`);
      return;
    }
    const setter = Object.getOwnPropertyDescriptor(
      field instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype
        : HTMLInputElement.prototype,
      'value',
    )?.set;
    if (setter) setter.call(field, name);
    else field.value = name;
    const inputEvent = typeof InputEvent === 'function'
      ? new InputEvent('input', { bubbles: true, inputType: 'insertText', data: name })
      : new Event('input', { bubbles: true });
    field.dispatchEvent(inputEvent);
    field.dispatchEvent(new Event('change', { bubbles: true }));
    field.blur();
    setTimeout(() => {
      if ((field.value || '').trim() === '') {
        deleted.add(name);
        clearSelectionIfDeleted(name);
        closeDropdown();
      } else {
        window.alert(`删除失败：${reason}（隐藏控件回退也未生效）`);
      }
    }, 2500);
  };

  const decorate = () => {
    for (const listbox of savedVoiceListboxes()) {
      const rect = listbox.getBoundingClientRect();
      if (!rect.width || !rect.height) continue;
      listbox.classList.add('minimax-saved-voice-options');
      for (const option of listbox.querySelectorAll('[role="option"]')) {
        const name = optionName(option);
        if (!name) continue;
        if (deleted.has(name)) { option.remove(); continue; }
        if (option.querySelector('.minimax-saved-voice-delete-action')) continue;

        option.style.position = 'relative';
        option.classList.add('minimax-saved-voice-option');
        const action = document.createElement('button');
        action.type = 'button';
        action.className = 'minimax-saved-voice-delete-action';
        action.textContent = '\\u00d7';
        action.title = `删除音色 ${name}`;
        action.setAttribute('aria-label', `删除音色 ${name}`);
        action.addEventListener('mousedown', (event) => {
          event.preventDefault();
          event.stopPropagation();
        });
        action.addEventListener('click', (event) => {
          event.preventDefault();
          event.stopPropagation();
          const confirmed = window.confirm(
            `确定删除音色「${name}」吗？\\n\\n` +
            '会先调用 MiniMax 官方接口删除云端音色，再移除本机索引。\\n' +
            '云端删除不可逆：删除后该 voice_id 将无法再次使用。'
          );
          if (!confirmed) return;
          requestDelete(name).then((status) => {
            deleted.add(name);
            option.remove();
            clearSelectionIfDeleted(name);
            closeDropdown();
            // 云端没删成功时必须当场告知：本机索引已经移除，而 voice_id 是
            // 找回那条云端音色的唯一线索，静默过去等于让用户丢东西还不知道。
            if (status && status.indexOf('云端未删除') >= 0) {
              window.alert(status);
            }
          }).catch((err) => {
            fallbackDelete(name, (err && err.message) ? err.message : String(err));
          });
        });
        option.appendChild(action);
      }
    }
  };

  const observer = new MutationObserver(decorate);
  observer.observe(document.body, { childList: true, subtree: true });
  document.addEventListener('click', () => setTimeout(decorate, 0), true);
  document.addEventListener('focusin', () => setTimeout(decorate, 0), true);
  decorate();
}
"""



# 记录云端配音文本的最后一个光标/选区位置；下拉菜单获得焦点后仍可原位插入语气词。
MINIMAX_TEXT_SELECTION_JS = """
() => {
  // Gradio 6 的 Tabs 懒渲染：MiniMax Tab 未激活时内容不在 DOM，
  // 页面 load 时直接初始化会因找不到 textarea 而静默失败。
  // 因此轮询等待 #minimax-dubbing-text 出现后再注册，并保证只注册一次。
  const init = () => {
    const root = document.getElementById("minimax-dubbing-text");
    const input = root && root.querySelector("textarea");
    if (!input) return false;
    // 回车绑定不依赖函数注册标记：每次轮询都尝试（dataset 保证只绑一次），
    // 这样即使 Accordion 折叠、input 后渲染也能补绑。
    const currentPauseInput = document.querySelector("#minimax-pause-seconds input");
    if (currentPauseInput && !currentPauseInput.dataset.minimaxPauseEnterBound) {
      currentPauseInput.dataset.minimaxPauseEnterBound = "1";
      currentPauseInput.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" || event.isComposing) return;
        event.preventDefault();
        const [next] = window.__minimaxInsertPauseAtSelection ? window.__minimaxInsertPauseAtSelection(input.value || "", currentPauseInput.value, false) : [input.value || ""];
        if (next === (input.value || "")) return;
        const valueSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
        if (valueSetter) valueSetter.call(input, next);
        else input.value = next;
        input.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: "" }));
        input.focus();
        input.setSelectionRange(window.__minimaxTextSelection?.start ?? next.length, window.__minimaxTextSelection?.end ?? next.length);
      });
    }
    if (window.__minimaxTextSelectionBound) return true;
    window.__minimaxTextSelectionBound = true;
    const save = () => {
      window.__minimaxTextSelection = {
        start: input.selectionStart ?? input.value.length,
        end: input.selectionEnd ?? input.selectionStart ?? input.value.length,
      };
    };
    ["focus", "keyup", "mouseup", "select", "input"].forEach((eventName) => input.addEventListener(eventName, save));
    save();
    window.__minimaxInsertExpressionAtSelection = (value, token) => {
      if (!token) return [value || "", ""];
      const saved = window.__minimaxTextSelection || {};
      const text = value || "";
      const start = Math.max(0, Math.min(Number.isInteger(saved.start) ? saved.start : text.length, text.length));
      const end = Math.max(start, Math.min(Number.isInteger(saved.end) ? saved.end : start, text.length));
      const before = text.slice(0, end);
      const after = text.slice(end);
      const leftSpace = before && !/\\s$/.test(before) ? " " : "";
      const rightSpace = after && !/^\\s/.test(after) ? " " : "";
      const inserted = `${leftSpace}${token}${rightSpace}`;
      const next = text.slice(0, start) + before.slice(start) + inserted + after;
      window.__minimaxTextSelection = { start: start + before.slice(start).length + inserted.length, end: start + before.slice(start).length + inserted.length };
      return [next, ""];
    };
    const pauseInput = document.querySelector("#minimax-pause-seconds input");
    window.__minimaxInsertPauseAtSelection = (value, seconds, insertNewline = false) => {
      const duration = Number(seconds);
      if (!Number.isFinite(duration) || duration < 0.01 || duration > 99.99) {
        return [value || "", seconds];
      }
      const text = value || "";
      const saved = window.__minimaxTextSelection || {};
      const start = Math.max(0, Math.min(Number.isInteger(saved.start) ? saved.start : text.length, text.length));
      const end = Math.max(start, Math.min(Number.isInteger(saved.end) ? saved.end : start, text.length));
      const prefix = insertNewline && start > 0 && !text.slice(0, start).endsWith("\\n") ? "\\n" : "";
      const token = `<#${duration.toFixed(2)}#>`;
      const next = text.slice(0, start) + prefix + token + text.slice(end);
      const caret = start + prefix.length + token.length;
      window.__minimaxTextSelection = { start: caret, end: caret };
      return [next, Number(duration.toFixed(2))];
    };
    return true;
  };
  if (init()) return;
  const timer = setInterval(() => {
    if (init()) clearInterval(timer);
  }, 500);
  // 兜底：最多轮询 30 秒，避免隐藏 Tab 场景下无限轮询
  setTimeout(() => clearInterval(timer), 30000);
  // 额外兜底：监听 Accordion 展开（用户点击展开时 input 才会渲染），
  // 展开后立即补绑回车监听；也监听 Tab 切换，保证任何时机都能补上。
  const rebind = () => {
    const root = document.getElementById("minimax-dubbing-text");
    if (root && root.querySelector("textarea")) init();
  };
  document.addEventListener("click", (event) => {
    const accBtn = event.target.closest && event.target.closest("#minimax-expression-accordion > button");
    if (accBtn) setTimeout(rebind, 300);
  }, true);
  const tabs = document.querySelectorAll(".tab-nav button, [role=tab]");
  tabs.forEach((t) => t.addEventListener("click", () => setTimeout(rebind, 500)));
}
"""

# 两个云端子页各自拥有右侧结果面板。这里完全在浏览器本地切换，
# 不依赖排队中的 Python 请求，避免生成时切换子页导致播放器消失。
MINIMAX_RESULT_PANEL_JS = """
() => {
  const setPanel = (name) => {
    const dubbing = document.getElementById("minimax-dubbing-result-panel");
    const design = document.getElementById("minimax-design-result-panel");
    if (dubbing) dubbing.style.display = name === "dubbing" ? "block" : "none";
    if (design) design.style.display = name === "design" ? "block" : "none";
  };
  window.__minimaxSetResultPanel = setPanel;
  setPanel("dubbing");
  setTimeout(() => setPanel("dubbing"), 200);
}
"""

MINIMAX_UPLOAD_TEXT_JS = """
() => {
  // 把参考音频 / SRT 上传区改写成紧凑单行：
  // 隐藏大拖拽按钮，把小上传/录音按钮放到文字行最左边，文字 16px。
  // 组件因上传/清除而重渲染后会再次改写（轮询，幂等）；
  // 已上传（出现波形/文件列表）时自动恢复并跳过。
  const TARGETS = [
    ["#minimax-clone-audio", "将参考音频拖放至此处，或点击上传", "16.5px", 153],
    ["#minimax-srt-file", "将 SRT 字幕文件拖放至此处，或点击上传", "16px", 120],
    ["#local-srt-file", "将 SRT 字幕文件拖放至此处，或点击上传", "16px", 120],
  ];
  const restyle = () => {
    for (const [selector, lineText, lineSize, lineH] of TARGETS) {
      const root = document.querySelector(selector);
      if (!root) continue;
      // 组件隐藏（display:none 的 tab）时跳过：不重建、不校正，
      // 避免尺寸为 0 时把自建行 top 改成 0，导致切回标签时文字闪到顶端再回中心
      if (!root.getClientRects().length) continue;
      const customRow = root.querySelector("[data-custom-row='1']");
      // 大上传按钮按提示文字识别（上传后该按钮会消失或被文件列表替换）
      const bigBtn = Array.from(root.querySelectorAll("button"))
        .find(b => b.textContent.includes("拖放") || b.textContent.includes("拖到"));
      if (!bigBtn) {
        // 上传后（大按钮消失）：移除自建行，恢复正常显示。
        // 注意：不得在此处绑定"点击自动清除"——用户上传后点击文件区域
        // 是查看/确认，自动清除会把刚上传的文件删掉（此前事故根因）。
        // 换文件走 Gradio 原生交互（右上角 × 清除后重新上传）。
        if (customRow) customRow.remove();
        // 空态为了保留原生上传事件而写入的内联布局，会压过 Gradio
        // 上传后的隐藏/播放器状态；上传后必须原样还原。
        const restyled = root.querySelector("[data-js-restyled='1']");
        if (restyled) {
          restyled.style.cssText = restyled.dataset.origStyle || "";
          delete restyled.dataset.origStyle;
          delete restyled.dataset.jsRestyled;
        }
        root.dataset.uploadState = "uploaded";
        continue;
      }
      delete root.dataset.uploadState;
      // 仅当“自建行存在 且 大按钮已隐藏”才是完成态；其余情况（残留行、
      // 重渲染把行丢了等）一律先清理再重建，保证最终状态正确
      // 居中由 wrap 的 CSS（flex + 定高）保证，无需 JS 校正
      const bigHidden = getComputedStyle(bigBtn).display === "none";
      if (customRow && bigHidden) continue;
      if (customRow) customRow.remove();
      const sourceRow = root.querySelector(".source-selection");
      const wrap = root.querySelector(".wrap, .audio-container") || root;
      // wrap 不能隐藏：Gradio 的原生点击/拖拽监听都挂在 wrap 上，隐藏后上传全失效。
      // 自建行放进 wrap 内，事件冒泡路径保持完整；居中交给 CSS（wrap flex + 定高），
      // 不做 JS 测量（测量在 Electron 布局时序下不可靠，曾导致行被裁掉不可见）。
      // 高度用内联样式钉死，不依赖样式表是否加载
      // 上传态需无损恢复，所以只在首次改写时记录原始内联样式。
      if (!wrap.dataset.jsRestyled) {
        wrap.dataset.origStyle = wrap.getAttribute("style") || "";
        wrap.dataset.jsRestyled = "1";
      }
      wrap.style.cssText +=
        ";position:relative!important;display:flex!important;" +
        "align-items:center!important;justify-content:center!important;" +
        "height:" + lineH + "px!important;min-height:" + lineH + "px!important;" +
        "max-height:" + lineH + "px!important;overflow:hidden!important;" +
        // Gradio 空状态会把 wrap 淡出(opacity:0)且 pointer-events:none——
        // 必须强制可见可点，否则自建行整行透明、点击无效
        "opacity:1!important;pointer-events:auto!important;";
      const fileInput = root.querySelector('input[type="file"]');
      // 自建单行：小按钮(s) + 16px 文字，整组水平居中（wrap 已 flex 居中）
      const row = document.createElement("div");
      row.dataset.customRow = "1";
      row.style.cssText =
        "margin:0;display:flex;align-items:center;justify-content:center;gap:8px;";
      if (sourceRow) {
        // 音频组件：隐藏大按钮，复用原生小按钮行（保留在 wrap 内，原生监听有效）
        // 必须强制按钮行收缩（原生样式带 flex 撑满/space-between，
        // 否则会把提示文字推到最右、中间留一大片空白）
        bigBtn.style.display = "none";
        sourceRow.style.cssText +=
          ";margin:0!important;display:flex!important;gap:6px!important;" +
          "flex:0 0 auto!important;width:auto!important;justify-content:flex-start!important;" +
          "border-top:0!important;";
        // 原生按钮行图标被 Gradio 样式压到 16px，与 SRT 箭头不一致——统一为 20px（用户指定）
        sourceRow.querySelectorAll("svg").forEach(s => {
          s.style.width = "20px";
          s.style.height = "20px";
        });
        row.appendChild(sourceRow);
      } else {
        // 文件组件：隐藏大按钮，自建小上传按钮——
        // 外观与参考音频卡片的原生上传按钮 1:1 一致（22px 圆形无框 + 同款箭头图标）
        bigBtn.style.display = "none";
        const refIconBtn = document.querySelector("#minimax-clone-audio .source-selection button");
        const icon = (refIconBtn ? refIconBtn.querySelector("svg, img") : null) || bigBtn.querySelector("svg, img");
        const smallBtn = document.createElement("button");
        smallBtn.type = "button";
        smallBtn.style.cssText =
          "display:inline-flex;align-items:center;justify-content:center;" +
          "width:22px;height:22px;border-radius:20px;border:0;padding:0;" +
          "background:transparent;cursor:pointer;";
        if (icon) {
          // 与参考音频箭头统一 20px（克隆的 svg 带 width:90%，显式覆盖）
          const iconEl = icon.cloneNode(true);
          iconEl.style.width = "20px";
          iconEl.style.height = "20px";
          smallBtn.appendChild(iconEl);
        }
        // 阻止冒泡到 wrap 的原生监听（避免双弹），直接触发 input
        smallBtn.onclick = (ev) => {
          ev.stopPropagation();
          // 点击时重新查询：创建行时缓存的 fileInput/root 可能已失效
          // （Gradio 组件重渲染时序），缓存失效会导致点击完全无效
          const rootEl = document.querySelector(selector) || root;
          const fi = rootEl.querySelector('input[type="file"]');
          if (fi) fi.click();
        };
        row.appendChild(smallBtn);
      }
      const span = document.createElement("span");
      span.textContent = lineText;
      span.style.cssText = "white-space:nowrap;font-size:" + (lineSize || "16px") + ";";
      row.appendChild(span);
      wrap.appendChild(row);
      // 点击文字/空白区域也触发上传（按钮点击除外，按钮走原生监听）
      row.onclick = (ev) => {
        if (ev.target.closest("button")) return;
        ev.stopPropagation();  // 自建处理，避免与 wrap 原生监听双弹
        const rootEl = document.querySelector(selector) || root;
        const fi = rootEl.querySelector('input[type="file"]');
        if (fi) fi.click();
      };
    }
    // 本地/云端试听结果：只有实际出现可播放音频时才扩至 250px。
    // 空态仍保持 153px，避免影响右列的初始布局。
    ["#local-dubbing-audio", "#local-design-audio", "#minimax-dubbing-audio", "#minimax-design-audio"].forEach((selector) => {
      const localResult = document.querySelector(selector);
      if (!localResult || !localResult.getClientRects().length) return;
      const hasResultAudio = !!localResult.querySelector(".waveform-container") ||
        Array.from(localResult.querySelectorAll("audio"))
          .some(audio => !!(audio.getAttribute("src") || audio.currentSrc || audio.querySelector("source[src]")));
      if (hasResultAudio) localResult.dataset.resultState = "uploaded";
      else delete localResult.dataset.resultState;
    });
  };
  // 右列顶部与左列「MiniMax API Key」卡片最顶边缘动态对齐（窗口缩放自适应）
  const alignRightColumn = () => {
    // 只匹配可见的标签元素（隐藏子页中的同名文字 rect 为 0，必须排除）
    const keyLabel = Array.from(document.querySelectorAll("label, span, p"))
      .find(e => {
        if (!e.textContent.trim().startsWith("MiniMax API Key")) return false;
        if (e.querySelector("*")) return false;
        const r = e.getBoundingClientRect();
        return r.width > 0 && r.height > 0;
      });
    const col = document.getElementById("minimax-result-column");
    if (!keyLabel || !col) return;
    // 对齐基准：API Key 卡片块的最顶边缘（含边框/内边距）
    const card = keyLabel.closest(".block, .form") || keyLabel;
    const targetTop = card.getBoundingClientRect().top;
    // 绝对计算：目标 margin = 卡片顶 - 右列无 margin 时的视口顶
    const currentMargin = parseFloat(getComputedStyle(col).marginTop) || 0;
    const colBase = col.getBoundingClientRect().top - currentMargin;
    const target = targetTop - colBase;
    if (Math.abs(target - currentMargin) > 0.5) {
      col.style.marginTop = target + "px";
    }
  };
  // Gradio 6 的 Accordion 没有稳定暴露展开属性；为情绪标签维护独立状态，
  // 让展开态的 CSS 能可靠增加底部留白。
  const bindLocalDesignTagsAccordion = () => {
    const root = document.querySelector("#local-design-tags");
    const header = root && root.querySelector(":scope > button");
    if (!root || !header || header.dataset.emotionStateBound) return;
    root.dataset.emotionExpanded = "false";
    header.dataset.emotionStateBound = "1";
    header.addEventListener("click", () => {
      window.setTimeout(() => {
        const current = document.querySelector("#local-design-tags");
        if (!current) return;
        current.dataset.emotionExpanded =
          current.dataset.emotionExpanded === "true" ? "false" : "true";
      }, 0);
    });
  };
  setInterval(() => { restyle(); alignRightColumn(); bindLocalDesignTagsAccordion(); }, 800);
  restyle();
  alignRightColumn();
  bindLocalDesignTagsAccordion();
}
"""

SEGMENT_GAP_SEC = 0.15

# 段间响度对齐的参数。数值依据见 09_docs 的实测记录：
# 实测本地长稿段间极差 5.5 dB（云端同文稿 1.7 dB），个别段峰值已达 0.997。
LOUDNESS_MATCH_MAX_GAIN_DB = 6.0    # 单段增益上限，防止极端段被放大成噪音墙
LOUDNESS_MATCH_PEAK_CEILING = 0.97  # 归一化后的峰值上限，留出削顶余量
LOUDNESS_MATCH_FLOOR_RATIO = 0.02   # 低于峰值这个比例的样本视为静音，不计入 RMS


def _segment_speech_rms(seg: "np.ndarray") -> Optional[float]:
    """一段音频里「有声部分」的 RMS。

    不能直接对整段求 RMS：各段的静音占比不同，静音会把均值稀释，
    静音多的段会被误判成「音量小」而被错误地放大。
    """
    if seg.size == 0:
        return None
    amp = np.abs(seg)
    peak = float(amp.max())
    if peak <= 0:
        return None
    voiced = seg[amp > max(peak * LOUDNESS_MATCH_FLOOR_RATIO, 1e-5)]
    if voiced.size < 100:
        return None
    return float(np.sqrt(np.mean(np.square(voiced.astype(np.float64)))))


def match_segment_loudness(segments: List["np.ndarray"]) -> List["np.ndarray"]:
    """把各段的响度对齐到中位数，返回新的段列表。

    取中位数而不是均值：均值会被个别过响的段拉高，结果是把其余段一起抬起来，
    整体更吵，段间差距却没解决。中位数只把离群段拽回大多数段所在的水平。

    只调整整段增益，不做压缩或限幅曲线，段内动态一概不变。
    """
    if len(segments) < 2:
        return segments

    levels = [_segment_speech_rms(seg) for seg in segments]
    usable = [lv for lv in levels if lv and lv > 0]
    if len(usable) < 2:
        return segments

    target = float(np.median(usable))
    max_gain = 10.0 ** (LOUDNESS_MATCH_MAX_GAIN_DB / 20.0)
    min_gain = 1.0 / max_gain

    adjusted: List["np.ndarray"] = []
    for seg, level in zip(segments, levels):
        if not level or level <= 0:
            adjusted.append(seg)
            continue
        gain = float(np.clip(target / level, min_gain, max_gain))
        peak = float(np.abs(seg).max())
        if peak * gain > LOUDNESS_MATCH_PEAK_CEILING:
            # 段 5、段 8 原本峰值就有 0.994，放大一点点就会削顶失真
            gain = LOUDNESS_MATCH_PEAK_CEILING / max(peak, 1e-9)
        if abs(gain - 1.0) < 1e-3:
            adjusted.append(seg)
            continue
        adjusted.append((seg.astype(np.float32) * gain).astype(np.float32))
    return adjusted

# 单条字幕的最大字符数：Whisper 识别出的一句话可能很长，超过此长度会被自动
# 拆分成多条更短的字幕（时间轴按字符数比例分配），避免一行显示不下。
MAX_SUBTITLE_CHARS = 20

# 生成历史每页展示的条数
HISTORY_PAGE_SIZE = 8

# 「声音设计」常用情绪 / 风格标签（点击填入音色描述框）
EMOTION_TAGS = [
    "开心", "愤怒", "悲伤", "惊讶", "恐惧", "温柔",
    "兴奋", "平静", "撒娇", "严肃", "紧张", "冷漠",
    "无奈", "深情", "语速快", "语速慢", "高声喊叫", "轻声细语",
]

# ---------- 文档（全部放入「最佳实践」Tab） ----------

_SPEED_NOTICE = """## 关于语速调节

**建议以 1.0 原速生成。** 原速输出的音色最准确，需要加速或减速时，
在剪辑软件里处理效果更好，也更可控。

本版的变速由 WSOLA 时间伸缩算法实现（不含 Rubber Band）。已知表现：

- **加速**（如 1.25×）：正常。
- **减速到 0.8× 左右**：可能出现轻微电流音。

这是时间伸缩算法在拉长音频时的固有特性，**不是文件损坏或安装错误**，
也不影响原速生成的音质。如果你对变速音质要求较高，请按上面的建议操作。
"""

_APIKEY_NOTICE = """## 关于你的 API 密钥

云端配音需要你自己的 MiniMax API Key。本整合包这样处理它：

- **整合包内不含任何密钥**，密钥不随软件分发。
- 只有你**明确点击「安全保存」**之后才会写入磁盘，不会偷偷保存。
- 落盘内容是 **Windows DPAPI 加密后的密文**，不是明文——
  只有你当前登录的这个 Windows 账户能解开，换个账户或换台电脑都读不了。
- 保存位置是 `%LOCALAPPDATA%\\YZYLauncher\\VoxCPM2\\`，
  **不在整合包目录里**。你复制、移动、打包或分享这个软件文件夹时，
  不会把密钥一起带走。

如果你要把整合包转给别人，密钥不会跟着走；反过来，重装或换目录后
需要重新填一次密钥，这是正常的。
"""


_USAGE_INSTRUCTIONS = (
    "## 使用说明\n\n"
    "🎙️ **配音**  \n"
    "上传参考音频后，系统会自动识别参考音频的文字内容，"
    "并完整还原参考音频中的所有声音细节进行配音。\n\n"
    "**使用步骤：**\n\n"
    "1. 在「参考音频」处上传或录制一段音频（必填）。\n"
    "2. 系统会自动识别参考音频文本，如识别有误可在「参考音频文本」中手动修正。\n"
    "3. 在「目标文本」中输入想要生成的内容，点击「提交到任务队列」即可。\n\n"
    "🎨 **声音设计**  \n"
    "无需参考音频，通过文字描述目标音色特征（性别、年龄、语气、情绪、语速等），"
    "从零创造出独一无二的声音。试听满意后，下载音频，到「配音」Tab 作为参考音频使用。\n\n"
    "---\n\n"
    "## 📏 长稿怎么做（重要）\n\n"
    "**先看该走哪条路：**\n\n"
    "| | 适合 | 说明 |\n"
    "| --- | --- | --- |\n"
    "| **本地配音** | 短稿、试音、批量小段、离线使用 | 免费、不限次数、音频不出本机 |\n"
    "| **MiniMax 云端 TTS** | 长稿、成品口播 | 长文本一致性更有保障，但按量计费 |\n\n"
    "**为什么长稿建议分段做：**\n\n"
    "本地模型生成长文本时会自动切片，每段独立生成后拼接。单段偶尔会出现音色或口音"
    "跑偏，概率不高，但**段数越多，整条里至少有一段出问题的概率就越高**——\n\n"
    "- 30 秒稿约 3～6 段，基本无感\n"
    "- 3 分钟稿约 8～13 段，风险明显上升\n\n"
    "所以稿子越长，越建议**按 1 分钟左右分段制作，再到剪辑软件里拼**。"
    "这样还有个好处：某一段不满意，只需重做那一段，不用整条重来。\n\n"
    "**「音色稳定性档位」怎么选：**\n\n"
    "这个档位决定每段切多少字。字数越少，单段越稳；但段数变多，拼接点也变多。"
    "两者要权衡：\n\n"
    "- **70 字**：最稳，适合对音色一致性要求最高的场合\n"
    "- **100 / 120 字**：适合连贯长句，段落更自然\n"
    "- **150 字**：段数最少、拼接点最少，建议在显存充足（12G 以上）时使用\n\n"
    "如果长稿仍有跑偏，优先调高 **CFG（引导强度）** 到 2.2～2.6——它直接控制"
    "「多贴近参考音色」，是对抗跑偏最有效的一个参数；其次把「生成迭代步数」提到 25 以上。"
    "两者都不额外占用显存。\n"
)

_EXAMPLES_FOOTER = (
    "## 🗣️ 方言 / 多语言生成指南\n\n"
    "生成方言或其他语言的语音时，请特别注意以下两点：\n\n"
    "1. **需要传入对应方言的参考音频。** 例如要生成粤语，就用一段粤语的参考音频；"
    "要生成四川话，就用四川话的参考音频。\n"
    "2. **不同语言要传入对应语言的参考音频。** 例如要生成英语，就用英语的参考音频；"
    "要生成日语，就用日语的参考音频。参考音频的语言应与目标文本的语言保持一致。\n\n"
    "同时，请在「目标文本」中直接使用对应方言 / 语言的词汇和句式。\n\n"
    "**示例 — 广东话**  \n"
    "✅ 正确（粤语表达）：伙計，唔該一個A餐，凍奶茶少甜！  \n"
    "❌ 错误（普通话原文）：伙计，麻烦来一个A餐，冻奶茶少甜！\n\n"
    "**示例 — 河南话**  \n"
    "✅ 正确（河南话表达）：恁这是弄啥嘞？晌午吃啥饭？  \n"
    "❌ 错误（普通话原文）：你这是在干什么呢？中午吃什么饭？\n\n"
    "🤖 **小技巧：** 不知道方言怎么写？可以用豆包、DeepSeek、Kimi 等 AI 助手"
    "将普通话翻译为方言文本，再粘贴到「目标文本」中即可。\n"
)


# ---------- 通用工具函数 ----------

def _now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _now_time() -> str:
    """只返回时间（不含日期），用于任务队列展示。"""
    return datetime.datetime.now().strftime("%H:%M:%S")


def save_wav(sr: int, wav: Any, prefix: str = "gen") -> str:
    """将音频数组写入 outputs 目录，返回保存路径。"""
    OUTPUTPUTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = OUTPUTPUTS / f"{prefix}_{ts}.wav"
    arr = np.asarray(wav, dtype=np.float32).squeeze()
    try:
        import soundfile as sf
        sf.write(str(path), arr, int(sr))
    except Exception:
        # soundfile 不可用时退回标准库 wave，按 16bit PCM 写出
        pcm = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(pcm.tobytes())
    return str(path)


def _encode_timeline_audio(
        sr: int,
        wav: Any,
        output_format: str,
        bitrate: int = 128000,
        channel: int = 1,
) -> bytes:
    """把时间轴音频编码到内存，避免失败时留下半成品文件。"""
    fmt = (output_format or "wav").strip().lower()
    if fmt not in {"mp3", "wav", "flac"}:
        raise ValueError("时间轴输出格式只支持 mp3、wav 或 flac。")
    if channel not in {1, 2}:
        raise ValueError("时间轴输出声道只能是单声道或双声道。")

    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 1:
        if channel == 2:
            arr = np.column_stack((arr, arr))
    elif arr.ndim == 2:
        if arr.shape[1] not in {1, 2}:
            raise ValueError("时间轴音频包含不支持的声道数。")
        if channel == 1 and arr.shape[1] == 2:
            arr = arr.mean(axis=1)
        elif channel == 2 and arr.shape[1] == 1:
            arr = np.repeat(arr, 2, axis=1)
    else:
        raise ValueError("时间轴音频数组维度无效。")
    if arr.size == 0:
        raise ValueError("时间轴音频为空。")
    arr = np.ascontiguousarray(np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0))

    if fmt == "mp3":
        if bitrate not in {32000, 64000, 128000, 256000}:
            raise ValueError("MP3 码率无效。")
        ffmpeg_exe = Path(current_dir) / "ffmpeg" / "bin" / "ffmpeg.exe"
        if not ffmpeg_exe.is_file():
            raise RuntimeError("未找到整合包内置 FFmpeg，无法编码 MP3 时间轴音频。")
        channels = 1 if arr.ndim == 1 else arr.shape[1]
        completed = subprocess.run(
            [
                str(ffmpeg_exe), "-hide_banner", "-loglevel", "error",
                "-f", "f32le", "-ar", str(int(sr)), "-ac", str(channels),
                "-i", "pipe:0", "-vn", "-codec:a", "libmp3lame",
                "-b:a", f"{int(bitrate) // 1000}k", "-f", "mp3", "pipe:1",
            ],
            input=arr.astype("<f4", copy=False).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            # 没有超时的话，ffmpeg 一旦卡住这个 Gradio 事件就永不返回。
            timeout=300,
        )
        if completed.returncode != 0 or not completed.stdout:
            raise RuntimeError("FFmpeg 未能编码 MP3 时间轴音频。")
        return completed.stdout

    import soundfile as sf
    buffer = io.BytesIO()
    sf.write(
        buffer,
        arr,
        int(sr),
        format=fmt.upper(),
        subtype="FLOAT" if fmt == "wav" else "PCM_16",
    )
    return buffer.getvalue()


def save_timeline_audio(
        sr: int,
        wav: Any,
        output_format: str,
        prefix: str = "timeline",
        bitrate: int = 128000,
        channel: int = 1,
) -> str:
    """按用户选择的格式保存时间轴音频，并返回最终路径。"""
    fmt = (output_format or "wav").strip().lower()
    payload = _encode_timeline_audio(sr, wav, fmt, bitrate, channel)
    OUTPUTPUTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = OUTPUTPUTS / f"{prefix}_{ts}.{fmt}"
    path.write_bytes(payload)
    return str(path)


def apply_speed(wav: Any, sr: int, speed: float) -> np.ndarray:
    """
    对音频做变速不变调处理（时间伸缩），speed>1 加快、speed<1 减慢，接近 1 时跳过。

    说明：语音对相位声码器（phase vocoder）很敏感，独立推进各频率 bin 的相位会导致
    谐波相位漂移（phasiness，金属感/水下感）并涂抹瞬态（咬字发糊）。因此这里按质量
    优先选择更适合语音的算法，逐级回退：
      1) pyrubberband —— 封装 Rubber Band，质量最佳、可保形峰，需系统安装 rubberband-cli
                         （apt install rubberband-cli）+ pip install pyrubberband
      2) audiotsm 的 WSOLA —— 时域算法，对语音瞬态/音色保持好，纯 pip 安装（pip install audiotsm）
      3) librosa 相位声码器 —— 兜底，可能发飘
      4) 全部不可用时返回原音频
    """
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    speed = float(speed or 1.0)
    if abs(speed - 1.0) < 1e-2 or arr.size == 0:
        return arr

    # 方案 1：pyrubberband（质量最佳，需 rubberband-cli 系统二进制）
    try:
        import pyrubberband as pyrb
        out = pyrb.time_stretch(arr, sr, speed)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        if out.size > 0:
            return out
    except Exception as e:
        logger.debug(f"pyrubberband 不可用，尝试 WSOLA：{e}")

    # 方案 2：audiotsm WSOLA（对语音更自然，纯 pip 依赖，推荐主力）
    try:
        from audiotsm import wsola
        from audiotsm.io.array import ArrayReader, ArrayWriter
        reader = ArrayReader(arr.reshape(1, -1))
        writer = ArrayWriter(channels=1)
        wsola(channels=1, speed=speed).run(reader, writer)
        out = np.asarray(writer.data, dtype=np.float32).reshape(-1)
        if out.size > 0:
            return out
    except Exception as e:
        logger.debug(f"audiotsm 不可用，回退 librosa：{e}")

    # 兜底：两级方案都没能产出有效音频时，返回未变速的原音频。
    # 这里必须显式 return——原来 WSOLA 正常执行但产出空数组时会从函数末尾掉出去
    # 隐式返回 None，下游 np.asarray(None) 抛 TypeError 让整条任务失败。
    return arr


# 兼容常见的非标准写法：省略小时位（MM:SS,mmm）、省略毫秒（HH:MM:SS）、
# 以及用点号作小数分隔（WebVTT 风格 HH:MM:SS.mmm）。
_SRT_TIME_PATTERN = re.compile(
    r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})(?:[,.](\d{1,3}))?\s*$"
)


def _srt_time_to_seconds(ts: str) -> float:
    """把 SRT 时间戳转换为秒，兼容常见的非标准写法。

    原实现用裸 split 解包，遇到「00:00:01 --> 00:00:04」这类省略毫秒的写法
    会直接抛 ValueError，用户看到的是英文异常而不是「字幕格式错误」。
    """
    raw = (ts or "").strip()
    matched = _SRT_TIME_PATTERN.match(raw)
    if not matched:
        raise RuntimeError(
            f"字幕时间码格式无法识别：「{raw}」。"
            "支持的写法为 HH:MM:SS,mmm / MM:SS,mmm / HH:MM:SS（毫秒可省略）。"
        )
    hours, minutes, seconds, millis = matched.groups()
    millis = (millis or "0").ljust(3, "0")[:3]
    return (
        int(hours or 0) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis) / 1000.0
    )


# 按可靠性排序：严格 UTF-8 能正确拒绝 GBK，必须排第一；gb18030 覆盖
# 简体中文的 ANSI 存法（剪映、Arctime、记事本另存都会产出这种）；
# cp1252 几乎不拒绝任何字节序列，只能垫底。
_SRT_ENCODINGS = ("utf-8-sig", "gb18030", "big5", "shift_jis", "cp1252")


def _read_srt_text(srt_path: str) -> str:
    """读取字幕文本并识别编码。

    原实现固定按 UTF-8 读且 errors="ignore"：GBK/ANSI 字幕不会报错，而是被
    静默替换成乱码后一路送去合成——本地产出读乱码的废音频，云端还会真金白银
    地计费。这里改为逐个编码严格尝试，全部失败就明确报错，绝不静默降级。
    """
    raw = Path(srt_path).read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for encoding in _SRT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(
        "无法识别该字幕文件的编码，已尝试 UTF-8 / GB18030 / Big5 / Shift-JIS / CP1252。"
        "请用记事本或字幕工具将其另存为 UTF-8 后重试。"
    )


def parse_srt(srt_path: str) -> List[Dict[str, Any]]:
    """解析 SRT 字幕文件，返回 [{start, end, text}, ...]。"""
    content = _read_srt_text(srt_path)
    blocks = re.split(r"\n\s*\n", content.strip())
    entries: List[Dict[str, Any]] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # 找到 "00:00:01,000 --> 00:00:04,000" 这一行
        time_line_idx = next(
            (i for i, ln in enumerate(lines) if "-->" in ln), None
        )
        if time_line_idx is None:
            continue
        start_str, end_str = lines[time_line_idx].split("-->")
        text = " ".join(lines[time_line_idx + 1:]).strip()
        if not text:
            continue
        try:
            start_sec = _srt_time_to_seconds(start_str)
            end_sec = _srt_time_to_seconds(end_str)
        except RuntimeError as error:
            # 把出错的字幕序号带出来，否则用户面对几百条字幕无从下手。
            raise RuntimeError(f"第 {len(entries) + 1} 条字幕解析失败：{error}") from None
        entries.append({
            "start": start_sec,
            "end": end_sec,
            "text": text,
        })
    return entries


# ===== P1-4：时间轴分配的内存上限 =====
# 时间轴按字幕里最大的结束时间预分配，时间码一旦手误（例如把 00:09:59 写成
# 09:59:59）就会一次性申请几个 GB。这里在分配前先算清楚要多少字节。
MAX_TIMELINE_BYTES = 1024 * 1024 * 1024   # 1 GB


def _check_timeline_budget(total_sec: float, sr: int, channels: int = 1) -> None:
    """在按时间轴预分配之前校验内存开销，超限就报清楚而不是让进程假死。"""
    needed = int(max(total_sec, 0.0) * sr) * max(channels, 1) * 4   # float32
    if needed > MAX_TIMELINE_BYTES:
        raise RuntimeError(
            f"字幕总时长 {total_sec / 60:.1f} 分钟，铺排时间轴需要约 "
            f"{needed / 1024 / 1024 / 1024:.1f} GB 内存，已超过 "
            f"{MAX_TIMELINE_BYTES / 1024 / 1024 / 1024:.0f} GB 上限。"
            "请检查是否有字幕的时间码写错（常见是小时位多打了一位）。"
        )


def _join_srt_texts(texts: Any) -> str:
    """用换行保留字幕块边界，避免英文单词或中文句子直接粘连。"""
    return "\n".join(str(text or "").strip() for text in texts if str(text or "").strip())


def _seconds_to_srt_time(sec: float) -> str:
    """秒转 SRT 时间戳 'HH:MM:SS,mmm'。"""
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# 优先在这些字符处断句（中英文标点 + 空格），让拆分点更自然
_SUBTITLE_BREAK_CHARS = "，。！？；：、,.!?;:… 　"


def _is_word_char(ch: str) -> bool:
    """是否为英文/数字字符（用于避免从英文单词中间切断）。"""
    return ch.isascii() and ch.isalnum()


def _split_text_by_length(text: str, max_chars: int) -> List[str]:
    """
    把一段文本按 max_chars 切成多块，规则：
    - 优先在标点处断开；其次在空格处断开；
    - 不切断英文单词（必要时把切点向后延伸到单词结束）；
    - 中文可逐字断开。
    返回非空文本块列表。
    """
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return [text] if text else []

    chunks: List[str] = []
    while len(text) > max_chars:
        window = text[:max_chars]
        cut = -1

        # 1) 在窗口内找最靠后的标点（标点归到当前块）
        for i in range(len(window) - 1, -1, -1):
            if window[i] in _SUBTITLE_BREAK_CHARS:
                cut = i + 1
                break

        # 2) 没有标点则尝试在空格处断（避免切断英文单词）
        if cut <= 0:
            sp = window.rfind(" ")
            if sp > 0:
                cut = sp + 1

        # 3) 仍无合适断点：默认在 max_chars 处切；若正处于英文单词中间，
        #    则把切点向后延伸到该单词结束，避免把单词拆散。
        if cut <= 0:
            cut = max_chars
            if cut < len(text) and _is_word_char(text[cut - 1]) and _is_word_char(text[cut]):
                j = cut
                while j < len(text) and _is_word_char(text[j]):
                    j += 1
                cut = j

        chunk = text[:cut].strip()
        if chunk:
            chunks.append(chunk)
        text = text[cut:].strip()

    if text:
        chunks.append(text)
    return chunks


def split_long_segments(
        segments: List[Dict[str, Any]],
        max_chars: int = MAX_SUBTITLE_CHARS,
) -> List[Dict[str, Any]]:
    """
    把过长的字幕条拆成多条更短的字幕，时间轴按字符数比例分配，
    使每条字幕的文本长度尽量不超过 max_chars，避免一行显示不下。
    """
    result: List[Dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        chunks = _split_text_by_length(text, max_chars)
        if len(chunks) <= 1:
            result.append({"start": start, "end": end, "text": text})
            continue

        total_len = sum(len(c) for c in chunks) or 1
        duration = max(0.0, end - start)
        cur = start
        for idx, chunk in enumerate(chunks):
            if idx == len(chunks) - 1:
                seg_end = end  # 最后一块对齐原始结束时间，避免累计误差
            else:
                seg_end = cur + duration * (len(chunk) / total_len)
            result.append({"start": cur, "end": seg_end, "text": chunk})
            cur = seg_end
    return result


def write_srt(segments: List[Dict[str, Any]], out_path: str) -> str:
    """把 [{start, end, text}] 列表写成 SRT 文件。"""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_seconds_to_srt_time(seg['start'])} --> {_seconds_to_srt_time(seg['end'])}")
        lines.append(str(seg["text"]).strip())
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def list_preset_voices() -> List[str]:
    """读取 voices 目录下的参考音频，返回不含扩展名的名称列表。"""
    VOICES.mkdir(parents=True, exist_ok=True)
    names = []
    for p in sorted(VOICES.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            names.append(p.stem)
    return names


def _load_personal_voice_names() -> set[str]:
    """读取用户保存音色的轻量索引；不把内置音色标记为可删除。"""
    names: set[str] = set()
    try:
        if PERSONAL_VOICE_META.exists():
            payload = json.loads(PERSONAL_VOICE_META.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                values = payload.get("names", [])
                if isinstance(values, list):
                    names = {str(value).strip() for value in values if str(value).strip()}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        logger.warning("无法读取个人音色索引，将按安全默认值处理。")
    # 该音色已存在于当前用户目录中，作为历史上由用户保存的音色保留删除能力。
    if any(name.casefold() == "juya" for name in list_preset_voices()):
        names.add(next(name for name in list_preset_voices() if name.casefold() == "juya"))
    return names


def _save_personal_voice_names(names: set[str]) -> None:
    """以可读 JSON 原子地保存个人音色索引，不触碰任何音频文件。

    原来是直接覆盖写：写到一半进程被杀（关窗口、蓝屏、杀软拦截）会留下半截
    JSON，下次启动解析失败被吞成空集合，此后界面上再也删不掉任何个人音色。
    同一整合包里 minimax_tts.py 对音色库用的就是「临时文件 + os.replace」，
    这里对齐。
    """
    VOICES.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"names": sorted(names, key=str.casefold)}, ensure_ascii=False, indent=2)
    temp_path = PERSONAL_VOICE_META.with_name(PERSONAL_VOICE_META.name + ".tmp")
    temp_path.write_text(payload, encoding="utf-8")
    os.replace(temp_path, PERSONAL_VOICE_META)


def is_personal_voice_name(name: str) -> bool:
    normalized = (name or "").strip().casefold()
    return bool(normalized) and any(item.casefold() == normalized for item in _load_personal_voice_names())


# 个人音色的隐形标记：U+2060 WORD JOINER。零宽、不可见、不影响换行，
# 前端据此判断哪些选项该显示行内“×”。原来是在标签末尾拼可见的「删除」两个字，
# 既跟行内“×”重复，又把功能标记暴露成了界面文案。
PERSONAL_VOICE_MARKER = "\u2060"


def preset_voice_choices() -> List[Any]:
    """返回下拉框选择项；个人音色带一个不可见标记，供前端渲染行内删除按钮。"""
    choices: List[Any] = []
    for name in list_preset_voices():
        label = f"{name}{PERSONAL_VOICE_MARKER}" if is_personal_voice_name(name) else name
        choices.append((label, name))
    return choices


def prune_voice_design_previews(keep: int = VOICE_DESIGN_PREVIEW_KEEP) -> None:
    """只保留最近 keep 个本地声音设计试听文件，其余删除。

    刻意写得很保守：只在 outputs 目录一层内、只匹配
    voice_design_preview_*.wav 这一种命名，绝不递归、绝不触碰其它前缀
    （gen_/multi_/srt_ 是「本地配音历史」的数据，minimax_* 是云端产物）。
    任何异常都吞掉——清理失败不该影响用户正在做的事。
    """
    try:
        candidates = [
            item for item in OUTPUTPUTS.glob("voice_design_preview_*.wav")
            if item.is_file()
        ]
    except OSError:
        return
    if len(candidates) <= keep:
        return
    try:
        candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    except OSError:
        return
    for stale in candidates[keep:]:
        try:
            stale.unlink()
        except OSError:
            logger.debug("试听文件清理失败（忽略）：%s", stale.name)


def scan_output_history() -> List[Dict[str, Any]]:
    """
    扫描 outputs 目录下已有的本地配音音频，构建初始历史记录。
    用于首次启动时把 gen_、multi_、srt_ 前缀的音频加载进「生成历史」。
    返回列表按生成时间升序排列（最旧在前），渲染时会再倒序展示。
    """
    try:
        OUTPUTPUTS.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        # 这个函数在 TaskManager 初始化时被调用，抛出去会让界面直接起不来。
        logger.error("无法创建 outputs 目录（%s）：%s", OUTPUTPUTS, error)
        return []
    items: List[Dict[str, Any]] = []
    local_dubbing_prefixes = ("gen_", "multi_", "srt_")
    for p in OUTPUTPUTS.iterdir():
        if (
            p.is_file()
            and p.suffix.lower() in AUDIO_EXTS
            and p.name.startswith(local_dubbing_prefixes)
        ):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            items.append({
                "time": datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "audio": str(p),
                "_mtime": mtime,
            })
    items.sort(key=lambda x: x["_mtime"])
    for it in items:
        it.pop("_mtime", None)
    return items


def resolve_preset_path(name: str) -> Optional[str]:
    """根据名称在 voices 目录中查找对应音频文件的完整路径。"""
    if not name or name == PRESET_PLACEHOLDER:
        return None
    for ext in AUDIO_EXTS:
        candidate = VOICES / f"{name}{ext}"
        if candidate.exists():
            return str(candidate)
    return None


# Windows 保留设备名：用它们做文件名时 shutil.copy 会「成功」但数据被丢弃，
# 于是界面提示「已保存」，下拉里却始终没有这个音色。
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def is_valid_voice_name(name: str) -> bool:
    """校验音色名称是否合法：非空、不含路径分隔符与非法文件名字符。"""
    name = (name or "").strip()
    if not name:
        return False
    if any(c in name for c in r'\/:*?"<>|'):
        return False
    # 保留名的判定要去掉扩展名，CON.wav 同样会被 Windows 当成设备。
    if name.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        return False
    # 结尾的点和空格会被 Windows 静默截断，导致存下来的文件名与界面显示不一致。
    if name != name.rstrip(" ."):
        return False
    return True


# 参考音频建议时长上限（秒）：VoxCPM2 官方推荐参考音频 10-20 秒，
# 超长音频整段作为条件序列会超出模型训练分布，导致音色/口音漂移。
REF_MAX_SECONDS = 20.0

# 原路径+mtime -> 裁剪临时文件路径；同一参考音频只裁剪一次
_crop_cache: Dict[str, str] = {}


def _cleanup_crop_cache() -> None:
    """退出时清掉本次会话裁剪出来的临时 wav。

    这些文件由 tempfile.mkstemp 创建，原来永不删除也无退出钩子，
    每上传一段超长参考音频就在 %TEMP% 留一个孤儿文件，长期使用会堆积成百上千个。
    """
    for temp_path in list(_crop_cache.values()):
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    _crop_cache.clear()


atexit.register(_cleanup_crop_cache)


def _reference_audio_problem(path: Optional[str]) -> Optional[str]:
    """返回参考音频不能进入 ASR/任务队列的原因；成功时返回 None。"""
    if not path:
        return "未收到参考音频文件。"
    try:
        size = os.path.getsize(path)
    except OSError:
        return "上传文件不存在或已无法访问。"
    if size == 0:
        return "上传文件为 0 字节，请重新选择音频文件。"
    try:
        import librosa
        duration = float(librosa.get_duration(path=path))
    except Exception:
        return "上传文件无法解码，请重新选择有效的音频文件。"
    if not np.isfinite(duration) or duration <= 0:
        return "上传文件没有可用的音频时长，请重新选择音频文件。"
    return None


def _crop_reference_audio(path: Optional[str]) -> Optional[str]:
    """参考音频超过 REF_MAX_SECONDS 时，跳过开头静音截取前 20 秒语音，
    返回裁剪后的临时 wav 路径（不修改原文件）；短音频原样返回。

    与模型 _encode_wav 相同采样率（16k），保证裁剪边界与编码一致。
    """
    if not path or not os.path.isfile(path):
        return path
    try:
        key = f"{path}|{os.path.getmtime(path)}"
    except OSError:
        return path
    cached = _crop_cache.get(key)
    if cached and os.path.isfile(cached):
        return cached
    try:
        import librosa
        audio, sr = librosa.load(path, sr=16000, mono=True)
    except Exception:
        return path
    if len(audio) / sr <= REF_MAX_SECONDS:
        return path

    # 跳过开头静音：找前 2 秒内能量超过阈值的第一个语音帧，向前留 0.3s 缓冲
    frame_len = int(sr * 0.025)
    usable = len(audio) // frame_len * frame_len
    rms = np.sqrt((audio[:usable].reshape(-1, frame_len) ** 2).mean(axis=1))
    thr = max(0.005, float(rms.max()) * 0.05)
    speech = np.where(rms > thr)[0]
    if len(speech):
        start = max(0, int((int(speech[0]) - 12) * frame_len))
    else:
        start = 0
    end = min(start + int(REF_MAX_SECONDS * sr), len(audio))
    crop = audio[start:end]

    try:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="voxcpm_ref_")
        os.close(fd)
        import soundfile as sf
        sf.write(tmp_path, crop, sr)
    except Exception:
        return path
    _crop_cache[key] = tmp_path
    logger.info(
        "参考音频 %.1fs 超过 %ds 上限，已自动截取前 %.1fs 语音用于克隆（原文件未修改）。",
        len(audio) / sr, REF_MAX_SECONDS, len(crop) / sr,
    )
    return tmp_path


def _split_keep_delims(text: str, delims: str) -> List[str]:
    """按 delims 中任意字符切分文本，并把分隔符保留在前一段末尾。"""
    out: List[str] = []
    buf = ""
    for ch in text:
        buf += ch
        if ch in delims:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def split_text_for_tts(text: str, max_chars: int = MAX_TTS_CHARS) -> List[str]:
    """
    将长文本切分为多段，便于逐段生成后再合并，避免一次性合成过长文本导致异常。
    切分原则（尽量从合理位置断开，减少读音异常）：
      1) 优先按强句末标点（。！？!?…；; 及换行）断句；
      2) 单句仍超长时，用次级标点（，,、：: 空格）继续细分；
      3) 实在没有标点时才按长度硬切；
      4) 最后贪心合并相邻短句，使每段尽量接近 max_chars。
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # 第一级：按强句末标点 / 换行断句（保留标点）
    pieces = _split_keep_delims(text, "。！？!?…；;\n")

    # 对仍然超长的片段，用次级标点继续细分；再不行则按长度硬切
    refined: List[str] = []
    for piece in pieces:
        if len(piece) <= max_chars:
            refined.append(piece)
            continue
        for sub in _split_keep_delims(piece, "，,、：:　 "):
            if len(sub) <= max_chars:
                refined.append(sub)
            else:
                for i in range(0, len(sub), max_chars):
                    refined.append(sub[i:i + max_chars])

    # 贪心合并：把相邻短片段聚合到接近 max_chars
    chunks: List[str] = []
    cur = ""
    for seg in refined:
        if not seg.strip():
            cur += seg  # 纯空白（如换行残留）并入当前块，不单独成段
            continue
        if cur and len(cur) + len(seg) > max_chars:
            if cur.strip():
                chunks.append(cur.strip())
            cur = seg
        else:
            cur += seg
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ---------- 模型 ----------

class VoxCPMDemo:
    def __init__(self, model_id: str = "openbmb/VoxCPM2", device: str = "auto") -> None:
        self.device = resolve_runtime_device(device, "cuda")
        logger.info(f"Running VoxCPM on device: {self.device}")
        self.optimize = self.device.startswith("cuda")

        self.asr_model_id = "./models/SenseVoiceSmall"
        self.asr_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
        # 注解写成字符串前向引用：funasr 已改为按需导入，模块顶层没有这个名字。
        self.asr_model: Optional["AutoModel"] = None

        self.voxcpm_model: Optional[voxcpm.VoxCPM] = None
        self._model_id = model_id

        # 推理锁：队列后台 worker 与「声音设计」同步生成可能并发访问模型，
        # 用同一把锁串行化，避免并发推理 / 重复加载导致显存暴涨或结果异常。
        self._infer_lock = threading.Lock()

        # Whisper（字幕识别）按需加载，用完即卸载
        self.whisper_model = None
        self.whisper_size = os.path.join(current_dir, "models", "whisper", "medium.pt")

    def get_or_load_voxcpm(self) -> voxcpm.VoxCPM:
        if self.voxcpm_model is not None:
            return self.voxcpm_model
        logger.info(f"Loading model: {self._model_id}")
        self.voxcpm_model = voxcpm.VoxCPM.from_pretrained(
            self._model_id,
            optimize=self.optimize,
            device=self.device,
        )
        logger.info("Model loaded successfully.")
        return self.voxcpm_model

    def unload_voxcpm_model(self) -> None:
        """卸载语音生成模型，释放显存（用于给 Whisper 字幕识别腾空间）。"""
        if self.voxcpm_model is None:
            return
        logger.info("Unloading VoxCPM model to free VRAM...")
        try:
            del self.voxcpm_model
        except Exception:
            pass
        self.voxcpm_model = None
        self._free_vram()
        logger.info("VoxCPM model unloaded.")

    def get_or_load_asr_model(self) -> "AutoModel":
        if self.asr_model is not None:
            return self.asr_model
        # 就地导入：funasr 的导入链很重，只有真正用到 ASR 时才值得付这个代价。
        from funasr import AutoModel
        logger.info(
            f"Loading ASR model: {self.asr_model_id} on device: {self.asr_device}"
        )
        self.asr_model = AutoModel(
            model=self.asr_model_id,
            disable_update=True,
            log_level="DEBUG",
            device=self.asr_device,
        )
        logger.info("ASR model loaded successfully.")
        return self.asr_model

    def unload_asr_model(self) -> None:
        """卸载语音转文字模型，释放显存，给语音生成腾出空间。"""
        if self.asr_model is None:
            return
        logger.info("Unloading ASR model to free VRAM...")
        try:
            del self.asr_model
        except Exception:
            pass
        self.asr_model = None
        self._free_vram()
        logger.info("ASR model unloaded.")

    @staticmethod
    def _free_vram() -> None:
        """统一的显存回收逻辑。"""
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 把进程间共享的缓存块也归还，进一步压低占用峰值
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    @staticmethod
    def _inference_ctx():
        """推理上下文：用 no_grad 避免推理过程中累积梯度相关显存（不影响生成结果）。"""
        if torch is not None:
            return torch.no_grad()
        return contextlib.nullcontext()

    # ----- Whisper 字幕 -----

    def get_or_load_whisper(self):
        """加载 OpenAI Whisper medium 模型。"""
        if self.whisper_model is not None:
            return self.whisper_model
        import whisper
        wdevice = "cuda" if self.device.startswith("cuda") else "cpu"
        logger.info(f"Loading Whisper '{self.whisper_size}' on {wdevice}...")
        self.whisper_model = whisper.load_model(self.whisper_size, device=wdevice)
        logger.info("Whisper model loaded.")
        return self.whisper_model

    def unload_whisper(self) -> None:
        """卸载 Whisper 模型，释放显存。"""
        if self.whisper_model is None:
            return
        logger.info("Unloading Whisper model...")
        try:
            del self.whisper_model
        except Exception:
            pass
        self.whisper_model = None
        self._free_vram()
        logger.info("Whisper model unloaded.")

    def generate_subtitle(self, audio_path: str) -> str:
        """
        为给定音频生成 SRT 字幕。
        按需求：先卸载语音生成模型 -> 加载 Whisper 识别 -> 卸载 Whisper -> 重新加载语音生成模型。
        返回生成的 .srt 文件路径。
        整个过程会换入/换出模型，故与其它推理共用同一把锁，避免与同步生成并发抢显存。
        """
        with self._infer_lock:
            # 1) 卸载 VoxCPM，给 Whisper 腾显存
            self.unload_voxcpm_model()
            try:
                model = self.get_or_load_whisper()
                result = model.transcribe(
                    audio_path,
                    task="transcribe",
                    # 不指定 language，让模型自动检测中英文
                    initial_prompt="以下是普通话的转写文本，简体中文，英文单词保持原文。The following is a transcription in Simplified Chinese with English words kept in their original form.",
                    verbose=False
                )
                segments = [
                    {"start": s.get("start", 0.0), "end": s.get("end", 0.0), "text": s.get("text", "")}
                    for s in result.get("segments", [])
                ]
            finally:
                # 2) 识别完立即卸载 Whisper
                self.unload_whisper()

            # 把过长的字幕条拆成更短的多条，避免一行显示不下
            segments = split_long_segments(segments, MAX_SUBTITLE_CHARS)

            srt_path = str(Path(audio_path).with_suffix(".srt"))
            write_srt(segments, srt_path)

            # 3) 重新加载语音生成模型，方便后续任务直接使用
            self.get_or_load_voxcpm()
            return srt_path

    # ----- 参考音频识别 -----

    def prompt_wav_recognition(self, prompt_wav: Optional[str]) -> str:
        if prompt_wav is None:
            return ""
        res = self.get_or_load_asr_model().generate(
            input=prompt_wav,
            language="auto",
            use_itn=True,
        )
        return res[0]["text"].split("|>")[-1]

    def _build_generate_kwargs(
            self,
            *,
            final_text: str,
            audio_path: Optional[str],
            prompt_text_clean: Optional[str],
            cfg_value_input: float,
            do_normalize: bool,
            denoise: bool,
            inference_timesteps: int = 10,
    ) -> dict:
        generate_kwargs = dict(
            text=final_text,
            reference_wav_path=audio_path,
            cfg_value=float(cfg_value_input),
            inference_timesteps=inference_timesteps,
            normalize=do_normalize,
            denoise=denoise,
        )
        if prompt_text_clean and audio_path:
            generate_kwargs["prompt_wav_path"] = audio_path
            generate_kwargs["prompt_text"] = prompt_text_clean
        return generate_kwargs

    def generate_tts_audio(
            self,
            text_input: str,
            control_instruction: str = "",
            reference_wav_path_input: Optional[str] = None,
            prompt_text: str = "",
            cfg_value_input: float = 2.0,
            do_normalize: bool = True,
            denoise: bool = True,
            inference_timesteps: int = 10,
            segment: bool = True,
            segment_max_chars: int = MAX_TTS_CHARS,
    ) -> Tuple[int, np.ndarray]:
        text = (text_input or "").strip()
        if len(text) == 0:
            raise ValueError("请输入要合成的目标文本。")

        control = (control_instruction or "").strip()
        # 去掉控制文本中的括号（半角/全角），避免破坏 "(control)text" 的提示格式
        control = re.sub(r"[()（）]", "", control).strip()

        audio_path = reference_wav_path_input if reference_wav_path_input else None
        # 兜底：超长参考音频自动截取（UI 未走识别流程时也生效，如多角色/SRT 模式）
        if audio_path:
            audio_path = _crop_reference_audio(audio_path)
        prompt_text_clean = (prompt_text or "").strip() or None

        if audio_path and prompt_text_clean:
            logger.info("[配音] 参考音频 + 参考文本续写")
        elif audio_path:
            logger.info("[配音] 仅参考音频")
        else:
            logger.info(f"[声音设计] 描述: {control[:50] if control else '无'}...")

        # segment=False 时整段一次性生成（用于「声音设计」短文本试听，不做分段）
        try:
            segment_max_chars = int(segment_max_chars)
        except (TypeError, ValueError):
            segment_max_chars = MAX_TTS_CHARS
        if segment_max_chars not in SEGMENT_LIMIT_CHOICES:
            segment_max_chars = MAX_TTS_CHARS

        if segment:
            # 长文本分段：从合理位置切分，逐段生成后再合并，避免一次性合成过长文本异常
            chunks = split_text_for_tts(text, segment_max_chars)
            if len(chunks) > 1:
                logger.info(f"目标文本较长（{len(text)} 字），已切分为 {len(chunks)} 段分别生成后合并。")
        else:
            chunks = [text]

        # 串行化模型访问：避免队列 worker 与同步生成并发推理 / 重复加载
        with self._infer_lock:
            current_model = self.get_or_load_voxcpm()
            sr: Optional[int] = None
            seg_wavs: List[np.ndarray] = []
            for idx, chunk in enumerate(chunks):
                final_text = f"({control}){chunk}" if control else chunk
                logger.info(
                    f"Generating audio [{idx + 1}/{len(chunks)}] for text: '{final_text[:80]}...'"
                )
                generate_kwargs = self._build_generate_kwargs(
                    final_text=final_text,
                    audio_path=audio_path,
                    prompt_text_clean=prompt_text_clean,
                    cfg_value_input=cfg_value_input,
                    do_normalize=do_normalize,
                    denoise=denoise,
                    inference_timesteps=inference_timesteps,
                )
                with self._inference_ctx():
                    w = current_model.generate(**generate_kwargs)
                sr = current_model.tts_model.sample_rate
                # 立即把结果搬到 CPU（numpy），并释放本段在 GPU 上的张量，
                # 避免随段数增加显存持续累积——这是把峰值压到 <8G 的关键。
                seg_wavs.append(np.asarray(w, dtype=np.float32).reshape(-1).copy())
                del w
                self._free_vram()
                # 段间静音改到拼接时再插：这里只留纯语音段，
                # 否则静音会混进段列表，响度对齐时被当成「一段音量极低的音频」。

        if sr is None or not seg_wavs:
            raise ValueError("未能生成任何音频，请检查目标文本。")

        # 拼接前先把各段拉到同一电平。模型每次输出的绝对电平有随机性，
        # 不对齐的话，某段「演」得用力就会比邻段高好几 dB，听感上像突然喊起来。
        seg_wavs = match_segment_loudness(seg_wavs)

        if len(seg_wavs) == 1:
            wav = seg_wavs[0]
        else:
            gap = (np.zeros(int(sr * SEGMENT_GAP_SEC), dtype=np.float32)
                   if SEGMENT_GAP_SEC > 0 else None)
            parts: List[np.ndarray] = []
            for idx, seg in enumerate(seg_wavs):
                parts.append(seg)
                if gap is not None and idx != len(seg_wavs) - 1:
                    parts.append(gap)
            wav = np.concatenate(parts)
        # 本次生成结束再回收一次，确保进入下一条任务/下一行台词前显存已释放
        self._free_vram()
        return (sr, wav)

# ---------- 任务队列 ----------

class TaskStopped(Exception):
    """用户点了「停止」时从逐条生成的循环里抛出，与真正的失败区分开。"""


class TaskManager:
    """
    服务端任务队列：所有「添加到任务队列」的请求按提交顺序串行执行。
    后台单线程 worker 逐个处理任务，前端通过 Timer 轮询 snapshot() 刷新状态表与结果。
    """

    # 状态对应的展示图标
    STATUS_ICON = {
        "排队": "⏳ 排队",
        "生成中": "🔄 生成中",
        "成功": "✅ 成功",
        "失败": "❌ 失败",
        "已取消": "⛔ 已取消",
    }

    def __init__(self, demo: VoxCPMDemo) -> None:
        self.demo = demo
        self._tasks: List[Dict[str, Any]] = []
        # 首次启动时加载 outputs 目录下已有的音频作为初始历史记录
        self._history: List[Dict[str, Any]] = scan_output_history()
        self._lock = threading.Lock()
        self._wake = threading.Event()      # 唤醒 worker
        self._stop_flag = threading.Event() # 请求停止：跳过后续排队任务
        self._worker: Optional[threading.Thread] = None
        self._revision = 0                  # 每次有任务完成 / 历史更新时自增，用于前端增量刷新
        self._last_audio: Optional[str] = None
        self._last_subtitle: Optional[str] = None

    # ----- 对外操作 -----

    def add_task(self, params: Dict[str, Any], note: str) -> str:
        task = {
            "id": uuid.uuid4().hex[:8],
            "submit_time": _now_time(),
            "status": "排队",
            "start_time": "",
            "duration": "",
            "note": note,
            "params": params,
            "done_event": threading.Event(),
            "audio_path": None,
            "subtitle_path": None,
            "error": "",
        }
        with self._lock:
            self._tasks.append(task)
            # 新任务入队本身就是一次可见状态变化，确保轮询立即刷新任务列表。
            self._revision += 1
        self._stop_flag.clear()
        self._ensure_worker()
        self._wake.set()
        return task["id"]

    def wait_for_task_result(
        self, task_id: Optional[str], timeout: float = 7200.0
    ) -> Dict[str, Any]:
        """等待指定队列任务的真实结束事件，不使用轮询估算进度。

        带超时是兜底：只要有任何一条路径漏了 done_event.set()，无超时的等待
        就会永久挂住一个并发槽，且无法自行恢复。超时后如实返回「等待超时」，
        任务本身不受影响，仍会在后台跑完并写入历史。
        """
        if not task_id:
            return {"status": "未提交"}
        with self._lock:
            task = next((t for t in self._tasks if t["id"] == task_id), None)
            if task is None:
                return {"status": "未找到"}
            done_event = task["done_event"]
        if not done_event.wait(timeout=timeout):
            return {"status": "等待超时"}
        with self._lock:
            return {
                "status": task["status"],
                "audio_path": task.get("audio_path"),
                "subtitle_path": task.get("subtitle_path"),
                "error": task.get("error", ""),
            }

    def clear_all(self) -> None:
        """清空所有任务（正在生成中的任务保留，无法强制中断）。"""
        with self._lock:
            for task in self._tasks:
                if task["status"] == "排队":
                    task["status"] = "已取消"
                    task["done_event"].set()
            self._tasks = [t for t in self._tasks if t["status"] == "生成中"]
            self._revision += 1

    def stop_current(self) -> None:
        """停止：取消所有排队任务；正在生成中的任务无法强行中断，会自然跑完。"""
        self._stop_flag.set()
        with self._lock:
            for t in self._tasks:
                if t["status"] == "排队":
                    t["status"] = "已取消"
                    t["done_event"].set()
            self._revision += 1

    def clear_history_records(self) -> None:
        """仅清空界面历史记录，不删除 outputs 中的音频或字幕文件。"""
        with self._lock:
            self._history = []
            self._last_audio = None
            self._last_subtitle = None
            self._revision += 1
        logger.info("已清空界面历史记录；outputs 中的音频与字幕文件保持不变。")

    def _sync_missing_output_history_locked(self) -> None:
        """在轮询时剔除已不在 outputs 的历史项，不对磁盘执行删除。"""
        changed = False
        synced_history: List[Dict[str, Any]] = []
        for record in self._history:
            audio_value = record.get("audio")
            try:
                audio_exists = bool(audio_value) and Path(audio_value).is_file()
            except (OSError, TypeError, ValueError):
                audio_exists = False
            if not audio_exists:
                changed = True
                continue
            synced_record = record
            subtitle_value = record.get("subtitle")
            if subtitle_value:
                try:
                    subtitle_exists = Path(subtitle_value).is_file()
                except (OSError, TypeError, ValueError):
                    subtitle_exists = False
                if not subtitle_exists:
                    synced_record = dict(record)
                    synced_record["subtitle"] = None
                    changed = True
            synced_history.append(synced_record)
        if self._last_audio:
            try:
                last_audio_exists = Path(self._last_audio).is_file()
            except (OSError, TypeError, ValueError):
                last_audio_exists = False
            if not last_audio_exists:
                self._last_audio = None
                self._last_subtitle = None
                changed = True
        if self._last_subtitle:
            try:
                last_subtitle_exists = Path(self._last_subtitle).is_file()
            except (OSError, TypeError, ValueError):
                last_subtitle_exists = False
            if not last_subtitle_exists:
                self._last_subtitle = None
                changed = True
        if changed:
            self._history = synced_history
            self._revision += 1
            logger.info("已同步 outputs：移除了不存在文件的历史引用。")

    # ----- worker -----

    def _ensure_worker(self) -> None:
        """确保后台 worker 存在。

        「检查 - 创建」必须在同一个锁区间内完成：提交按钮是
        trigger_mode="multiple" + queue=False，连点会真并发跑在线程池里，
        无锁时两个线程会同时看到 _worker is None 而各起一个 worker。
        本方法只在 add_task 出锁之后调用，不存在锁嵌套。
        """
        with self._lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_loop, daemon=True)
                self._worker.start()

    def _next_pending(self) -> Optional[Dict[str, Any]]:
        """在锁内原子地认领一条排队任务：取出并就地置为「生成中」。

        取任务与改状态必须在同一个锁区间内完成，否则出锁的那一瞬间另一个
        worker 会拿到同一条任务（check-then-act），同一段文本被生成两遍。

        处于停止状态时在这里集中把排队任务取消掉，并逐条 set 各自的
        done_event —— 漏掉这一步会让等待方永久挂起、占死并发槽。
        """
        with self._lock:
            if self._stop_flag.is_set():
                cancelled = False
                for t in self._tasks:
                    if t["status"] == "排队":
                        t["status"] = "已取消"
                        t["done_event"].set()
                        cancelled = True
                if cancelled:
                    self._revision += 1
                return None

            for t in self._tasks:
                if t["status"] == "排队":
                    t["status"] = "生成中"
                    t["start_time"] = _now_time()
                    self._revision += 1
                    return t
        return None

    def _run_loop(self) -> None:
        while True:
            # 认领已在锁内完成，状态此时必定是「生成中」，无需再改。
            task = self._next_pending()
            if task is None:
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue

            t0 = time.time()
            try:
                audio_path, subtitle_path, history_rec = self._execute(task["params"])
                with self._lock:
                    task["status"] = "成功"
                    task["duration"] = f"{time.time() - t0:.1f}s"
                    task["audio_path"] = audio_path
                    task["subtitle_path"] = subtitle_path
                    self._last_audio = audio_path
                    self._last_subtitle = subtitle_path
                    self._history.append(history_rec)
                    self._revision += 1
            except TaskStopped as e:
                # 用户主动停止，不是故障：标成「已取消」而不是「失败」。
                logger.info("任务被停止：%s", e)
                with self._lock:
                    task["status"] = "已取消"
                    task["duration"] = f"{time.time() - t0:.1f}s"
                    task["note"] = f"{task['note']} | {e}"
                    self._revision += 1
            except Exception as e:
                logger.exception("任务执行失败")
                with self._lock:
                    task["status"] = "失败"
                    task["duration"] = f"{time.time() - t0:.1f}s"
                    task["error"] = str(e)
                    task["note"] = f"{task['note']} | 错误：{e}"
                    self._revision += 1
            finally:
                # 每条任务结束后兜底回收一次显存，避免跨任务累积
                self.demo._free_vram()
                task["done_event"].set()

    # ----- 三种生成方式分派 -----


    def _execute(self, p: Dict[str, Any]) -> Tuple[str, Optional[str], Dict[str, Any]]:
        if p["srt_mode"]:
            audio_path = self._exec_srt(p)
            detail_label, detail_value = "SRT 配音", Path(p.get("srt_file") or "").name
            subtitle_path = None  # SRT 模式本身已有字幕，不再额外生成
        elif p["multi_role"]:
            audio_path = self._exec_multi_role(p)
            detail_label, detail_value = "多角色配音", "按台词逐角色合成"
            subtitle_path = self.demo.generate_subtitle(audio_path) if p["subtitle"] else None
        else:
            audio_path = self._exec_single(p)
            if p["mode"] == "极致克隆":
                detail_label, detail_value = "参考文本", (p.get("prompt_text") or "").strip()
            else:
                detail_label, detail_value = "情绪控制", (p.get("control") or "").strip()
            subtitle_path = self.demo.generate_subtitle(audio_path) if p["subtitle"] else None

        history_rec = {
            "time": _now_str(),
            "mode": p.get("mode", ""),
            "detail_label": detail_label,
            "detail_value": detail_value,
            "text": (p.get("text") or "").strip()[:200],
            "audio": audio_path,
            "subtitle": subtitle_path,
        }
        return audio_path, subtitle_path, history_rec

    def _exec_single(self, p: Dict[str, Any]) -> str:
        sr, wav = self.demo.generate_tts_audio(
            text_input=p["text"],
            control_instruction=p["control"] if p["mode"] == "情绪控制" else "",
            reference_wav_path_input=p.get("ref_wav"),
            prompt_text=p["prompt_text"] if p["mode"] == "极致克隆" else "",
            cfg_value_input=p["cfg"],
            do_normalize=p["normalize"],
            denoise=p["denoise"],
            inference_timesteps=p["steps"],
            segment_max_chars=p["segment_limit"],
        )
        wav = apply_speed(wav, sr, p.get("speed", 1.0))
        return save_wav(sr, wav)

    def _exec_multi_role(self, p: Dict[str, Any]) -> str:
        """
        多角色配音：目标文本一行一个角色，格式 [角色名]: 台词。
        每个角色使用各自的参考音频与语速，最终把所有片段拼接为一段音频。
        """
        # name -> {ref, speed}
        role_map = {
            r["name"].strip(): r
            for r in p.get("roles", [])
            if r.get("name") and r["name"].strip()
        }
        segments: List[np.ndarray] = []
        sr: Optional[int] = None
        for line in (p["text"] or "").splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^\[?\s*([^\]:：]+?)\s*\]?\s*[:：]\s*(.*)$", line)
            if not m:
                continue
            name, content = m.group(1).strip(), m.group(2).strip()
            if not content:
                continue
            # 多角色是唯一真正耗时的任务，原来整个循环不看停止标志，
            # 「停止」对它完全无效——点了也要等几百行跑完。
            if self._stop_flag.is_set():
                raise TaskStopped("多角色配音已按停止请求中断。")
            cfg = role_map.get(name)
            ref = cfg.get("ref") if cfg else None
            spd = cfg.get("speed", 1.0) if cfg else 1.0
            s, w = self.demo.generate_tts_audio(
                text_input=content,
                reference_wav_path_input=ref,
                cfg_value_input=p["cfg"],
                do_normalize=p["normalize"],
                denoise=p["denoise"],
                inference_timesteps=p["steps"],
                segment_max_chars=p["segment_limit"],
            )
            sr = s
            w = apply_speed(w, s, spd)
            segments.append(np.asarray(w, dtype=np.float32).reshape(-1))
            segments.append(np.zeros(int(s * 0.1), dtype=np.float32))  # 角色之间插入 0.1s 停顿

        if not segments or sr is None:
            raise ValueError("多角色模式未解析到有效台词，请检查文本格式：[角色名]: 内容")
        full = np.concatenate(segments)
        return save_wav(sr, full, prefix="multi")

    def _exec_srt(self, p: Dict[str, Any]) -> str:
        """
        SRT 一键配音：解析字幕文件，逐条生成语音并按时间轴铺排为一整段音频。
        若某条语音超出其时间槽，则顺延到上一条结束之后，避免重叠。

        稳定性处理：逐条独立生成时每条采样随机，短句音色波动大（实测相邻条
        基频可跳变 60-80Hz，听感即"漂移"）。因此把相邻字幕合并为批次
        （每批 ≤ 用户选择的长段生成上限）一次生成，再按各字幕字符数比例切分回填，
        采样次数从"每条一次"降为"每批一次"，音色一致性显著提升。
        """
        srt_path = p.get("srt_file")
        if not srt_path:
            raise ValueError("SRT 模式请先上传 .srt 字幕文件。")
        entries = parse_srt(srt_path)
        if not entries:
            raise ValueError("SRT 文件解析为空，请检查文件格式。")

        # 合并相邻字幕为批次（保持文本顺序与标点，批次内一次生成）
        batches: List[Tuple[int, int, str]] = []
        cur_start = 0
        cur_text = ""
        for i, e in enumerate(entries):
            line_text = e["text"].strip()
            if not line_text:
                continue
            candidate_text = _join_srt_texts((cur_text, line_text)) if cur_text else line_text
            if cur_text and len(candidate_text) > p["segment_limit"]:
                batches.append((cur_start, i - 1, cur_text))
                cur_start = i
                cur_text = line_text
            else:
                cur_text = candidate_text
        if cur_text:
            batches.append((cur_start, len(entries) - 1, cur_text))

        clips: List[Tuple[int, np.ndarray]] = []  # (entry_index, wav)
        sr: Optional[int] = None
        for start_idx, end_idx, batch_text in batches:
            if self._stop_flag.is_set():
                raise TaskStopped("SRT 配音已按停止请求中断。")
            s, w = self.demo.generate_tts_audio(
                text_input=batch_text,
                control_instruction=p["control"] if p["mode"] == "情绪控制" else "",
                reference_wav_path_input=p.get("ref_wav"),
                # 保留极致克隆的参考文本条件（combined 模式）：实测对长批次文本
                # 音色最贴近参考且稳定；去掉文本条件（reference-only）在长文本上
                # 反而音高漂移（整批可偏移 100Hz 以上）。
                prompt_text=p["prompt_text"] if p["mode"] == "极致克隆" else "",
                cfg_value_input=p["cfg"],
                do_normalize=p["normalize"],
                denoise=p["denoise"],
                inference_timesteps=p["steps"],
                segment_max_chars=p["segment_limit"],
            )
            sr = s
            w = apply_speed(w, s, p.get("speed", 1.0))
            w = np.asarray(w, dtype=np.float32).reshape(-1)
            # 按各字幕字符数比例切分批次音频
            total_chars = sum(len(entries[j]["text"]) for j in range(start_idx, end_idx + 1)) or 1
            pos = 0
            for j in range(start_idx, end_idx + 1):
                seg_len = int(len(w) * len(entries[j]["text"]) / total_chars)
                clips.append((j, w[pos:pos + seg_len]))
                pos += seg_len
            # 取整误差并入本批最后一条，避免丢帧
            if pos < len(w) and clips:
                last_idx, last_w = clips[-1]
                clips[-1] = (last_idx, np.concatenate([last_w, w[pos:]]))

        if sr is None or not clips:
            raise ValueError("SRT 配音未生成任何片段。")

        total_sec = max(e["end"] for e in entries)
        _check_timeline_budget(total_sec, sr)
        longest = max(len(w) for _, w in clips)
        track = np.zeros(int(total_sec * sr) + longest + sr, dtype=np.float32)
        cursor = 0
        for idx, w in clips:
            start = entries[idx]["start"]
            pos = max(int(start * sr), cursor)
            end = pos + len(w)
            if end > len(track):
                track = np.concatenate([track, np.zeros(end - len(track), dtype=np.float32)])
            track[pos:end] += w
            cursor = end
        return save_wav(sr, track, prefix="srt")

    # ----- 状态快照 -----

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._sync_missing_output_history_locked()
            tasks = list(self._tasks)
            history = list(self._history)
            revision = self._revision
            last_audio = self._last_audio
            last_subtitle = self._last_subtitle

        total = len(tasks)
        success = sum(1 for t in tasks if t["status"] == "成功")
        failed = sum(1 for t in tasks if t["status"] == "失败")
        queued = sum(1 for t in tasks if t["status"] == "排队")
        running = sum(1 for t in tasks if t["status"] == "生成中")

        # 闲置（无任务）时只显示计数，不显示输出目录：路径很长，闲置状态下
        # 既没有信息量，又会被渲染成带横向滚动条的代码块。
        if tasks:
            summary_md = (
                f"成功：{success} | 失败：{failed} | 排队：{queued}\n\n"
                f"输出目录：`{OUTPUTPUTS}`"
            )
        else:
            summary_md = f"成功：{success} | 失败：{failed} | 排队：{queued}"

        if tasks:
            rows = ["| 提交时间 | 状态 | 开始时间 | 耗时 | 备注 |", "| --- | --- | --- | --- | --- |"]
            # 最新提交的任务排在最前面
            for t in reversed(tasks):
                note = (t["note"] or "").replace("|", "／").replace("\n", " ")
                # 备注最多展示 7 个字符
                # 失败任务优先显示实际原因，避免把“请输入目标文本”等可处理提示截断。
                if t["status"] == "失败" and "错误：" in note:
                    note = "错误：" + note.split("错误：", 1)[1].strip()
                    note = note[:80]
                else:
                    note = note[:30]
                rows.append(
                    f"| {t['submit_time']} | {self.STATUS_ICON.get(t['status'], t['status'])} "
                    f"| {t['start_time'] or '-'} | {t['duration'] or '-'} | {note} |"
                )
            table_md = "\n".join(rows)
        else:
            table_md = "暂无任务。填写参数后点击「提交到任务队列」。"

        return {
            "summary_md": summary_md,
            "table_md": table_md,
            "revision": revision,
            "latest_audio": last_audio,
            "latest_subtitle": last_subtitle,
            "history": history,
        }


# ---------- UI ----------

def create_demo_interface(demo: VoxCPMDemo):

    task_manager = TaskManager(demo)

    def _append_tag(tag: str, current: str) -> str:
        """「声音设计」点击情绪/风格标签时，追加到音色描述框（允许重复、逗号分隔）。
        API 端音色描述为自由文本，重复情绪词不影响请求合法性。"""
        current = (current or "").strip()
        if not current:
            return tag
        return current + "，" + tag

    def _design_generate(design_text, design_desc, cfg, steps, normalize):
        """
        「声音设计」同步生成：使用情绪控制逻辑，根据音色描述从零生成语音。
        不分段、不入队列、不写入 outputs、不计入配音历史；结果直接返回给试听组件。
        """
        if not (design_text or "").strip():
            gr.Warning("请输入待生成文本")
            return gr.update()
        try:
            sr, wav = demo.generate_tts_audio(
                text_input=design_text,
                control_instruction=design_desc or "",
                reference_wav_path_input=None,   # 声音设计无需参考音频
                prompt_text="",
                cfg_value_input=float(cfg),
                do_normalize=bool(normalize),
                denoise=False,                   # 无参考音频，降噪不适用
                inference_timesteps=int(steps),
                segment=False,                   # 声音设计不分段，整段一次生成
            )
        except Exception as e:
            logger.exception("声音设计生成失败")
            gr.Warning(f"生成失败：{e}")
            return gr.update()
        # 直接返回 (采样率, 波形) 给 gr.Audio：仅在内存中，不落地到 outputs
        return (sr, wav)

    def _design_generate_with_download(design_text, design_desc, cfg, steps, normalize):
        result = _design_generate(design_text, design_desc, cfg, steps, normalize)
        if not isinstance(result, tuple) or len(result) != 2:
            return result, gr.update()
        sr, wav = result
        download_path = save_wav(sr, wav, prefix="voice_design_preview")
        # 试听文件不进历史列表也没有清理入口，每次生成后顺手修剪一次。
        prune_voice_design_previews()
        return result, gr.update(value=download_path, visible=True, interactive=True)

    def _load_recognize_unload(audio_path: Optional[str], preset_name: Optional[str] = None):
        """
        存在参考音频时：加载 ASR 模型 -> 识别 -> 立即卸载，释放显存给语音生成。
        通过生成器分阶段在前端提示「语音模型加载中」「语音识别中」。
        参考音频超过 REF_MAX_SECONDS 时先自动截取（避免长音频导致克隆漂移），
        并用截取后的音频重新识别文本，保证文本与音频对齐。
        """
        if not audio_path:
            yield gr.update(value="", visible=False), gr.update(), gr.update()
            return
        problem = _reference_audio_problem(audio_path)
        if problem:
            logger.warning("参考音频拒绝识别：%s", problem)
            yield (
                gr.update(value=f"❌ 参考音频无效：{problem}", visible=True),
                gr.update(value=""),
                gr.update(value=None),
            )
            return

        # 超长参考音频自动截取；截取后的路径回填到参考音频组件（原文件不变）
        cropped = _crop_reference_audio(audio_path)
        if cropped != audio_path:
            gr.Info(
                f"参考音频超过 {REF_MAX_SECONDS:.0f} 秒，已自动截取前 {REF_MAX_SECONDS:.0f} 秒语音用于克隆"
                "（原文件未修改）；识别文本已按截取后的音频更新，如有误请手动修正。"
            )
            audio_path = cropped

        # 用户保存的个人音色（包括 Juya）不自动写入参考音频文本。
        if preset_name and is_personal_voice_name(preset_name):
            yield gr.update(value="", visible=False), gr.update(value=""), gr.update(value=cropped)
            return

        # 阶段 1：加载模型
        yield gr.update(value="⏳ 语音模型加载中…", visible=True), gr.update(), gr.update()
        asr_text, ok = "", False
        try:
            demo.get_or_load_asr_model()
            # 阶段 2：识别
            yield gr.update(value="🎧 语音识别中…", visible=True), gr.update(), gr.update()
            asr_text = demo.prompt_wav_recognition(audio_path)
            ok = True
        except Exception as e:
            logger.warning(f"ASR recognition failed: {e}")
        finally:
            # 无论成功与否，立即卸载 ASR 模型释放显存
            demo.unload_asr_model()

        if ok:
            yield (
                gr.update(value="", visible=False),
                gr.update(value=asr_text),
                gr.update(value=cropped),
            )
        else:
            yield (
                gr.update(value="❌ 语音识别失败，请手动填写参考音频文本", visible=True),
                gr.update(),
                gr.update(value=cropped),
            )

    def _on_preset_change(name: str):
        """选择预设音色时，把对应音频填入参考音频组件（占位符则清空）。"""
        path = resolve_preset_path(name)
        return gr.update(value=path)

    def _save_voice(name: str, ref_wav: Optional[str]):
        """
        保存当前参考音频为预设音色：校验名称合法性、避免重名覆盖，
        复制到 voices 目录后刷新下拉框并清空名称输入框。
        """
        if not ref_wav:
            gr.Warning("请先上传或录制参考音频再保存。")
            return gr.update(), gr.update(), *[gr.update() for _ in role_components]
        if not is_valid_voice_name(name):
            gr.Warning("音色名称非法：不能为空，且不能包含 \\ / : * ? \" < > | 等字符。")
            return gr.update(), gr.update(), *[gr.update() for _ in role_components]

        name = name.strip()
        if resolve_preset_path(name) is not None:
            gr.Warning(f"音色「{name}」已存在，请换一个名称。")
            return gr.update(), gr.update(), *[gr.update() for _ in role_components]

        VOICES.mkdir(parents=True, exist_ok=True)
        suffix = Path(ref_wav).suffix.lower()
        if suffix not in AUDIO_EXTS:
            suffix = ".wav"
        dest = VOICES / f"{name}{suffix}"
        try:
            shutil.copy(ref_wav, dest)
        except Exception as e:
            gr.Warning(f"保存失败：{e}")
            return gr.update(), gr.update(), *[gr.update() for _ in role_components]

        personal_names = _load_personal_voice_names()
        personal_names.add(name)
        try:
            _save_personal_voice_names(personal_names)
        except OSError as e:
            logger.warning("个人音色索引保存失败：%s", e)
        gr.Info(f"音色「{name}」已保存。")
        choices = [PRESET_PLACEHOLDER] + preset_voice_choices()
        # 刷新主下拉与多角色面板的角色下拉选项，并清空名称输入框
        role_updates = [gr.update(choices=choices) for _ in role_components]
        return gr.update(choices=choices), gr.update(value=""), *role_updates

    def _move_file_to_recycle_bin(path: Path) -> None:
        """将文件移入 Windows 回收站，而不是不可恢复地直接删除。"""
        if os.name != "nt":
            raise OSError("仅支持 Windows 回收站操作")

        import ctypes
        from ctypes import wintypes

        class _SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", wintypes.WORD),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", wintypes.LPVOID),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        # FO_DELETE + FOF_ALLOWUNDO = 发送到回收站；其余标志关闭额外确认窗口。
        operation = _SHFILEOPSTRUCTW()
        operation.wFunc = 3  # FO_DELETE
        # SHFileOperationW 的 pFrom 是双空字符结尾的文件列表；单空字符
        # 可能返回成功但不实际移动文件，导致界面已刷新而源文件仍存在。
        source_buffer = ctypes.create_unicode_buffer(f"{path.resolve()}\0\0")
        operation.pFrom = ctypes.cast(source_buffer, wintypes.LPCWSTR)
        operation.fFlags = 0x0040 | 0x0010 | 0x0004 | 0x0400
        result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
        if result != 0 or operation.fAnyOperationsAborted:
            raise OSError(int(result or 1), f"无法将文件移入回收站：{path.name}")

    def _delete_voice(name: str, current_name: str):
        """只删除个人保存的音色文件，并同步刷新下拉框。"""
        name = (name or "").strip()
        if not is_personal_voice_name(name):
            raise gr.Error("只能删除个人保存的音色。")

        voices_root = VOICES.resolve()
        target = None
        for candidate in VOICES.iterdir():
            if (
                candidate.is_file()
                and candidate.suffix.lower() in AUDIO_EXTS
                and candidate.stem.casefold() == name.casefold()
            ):
                resolved = candidate.resolve()
                if resolved.parent == voices_root:
                    target = candidate
                    break
        if target is None:
            raise gr.Error(f"未找到个人音色「{name}」。")

        try:
            _move_file_to_recycle_bin(target)
            if target.exists():
                raise OSError(f"文件仍存在，未能移入回收站：{target.name}")
        except OSError as e:
            raise gr.Error(f"删除音色失败：{e}") from e

        remaining = {
            item for item in _load_personal_voice_names()
            if item.casefold() != name.casefold()
        }
        try:
            _save_personal_voice_names(remaining)
            if any(item.casefold() == name.casefold() for item in _load_personal_voice_names()):
                raise OSError("个人音色索引写入后校验失败")
        except OSError as e:
            logger.error("个人音色索引更新失败：%s", e)
            raise gr.Error(f"音色文件已移入回收站，但索引更新失败：{e}") from e

        choices = [PRESET_PLACEHOLDER] + preset_voice_choices()
        role_updates = [gr.update(choices=choices) for _ in role_components]
        current_value = (current_name or "").strip()
        if current_value.casefold() == name.casefold() or current_value not in {item[1] if isinstance(item, tuple) else item for item in choices}:
            return gr.update(choices=choices, value=PRESET_PLACEHOLDER), gr.update(value=None), *role_updates
        return gr.update(choices=choices, value=current_value), gr.update(), *role_updates

    def _local_player_cleared():
        """把播放器 / 下载按钮 / 字幕框恢复到「本次还没有结果」的状态。

        不清空的话，它们会继续指向上一次成功的文件，用户会以为那就是这次的结果。
        """
        return (
            gr.update(value=None),
            gr.update(value=None, interactive=False),
            gr.update(value=None, visible=False),
        )

    def _enqueue_and_wait(
            text, srt_file, ref_wav, prompt_text_value,
            cfg, normalize, denoise, steps, speed,
            subtitle, multi_role, srt_mode, segment_limit,
            *role_values,
    ):
        """收集参数入队，然后等这一条任务自己的结果。

        关键是 task_id 留在函数局部变量里：它以前经过一个全会话共用的 gr.State
        传给下一段事件，连点提交时会被后一次覆盖，两个等待方最终等同一条任务。
        合并进同一个函数之后每次点击各持一份，共享状态从根上消失。

        必须是普通函数，不能写成生成器：Gradio 只要见到函数 yield 过一次，
        就把事件从「等待中」切到「流式输出中」，show_progress 的等待动画随之结束
        （跟 yield 的内容无关，gr.skip() 也一样）。这个事件要阻塞好几分钟，
        那个正在计时的等待动画是用户唯一的进度反馈，不能丢。
        队列表格的刷新交给 poll_timer，5 秒一次，够用。
        """
        # 参考音频为必填项；多角色配音由各角色单独选择预设音色，故跳过此校验
        if not multi_role and not ref_wav:
            gr.Warning("上传参考音频")
            snap = task_manager.snapshot()
            return (snap["summary_md"], snap["table_md"], gr.skip(), gr.skip(), gr.skip())
        if not multi_role:
            problem = _reference_audio_problem(ref_wav)
            if problem:
                logger.warning("拒绝提交无效参考音频：%s", problem)
                gr.Warning(f"参考音频无效：{problem}")
                snap = task_manager.snapshot()
                return (snap["summary_md"], snap["table_md"], gr.skip(), gr.skip(), gr.skip())

        # role_values 依次为 8 组 (名称, 参考音色下拉框, 语速)
        roles = []
        for i in range(MAX_ROLES):
            nm = role_values[i * 3]
            rf = role_values[i * 3 + 1]
            sp = role_values[i * 3 + 2]
            if nm and str(nm).strip():
                # 下拉框存的是预设音色名称，转换为实际音频路径
                ref_path = resolve_preset_path(rf)
                roles.append({"name": str(nm).strip(), "ref": ref_path, "speed": float(sp or 1.0)})

        # 多角色配音的额外校验
        if multi_role:
            defined_names = {r["name"] for r in roles}
            # 解析目标文本中实际用到的角色名（格式：[角色名]: 台词）
            used_names: List[str] = []
            for line in (text or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^\[?\s*([^\]:：]+?)\s*\]?\s*[:：]\s*(.*)$", line)
                if m and m.group(2).strip():
                    nm2 = m.group(1).strip()
                    if nm2 not in used_names:
                        used_names.append(nm2)

            if not used_names:
                gr.Warning("请按「[角色名]: 台词」的格式填写目标文本")
                snap = task_manager.snapshot()
                return (snap["summary_md"], snap["table_md"], gr.skip(), gr.skip(), gr.skip())

            # 1) 文本里用到、但下面没有定义的角色名
            undefined = [n for n in used_names if n not in defined_names]
            if undefined:
                gr.Warning(f"在下面定义角色名：{'、'.join(undefined)}")
                snap = task_manager.snapshot()
                return (snap["summary_md"], snap["table_md"], gr.skip(), gr.skip(), gr.skip())

            # 2) 每个已定义的角色都必须选择预设音频
            no_preset = [r["name"] for r in roles if not r["ref"]]
            if no_preset:
                gr.Warning(f"请为角色选择预设音频：{'、'.join(no_preset)}")
                snap = task_manager.snapshot()
                return (snap["summary_md"], snap["table_md"], gr.skip(), gr.skip(), gr.skip())

        try:
            segment_limit = int(segment_limit)
        except (TypeError, ValueError):
            segment_limit = MAX_TTS_CHARS
        if segment_limit not in SEGMENT_LIMIT_CHOICES:
            segment_limit = MAX_TTS_CHARS

        params = {
            "text": text,
            "srt_file": srt_file,
            "mode": "极致克隆",
            "control": "",
            "ref_wav": ref_wav,
            "prompt_text": prompt_text_value,
            "cfg": float(cfg),
            "normalize": bool(normalize),
            "denoise": bool(denoise),
            "steps": int(steps),
            "speed": float(speed or 1.0),
            "subtitle": bool(subtitle),
            "multi_role": bool(multi_role),
            "srt_mode": bool(srt_mode),
            "roles": roles,
            "segment_limit": segment_limit,
        }

        # 生成一条简短备注，便于在队列表中辨认
        if srt_mode:
            note = f"[SRT] {Path(srt_file).name if srt_file else '未上传字幕'}"
        elif multi_role:
            note = f"[多角色] {len(roles)} 个角色"
        else:
            note = f"[配音] {(text or '').strip()[:20]}"

        task_id = task_manager.add_task(params, note)
        snap = task_manager.snapshot()
        # task_id 是局部变量：连点提交时每次点击各等各的，不会串台。
        result = task_manager.wait_for_task_result(task_id)
        snap = task_manager.snapshot()
        if result["status"] == "成功" and result.get("audio_path"):
            subtitle_path = result.get("subtitle_path")
            return (
                snap["summary_md"], snap["table_md"],
                gr.update(value=result["audio_path"]),
                gr.update(value=result["audio_path"], visible=True, interactive=True),
                gr.update(value=subtitle_path, visible=bool(subtitle_path)),
            )
        gr.Warning(f"本地任务未生成音频：{result['status']}")
        return (snap["summary_md"], snap["table_md"], *_local_player_cleared())

    def _poll(prev_rev: int, expanded=None):
        """仅刷新任务表与历史；播放器由对应任务的完成事件更新。

        expanded 是「历史」页里当前展开的音频路径列表。它非空就说明页面上有
        活着的 gr.Audio，而历史区是 @gr.render 渲染的——一旦 history_state 变化，
        整块会被拆掉重建，正在播放的音频会被打断。所以此时推迟刷新。

        推迟时必须把 prev_rev 原样退回去：如果照常返回新的 rev，这一轮就把版本号
        对齐了，等用户收起播放器之后「内容没变过」，历史列表会一直停在旧数据上，
        直到下一条任务完成才补上。退回 prev_rev 才能在收起后的下一次轮询立刻补刷。
        """
        snap = task_manager.snapshot()
        rev = snap["revision"]
        if rev == prev_rev:
            return snap["summary_md"], snap["table_md"], gr.skip(), rev
        if expanded:
            return snap["summary_md"], snap["table_md"], gr.skip(), prev_rev
        return snap["summary_md"], snap["table_md"], snap["history"], rev

    def _on_multi_role_change(checked: bool, current_text: str = ""):
        """
        勾选「多角色配音」后：禁用并取消「SRT 一键配音」，禁用「语速调节」、
        「参考音频上传」「预设音频下拉框」，自动展开多角色面板
        （参考音频改由每个角色单独提供）。
        仅在待生成文本为空时填入三角色示例，不覆盖用户已输入的内容。
        """
        if checked:
            # 勾选多角色配音时，仅当文本为空才填入三角色示例
            multi_role_example = (
                "[角色1]: 你好，很高兴认识你，欢迎来到我们的节目。\n"
                "[角色2]: 我也很高兴，今天的话题听起来非常有意思。\n"
                "[角色3]: 那我们就别浪费时间，赶紧开始吧！"
            )
            text_update = gr.update(
                visible=True,
                value=multi_role_example if not (current_text or "").strip() else None,
            )
            return (
                gr.update(interactive=False, value=False),  # srt_cb
                gr.update(interactive=False),               # speed_slider
                gr.update(interactive=False),               # reference_wav
                gr.update(interactive=False),               # preset_dropdown
                gr.update(open=True),                       # multi_role_accordion
                text_update,                                # text
                gr.update(visible=False),                   # srt_file
            )
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(open=False),
            gr.update(),
            gr.update(),
        )

    def _on_srt_change(checked: bool):
        """
        勾选「SRT 一键配音」后：禁用并取消「字幕生成」「多角色配音」，
        把「目标文本」切换为字幕文件上传；取消勾选则恢复。
        """
        if checked:
            return (
                gr.update(interactive=False, value=False),  # subtitle_cb
                gr.update(interactive=False, value=False),  # multi_role_cb
                gr.update(visible=False),                   # text
                gr.update(visible=True),                    # srt_file
                gr.update(interactive=True),                # speed_slider（恢复，因多角色被取消）
                gr.update(interactive=True),                # reference_wav
                gr.update(interactive=True),                # preset_dropdown
                gr.update(open=False),                      # multi_role_accordion
            )
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    # 队列表格限高 + 超出滚动 + 表格铺满宽度
    custom_css = """
    /* ===== 主题自适应色板 =====
       这些块原来把颜色写死，Light 模式下不跟着变，黑卡片浮在白底上。
       每一项都指向 Gradio 自己的主题变量——它本身就是明暗两套，
       主题切换时自己会变，这里不需要（也没法可靠地）判断当前是哪种模式。

       上一版栽在这上面：用 .gradio-container:not(.dark) 当亮色判据，
       而 .dark 是加在祖先上的，这个元素自己永远不带 .dark，
       于是暗色模式下它照样匹配，把暗色值盖成了浅色。

       每项都留了 fallback = 改动前的固定深色：万一变量名对不上，
       那一项退回现状，而不是变成一片白。

       :root 与 .gradio-container 两处都定义：Gradio 把主题变量挂在哪一层
       并不确定，而变量只从祖先继承给后代——只写 :root 的话，若它其实
       定义在 .gradio-container 上，这里就取不到值、静默落到 fallback。 */
    :root, .gradio-container {
        --lemon-page: var(--body-background-fill, #0f0f11);   /* 最外层容器底 */
        --lemon-card: var(--block-background-fill, #27272a);   /* 卡片 / 面板底 */
        --lemon-ctl: var(--input-background-fill, #3f3f46);   /* 控件填充底 */
        --lemon-ctl-mid: var(--button-secondary-background-fill-hover, #52525b);   /* 控件渐变中段 */
        --lemon-ctl-hi: var(--button-secondary-background-fill, #454549);   /* 未选中 / 次级按钮底 */
        --lemon-line: var(--border-color-primary, #3f3f46);   /* 边框与描边 */
        --lemon-sunken: var(--background-fill-secondary, #24242a);   /* 重置按钮等凹陷块 */
        --lemon-sunken-hover: var(--background-fill-primary, #18181b);   /* 凹陷块 hover */
        --lemon-muted: var(--button-secondary-background-fill, #4a4a55);   /* 禁用 / 弱化底 */
        --lemon-on-ctl: var(--body-text-color-subdued, #d4d4d8);   /* 控件上的次要文字 */
        --lemon-on-muted: var(--body-text-color-subdued, #d0d0d6);   /* 禁用态文字 */
        --lemon-on-menu: var(--body-text-color, #ffffff);   /* 右键菜单文字 */
    }
    /* 任务表格随任务数自然增长，不限高、不出滚动条。 */
    #page-local-dubbing .queue-table { max-height: none; overflow-y: visible; }
    /* 输出目录是长路径，让它折行显示，不要变成带横向滚动条的代码块。 */
    #local-task-results-panel code {
        white-space: normal !important;
        word-break: break-all;
        overflow-x: visible !important;
    }
    #page-local-dubbing .queue-table table { width: 100% !important; table-layout: auto; }
    #page-local-dubbing .small_checkbox { display: flex;  height: 40px; }
    /* 生成历史：紧凑卡片，单行展示，节省空间，一页可放更多条目 */
    .hist-count { margin: 4px 0 8px 0 !important; opacity: 0.85; font-size: 0.9em; }
    .hist-item { padding: 4px 10px !important; margin-bottom: 6px !important; border-radius: 8px; }
    .hist-item .gr-group, .hist-item > div { gap: 4px !important; }
    .hist-meta { display: flex; align-items: center; min-height: 32px; font-size: 0.92em; }
    .hist-meta p { margin: 0 !important; }
    .hist-toggle { align-self: center; }
    .hist-audio { margin-top: 4px; }
    .hist-audio .waveform-container, .hist-audio canvas { max-height: 48px !important; }
    /* WaveSurfer 的横向滚动条会覆盖紧随其后的时间戳；统一留出下方间距。 */
    #local-dubbing-audio .component-wrapper .timestamps,
    #local-design-audio .component-wrapper .timestamps,
    #minimax-dubbing-audio .component-wrapper .timestamps,
    #minimax-design-audio .component-wrapper .timestamps {
        padding-top: 16px !important;
        padding-bottom: 8px !important;
    }
    /* 云端下载区固定为与播放器同宽的彩色扁平长条，禁止被右侧列拉伸成大卡片。 */
    #minimax-current-audio-download,
    #minimax-design-audio-download,
    #minimax-current-audio-download > button,
    #minimax-design-audio-download > button,
    #minimax-current-audio-download > a,
    #minimax-design-audio-download > a,
    #minimax-current-audio-download button,
    #minimax-design-audio-download button,
    #minimax-current-audio-download a,
    #minimax-design-audio-download a {
        width: 100% !important;
        min-height: 52px !important;
        height: 52px !important;
        max-height: 52px !important;
        flex: 0 0 52px !important;
        align-self: stretch !important;
        box-sizing: border-box !important;
    }
    #minimax-current-audio-download,
    #minimax-design-audio-download {
        display: block !important;
        margin: 12px 0 0 !important;
        overflow: hidden !important;
        border-radius: 16px !important;
    }
    /* 云端配音下载按钮单独 40px（用户指定；云端音色设计下载按钮保持 52px） */
    #minimax-current-audio-download,
    #minimax-current-audio-download > button,
    #minimax-current-audio-download > a,
    #minimax-current-audio-download button,
    #minimax-current-audio-download a {
        min-height: 40px !important;
        height: 40px !important;
        max-height: 40px !important;
        flex: 0 0 40px !important;
    }
    #minimax-current-audio-download button:disabled,
    #minimax-current-audio-download a[aria-disabled="true"],
    #minimax-design-audio-download button:disabled,
    #minimax-design-audio-download a[aria-disabled="true"] {
        opacity: 0.58 !important;
        background: var(--lemon-muted) !important;
        color: var(--lemon-on-muted) !important;
        cursor: not-allowed !important;
    }
    #local-dubbing-audio-download,
    #local-design-audio-download,
    #local-dubbing-audio-download > button,
    #local-design-audio-download > button,
    #local-dubbing-audio-download > a,
    #local-design-audio-download > a,
    #local-dubbing-audio-download button,
    #local-design-audio-download button,
    #local-dubbing-audio-download a,
    #local-design-audio-download a {
        width: 100% !important;
        min-height: 52px !important;
        height: 52px !important;
        max-height: 52px !important;
        flex: 0 0 52px !important;
        align-self: stretch !important;
        box-sizing: border-box !important;
    }
    #local-dubbing-audio-download,
    #local-design-audio-download {
        display: block !important;
        margin: 12px 0 0 !important;
        overflow: hidden !important;
        border-radius: 16px !important;
    }
    /* 本地配音下载按钮单独 40px（照抄云端配音下载按钮 #minimax-current-audio-download；
       本地音色设计下载按钮保持 52px） */
    #local-dubbing-audio-download,
    #local-dubbing-audio-download > button,
    #local-dubbing-audio-download > a,
    #local-dubbing-audio-download button,
    #local-dubbing-audio-download a {
        min-height: 40px !important;
        height: 40px !important;
        max-height: 40px !important;
        flex: 0 0 40px !important;
    }
    #local-dubbing-text textarea,
    #local-reference-text textarea,
    #local-design-text textarea,
    #local-design-description textarea,
    #minimax-dubbing-text textarea,
    #minimax-design-prompt textarea,
    #minimax-design-preview textarea {
        resize: none !important;
    }
    #local-dubbing-text textarea,
    #local-reference-text textarea,
    #local-design-text textarea,
    #local-design-description textarea,
    #minimax-dubbing-text textarea,
    #minimax-design-prompt textarea,
    #minimax-design-preview textarea {
        overflow-y: auto !important;
    }
    /* 云端音色设计：音色描述 + 试听文本，固定 8 行高度，超过显示滚动条 */
    #minimax-design-prompt textarea,
    #minimax-design-preview textarea {
        height: 192px !important;
        min-height: 192px !important;
        max-height: 192px !important;
        overflow-y: auto !important;
    }
    /* 抽取按钮：40px 高与参考图一致，圆角矩形，收短为 88px */
    #minimax-design-generate-button {
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        line-height: 40px !important;
        padding: 0 12px !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        margin-bottom: 0 !important;
        min-width: 88px !important;
        width: 88px !important;
        margin-left: 0 !important;
    }
    /* 输入框→保存按钮组间距 14px（row gap 8 + margin 6） */
    #minimax-design-save-btn {
        margin-left: 6px !important;
    }
    /* 抽卡大卡片：按 Photoshop 参考图 1363×207（=页面 909×138 @1.5x），高度锁定 138px，内边距清零改绝对定位 */
    #minimax-design-draw-group {
        overflow: hidden !important;
        position: relative !important;
        height: 138px !important;
        min-height: 138px !important;
        max-height: 138px !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    /* 抽卡大卡片：底色统一深灰 #27272A，无分界线、无浅灰卡片 */
    #minimax-design-draw-group,
    #minimax-design-draw-group > .styler,
    #minimax-design-draw-group > .form,
    #minimax-design-draw-group .row,
    #minimax-design-draw-group .form {
        background: var(--lemon-card) !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 10px !important;
    }
    /* 水印选项行：分隔线由行1承载，checkbox 压成紧凑单行 */
    #minimax-design-watermark {
        border: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        height: 20px !important;
    }
    #minimax-design-watermark label {
        height: 20px !important;
        min-height: 20px !important;
        line-height: 20px !important;
        padding: 0 !important;
        gap: 10px !important;
        font-size: 14px !important;
    }
    #minimax-design-watermark label input {
        width: 16px !important;
        height: 16px !important;
        position: relative !important;
        top: -1px !important;
    }
    #minimax-design-draw-group .row > .form {
        flex: 0 0 auto !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    #minimax-design-candidate-count {
        width: 316px !important;
        max-width: 316px !important;
        min-width: 316px !important;
        padding-right: 0 !important;
        position: relative !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        overflow: visible !important;
        padding: 0 !important;
    }
    #minimax-design-candidate-count > .wrap {
        position: static !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
    }
    /* 内层 wrap（含抽卡按钮 label）：绝对定位贴底，保证与同行按钮底部对齐 */
    #minimax-design-candidate-count .wrap:not(.default) {
        position: absolute !important;
        bottom: 0 !important;
        left: 0 !important;
        gap: 8px !important;
        overflow: hidden !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        width: 316px !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
    }
    /* 抽卡按钮 label：统一 40px 高（参考图），hover/focus 用 inset box-shadow 防频闪 */
    #minimax-design-candidate-count label {
        outline: none !important;
        box-shadow: inset 0 0 0 0 var(--lemon-line) !important;
        transition: box-shadow 0.1s ease-in-out !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        line-height: 28px !important;
        box-sizing: border-box !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 12px !important;
        min-width: 100px !important;
        width: 100px !important;
        max-width: 100px !important;
        padding: 0 8px !important;
        font-size: 14px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    /* 三个抽卡按钮统一 100px（文字已去"色"字收短） */
    #minimax-design-candidate-count label:nth-child(1) {
        min-width: 100px !important;
        width: 100px !important;
    }
    #minimax-design-candidate-count label:nth-child(2) {
        min-width: 100px !important;
        width: 100px !important;
    }
    /* 卡片内所有文字块禁止滚动条：换行可、滚动不可 */
    #minimax-design-draw-group .block,
    #minimax-design-draw-group .block .prose,
    #minimax-design-draw-group .form,
    #minimax-design-watermark {
        overflow: visible !important;
        overflow-x: visible !important;
        overflow-y: visible !important;
        scrollbar-width: none !important;
    }
    #minimax-design-draw-group .block::-webkit-scrollbar,
    #minimax-design-draw-group .form::-webkit-scrollbar,
    #minimax-design-draw-group .wrap::-webkit-scrollbar {
        display: none !important;
    }
    /* 行2：标题 14px 加粗贴顶，说明 11.6px（参考图宽度 84px / 270px） */
    #minimax-design-draw-group .row:nth-of-type(2) .column > .block {
        padding: 0 !important;
        margin: 0 !important;
    }
    #minimax-design-draw-group .row:nth-of-type(2) .column > .block .prose {
        margin: 0 !important;
    }
    #minimax-design-draw-group .row:nth-of-type(2) .column > .block:nth-child(1) .prose p {
        margin: 0 !important;
        font-size: 14px !important;
        line-height: 14px !important;
    }
    #minimax-design-draw-group .row:nth-of-type(2) .column > .block:nth-child(2) {
        margin-top: 3.7px !important;
    }
    #minimax-design-draw-group .row:nth-of-type(2) .column > .block:nth-child(2) .prose p {
        margin: 0 !important;
        font-size: 11.6px !important;
        line-height: 13px !important;
    }
    #minimax-design-candidate-count label:not(.selected) {
        background: var(--lemon-ctl-hi) !important;
        background-image: none !important;
    }
    #minimax-design-candidate-count label:hover,
    #minimax-design-candidate-count label:focus-visible {
        box-shadow: inset 0 0 0 2px var(--lemon-line) !important;
        outline: none !important;
    }
    #minimax-design-draw-group .row {
        gap: 8px !important;
        align-items: flex-start !important;
        flex-wrap: nowrap !important;
    }
    /* 行1（水印行）：绝对定位 y8-31，右边界与保存按钮右缘对齐（16px），分隔线延伸到 x893 */
    #minimax-design-draw-group .row:nth-of-type(1) {
        position: absolute !important;
        top: 8px !important;
        left: 11px !important;
        right: 16px !important;
        width: auto !important;
        height: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        margin: 0 !important;
        gap: 0 !important;
        align-items: flex-start !important;
        border-bottom: 1px solid var(--lemon-line) !important;
        border-radius: 0 !important;
    }
    /* 行2（标题说明行）：绝对定位 y39-72，右边界 x697（参考图官网注释右缘） */
    #minimax-design-draw-group .row:nth-of-type(2) {
        position: absolute !important;
        top: 39px !important;
        left: 11px !important;
        right: 212px !important;
        width: auto !important;
        height: 33px !important;
        min-height: 33px !important;
        max-height: 33px !important;
        margin: 0 !important;
        gap: 0 !important;
        align-items: flex-start !important;
        border: 0 !important;
    }
    /* 行3（控制条）：绝对定位 y88-128，右缘留 16px 空隙（参照卡片与上方试听文本模块间距） */
    #minimax-design-draw-group .row:nth-of-type(3) {
        position: absolute !important;
        top: 88px !important;
        left: 11px !important;
        right: 16px !important;
        width: auto !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        margin: 0 !important;
        gap: 8px !important;
        align-items: flex-start !important;
    }
    /* 播放器按钮不显示焦点方框。
       原来逐个列 elem_id，漏掉了两个参考音频播放器和历史 Tab 动态创建的那个；
       改成按后缀匹配——本文件里每个 gr.Audio 的 elem_id 都以 -audio 结尾，
       新增播放器沿用这个命名即自动生效。
       不写成全局 button:focus-visible：那会干掉键盘导航的焦点提示，
       也会覆盖抽卡按钮故意用 inset box-shadow 做的 focus 效果。 */
    [id$="-audio"] button:focus,
    [id$="-audio"] button:focus-visible,
    [id$="-audio"] [role="button"]:focus,
    [id$="-audio"] [role="button"]:focus-visible,
    .hist-audio button:focus,
    .hist-audio button:focus-visible,
    .hist-audio [role="button"]:focus,
    .hist-audio [role="button"]:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    /* 波形区本身与它的可聚焦子节点同样不加框（Gradio 的 waveform 会给
       容器和拖动手柄加 outline，点一下播放就会出现一圈方块）。 */
    [id$="-audio"] :focus,
    [id$="-audio"] :focus-visible,
    [id$="-audio"] :focus-within,
    .hist-audio :focus,
    .hist-audio :focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    /* 本地生成结果播放器与本地音色设计播放器保持一致，不显示白色焦点边框。 */
    #local-dubbing-audio:focus,
    #local-dubbing-audio:focus-within,
    #local-dubbing-audio *:focus,
    #local-dubbing-audio *:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }
    #local-dubbing-audio .component-wrapper {
        outline: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    /* 本地试听结果面板：照抄云端 #minimax-dubbing-result-panel——
       播放器灰卡与下载按钮平级独立，容器全透明无边框无内边距（不融合）；
       gap 0 去掉 Gradio Column 默认 16px 组件间距（云端间距 = 按钮 margin 12px） */
    #local-generated-result-panel,
    #local-generated-result-panel > .form,
    #local-generated-result-panel .form {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
        gap: 0 !important;
    }
    /* 两个本地试听播放器共用同一视觉基类；结果播放器只额外增加高度。 */
    .local-preview-player,
    .local-preview-player.block,
    .local-preview-player.block.border_focus,
    .local-preview-player .block,
    .local-preview-player > .block {
        border-color: transparent !important;
        outline: none !important;
        box-shadow: none !important;
    }
    #page-local-dubbing #local-dubbing-audio,
    #page-local-dubbing #local-dubbing-audio.block,
    #page-local-dubbing #local-dubbing-audio .block,
    #local-design-audio,
    #local-design-audio.block,
    #local-design-audio .block {
        width: 100% !important;
        max-width: none !important;
        height: 153px !important;
        min-height: 153px !important;
        max-height: 153px !important;
        box-sizing: border-box !important;
    }
    /* 本地配音/本地音色设计试听播放器：照抄云端 #minimax-dubbing-audio——
       四圆角 10px + 1px 灰边框 + 微阴影（玻璃质感） */
    #page-local-dubbing #local-dubbing-audio,
    #page-local-dubbing #local-dubbing-audio.block,
    #local-design-audio,
    #local-design-audio.block {
        border-radius: 10px !important;
        border: 0.667px solid var(--lemon-line) !important;
        box-shadow: rgba(0, 0, 0, 0.1) 0px 1px 3px 0px,
            rgba(0, 0, 0, 0.1) 0px 1px 2px -1px !important;
    }
    #page-local-dubbing #local-dubbing-audio .component-wrapper,
    #local-design-audio .component-wrapper {
        min-height: 153px !important;
        box-sizing: border-box !important;
    }
    #page-local-dubbing #local-right-column {
        gap: 15px !important;
    }
    /* 空播放器内部容器也沿用“本地音色设计 / 试听结果”的无白框状态。 */
    .local-preview-player .wrap,
    .local-preview-player .audio-container,
    .local-preview-player .waveform-container,
    .local-preview-player .component-wrapper:focus-within {
        outline: none !important;
        border-color: transparent !important;
        box-shadow: none !important;
    }
    /* 本地页左列的两个折叠模块：高级设置在上，多角色配音在下。 */
    #page-local-dubbing #local-advanced-settings { order: 1 !important; }
    #page-local-dubbing #local-multirole-panel { order: 2 !important; }
    /* 云端音色设计的底部注释区：紧凑排布，不额外撑高页面。 */
    #minimax-design-notes ol {
        margin: 4px 0 0 0 !important;
        padding-left: 20px !important;
    }
    #minimax-design-notes li {
        margin: 0 !important;
        line-height: 1.6 !important;
    }
    /* 「云端克隆音色」下方的计费提示：强制单行完整显示，不折行、不截断。 */
    #minimax-clone-api-note,
    #minimax-clone-api-note * {
        white-space: nowrap !important;
    }
    #minimax-clone-api-note {
        overflow: visible !important;
    }
    #minimax-clone-api-note p {
        font-size: 12px !important;
        line-height: 1.5 !important;
        margin: 0 !important;
    }
    /* 云端“已保存音色”下拉：宽度沿用 Gradio 原生（与下拉框同宽），
       只给每行右侧留出行内“×”删除动作的位置，右缘正好收在“×”上。 */
    .minimax-saved-voice-option {
        padding-right: 24px !important;
    }
    .minimax-saved-voice-delete-action {
        position: absolute !important;
        right: 4px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        margin: 0 !important;
        padding: 0 6px !important;
        line-height: 18px !important;
        font-size: 16px !important;
        border: 0 !important;
        border-radius: 4px !important;
        background: transparent !important;
        color: #d0d0d6 !important;
        cursor: pointer !important;
        z-index: 2 !important;
    }
    .minimax-saved-voice-delete-action:hover,
    .minimax-saved-voice-delete-action:focus-visible {
        background: #b3413f !important;
        color: #ffffff !important;
        outline: none !important;
    }
    /* 删除由下拉行内动作触发；该输入框保持可渲染但移到屏幕外，
       确保浏览器端事件与 Gradio 状态同步仍然有效。 */
    #page-minimax-dubbing #minimax-delete-voice-name {
        position: fixed !important;
        left: -10000px !important;
        top: -10000px !important;
        width: 1px !important;
        height: 1px !important;
        min-width: 1px !important;
        min-height: 1px !important;
        opacity: 0 !important;
        overflow: hidden !important;
        pointer-events: auto !important;
    }
    .local-preset-delete-action {
        position: absolute !important;
        right: 8px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        margin: 0 !important;
        padding: 1px 6px !important;
        border: 0 !important;
        border-radius: 4px !important;
        background: transparent !important;
        color: #d0d0d6 !important;
        cursor: pointer !important;
        z-index: 2 !important;
    }
    .local-preset-delete-action:hover,
    .local-preset-delete-action:focus-visible {
        background: var(--lemon-muted) !important;
        color: var(--lemon-on-menu) !important;
        outline: none !important;
    }
    /* 删除动作由下拉项触发；内部 Gradio 控件保持可渲染但移到屏幕外，
       确保浏览器端事件和 Gradio 状态同步仍然有效。 */
    #page-local-dubbing #local-delete-voice-name,
    #page-local-dubbing #local-delete-voice-button {
        position: fixed !important;
        left: -10000px !important;
        top: -10000px !important;
        width: 1px !important;
        height: 1px !important;
        min-width: 1px !important;
        min-height: 1px !important;
        opacity: 0 !important;
        overflow: hidden !important;
        pointer-events: auto !important;
    }
    #page-local-dubbing .local-result-row {
        align-items: flex-start !important;
    }
    #page-local-dubbing .local-result-row > .column {
        align-self: flex-start !important;
    }
    /* 获取音色按钮在下方区域框中复用保存 API key 的按钮规范。 */
    /* 密钥使用单行原生输入框；内容超过可视宽度时保留横向滚动而非截断。 */
    /* 统一 API 输入区与保存按钮周围的底框，避免右侧出现独立灰色分块。 */
    #minimax-api-key-group,
    #minimax-api-key-group .gr-group,
    #minimax-api-key-group .styler,
    #minimax-api-key-group .row {
        background: var(--lemon-card) !important;
    }
    #minimax-api-key-group .row {
        align-items: flex-end !important;
        gap: 15px !important;
        padding-right: 18px !important;
        box-sizing: border-box !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* Gradio 会给同一 .form 内相邻控件加一条竖线做分隔。
       这两处要的是「输入框与按钮连成一条」的观感，那条线是多余的：
       API Key 输入框与「保存 API Key」之间、自定义 voice_id 的左右两侧。

       用 transparent 而不是 border: 0——边框宽度仍占位，不会因为少 1px 造成
       控件错位；也与主题无关，明暗两种模式一起生效（暗色下这条线同样存在，
       只是深底配深线看不出来）。

       只作用于 .form / .row 的直接子元素，即各控件的外层容器，
       不下探到输入框内部，避免把控件自身该有的边框也抹掉。 */
    #minimax-api-key-group .form > *,
    #minimax-api-key-group .row > *,
    #minimax-voice-select-group .form > *,
    #minimax-voice-select-group .row > * {
        border-left-color: transparent !important;
        border-right-color: transparent !important;
    }
    /* 这两个控件的兄弟规则都写了 border: 0，唯独它俩漏了，线就从这里露出来。 */
    #minimax-voice-select-group #minimax-fetched-voice-id,
    #minimax-voice-select-group #minimax-manual-voice-id,
    #minimax-api-key-group #minimax-api-key {
        border: 0 !important;
        box-shadow: none !important;
    }
    /* 上面两条只关掉了控件自身、以及 .form / .row 直接子元素的边框，
       实测都没消掉那条线——说明它画在这之外的某一层容器上
       （那条线贯穿整行高度，本来就比输入框高，早该想到）。

       这两块区域里的控件都是「浅灰填充、无可见边框」的设计，
       左右方向的边框整片关掉不会损失任何视觉元素，因此直接覆盖所有层级。
       只动左右：上下边框留着（有的控件靠它撑高度），
       圆角、底色、内外边距一律不碰，布局不会位移。
       宽度和阴影都处理，是因为这类分隔线两种画法都常见。 */
    #minimax-api-key-group .row *,
    #minimax-voice-select-group .row * {
        border-left-width: 0 !important;
        border-right-width: 0 !important;
        box-shadow: none !important;
    }

    #minimax-api-key {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        padding: 10px 0 10px 8px !important;
        background: var(--lemon-card) !important;
    }
    #minimax-api-key input {
        overflow-x: auto !important;
        white-space: nowrap !important;
        border-radius: 12px !important;
    }
    #minimax-save-api-key {
        flex: 0 0 152px !important;
        min-width: 152px !important;
        align-self: flex-end !important;
        width: 152px !important;
        min-height: 39.59375px !important;
        height: 39.59375px !important;
        padding: 0 12px !important;
        border-radius: 12px !important;
        margin-bottom: 10px !important;
        border: 0 !important;
        box-shadow: none !important;
        font-size: 14px !important;
    }
    /* 音色选择行与上方 API key 行使用同一深色区域框、间距和圆角逻辑。 */
    #minimax-voice-select-group,
    #minimax-voice-select-group .gr-group,
    #minimax-voice-select-group .styler,
    #minimax-voice-select-group .row {
        background: var(--lemon-card) !important;
    }
    /* 新设计音色的只读 voice_id 显示框：候选生成后显示，不能点击或触发保存。 */
    #minimax-design-voice-id-display {
        flex: 1 1 210px !important;
        min-width: 210px !important;
        max-width: none !important;
        align-self: flex-start !important;
        width: auto !important;
        min-height: 40px !important;
        height: 40px !important;
        padding: 0 10px !important;
        border-radius: 12px !important;
        margin-bottom: 0 !important;
        margin-left: 2px !important;
        border: 1px solid rgba(255, 255, 255, 0.10) !important;
        box-shadow: none !important;
        font-size: 14px !important;
        background: var(--lemon-ctl-hi) !important;
        background-image: none !important;
    }
    #minimax-design-voice-id-display textarea,
    #minimax-design-voice-id-display input {
        min-height: 38px !important;
        height: 38px !important;
        padding: 0 !important;
        color: rgba(255, 255, 255, 0.82) !important;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
        font-size: 12px !important;
    }
    .minimax-saved-voice-context-menu {
        position: fixed;
        /* Gradio 下拉选项在 portal 浮层中，层级高于普通页面内容。 */
        z-index: 2147483647 !important;
        min-width: 196px;
        height: 34px;
        padding: 0 12px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        color: var(--lemon-on-menu);
        background: var(--lemon-ctl);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        text-align: left;
        cursor: pointer;
    }
    .minimax-saved-voice-context-menu:hover {
        background: #b42318;
    }
    #minimax-design-save-btn {
        flex: 0 0 126px !important;
        min-width: 126px !important;
        max-width: 126px !important;
        align-self: flex-start !important;
        width: 126px !important;
        min-height: 40px !important;
        height: 40px !important;
        padding: 0 12px !important;
        border-radius: 12px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        margin-bottom: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        font-size: 14px !important;
    }
#minimax-voice-select-group .row {
    align-items: flex-end !important;
    flex-wrap: nowrap !important;
    gap: 15px !important;
        padding-right: 18px !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        border: 0 !important;
        box-shadow: none !important;
    }
#minimax-voice-select-group .row > .form {
    display: flex !important;
    flex-wrap: nowrap !important;
    flex: 1 1 582px !important;
        width: auto !important;
        min-width: 0 !important;
    gap: 15px !important;
    align-items: flex-end !important;
    background: var(--lemon-card) !important;
    border: 0 !important;
    box-shadow: none !important;
}
#minimax-voice-select-group #minimax-fetched-voice-id {
    flex: 0 1 419px !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 10px 0 10px 8px !important;
        background: var(--lemon-card) !important;
    }
#minimax-voice-select-group #minimax-saved-voice-id {
    flex: 0 1 164px !important;
        width: auto !important;
        min-width: 0 !important;
    padding: 10px 0 !important;
    background: var(--lemon-card) !important;
    border: 0 !important;
    box-shadow: none !important;
}
#minimax-voice-select-group #minimax-manual-voice-wrap {
        flex: 0 0 144px !important;
        width: 144px !important;
    min-width: 144px !important;
    max-width: 144px !important;
    background: var(--lemon-card) !important;
    border: 0 !important;
    box-shadow: none !important;
}
#minimax-voice-select-group #minimax-manual-voice-id {
    width: 100% !important;
    min-width: 0 !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    padding: 10px 0 !important;
    background: var(--lemon-card) !important;
    }
#minimax-voice-select-group #minimax-fetched-voice-id .wrap-inner,
#minimax-voice-select-group #minimax-saved-voice-id .wrap-inner,
#minimax-voice-select-group #minimax-manual-voice-id .input-container {
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    box-sizing: border-box !important;
    border-radius: 12px !important;
}
#minimax-voice-select-group #minimax-manual-voice-id input,
#minimax-voice-select-group #minimax-manual-voice-id textarea {
    height: 40px !important;
    min-height: 40px !important;
    max-height: 40px !important;
    box-sizing: border-box !important;
    line-height: 20px !important;
    resize: none !important;
    white-space: nowrap !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
}
#minimax-voice-select-group #minimax-fetched-voice-id,
#minimax-voice-select-group #minimax-saved-voice-id,
#minimax-voice-select-group #minimax-manual-voice-id {
    border: 0 !important;
    box-shadow: none !important;
}
    #minimax-voice-select-group #minimax-fetch-voices {
        flex: 0 1 122px !important;
        width: auto !important;
        min-width: 0 !important;
        align-self: flex-end !important;
        min-height: 39.59375px !important;
        height: 39.59375px !important;
        margin-bottom: 10px !important;
        padding: 0 12px !important;
        border-radius: 12px !important;
        border: 0 !important;
        box-shadow: none !important;
        font-size: 14px !important;
    }
    /* 模型、输出格式与模型 ID 获取区：三个独立区域框沿用 API key 行的间距与底色。 */
    #minimax-model-layout,
    #minimax-model-layout > .form {
        gap: 15px !important;
        align-items: flex-start !important;
    }
    /* 模型按钮始终为两列等宽网格；模型名称长短不影响第二列的垂直基线。 */
    #minimax-model-group .wrap {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        grid-auto-flow: row !important;
        column-gap: 10px !important;
        row-gap: 10px !important;
        align-items: stretch !important;
    }
    #minimax-model-group .wrap > label {
        width: 100% !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
        margin: 0 !important;
    }
    #minimax-model-group,
    #minimax-output-format-group,
    #minimax-model-fetch-group {
        background: var(--lemon-card) !important;
        border: 1px solid var(--lemon-line) !important;
        border-radius: 12px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        align-self: flex-start !important;
        justify-content: flex-start !important;
        align-content: flex-start !important;
        overflow: hidden !important;
    }
    /* 输出格式卡：深灰底框与模型卡、模型ID卡等高同底，保持同一水平线。 */
    #minimax-output-format-group {
        align-self: stretch !important;
    }
    /* 高级模型 ID 区域：标题与前两栏同基线，功能控件独立成下一行。 */
    #minimax-model-fetch-group {
        position: relative !important;
        padding: 12px !important;
        gap: 0 !important;
        /* 深灰底框与左右卡片底部平齐：stretch 覆盖三卡共用块的 flex-start，
           使底框拉伸到父行最高卡片高度，内容仍保持顶部对齐。 */
        align-self: stretch !important;
    }
    #minimax-model-fetch-group > .form {
        width: 100% !important;
        flex: 0 0 auto !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    #minimax-model-fetch-title,
    #minimax-model-fetch-title > .prose,
    #minimax-model-fetch-title p {
        min-height: 22px !important;
        height: 22px !important;
        /* Gradio .block.padded 默认高度会撑到 57px，导致标题占用过高、
           功能控件行下沉 35px，与左侧模型/输出格式的按钮行不在同一水平线。
           用 max-height + overflow 强制压回 22px（inline style 实测有效）。 */
        max-height: 22px !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        font-size: 14px !important;
        line-height: 22px !important;
    }
    #minimax-model-fetch-controls {
        position: static !important;
        top: auto !important;
        right: auto !important;
        left: auto !important;
        bottom: auto !important;
        width: auto !important;
        flex: 0 0 40px !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        /* 标题下的独立功能行：与输出格式的首行按钮共享同一 40px 基线。 */
        margin: 5px 0 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        z-index: 1 !important;
        align-items: stretch !important;
        flex-wrap: nowrap !important;
        gap: 15px !important;
        min-width: 0 !important;
    }
    /* 情绪、语气词、停顿值和快捷按钮共用一条 40px 功能线。
       控件间距参考输出参数卡"语言增强与音量"间距（10px）；
       三控件均分加长，自然把按钮组推到行尾与"获取"按钮右缘齐平。 */
    #minimax-expression-controls,
    #minimax-expression-controls > .form {
        align-items: flex-end !important;
        flex-wrap: nowrap !important;
        gap: 10px !important;
    }
    /* 控件 form 内边距归零：下拉框/输入框本体左缘与标题"高级"二字左缘平齐 */
    #minimax-emotion,
    #minimax-expression,
    #minimax-pause-seconds {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    #minimax-emotion .wrap-inner,
    #minimax-expression .wrap-inner {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }
    /* Gradio 内层 .form 默认 min-width: min(478px, 100%) 且 overflow-x: auto，
       会把三个控件压进 478px 并出现横向滚动条；改为按内容自适应撑开。
       注意：overflow-x 不能设 visible——与 Gradio 自带 overflow-y: hidden 组合时
       浏览器会强制把 visible 计算为 auto，必须显式 hidden 才能消除滚动条。
       flex 用 1 1 0（增长）：三控件均分撑满剩余宽度，
       右侧按钮组被自然推到行尾，右缘与"获取"按钮齐平。 */
    #minimax-expression-controls > .form {
        min-width: 0 !important;
        width: auto !important;
        flex: 1 1 0 !important;
        overflow-x: hidden !important;
        /* Gradio 默认 .form 带亮灰底/边框/阴影，会包出一块与底色不一致的大卡；
           按设计（卡片与底色一致）透明化，仅保留各控件自身的 #3F3F46 轮廓。 */
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* 情绪风格、语气词、停顿时长：参考多角色配音三层灰度配色——
       大卡 #27272A（中灰）→ 控件外描边 #202028（稍深内框）→ 控件本体 #3F3F46（最浅）。
       内框用 box-shadow 描边紧贴控件（约 3px），不做大色块，控件框贴合按钮本体。 */
    #minimax-expression-controls #minimax-emotion,
    #minimax-expression-controls #minimax-expression,
    #minimax-expression-controls #minimax-pause-seconds {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    #minimax-expression-controls #minimax-emotion .wrap,
    #minimax-expression-controls #minimax-expression .wrap,
    #minimax-expression-controls #minimax-pause-seconds .wrap {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* 情绪风格、语气词、停顿时长：三控件均分加长（参考顶部音色下拉的宽裕感），
       宽度由 flex 均分，不再固定 150/175/150。 */
    #minimax-expression-controls #minimax-emotion {
        flex: 1 1 0 !important;
        width: auto !important;
        min-width: 0 !important;
    }
    #minimax-expression-controls #minimax-expression {
        flex: 1 1 0 !important;
        width: auto !important;
        min-width: 0 !important;
    }
    #minimax-expression-controls #minimax-pause-seconds {
        flex: 1 1 0 !important;
        width: auto !important;
        min-width: 0 !important;
    }
    #minimax-expression-controls #minimax-emotion .wrap-inner,
    #minimax-expression-controls #minimax-expression .wrap-inner,
    #minimax-expression-controls #minimax-pause-seconds .input-container,
    #minimax-expression-controls #minimax-pause-seconds input {
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        box-sizing: border-box !important;
        /* 与上方"已获取音色/自定义 voice_id"一致：明显的条状选择框 + 12px 圆角 */
        border-radius: 12px !important;
    }
    /* 情绪风格、语气词、停顿时长：对应多角色配音的 Photoshop 标注色——
       卡片底框 R39,G39,B42（#27272A，即 Accordion 大卡背景，与底色一致）；
       三个功能卡片本体不填充（背景与卡片底色一致），
       仅外边轮廓为浅色 R63,G63,B70（#3F3F46），40px 高圆角矩形。 */
    #minimax-expression-controls #minimax-emotion .wrap-inner,
    #minimax-expression-controls #minimax-expression .wrap-inner,
    #minimax-expression-controls #minimax-pause-seconds .input-container {
        background: transparent !important;
        border: 1px solid var(--lemon-line) !important;
        box-shadow: none !important;
        border-radius: 10px !important;
    }
    #minimax-expression-controls #minimax-pause-seconds input {
        line-height: 40px !important;
        padding: 0 10px !important;
        border-radius: 10px !important;
        background: transparent !important;
        /* 与情绪风格/语气词下拉框一致：1px #3F3F46 轮廓边框（线框样式）。
           Number 组件没有 .input-container 层，边框直接加在 input 上。 */
        border: 1px solid var(--lemon-line) !important;
    }
    /* 四个预设按钮：两两并排、两行堆叠成 2x2 网格，两列均分；
       按钮间距参考输出格式卡 mp3 与 wav 的间距（8px）。
       行尾位置由 .form 的 flex 增长自然推到行尾（右缘与"获取"按钮齐平），
       不再使用 margin-left 硬编码。 */
    #minimax-pause-preset-grid {
        flex: 0 0 150px !important;
        width: 150px !important;
        min-width: 150px !important;
        align-self: flex-end !important;
        gap: 8px !important;
    }
    #minimax-pause-preset-grid > .form {
        min-width: 0 !important;
        overflow-x: hidden !important;
        gap: 8px !important;
        /* 与情绪卡一致：透明化 Gradio 默认亮灰底 */
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    #minimax-pause-preset-grid .gr-button,
    #minimax-pause-preset-grid button {
        width: 100% !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        box-sizing: border-box !important;
        padding: 0 6px !important;
        font-size: 13px !important;
        border-radius: 8px !important;
    }
    #minimax-pause-preset-025,
    #minimax-pause-preset-05,
    #minimax-pause-preset-10,
    #minimax-pause-preset-15 {
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }
    #minimax-pause-insert {
        flex: 0 0 96px !important;
        width: 96px !important;
        min-width: 96px !important;
        align-self: flex-end !important;
        /* controls 行 gap 15px（控件间距参考），但按钮组内部间距要 8px
           （参考 mp3 与 wav 的间距），用负 margin 把 15px 收窄到 8px。 */
        margin-left: -7px !important;
    }
    /* 换行插入按钮与四个预设按钮同为 40px 高、方形圆角矩形（非胶囊形）。 */
    #minimax-pause-insert {
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        box-sizing: border-box !important;
        padding: 0 8px !important;
        border-radius: 8px !important;
    }
    /* 输出参数内部控件缩短间距（5px），不改变参数顺序与功能；
       强制单行排列（flex-wrap: nowrap），且不产生横向滚动条。
       注意：行容器是 Accordion 内的 .form（非直接子元素），用后代选择器。
       Gradio 默认 .form 带浅灰底 rgb(63,63,70)——控件间 gap 会露出浅色竖杠，
       必须透明化让底色统一（与卡片一致），竖杠即消失。 */
    #minimax-output-params-accordion .form {
        /* 语言增强/音量/音高/MP3码率/采样率/声道 之间的间距 =
           模型卡两个按钮之间的间距（10px），视觉统一。 */
        gap: 10px !important;
        flex-wrap: nowrap !important;
        overflow-x: hidden !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        /* 整体左移 13px：贴到 Accordion 内容区最左，
           让最左侧"语速"圆角矩形的左边弧线完整显示；
           右侧 padding-right 留出空间，让"声道"圆角矩形右弧线完整显示。 */
        margin-left: -13px !important;
        padding-right: 14px !important;
    }
    /* 语言增强/音量/音高/MP3码率/采样率/声道：与语速一致——
       二级卡片区域（block/wrap）深灰 rgb(39,39,42)，与卡片底色统一，
       无分割、是一个整体；浅色只存在于圆角矩形按钮本体上：
       填充 #3F3F46 + 1px 同色细外框，40px 高、10px 圆角。
       注意：不要在 wrap-inner 上设 padding-right——
       会把下拉箭头挤出容器导致点不开；箭头空间在 input 上留。 */
    #minimax-output-params-accordion #minimax-language .wrap-inner,
    #minimax-output-params-accordion #minimax-volume .wrap-inner,
    #minimax-output-params-accordion #minimax-pitch .wrap-inner,
    #minimax-output-params-accordion #minimax-bitrate .wrap-inner,
    #minimax-output-params-accordion #minimax-sample-rate .wrap-inner,
    #minimax-output-params-accordion #minimax-channel .wrap-inner {
        background: var(--lemon-ctl) !important;
        border: 1px solid var(--lemon-line) !important;
        box-shadow: none !important;
        border-radius: 10px !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
        box-sizing: border-box !important;
        padding: 0 !important;
    }
    /* 下拉框内层文字输入留出箭头空间（数字/文字不与箭头重叠），
       并水平居中显示（音量/音高/码率等数字在框内居中）。 */
    #minimax-output-params-accordion #minimax-language input,
    #minimax-output-params-accordion #minimax-volume input,
    #minimax-output-params-accordion #minimax-pitch input,
    #minimax-output-params-accordion #minimax-bitrate input,
    #minimax-output-params-accordion #minimax-sample-rate input,
    #minimax-output-params-accordion #minimax-channel input {
        padding-right: 26px !important;
        text-align: center !important;
    }
    /* 语速滑块：数值输入框浅灰填充+细外框（与下拉按钮一致）；
       滑块轨道本身参考本地"多角色配音"的语速样式——
       细条（8px 高）、透明背景、胶囊圆角（28px），不被浅灰填充误伤。 */
    #minimax-output-params-accordion #minimax-speed input[type="number"] {
        background: var(--lemon-ctl) !important;
        border: 1px solid var(--lemon-line) !important;
        border-right: 0 !important;
        box-shadow: none !important;
        /* 左侧圆角、右侧方角：与右侧重置按钮拼合成一个整体圆角矩形 */
        border-radius: 10px 0 0 10px !important;
        /* 行高保持 Gradio 原样（head 容器 24px），不撑高行；
           数字水平居中：去掉步进箭头 + 零左右 padding + text-align center，
           垂直方向用 flex 居中。22px = 容器 24px - 上下 1px 边框，完全收在容器内。 */
        height: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        box-sizing: border-box !important;
        text-align: center !important;
        padding: 0 4px !important;
        line-height: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        -webkit-appearance: none !important;
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    /* 隐藏 number input 的步进箭头（否则数字被挤偏左） */
    #minimax-output-params-accordion #minimax-speed input[type="number"]::-webkit-outer-spin-button,
    #minimax-output-params-accordion #minimax-speed input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }
    /* 重置按钮：深色背景，与浅色输入框形成左右两半；
       只保留左侧深色分界线（中间一条线），去掉按钮自己的左右边框，
       避免"两条边线"视觉——按钮是无框深灰块，右侧圆角收尾。
       22px 与输入框一致，完全收在 head 容器内，内容垂直居中。 */
    #minimax-output-params-accordion #minimax-speed .reset-button {
        background: var(--lemon-sunken) !important;
        border: 0 !important;
        border-left: 1px solid rgb(24, 24, 28) !important;
        box-shadow: none !important;
        border-radius: 0 10px 10px 0 !important;
        height: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        padding: 0 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    /* 语速重置按钮 hover：深色下沉，照搬 CFG 引导强度重置按钮的 Gradio 默认逻辑
       （.reset-button:hover:not(:disabled) -> background-color: var(--background-fill-secondary) = #18181b） */
    #minimax-output-params-accordion #minimax-speed .reset-button:hover:not(:disabled) {
        background: var(--lemon-sunken-hover) !important;
    }
    /* 本地语速条数值控件：1:1 照抄云端「音频输出参数」语速条
       （数值框 63,63,70 左圆角 + 深色重置块 36,36,42 右圆角 + 中间深色分界线） */
    .wrap:has(input[aria-label*="语速调节"]) input[type="number"] {
        background: var(--lemon-ctl) !important;
        border: 1px solid var(--lemon-line) !important;
        border-right: 0 !important;
        box-shadow: none !important;
        border-radius: 10px 0 0 10px !important;
        height: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        box-sizing: border-box !important;
        text-align: center !important;
        padding: 0 4px !important;
        line-height: 24px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        -webkit-appearance: none !important;
        -moz-appearance: textfield !important;
        appearance: textfield !important;
    }
    .wrap:has(input[aria-label*="语速调节"]) input[type="number"]::-webkit-outer-spin-button,
    .wrap:has(input[aria-label*="语速调节"]) input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none !important;
        margin: 0 !important;
    }
    .wrap:has(input[aria-label*="语速调节"]) .reset-button {
        background: var(--lemon-sunken) !important;
        border: 0 !important;
        border-left: 1px solid rgb(24, 24, 28) !important;
        box-shadow: none !important;
        border-radius: 0 10px 10px 0 !important;
        height: 22px !important;
        min-height: 22px !important;
        max-height: 22px !important;
        box-sizing: border-box !important;
        margin: 0 !important;
        padding: 0 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .wrap:has(input[aria-label*="语速调节"]) .reset-button:hover:not(:disabled) {
        background: var(--lemon-sunken-hover) !important;
    }
    /* 所有重置小按钮：鼠标悬停时在按钮下方显示"重置"小浮标（字号与数值框一致 12px） */
    .reset-button {
        position: relative !important;
    }
    /* Gradio 默认给数字框+重置按钮的容器 overflow:hidden，会裁掉按钮下方的浮标；改为可见 */
    .tab-like-container {
        overflow: visible !important;
    }
    /* 多角色配音代码块复制按钮：hover 深色按下 + "复制"小浮标（与重置按钮逻辑一致）。
       注意：不要动 position——Gradio 默认 absolute 定位在代码块右上角，覆盖成 relative 会掉回文档流。 */
    #local-multirole-panel button.copy_code_button:hover:not(:disabled) {
        background-color: var(--lemon-sunken-hover) !important;
    }
    #local-multirole-panel button.copy_code_button:hover::after {
        content: "复制" !important;
        position: absolute !important;
        top: calc(100% + 3px) !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        line-height: 12px !important;
        color: rgb(198, 198, 204) !important;
        white-space: nowrap !important;
        pointer-events: none !important;
        z-index: 9999 !important;
    }
    .reset-button:hover::after {
        content: "重置" !important;
        position: absolute !important;
        top: calc(100% + 3px) !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        line-height: 12px !important;
        color: rgb(198, 198, 204) !important;
        white-space: nowrap !important;
        pointer-events: none !important;
        z-index: 9999 !important;
    }
    #minimax-output-params-accordion #minimax-speed input[type="range"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 28px !important;
        height: 8px !important;
        min-height: 8px !important;
        max-height: 8px !important;
        box-sizing: border-box !important;
        padding: 0 !important;
        margin-left: 0 !important;
        /* 滑块行自适应收缩：容器变窄时滑块让位给两端刻度，
           避免内容超出容器产生横向滚动条（语速底下滚动条问题）。 */
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }
    /* 语速两端刻度标签：不可收缩，保持固定。 */
    #minimax-output-params-accordion #minimax-speed .max_value {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
    }
    /* 最小值刻度隐藏：滑块轨道左端与"语速"标签左缘对齐（用户要求）。 */
    #minimax-output-params-accordion #minimax-speed .min_value {
        display: none !important;
    }
    /* 滑块行：0.5 刻度左缘对齐黄线（往左移），滑块轨道左端对齐红线。 */
    #minimax-output-params-accordion #minimax-speed .slider_input_container {
        padding-left: 0px !important;
        box-sizing: border-box !important;
    }
    #minimax-output-params-accordion #minimax-speed .min_value {
        margin-right: 20px !important;
    }
    /* 语速数值条往左：标签缩窄，数值条（数字框+重置按钮）左移占位，
       更贴近左侧；右侧与滑块保持对齐。 */
    #minimax-output-params-accordion #minimax-speed .head label {
        flex: 0 0 52px !important;
        min-width: 52px !important;
        max-width: 52px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    #minimax-output-params-accordion #minimax-speed .tab-like-container {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        justify-content: stretch !important;
    }
    /* 数值框拉伸占满容器：重置按钮贴到容器最右缘
       （与下方滑块"2"刻度的竖向垂直线对齐）。 */
    #minimax-output-params-accordion #minimax-speed .tab-like-container input {
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }
    /* 语言增强/音量/音高/MP3码率/采样率/声道：与语速一致——
       block 层深灰底 rgb(39,39,42)（同语速），
       控件本体细外框 1px #3F3F46（参考语气词下拉按钮的外框，很细）。
       注意：block 默认有 12px 左右 padding，导致浅灰按钮不撑满、
       视觉间距比 gap 大很多（实测 43px 而非 10px）——
       padding 收为 0，让按钮撑满，视觉间距 = form gap = 模型按钮间距。 */
    #minimax-output-params-accordion #minimax-language,
    #minimax-output-params-accordion #minimax-volume,
    #minimax-output-params-accordion #minimax-pitch,
    #minimax-output-params-accordion #minimax-bitrate,
    #minimax-output-params-accordion #minimax-sample-rate,
    #minimax-output-params-accordion #minimax-channel {
        background: var(--lemon-card) !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    #minimax-output-params-accordion #minimax-language .wrap,
    #minimax-output-params-accordion #minimax-volume .wrap,
    #minimax-output-params-accordion #minimax-pitch .wrap,
    #minimax-output-params-accordion #minimax-bitrate .wrap,
    #minimax-output-params-accordion #minimax-sample-rate .wrap,
    #minimax-output-params-accordion #minimax-channel .wrap {
        padding: 0 !important;
    }
    /* 高级卡按钮行：第一行「生成字幕 + 获取」各占半列宽；
       尺寸 33px 高、10px 圆角、选中翡翠绿、未选深灰 */
    #minimax-model-fetch-controls > .form {
        flex: 1 1 0% !important;
        align-self: center !important;
        min-width: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    #minimax-model-fetch-controls .form,
    #minimax-model-fetch-controls button {
        width: 100% !important;
        min-width: 0 !important;
        height: 33px !important;
        min-height: 33px !important;
        max-height: 33px !important;
        margin: 0 !important;
        box-sizing: border-box !important;
        border-radius: 10px !important;
    }
    #minimax-model-fetch-controls button.minimax-toggle-btn {
        padding: 0 14px !important;
        border: 0 !important;
        background: var(--lemon-ctl) !important;
        background-image: none !important;
        box-shadow: none !important;
        color: var(--lemon-on-ctl) !important;
        font-size: 13px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
    }
    #minimax-model-fetch-controls button.minimax-toggle-btn.minimax-toggle-on {
        /* 与同页的模型按钮统一，别再自己写一个绿。 */
        background: var(--checkbox-label-background-fill-selected, var(--button-primary-background-fill, linear-gradient(120deg, rgb(5, 150, 105) 0%, rgb(16, 185, 129) 60%, rgb(5, 150, 105) 100%))) !important;
        color: #ffffff !important;
    }
    /* 右列模块间距统一 15px（与左列标签间距一致）：
       克隆面板内部由 gap 控制（实测 15px）；
       结果面板内 gap 对部分子元素不生效，改用相邻兄弟 margin 兜底 */
    #minimax-clone-panel {
        gap: 15px !important;
    }
    #minimax-clone-panel > *,
    #minimax-clone-panel > * > * {
        margin: 0 !important;
    }
    #minimax-dubbing-result-panel {
        gap: 0 !important;
    }
    #minimax-dubbing-result-panel > * {
        margin: 0 !important;
    }
    #minimax-dubbing-result-panel > * + * {
        margin-top: 15px !important;
    }
    #minimax-dubbing-result-panel > * > * {
        margin: 0 !important;
    }
    /* 云端请求进行时，状态信息只能占用一行紧凑提示，不能把右侧结果区撑成可滚动大块。 */
    #minimax-status,
    #minimax-status > .prose,
    #minimax-status .prose {
        min-height: 0 !important;
        max-height: 48px !important;
        overflow: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 20px !important;
    }
    /* wrap 保留（事件载体），清掉 Gradio 默认视觉（边框/背景/阴影），视觉上只剩自建行 */
    #minimax-clone-audio .wrap,
    #minimax-srt-file .wrap {
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    /* ===== 本地配音页高级设置三卡片（对齐云端 模型/输出格式/获取ID 卡片） ===== */
    /* 常驻卡片：隐藏 Accordion 标题按钮（含箭头），内容永远展开显示 */
    #local-advanced-settings > button {
        display: none !important;
    }
    /* 黑河：accordion 容器+内容区 = 页面黑底，三卡片如船浮在黑底上；
       左右 0 内边距（覆盖 Gradio Accordion 默认 10px 12px），
       让三卡片左右边缘与「参考音频文本」左右边缘垂直对齐；
       border 0 去掉 Accordion 容器自带的细线范围框 */
    #local-advanced-settings {
        background: var(--lemon-page) !important;
        box-sizing: border-box !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 10px 0 !important;
        /* 禁止横向滚动条：Gradio Accordion 默认 overflow-x auto，
           内容微溢出会在容器底部（注释下方）出滚动条 */
        overflow-x: hidden !important;
    }
    /* 卡片内层 form 透明且无边框（去 Gradio 默认 0.67px 灰细线 + 外阴影，
       即用户所见的「内圈圆角细线」），只留卡片一层 */
    #local-advanced-toggles > .form,
    #local-advanced-speed > .form,
    #local-advanced-cfg > .form,
    #local-advanced-steps > .form {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    #local-advanced-layout,
    #local-advanced-layout > .form {
        gap: 15px !important;
        align-items: flex-start !important;
        flex-wrap: nowrap !important;
    }
    #local-advanced-toggles,
    #local-advanced-speed,
    #local-advanced-cfg,
    #local-advanced-steps {
        /* 与云端「模型 / 输出格式」卡同款：实心底 + 一像素边框。
           原来是写死的半透明毛玻璃（rgba(39,39,42,.72)），主题一变就还是灰的；
           毛玻璃本身也依赖深色背景才成立，浅色下只会糊成一团。 */
        background: var(--lemon-card) !important;
        border: 1px solid var(--lemon-line) !important;
        border-radius: 12px !important;
        box-sizing: border-box !important;
        display: flex !important;
        flex-direction: column !important;
        align-self: stretch !important;
        justify-content: flex-start !important;
        overflow: hidden !important;
        padding: 12px !important;
        /* 高光/暗边那一套是给深色底调的，浅色下会发脏；
           只留一层淡投影拉开卡片与页面的层次，明暗都成立。 */
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12) !important;
    }
    /* CFG / 步数卡内部组件块透明：去掉 Slider block 的不透明底衬，
       卡片只剩一层毛玻璃（照抄云端模型卡内部无衬底） */
    #local-advanced-cfg > .form > .block,
    #local-advanced-steps > .form > .block,
    #local-advanced-cfg > .form > .block .wrap,
    #local-advanced-steps > .form > .block .wrap {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* 功能开关卡（2×2 网格，列宽贴合内容）+ 语速卡收窄：不撑满 Row，给 CFG/步数让空间 */
    #local-advanced-toggles,
    #local-advanced-speed {
        flex: 0 1 auto !important;
        min-width: 0 !important;
        max-width: fit-content !important;
    }
    /* 卡片顶部空余处加标题「高级设置」；按钮组贴卡片下部 */
    #local-advanced-toggles::before {
        content: "高级设置" !important;
        display: block !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: var(--body-text-color) !important;
        white-space: nowrap !important;
        margin-bottom: 8px !important;
        line-height: 19.6px !important;
    }
    #local-advanced-toggles > .form {
        margin-top: auto !important;
        display: grid !important;
        grid-template-columns: repeat(2, auto) !important;
        column-gap: 10px !important;
        row-gap: 10px !important;
        align-content: start !important;
        justify-content: start !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }
    /* 语速卡收短：滑块宽度收窄（原 max-width 190 会裁掉滑块行——
       内容最小宽 169px，容器只有 142px，横向溢出被裁剪导致绿色轨道显示不全。
       220 = 169 内容 + 48 卡片 padding，仍比 CFG 卡（230）窄） */
    #local-advanced-speed {
        flex: 0 0 245px !important;
        max-width: 245px !important;
    }
    /* CFG 卡略收短，把第一行空间优先留给语速滑块。 */
    #local-advanced-cfg {
        flex: 0 0 205px !important;
        max-width: 205px !important;
    }
    /* 参数注释：卡片外侧黑色空余处的小字说明 */
    #local-advanced-info,
    #local-stability-tier-note {
        color: var(--body-text-color-subdued) !important;
        font-size: 12px !important;
        margin: 4px 2px 0 !important;
        padding: 0 !important;
        background: transparent !important;
        overflow-x: hidden !important;
    }
    #local-advanced-info p,
    #local-stability-tier-note p {
        margin: 0 !important;
    }
    #local-advanced-info .block,
    #local-advanced-info .prose,
    #local-stability-tier-note .block,
    #local-stability-tier-note .prose {
        overflow-x: hidden !important;
    }
    /* 音色稳定性档位：尺寸与云端「输出格式」卡一致，作为高级设置第二行的同排卡。 */
    #local-stability-tier-card {
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        min-height: 136px !important;
        margin: 0 !important;
        padding: 12px !important;
        box-sizing: border-box !important;
        background: var(--lemon-card) !important;
        border: 1px solid var(--lemon-line) !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        overflow: hidden !important;
    }
    #local-stability-tier-card > .form,
    #local-stability-tier-card .block,
    #local-stability-tier-card .wrap,
    #local-stability-tier-card .wrap-inner {
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        overflow: visible !important;
        overflow-y: visible !important;
        max-height: none !important;
        scrollbar-width: none !important;
    }
    #local-advanced-secondary-layout,
    #local-advanced-secondary-layout > .form {
        gap: 15px !important;
        align-items: stretch !important;
    }
    /* 左侧开关卡随按钮加高加宽，但只包住标题与 2×2 按钮组。 */
    #local-advanced-toggles {
        width: 250px !important;
        min-width: 250px !important;
        max-width: 250px !important;
        min-height: 136px !important;
    }
    #local-stability-tier-title,
    #local-stability-tier-title p {
        margin: 0 0 8px !important;
        padding-left: 6px !important;
        color: var(--body-text-color) !important;
        font-size: 14px !important;
        line-height: 19.6px !important;
    }
    #local-stability-tier-buttons .wrap {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 8px !important;
        align-items: stretch !important;
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
    }
    #local-stability-tier-buttons,
    #local-stability-tier-buttons .block,
    #local-stability-tier-buttons .form {
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
    }
    /* Gradio 6 在 Radio 的中间包装层保留 overflow-y:auto；仅对本 2×2 档位组
       展开全部包装层并隐藏滚动条，避免鼠标悬停时出现无意义的滚动轨道。 */
    #local-stability-tier-buttons *,
    #local-stability-tier-buttons .wrap-inner {
        overflow-y: visible !important;
        max-height: none !important;
        scrollbar-width: none !important;
    }
    #local-stability-tier-buttons *::-webkit-scrollbar {
        width: 0 !important;
        height: 0 !important;
        display: none !important;
    }
    #local-stability-tier-card *::-webkit-scrollbar {
        width: 0 !important;
        height: 0 !important;
        display: none !important;
    }
    #local-stability-tier-buttons .wrap > label {
        height: 33px !important;
        min-height: 33px !important;
        margin: 0 !important;
        padding: 6px 10px !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 6.5px !important;
        background-color: var(--lemon-ctl) !important;
        background-image: linear-gradient(120deg, var(--lemon-ctl) 0%, var(--lemon-ctl-mid) 60%, var(--lemon-ctl) 100%) !important;
        /* 亮色下按钮底与卡片底很接近，没有边框就只剩文字浮着 */
        border: 1px solid var(--lemon-line) !important;
        border-radius: 10px !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.09),
            inset 0 -1px 0 rgba(0, 0, 0, 0.28),
            0 1px 2px rgba(0, 0, 0, 0.35) !important;
        cursor: pointer !important;
    }
    /* 选中态的绿交给 Gradio 主题，与云端「模型」按钮同源。
       原来写死 rgb(5,150,105)，和云端那个绿不是一个颜色。 */
    #local-stability-tier-buttons .wrap > label:has(input:checked) {
        background: var(--checkbox-label-background-fill-selected, var(--button-primary-background-fill, linear-gradient(120deg, rgb(5, 150, 105) 0%, rgb(16, 185, 129) 60%, rgb(5, 150, 105) 100%))) !important;
    }
    #local-stability-tier-buttons input[type="radio"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        margin: 0 !important;
        border-radius: 9999px !important;
        background: var(--lemon-card) !important;
        border: 1px solid var(--lemon-line) !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
    }
    #local-stability-tier-buttons input[type="radio"]:checked {
        background: radial-gradient(circle at center,
            rgb(255, 255, 255) 0%,
            rgb(255, 255, 255) 36%,
            rgba(255, 255, 255, 0.9) 38%,
            rgba(255, 255, 255, 0.15) 43%,
            var(--button-primary-background-fill, rgb(5, 150, 105)) 46%) !important;
        /* 圆点描边也跟着主题走，否则底色统一了、描边还是旧绿。 */
        border-color: var(--checkbox-border-color-selected,
            var(--button-primary-border-color, rgb(16, 185, 129))) !important;
    }
    #local-stability-tier-buttons .label-wrap,
    #local-stability-tier-buttons .label-text {
        margin: 0 !important;
        color: var(--body-text-color) !important;
        font-size: 12px !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }
    /* 三张滑块卡：标题行纯文字（第一行），数字显示图块另起一行（第二行），
       滑动条第三行——head 换行 + 数值框占满第二行 */
    #local-advanced-speed .head,
    #local-advanced-cfg .head,
    #local-advanced-steps .head {
        flex-wrap: wrap !important;
        row-gap: 6px !important;
    }
    #local-advanced-speed .head .tab-like-container,
    #local-advanced-cfg .head .tab-like-container,
    #local-advanced-steps .head .tab-like-container {
        flex-basis: 100% !important;
        justify-content: flex-end !important;
    }
    /* 三张滑块卡的数字显示框加宽 10px（56 → 66） */
    #local-advanced-speed .head .tab-like-container input,
    #local-advanced-cfg .head .tab-like-container input,
    #local-advanced-steps .head .tab-like-container input {
        width: 66px !important;
        min-width: 66px !important;
    }
    /* 绿色滑块轨道在卡片内下移 10px（卡片高度固定不变，
       transform 不改变文档流/卡片高度） */
    #local-advanced-speed .slider_input_container,
    #local-advanced-cfg .slider_input_container,
    #local-advanced-steps .slider_input_container {
        transform: translateY(10px) !important;
    }
    /* 稳定性档位说明下方再保留少量呼吸空间。 */
    #page-local-dubbing #local-multirole-panel {
        margin-top: 10px !important;
    }
    /* 语速卡内禁止横向滚动条：block 只裁横向；竖向必须 visible——
       wrap 默认 visible 时滑块下移 10px 超出容器底边也能完整显示；
       若整层 overflow:hidden 会把滑块下半截裁掉（上下显示一半） */
    #local-advanced-speed .block {
        overflow-x: hidden !important;
        overflow-y: visible !important;
    }
    #local-advanced-speed .wrap,
    #local-advanced-speed .slider_input_container,
    #local-advanced-speed .head {
        overflow: visible !important;
    }
    /* 第一行三个参数卡等宽：语速 / CFG / 迭代步数共用可用宽度。 */
    #local-advanced-speed,
    #local-advanced-cfg,
    #local-advanced-steps {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        max-width: none !important;
    }
    /* 三张滑块卡标题与「高级设置」标题同一水平线：
       Gradio Slider block 自带 10px 顶部内边距（padded），
       去掉后 label 顶部 = 卡片 padding 顶（与 toggles ::before 对齐） */
    #local-advanced-speed .block,
    #local-advanced-cfg .block,
    #local-advanced-steps .block {
        padding-top: 0 !important;
    }
    /* 语速卡内部组件块透明：去掉 Slider block 底衬（与 CFG/步数一致） */
    #local-advanced-speed > .form > .block,
    #local-advanced-speed > .form > .block .wrap {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* 提交任务队列按钮：圆角矩形（照抄「下载当前生成音频」按钮样式） */
    #local-submit-button,
    #local-submit-button > button,
    #local-submit-button > a,
    #local-submit-button button,
    #local-submit-button a {
        width: 100% !important;
        min-height: 40px !important;
        height: 40px !important;
        max-height: 40px !important;
        flex: 0 0 40px !important;
        box-sizing: border-box !important;
        border-radius: 16px !important;
        margin: 12px 0 0 !important;
        overflow: hidden !important;
    }
    /* 功能开关卡：按钮 = 圆角矩形渐变按钮（1:1 照抄云端「输出格式」卡 MP3/wav 按钮：
       高 33 / 圆角 10 / padding 6x12 / 未选中深灰渐变 / 选中绿渐变 / 圆点 16px） */
    #local-advanced-toggles .local-radio-btn {
        height: 33px !important;
        min-height: 33px !important;
        padding: 6px 12px !important;
        background-color: var(--lemon-ctl) !important;
        background-image: linear-gradient(120deg, var(--lemon-ctl) 0%, var(--lemon-ctl-mid) 60%, var(--lemon-ctl) 100%) !important;
        border-radius: 10px !important;
        /* 亮色下按钮底与卡片底很接近，没有边框就只剩文字浮着 */
        border: 1px solid var(--lemon-line) !important;
        box-sizing: border-box !important;
        display: flex !important;
        align-items: center !important;
        cursor: pointer !important;
        overflow: visible !important;
        /* 立体感（非纸片）：顶部内高光 + 底部内暗 + 外投影 */
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.09),
            inset 0 -1px 0 rgba(0, 0, 0, 0.28),
            0 1px 2px rgba(0, 0, 0, 0.35) !important;
    }
    /* 同上：与云端按钮同源，避免同一界面里出现两个绿。 */
    #local-advanced-toggles .local-radio-btn:has(input:checked) {
        background: var(--checkbox-label-background-fill-selected, var(--button-primary-background-fill, linear-gradient(120deg, rgb(5, 150, 105) 0%, rgb(16, 185, 129) 60%, rgb(5, 150, 105) 100%))) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.16),
            inset 0 -1px 0 rgba(0, 0, 0, 0.22),
            0 1px 2px rgba(0, 0, 0, 0.35) !important;
    }
    #local-advanced-toggles .local-radio-btn .checkbox-container {
        display: flex !important;
        align-items: center !important;
        gap: 6.5px !important;
        width: 100% !important;
        margin: 0 !important;
        cursor: pointer !important;
    }
    /* 按钮内层 wrap 清背景（去掉底衬，按钮一层渐变底即可） */
    #local-advanced-toggles .local-radio-btn .wrap {
        background: transparent !important;
        border: 0 !important;
    }
    #local-advanced-toggles .local-radio-btn .label-text {
        font-size: 13px !important;
        color: var(--body-text-color) !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        /* 文字中心与圆点圆心同一水平线（垂直居中）；
           行高保持正常（line-height:1 会裁剪中文字形上下边缘） */
        line-height: normal !important;
        display: inline-flex !important;
        align-items: center !important;
    }
    #local-advanced-toggles .local-radio-btn input[type="checkbox"] {
        appearance: none !important;
        -webkit-appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        min-width: 16px !important;
        border-radius: 9999px !important;
        background: var(--lemon-card) !important;
        border: 1px solid var(--lemon-line) !important;
        margin: 0 !important;
        cursor: pointer !important;
        box-sizing: border-box !important;
        flex: 0 0 auto !important;
    }
    #local-advanced-toggles .local-radio-btn input[type="checkbox"]:checked {
        /* 选中态：绿色圆 + 中心白色实心圆点（照抄云端选中圆点——
           白心约 36%（≈5.8px，与云端模型卡白点同尺寸），
           边缘 36%→46% 平滑过渡，圆润不毛躁） */
        background: radial-gradient(circle at center,
            rgb(255, 255, 255) 0%,
            rgb(255, 255, 255) 36%,
            rgba(255, 255, 255, 0.9) 38%,
            rgba(255, 255, 255, 0.15) 43%,
            var(--button-primary-background-fill, rgb(5, 150, 105)) 46%) !important;
        /* 圆点描边也跟着主题走，否则底色统一了、描边还是旧绿。 */
        border-color: var(--checkbox-border-color-selected,
            var(--button-primary-border-color, rgb(16, 185, 129))) !important;
    }
    /* CFG / 迭代步数卡片：滑块收短（对齐云端语速滑块样式） */
    #local-advanced-cfg input[type="range"],
    #local-advanced-steps input[type="range"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        border-radius: 28px !important;
        height: 8px !important;
        min-height: 8px !important;
        max-height: 8px !important;
        box-sizing: border-box !important;
        padding: 0 !important;
        margin-left: 0 !important;
        flex: 1 1 auto !important;
        min-width: 0 !important;
        width: auto !important;
    }
    #local-advanced-cfg .max_value,
    #local-advanced-steps .max_value {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
    }
    #local-advanced-cfg .min_value,
    #local-advanced-steps .min_value {
        display: none !important;
    }
    #local-advanced-cfg .slider_input_container,
    #local-advanced-steps .slider_input_container {
        padding-left: 0px !important;
        box-sizing: border-box !important;
    }
    #local-advanced-cfg .head label,
    #local-advanced-steps .head label {
        flex: 0 0 104px !important;
        min-width: 104px !important;
        max-width: 104px !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }
    /* 本地 SRT 上传卡片：高度 120px（与云端一致），wrap 只作事件载体 */
    #local-srt-file,
    #local-srt-file .wrap,
    #local-srt-file .wrap-inner {
        height: 120px !important;
        min-height: 120px !important;
        max-height: 120px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    #local-srt-file .wrap {
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        outline: none !important;
    }
    /* SRT 字幕上传模块压缩为紧凑条（120px）；提示文字由前端 JS 横排成一行 */
    #minimax-srt-file,
    #minimax-srt-file .wrap,
    #minimax-srt-file .wrap-inner {
        height: 150px !important;
        min-height: 150px !important;
        max-height: 150px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    /* SRT 卡片最终高度 120px（用户指定；覆盖上面的 150px 与统一 153px 规则） */
    #minimax-srt-file,
    #minimax-srt-file .wrap,
    #minimax-srt-file .wrap-inner {
        height: 120px !important;
        min-height: 120px !important;
        max-height: 120px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    #minimax-srt-file .wrap {
        align-items: center !important;
        justify-content: center !important;
    }
    /* 本地参考音频原生空态：提示语横排，来源按钮即可在 153px 内完整显示。 */
    #local-reference-audio,
    #local-reference-audio .wrap,
    #local-reference-audio .wrap-inner {
        height: 153px !important;
        min-height: 153px !important;
        max-height: 153px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    /* 仅改变 Gradio 原生提示的排版，不移动/隐藏上传或录音按钮，
       从而保持 Audio 组件自身的文件上传事件不变。 */
    #local-reference-audio .wrap:has(> .icon-wrap) {
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        white-space: nowrap !important;
    }
    #local-reference-audio .wrap:has(> .icon-wrap) .or {
        margin: 0 !important;
    }
    #local-reference-audio .wrap:has(> .icon-wrap) .icon-wrap svg {
        width: 20px !important;
        height: 20px !important;
    }
    /* 原生来源行含其顶部边线一起上移，完整露出两个按钮并留下底部留白。 */
    #local-reference-audio .source-selection {
        transform: translateY(-14px) !important;
    }
    /* 本地参考音频使用 Gradio 原生上传交互；出现原生波形后才扩高，
       绝不通过 JS 移动或隐藏其上传按钮。 */
    #local-reference-audio:has(.waveform-container),
    #local-reference-audio:has(.waveform-container) .wrap,
    #local-reference-audio:has(.waveform-container) .wrap-inner {
        height: 250px !important;
        min-height: 250px !important;
        max-height: 250px !important;
        overflow: visible !important;
    }
    #local-srt-file[data-upload-state="uploaded"],
    #local-srt-file[data-upload-state="uploaded"] .wrap,
    #local-srt-file[data-upload-state="uploaded"] .wrap-inner {
        overflow: visible !important;
    }
    #page-local-dubbing #local-dubbing-audio[data-result-state="uploaded"],
    #page-local-dubbing #local-dubbing-audio[data-result-state="uploaded"].block,
    #page-local-dubbing #local-dubbing-audio[data-result-state="uploaded"] .block,
    #page-local-dubbing #local-dubbing-audio[data-result-state="uploaded"] .component-wrapper,
    #local-design-audio[data-result-state="uploaded"],
    #local-design-audio[data-result-state="uploaded"].block,
    #local-design-audio[data-result-state="uploaded"] .block,
    #local-design-audio[data-result-state="uploaded"] .component-wrapper {
        height: 250px !important;
        min-height: 250px !important;
        max-height: 250px !important;
    }
    /* 结果播放器由 Gradio/WaveSurfer 绘出波形画布即代表已有音频。
       直接以该可见 DOM 状态扩高，不依赖 Electron 中可能缺失的 audio[src]。 */
    #page-local-dubbing #local-dubbing-audio:has(.waveform-container),
    #page-local-dubbing #local-dubbing-audio:has(.waveform-container).block,
    #page-local-dubbing #local-dubbing-audio:has(.waveform-container) .block,
    #page-local-dubbing #local-dubbing-audio:has(.waveform-container) .component-wrapper,
    #local-design-audio:has(.waveform-container),
    #local-design-audio:has(.waveform-container).block,
    #local-design-audio:has(.waveform-container) .block,
    #local-design-audio:has(.waveform-container) .component-wrapper {
        height: 250px !important;
        min-height: 250px !important;
        max-height: 250px !important;
    }
    /* 右列参考音频/结果模块高度统一：代码 153px，150% DPI 屏幕实际渲染 ≈230 物理像素。
       右列顶边线与左列 API Key 卡片由 JS 动态对齐；SRT 卡片单独 120px（见上） */
    #minimax-clone-audio,
    #minimax-clone-audio .wrap,
    #minimax-clone-audio .wrap-inner,
    #minimax-dubbing-audio,
    #minimax-dubbing-audio .wrap,
    #minimax-dubbing-audio .wrap-inner,
    #minimax-design-audio,
    #minimax-design-audio .wrap,
    #minimax-design-audio .wrap-inner {
        height: 153px !important;
        min-height: 153px !important;
        max-height: 153px !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }
    /* 云端三种播放器沿用本地播放器标准：10px 圆角、细灰边框、微阴影。 */
    #minimax-clone-audio,
    #minimax-clone-audio.block,
    #minimax-dubbing-audio,
    #minimax-dubbing-audio.block,
    #minimax-design-audio,
    #minimax-design-audio.block {
        border-radius: 10px !important;
        border: 0.667px solid var(--lemon-line) !important;
        box-shadow: rgba(0, 0, 0, 0.1) 0px 1px 3px 0px,
            rgba(0, 0, 0, 0.1) 0px 1px 2px -1px !important;
    }
    #minimax-clone-audio .component-wrapper,
    #minimax-dubbing-audio .component-wrapper,
    #minimax-design-audio .component-wrapper {
        min-height: 153px !important;
        box-sizing: border-box !important;
    }
    /* 参考音频上传、或云端结果出现 WaveSurfer 波形后，扩为完整 250px 播放器。 */
    #minimax-clone-audio[data-upload-state="uploaded"],
    #minimax-clone-audio[data-upload-state="uploaded"] .wrap,
    #minimax-clone-audio[data-upload-state="uploaded"] .wrap-inner,
    #minimax-clone-audio[data-upload-state="uploaded"] .component-wrapper,
    #minimax-dubbing-audio[data-result-state="uploaded"],
    #minimax-dubbing-audio[data-result-state="uploaded"] .wrap,
    #minimax-dubbing-audio[data-result-state="uploaded"] .wrap-inner,
    #minimax-dubbing-audio[data-result-state="uploaded"] .component-wrapper,
    #minimax-design-audio[data-result-state="uploaded"],
    #minimax-design-audio[data-result-state="uploaded"] .wrap,
    #minimax-design-audio[data-result-state="uploaded"] .wrap-inner,
    #minimax-design-audio[data-result-state="uploaded"] .component-wrapper,
    #minimax-clone-audio:has(.waveform-container),
    #minimax-dubbing-audio:has(.waveform-container),
    #minimax-design-audio:has(.waveform-container) {
        height: 250px !important;
        min-height: 250px !important;
        max-height: 250px !important;
        overflow: visible !important;
    }
    /* 右列顶部与左列「MiniMax API Key」模块顶线对齐：
       CSS 提供初始值，前端 JS 按实际布局动态微调（内联样式覆盖此值） */
    #minimax-result-column {
        margin-top: 117px;
    }
    /* 参考音频模块内（大按钮已由前端隐藏），小按钮行紧凑贴左 */
    #minimax-clone-audio .source-selection {
        margin: 0 !important;
        display: flex !important;
        gap: 6px !important;
    }
    /* 试听文本：内容不超出时隐藏滚动条，超出才显示 */
    #minimax-clone-preview-input textarea {
        overflow-y: auto !important;
    }
    /* 云端两个子页始终挂载各自的播放器；仅在浏览器端切换显示，
       防止异步请求或重启后因为服务端可见性状态而丢失播放器。 */
    #minimax-design-result-panel { display: none; }
    /* 音色设计标签与文字保留呼吸空间，避免贴边。 */
    /* 常用情绪卡片底色与"试听文本"卡片一致：#27272A（Gradio 默认卡底色） */
    #local-design-tags {
        padding: 0 !important;
        background: var(--lemon-card) !important;
    }
    #local-design-tags .gr-group,
    #local-design-tags > .gr-group {
        background: var(--lemon-card) !important;
    }
    /* 折叠卡内容容器：取消旧 Group 的负边距，避免 Accordion 内产生横向滚动条。 */
    #local-design-tags .styler {
        margin-left: 0 !important;
        margin-right: 0 !important;
        border-radius: 10px !important;
        background: transparent !important;
        overflow-x: hidden !important;
    }
    /* Gradio Accordion 的滚动层不在 .styler，而在 form / wrap 包装链；
       折叠时也强制禁止横向滚动，展开内容仍可按行自然换行。 */
    #local-design-tags,
    #local-design-tags > .form,
    #local-design-tags .form,
    #local-design-tags .wrap,
    #local-design-tags .wrap-inner {
        max-width: 100% !important;
        overflow-x: hidden !important;
    }
    /* 展开时额外保留 16px 底部空间，并取消内容层的竖向滚动/高度截断。 */
    #local-design-tags[data-emotion-expanded="true"] .form,
    #local-design-tags[data-emotion-expanded="true"] .wrap,
    #local-design-tags[data-emotion-expanded="true"] .wrap-inner {
        max-height: none !important;
        overflow-y: visible !important;
    }
    #local-design-tags[data-emotion-expanded="true"] .form {
        padding-bottom: 16px !important;
    }
    /* 余白加在 Accordion 外卡本身，确保最后一排标签与底边框保持可见距离。 */
    #local-design-tags[data-emotion-expanded="true"] {
        padding-bottom: 16px !important;
    }
    /* Accordion 标题完全沿用下方「高级设置」：透明内层、相同文字/箭头对齐。
       情绪标签的浅灰按钮样式只作用于展开后的内容，不能覆盖标题按钮。 */
    #local-design-tags > button {
        min-height: 40px !important;
        height: 40px !important;
        box-sizing: border-box !important;
        padding: 0 31px 0 20px !important;
        align-items: center !important;
        border-radius: unset !important;
        background: transparent !important;
        background-image: none !important;
        border: 0 !important;
        box-shadow: none !important;
    }
    /* 情绪标签按钮：圆角矩形 12px（与其他圆角按钮一致）、浅灰底（与抽卡未选中按钮同色）；
       行列间距：行距 8px（=输出格式卡 mp3/wav 按钮上下间距），列距 6px（用户要求 5~8px） */
    #local-design-tags .row button {
        min-height: 36px !important;
        padding: 7px 14px !important;
        border-radius: 12px !important;
        /* 亮色下这个底色与卡片底几乎同色，原来 border:0 + 无阴影，
           结果就是「只看得见字、看不见按钮」。补边框与淡投影。 */
        background: var(--lemon-ctl-hi) !important;
        background-image: none !important;
        border: 1px solid var(--lemon-line) !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
    }
    #local-design-tags .row {
        gap: 8px 6px !important;
        row-gap: 8px !important;
        column-gap: 6px !important;
        width: calc(100% - 36px) !important;
        margin-left: 18px !important;
        margin-right: 18px !important;
        box-sizing: border-box !important;
    }
    /* 与情绪标签标题箭头共用同一右侧竖线；仅右移内边距，不移动标题文字。 */
    #local-design-advanced-settings > button {
        padding-right: 20px !important;
    }
    """
    page_css = custom_css + STATUS_CSS
    blocks_kwargs = {"title": "YZY启动器"}
    if not _LAUNCH_SUPPORTS_CSS:
        blocks_kwargs["css"] = page_css

    with gr.Blocks(**blocks_kwargs) as interface:

        # 顶部状态栏
        brand_name = gr.State("VoxCPM 2.0") # 每个项目都不同
        brand_url = gr.State("https://github.com/OpenBMB/VoxCPM") # 每个项目都不同
        status = gr.HTML(initial_status(), elem_id="yzystatus-bar", container=False)
        # 状态栏每 2 秒刷新一次，属于轻量 UI 更新，不应占用队列并发槽。
        gr.Timer(2).tick(
            refresh_status, inputs=[brand_name, brand_url], outputs=status,
            queue=False,
        )
        # Gradio 6.18 对无后端 fn 的 Blocks.load(js=...) 不会稳定触发；绑定无副作用函数，
        # 让页面初始化脚本在每次完整加载时可靠执行。
        interface.load(
            lambda: None,
            inputs=[],
            outputs=[],
            queue=False,
            api_name=False,
            js="() => { (" + STATUS_JS + ")(); (" + MINIMAX_DESIGN_COUNTER_JS + ")(); (" + LOCAL_TEXTAREA_AUTOGROW_JS + ")(); (" + LOCAL_REFERENCE_AUDIO_TOOLTIP_JS + ")(); (" + LOCAL_RESET_TOOLTIP_JS + ")(); (" + LOCAL_MULTIROLE_COPY_JS + ")(); (" + LOCAL_PRESET_DELETE_JS + ")(); (" + MINIMAX_SAVED_VOICE_DELETE_JS + ")(); (" + MINIMAX_TEXT_SELECTION_JS + ")(); (" + MINIMAX_RESULT_PANEL_JS + ")(); (" + MINIMAX_UPLOAD_TEXT_JS + ")(); }",
        )

        # 首次启动即加载 outputs 目录下已有的历史记录
        history_state = gr.State(task_manager.snapshot()["history"])
        rev_state = gr.State(-1)  # 记录上一次轮询到的队列版本号
        hist_page_state = gr.State(0)        # 生成历史当前页码（从 0 开始）
        hist_expanded_state = gr.State([])   # 已点击「播放」展开（懒加载）的音频路径列表

        with gr.Tabs():
            # ===== Tab 1：语音生成 =====
            with gr.Tab("本地配音", elem_id="page-local-dubbing"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2) as local_left_column:
                        text = gr.Textbox(
                            value="",
                            label="待生成文本（本地生成，使用右侧参考音频）",
                            lines=8,
                            max_lines=8,
                            show_label=True,
                            placeholder="输入待转换文本...",
                            elem_id="local-dubbing-text",
                            autofocus=True,
                        )
                        # SRT 模式下「目标文本」会切换为此字幕文件上传框
                        srt_file = gr.File(
                            label="上传 SRT 字幕文件",
                            file_types=[".srt"],
                            visible=False,
                        )
                        asr_status = gr.Markdown(value="", visible=False) # 语音识别状态提示
                        prompt_text = gr.Textbox(
                            value="",
                            show_label=True,
                            label="参考音频文本（仅供本地克隆校验，不参与生成）",
                            placeholder="自动识别参考音频文本填充，如果有错，请手动修正......",
                            lines=8,
                            max_lines=8,
                            elem_id="local-reference-text",
                            visible=True,
                        )
                        with gr.Row():
                            # SRT 字幕配音已改为右列「上传 SRT 字幕文件」卡片触发（local_srt_file），
                            # 此复选框隐藏保留（事件/联动引用），不再展示
                            srt_cb = gr.Checkbox(value=False, elem_classes="small_checkbox", label="SRT字幕配音", visible=False)

                    with gr.Column(scale=1, elem_id="local-right-column"):
                        preset_dropdown = gr.Dropdown(
                            label="预设参考音色音频",
                            show_label=False,
                            choices=[PRESET_PLACEHOLDER] + preset_voice_choices(),
                            value=PRESET_PLACEHOLDER,
                            elem_id="local-preset-voice-dropdown",
                        )
                        delete_voice_name_box = gr.Textbox(
                            value="",
                            visible=True,
                            elem_id="local-delete-voice-name",
                        )
                        delete_voice_btn = gr.Button(
                            "删除",
                            visible=True,
                            elem_id="local-delete-voice-button",
                        )
                        reference_wav = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="参考音频（必填 – 上传后用于克隆）",
                            elem_id="local-reference-audio",
                        )
                        save_name_box = gr.Textbox(
                            label="保存音色", placeholder="输入音色名称", show_label=False,
                            lines=1,
                            max_lines=1,
                            elem_id="local-voice-name-input",
                        )
                        save_voice_btn = gr.Button("💎 保存当前音色", elem_id="local-save-voice-button")

                        # 上传 SRT 字幕文件卡片：1:1 复刻云端配音页的 SRT 卡片
                        # （高度 120px、单行提示、样式与功能一致）
                        local_srt_file = gr.File(
                            label="上传 SRT 字幕文件",
                            file_types=[".srt"],
                            visible=True,
                            elem_id="local-srt-file",
                        )

                        # 提交任务队列按钮：从左侧模块移到右侧（SRT 上传下方），
                        # 样式与「下载当前生成音频」一致（圆角矩形）
                        run_btn = gr.Button(
                            "⚡ 提交到任务队列",
                            variant="primary",
                            elem_id="local-submit-button",
                        )

                        # 试听结果播放器保持与“本地音色设计”页面相同的右侧播放器结构。
                        # 用 gr.Column（照抄云端 #minimax-dubbing-result-panel）——
                        # Group 会带 .styler 灰底把播放器与按钮之间染灰，Column 不会。
                        with gr.Column(elem_id="local-generated-result-panel"):
                            audio_output = gr.Audio(
                                label="试听结果",
                                elem_id="local-dubbing-audio",
                                elem_classes="local-preview-player",
                            )
                            local_dubbing_download = gr.DownloadButton(
                                "下载当前生成音频",
                                visible=True,
                                interactive=False,
                                variant="primary",
                                size="lg",
                                elem_id="local-dubbing-audio-download",
                            )
                            subtitle_output = gr.File(label="字幕文件下载", visible=False)

                        # 任务结果紧跟在生成结果播放器下方，保持同一右列宽度。
                        with gr.Accordion(label="🎧 *任务&结果*", visible=True, elem_id="local-task-results-panel"):
                            queue_summary = gr.Markdown(
                                value=task_manager.snapshot()["summary_md"]
                            )
                            queue_table = gr.Markdown(
                                value=task_manager.snapshot()["table_md"],
                                elem_classes=["queue-table"],
                            )
                            gr.Markdown("", height=8) # 和下面的两个按钮拉开间隔
                            with gr.Row():
                                clear_tasks_btn = gr.Button("✨ 清空所有任务", size="sm")
                                stop_task_btn = gr.Button("❎ 取消所有排队任务", size="sm")

                with local_left_column, gr.Accordion(
                    label="🎭 *多角色配音*", visible=True, open=False,
                    elem_id="local-multirole-panel",
                ) as multi_role_accordion:
                    gr.Markdown(
                        "**目标文本格式：** 每行一个角色，开头用 `[角色名]:` 标记，例如：\n\n"
                        "```\n[角色1]: 你好，很高兴认识你。\n[角色2]: 我也是，欢迎欢迎！\n```\n\n"
                    )
                    role_components = []  # [(name, ref_dropdown, speed), ...]
                    _preset_choices = [PRESET_PLACEHOLDER] + preset_voice_choices()
                    for i in range(MAX_ROLES):
                        with gr.Row(equal_height=False):
                            r_name = gr.Textbox(show_label=False, scale=2, placeholder=f"角色{i + 1}")
                            r_ref = gr.Dropdown(
                                label="参考音频",
                                choices=_preset_choices,
                                value=PRESET_PLACEHOLDER,
                                scale=3,
                                show_label=False,
                            )
                            r_speed = gr.Slider(
                                label="语速", value=1.0, step=0.05, maximum=1.5, minimum=0.5, scale=2
                            )
                        role_components.append((r_name, r_ref, r_speed))

                with local_left_column, gr.Accordion("📚 **高级设置**", open=True, elem_id="local-advanced-settings"):
                    # 横向四卡片：功能开关卡（2×2）+ 语速卡 + CFG 卡 + 迭代步数卡
                    with gr.Row(equal_height=True, elem_id="local-advanced-layout"):
                        with gr.Column(scale=4, min_width=300, elem_id="local-advanced-speed"):
                            speed_slider = gr.Slider(
                                label="语速调节", value=1.0, step=0.05, maximum=1.5, minimum=0.5
                            )
                        with gr.Column(scale=2, min_width=110, elem_id="local-advanced-cfg"):
                            cfg_value = gr.Slider(
                                minimum=1.0,
                                maximum=3.0,
                                value=1.6,
                                step=0.1,
                                label="CFG（引导强度）",
                            )
                        with gr.Column(scale=3, min_width=180, elem_id="local-advanced-steps"):
                            dit_steps = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=15,
                                step=1,
                                label="生成迭代步数",
                            )
                    # 参数注释：在四张卡片下方另起一行（不占卡片行空间）
                    gr.Markdown(
                        "CFG：数值越高越贴合提示/参考音色，数值越低风格更自由 　|　 "
                        "迭代步数：步数越多音质可能更好，但速度更慢",
                        elem_id="local-advanced-info",
                    )
                    # 第二行：两个同宽的独立卡片，避免挤压上方语速 / CFG / 步数参数卡。
                    with gr.Row(equal_height=False, elem_id="local-advanced-secondary-layout"):
                        with gr.Column(scale=0, min_width=216, elem_id="local-advanced-toggles"):
                            DoDenoisePromptAudio = gr.Checkbox(
                                value=False,
                                label="音频降噪",
                                elem_classes=["local-radio-btn"],
                            )
                            DoNormalizeText = gr.Checkbox(
                                value=False,
                                label="文本规范化",
                                elem_classes=["local-radio-btn"],
                            )
                            gen_subtitle_cb = gr.Checkbox(
                                value=False,
                                label="生成字幕",
                                elem_classes=["local-radio-btn"],
                            )
                            multi_role_cb = gr.Checkbox(
                                value=False,
                                label="多角色配音",
                                elem_classes=["local-radio-btn"],
                            )
                        with gr.Column(scale=0, min_width=216, elem_id="local-stability-tier-card"):
                            gr.Markdown("音色稳定性档位", elem_id="local-stability-tier-title")
                            segment_limit_radio = gr.Radio(
                                choices=[("70 字", 70), ("100 字", 100), ("120 字", 120), ("150 字", 150)],
                                value=70,
                                show_label=False,
                                elem_id="local-stability-tier-buttons",
                            )
                    gr.Markdown(
                        "音色稳定性：70 字最稳；100/120 字适合连贯长句；150 字仅建议显存充足时使用。",
                        elem_id="local-stability-tier-note",
                    )

            # ===== Tab：声音设计（无需参考音频，通过描述从零生成音色） =====
            with gr.Tab("本地音色设计", elem_id="page-local-voice-design"):
                gr.Markdown(
                    "🎨 **声音设计**：无需参考音频，通过文字描述目标音色，从零创造声音。\n\n"
                    "👉 **声音设计满意后，请下载音频到「配音」Tab 进行配音。**"
                )
                # 右侧原生播放器保持紧凑高度，不随左侧文本与标签区域拉伸。
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        design_desc = gr.Textbox(
                            label="音色描述（本地声音设计：用文字描述目标音色）",
                            lines=8,
                            max_lines=8,
                            elem_id="local-design-description",
                            placeholder="如：年轻女性，温柔甜美 / 暴躁老哥，语速飞快 / 兴奋，语速快",
                        )
                        design_text = gr.Textbox(
                            label="试听文本（本地声音设计的试听）",
                            lines=8,
                            max_lines=8,
                            elem_id="local-design-text",
                            placeholder="输入一段用于试听的文本……",
                            info="文本不要太长，生成的语音在 10 秒以内效果最好。",
                        )
                        with gr.Accordion(
                            "📚 常用情绪 / 风格标签（点击填入上方「音色描述」）",
                            open=False,
                            elem_id="local-design-tags",
                        ):
                            design_tag_buttons = []
                            with gr.Row():
                                for _dtag in EMOTION_TAGS:
                                    _dbtn = gr.Button(_dtag, size="sm")
                                    design_tag_buttons.append((_dbtn, _dtag))
                        with gr.Accordion(
                            "📚 高级设置", open=False,
                            elem_id="local-design-advanced-settings",
                        ):
                            design_cfg = gr.Slider(
                                minimum=1.0, maximum=3.0, value=2.6, step=0.1,
                                label="CFG（引导强度）",
                                info="数值越高越贴合描述；数值越低生成风格更自由",
                            )
                            design_steps = gr.Slider(
                                minimum=1, maximum=50, value=15, step=1,
                                label="生成迭代步数",
                                info="步数越多音质可能更好，但速度更慢",
                            )
                            design_normalize = gr.Checkbox(
                                value=False, label="文本规范化",
                                info="自动规范化数字、日期及缩写",
                            )
                        design_run_btn = gr.Button("生成", variant="primary")
                    with gr.Column(scale=1):
                        design_audio = gr.Audio(
                            label="试听结果",
                            elem_id="local-design-audio",
                            elem_classes="local-preview-player",
                        )
                        local_design_download = gr.DownloadButton(
                            "下载当前试听音频",
                            visible=True,
                            interactive=False,
                            variant="primary",
                            size="lg",
                            elem_id="local-design-audio-download",
                        )
                        gr.Markdown(
                            "✅ 满意后，点击下方“下载当前试听音频”按钮保存音频，"
                            "再到「配音」Tab 上传为参考音频进行配音。"
                        )

                # 情绪 / 风格标签：点击填入音色描述框
                for _dbtn, _dtag in design_tag_buttons:
                    _dbtn.click(
                        fn=lambda cur, t=_dtag: _append_tag(t, cur),
                        inputs=[design_desc],
                        outputs=[design_desc],
                        queue=False,
                        api_name=False,
                    )

                # 同步生成（不入队列、不保存到 outputs、不计入配音历史）
                design_run_btn.click(
                    fn=_design_generate_with_download,
                    inputs=[design_text, design_desc, design_cfg,
                            design_steps, design_normalize],
                    outputs=[design_audio, local_design_download],
                )

            # ===== Tab：MiniMax 云端 TTS（独立服务商，不影响本地 VoxCPM 队列） =====
            with gr.Tab("MiniMax 云端 TTS"):
                minimax_saved_key_hint = (
                    "已检测到当前 Windows 用户保存的 DPAPI 密钥：可将输入框留空后生成。"
                    if has_saved_api_key()
                    else "尚未保存密钥：可本次直接使用，或粘贴后点击下方安全保存。"
                )
                gr.Markdown(
                    "### MiniMax 云端语音工作台\n\n"
                    "先在“云端音色设计”试听并确认，再保存到本机音色库；随后会自动带入“云端配音”。"
                    "云端配音与云端音色设计各有独立的结果播放器与下载条，所有音频均保存到本地 `整合包\\win-unpacked\\python\\outputs` 目录。"
                )
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2):
                        with gr.Tabs(selected="cloud_dubbing") as minimax_cloud_tabs:
                            with gr.Tab(
                                "云端配音", id="cloud_dubbing",
                                elem_id="page-minimax-dubbing",
                            ) as minimax_dubbing_tab:
                                gr.Markdown("云端配音结果固定显示在右侧播放器；未生成时下载按钮为灰色。")
                                with gr.Group(elem_id="minimax-api-key-group"):
                                    with gr.Row(equal_height=False):
                                        minimax_api_key = gr.Textbox(
                                            label="MiniMax API Key（可选：留空时使用已安全保存的密钥）",
                                            type="password",
                                            placeholder="首次粘贴后点击“安全保存”；保存后下次可留空直接生成",
                                            elem_id="minimax-api-key",
                                            min_width=0,
                                            scale=7,
                                        )
                                        minimax_save_key_btn = gr.Button(
                                            "保存 API Key", variant="primary", size="sm",
                                            min_width=150, scale=0, elem_id="minimax-save-api-key",
                                        )
                                minimax_fetched_voice_choices_state = gr.State(value=[])
                                minimax_saved_voice_choices_state = gr.State(value=saved_voice_choices())
                                with gr.Group(elem_id="minimax-voice-select-group"):
                                    with gr.Row(equal_height=True):
                                        minimax_fetched_voice_id = gr.Dropdown(
                                            choices=[],
                                            value=None,
                                            allow_custom_value=False,
                                            filterable=True,
                                            label="已获取音色（即时匹配）",
                                            min_width=250,
                                            scale=0,
                                            elem_id="minimax-fetched-voice-id",
                                        )
                                        minimax_saved_voice_id = gr.Dropdown(
                                            choices=saved_voice_choices(),
                                            value=None,
                                            allow_custom_value=False,
                                            label="已保存音色",
                                            min_width=164,
                                            scale=0,
                                            elem_id="minimax-saved-voice-id",
                                        )
                                        with gr.Column(
                                            min_width=144,
                                            scale=0,
                                            elem_id="minimax-manual-voice-wrap",
                                        ):
                                            minimax_manual_voice_id = gr.Textbox(
                                                label="自定义 voice_id",
                                                placeholder="填写后优先使用",
                                                lines=1,
                                                max_lines=1,
                                                elem_id="minimax-manual-voice-id",
                                            )
                                        minimax_fetch_voices_btn = gr.Button(
                                            "获取", variant="primary", size="sm", min_width=122, scale=0,
                                            elem_id="minimax-fetch-voices",
                                        )
                                # 下拉行内“×”把待删除音色写入该输入框并触发 input 事件；
                                # 控件保持可渲染但由 CSS 移到屏幕外，确保事件桥接有效。
                                minimax_delete_voice_name = gr.Textbox(
                                    value="",
                                    visible=True,
                                    elem_id="minimax-delete-voice-name",
                                )
                                minimax_prompt_text = gr.Textbox(
                                    value="",
                                    show_label=True,
                                    label="云端克隆参考音频文本（仅供克隆校验，不参与生成）",
                                    info="仅填写参考音频的逐字内容，用于克隆校验；不要在此加入情绪词、语气词或停顿标签。",
                                    placeholder="上传克隆音频后自动识别填充，如有错请手动修正；克隆时用于校验与提升相似度",
                                    lines=8,
                                    max_lines=8,
                                    elem_id="minimax-reference-text",
                                    visible=True,
                                )
                                minimax_text = gr.Textbox(
                                    label=MINIMAX_TEXT_LABEL,
                                    lines=8,
                                    max_lines=8,
                                    placeholder="输入待合成文本；单次最多 10,000 个输入字符。上传 SRT 后此处显示字幕预览。",
                                    elem_id="minimax-dubbing-text",
                                )
                                minimax_dubbing_text_counter = gr.Markdown(
                                    character_counter_markdown("", 10000, "待生成文本"),
                                    elem_id="minimax-dubbing-text-counter",
                                )
                                minimax_clone_preview = gr.Textbox(
                                    label="试听文本（上传参考音频克隆完成后的试听）",
                                    placeholder="克隆完成后用该文本生成试听音频（可留空，不试听）",
                                    show_label=True,
                                    lines=5,
                                    max_lines=5,
                                    value="大家好，这是新克隆音色的试听。",
                                    elem_id="minimax-clone-preview-input",
                                )
                                minimax_model_choices_state = gr.State(value=[
                                    "speech-2.8-turbo", "speech-2.8-hd",
                                    "speech-2.6-turbo", "speech-2.6-hd",
                                ])
                                with gr.Row(equal_height=True, elem_id="minimax-model-layout"):
                                    with gr.Column(scale=4, min_width=360, elem_id="minimax-model-group"):
                                        minimax_model = gr.Radio(
                                            choices=["speech-2.8-turbo", "speech-2.8-hd", "speech-2.6-turbo", "speech-2.6-hd"],
                                            value="speech-2.8-turbo",
                                            label="模型",
                                        )
                                    with gr.Column(scale=2, min_width=180, elem_id="minimax-output-format-group"):
                                        minimax_format = gr.Radio(
                                            choices=["mp3", "wav", "flac"], value="mp3", label="输出格式"
                                        )
                                    with gr.Column(scale=3, min_width=240, elem_id="minimax-model-fetch-group"):
                                        gr.Markdown("高级：获取其他模型ID", elem_id="minimax-model-fetch-title")
                                        # 第一行「生成字幕 + 获取」各占半列宽（SRT 模式由右列上传文件自动触发）
                                        with gr.Row(equal_height=False, elem_id="minimax-model-fetch-controls"):
                                            minimax_gen_subtitle_btn = gr.Button(
                                                "生成字幕", size="sm", scale=1,
                                                elem_id="minimax-gen-subtitle-toggle",
                                                elem_classes=["minimax-toggle-btn"],
                                            )
                                            minimax_fetch_models_btn = gr.Button(
                                                "获取", variant="primary", size="sm", scale=1,
                                                elem_id="minimax-fetch-models",
                                            )
                                        minimax_gen_subtitle_state = gr.State(False)
                                with gr.Accordion(
                                    "高级：情绪、断句与语气词", open=False,
                                    elem_id="minimax-expression-accordion",
                                ):
                                    with gr.Row(equal_height=False, elem_id="minimax-expression-controls"):
                                        minimax_emotion = gr.Dropdown(
                                            choices=[
                                                ("自动", ""),
                                                ("开心", "happy"), ("悲伤", "sad"), ("愤怒", "angry"),
                                                ("恐惧", "fearful"), ("厌恶", "disgusted"),
                                                ("惊讶", "surprised"), ("平静", "calm"),
                                            ],
                                            value="",
                                            label="情绪风格（整段 API 参数）",
                                            min_width=150,
                                            scale=0,
                                            elem_id="minimax-emotion",
                                        )
                                        minimax_expression = gr.Dropdown(
                                            choices=[
                                                ("无", ""),
                                                ("笑声", "(laughs)"), ("轻笑", "(chuckle)"), ("咳嗽", "(coughs)"),
                                                ("清嗓子", "(clear-throat)"), ("呻吟", "(groans)"), ("换气", "(breath)"),
                                                ("喘气", "(pant)"), ("吸气", "(inhale)"), ("呼气", "(exhale)"),
                                                ("倒吸气", "(gasps)"), ("吸鼻子", "(sniffs)"), ("叹气", "(sighs)"),
                                                ("喷鼻息", "(snorts)"), ("打嗝", "(burps)"), ("咂嘴", "(lip-smacking)"),
                                                ("哼唱", "(humming)"), ("嘶嘶声", "(hissing)"), ("嗯", "(emm)"), ("喷嚏", "(sneezes)"),
                                            ],
                                            value="",
                                            label="语气词（仅 2.8 模型）",
                                            min_width=175,
                                            scale=0,
                                            elem_id="minimax-expression",
                                        )
                                        minimax_pause_seconds = gr.Number(
                                            value=0.5,
                                            minimum=0.01,
                                            maximum=99.99,
                                            precision=2,
                                            label="停顿时长（秒）",
                                            min_width=150,
                                            scale=0,
                                            elem_id="minimax-pause-seconds",
                                        )
                                        pause_preset_buttons = []
                                        # 四个预设按钮两两并排、两行堆叠（2x2 网格），
                                        # 样式参考输出格式卡的 mp3/flac 堆叠；总宽不超过情绪风格/停顿时长区域。
                                        with gr.Column(
                                            scale=0, min_width=150,
                                            elem_id="minimax-pause-preset-grid",
                                        ):
                                            with gr.Row(
                                                equal_height=True, elem_id="minimax-pause-preset-row1",
                                            ):
                                                for _pause_label, _pause_value, _pause_id in [
                                                    ("0.25s", 0.25, "minimax-pause-preset-025"),
                                                    ("0.5s", 0.5, "minimax-pause-preset-05"),
                                                ]:
                                                    _pause_btn = gr.Button(
                                                        _pause_label, size="sm", min_width=70, scale=1,
                                                        elem_id=_pause_id,
                                                    )
                                                    pause_preset_buttons.append((_pause_btn, _pause_value))
                                            with gr.Row(
                                                equal_height=True, elem_id="minimax-pause-preset-row2",
                                            ):
                                                for _pause_label, _pause_value, _pause_id in [
                                                    ("1.0s", 1.0, "minimax-pause-preset-10"),
                                                    ("1.5s", 1.5, "minimax-pause-preset-15"),
                                                ]:
                                                    _pause_btn = gr.Button(
                                                        _pause_label, size="sm", min_width=70, scale=1,
                                                        elem_id=_pause_id,
                                                    )
                                                    pause_preset_buttons.append((_pause_btn, _pause_value))
                                        minimax_insert_pause_btn = gr.Button(
                                            "换行插入", size="sm", min_width=96, scale=0,
                                            elem_id="minimax-pause-insert",
                                        )
                                    gr.Markdown(
                                        "可直接手写精确停顿，如 `第一句<#0.80#>第二句`。停顿必须位于两段可发音文本之间；"
                                        "2.8 模型还支持 `(laughs)`、`(sighs)`、`(breath)` 等语气词标签。"
                                    )
                                    gr.Markdown(
                                        "ℹ️ 以上功能只插入「待生成文本」。上传 SRT 后，正式生成将以 SRT 的文本和时间轴为准；"
                                        "参考音频文本仅用于克隆校验。"
                                    )
                                with gr.Accordion(
                                    "输出音频参数（语速、语言增强与高级设置）", open=False,
                                    elem_id="minimax-output-params-accordion",
                                ):
                                    with gr.Row(equal_height=True):
                                        minimax_speed = gr.Slider(
                                            minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                                            label="语速", min_width=160, scale=2,
                                            elem_id="minimax-speed",
                                        )
                                        minimax_language = gr.Dropdown(
                                            choices=["auto", "Chinese", "Chinese,Yue", "English"],
                                            value="auto", label="语言增强", min_width=96, scale=1,
                                            elem_id="minimax-language",
                                        )
                                        minimax_volume = gr.Dropdown(
                                            choices=[0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
                                            value=1.0, label="音量", min_width=100, scale=1,
                                            elem_id="minimax-volume",
                                        )
                                        minimax_pitch = gr.Dropdown(
                                            choices=[-12, -6, -3, 0, 3, 6, 12],
                                            value=0, label="音高", min_width=84, scale=1,
                                            elem_id="minimax-pitch",
                                        )
                                        minimax_bitrate = gr.Dropdown(
                                            choices=[32000, 64000, 128000, 256000],
                                            value=128000, label="码率", min_width=118, scale=1,
                                            elem_id="minimax-bitrate",
                                        )
                                        minimax_sample_rate = gr.Dropdown(
                                            choices=[8000, 16000, 22050, 24000, 32000, 44100],
                                            value=44100, label="采样率", min_width=118, scale=1,
                                            elem_id="minimax-sample-rate",
                                        )
                                        minimax_channel = gr.Dropdown(
                                            choices=[("单声道", 1), ("双声道", 2)],
                                            value=2, label="声道", min_width=120, scale=1,
                                            elem_id="minimax-channel",
                                        )
                                gr.Markdown("以 MiniMax 中国区官网为准。")
                                minimax_run_btn = gr.Button(
                                    "生成云端音频（调用 MiniMax）", variant="primary"
                                )

                            with gr.Tab(
                                "云端音色设计", id="cloud_design",
                                elem_id="page-minimax-voice-design",
                            ) as minimax_design_tab:
                                gr.Markdown("首次使用请先在“云端配音”页安全保存 MiniMax API Key。")
                                gr.Markdown(
                                    "用文字描述新音色并生成试听。试听满意后点击“保存到本机音色库”，"
                                    "即可将返回的 `voice_id` 放入本机音色库并切回配音标签。"
                                )
                                gr.Markdown("候选音色的试听结果固定显示在右侧播放器；生成前下载按钮为灰色。")
                                minimax_design_prompt = gr.Textbox(
                                    label="音色描述（最多 1000 字符）",
                                    lines=8,
                                    max_lines=8,
                                    max_length=1000,
                                    placeholder="例如：成熟温和的女性讲述者，音色清澈有亲和力，平静而有叙事感，语速中等偏慢。",
                                    info="官方网页的输入上限为 1000 字符；建议写清性别、年龄感、音质、情绪、语速与表达方式。",
                                    elem_id="minimax-design-prompt",
                                )
                                minimax_design_prompt_counter = gr.Markdown(
                                    character_counter_markdown("", 1000, "音色描述"),
                                    elem_id="minimax-design-prompt-counter",
                                )
                                minimax_design_preview_text = gr.Textbox(
                                    label="试听文本（最多 500 字符）",
                                    lines=8,
                                    max_lines=8,
                                    max_length=500,
                                    placeholder="输入一段能代表实际使用场景的短文本。",
                                    elem_id="minimax-design-preview",
                                )
                                minimax_design_preview_counter = gr.Markdown(
                                    character_counter_markdown("", 500, "试听文本"),
                                    elem_id="minimax-design-preview-counter",
                                )
                                with gr.Group(elem_id="minimax-design-draw-group"):
                                    # 说明性文字一律不再和控件挤在同一行——它们没有
                                    # 稳定的对齐基准，只能靠脚本硬校正。统一收到本页
                                    # 底部的编号注释区（#minimax-design-notes）。
                                    # 行1：水印 checkbox
                                    with gr.Row(equal_height=False):
                                        minimax_design_watermark = gr.Checkbox(
                                            value=False,
                                            label="试听末尾加水印",
                                            elem_id="minimax-design-watermark",
                                        )
                                    # 行2：本次抽取数量标题 + 说明
                                    with gr.Row(equal_height=False):
                                        with gr.Column(scale=3, min_width=300):
                                            gr.Markdown("**本次抽取数量**")
                                            gr.Markdown("每个音色槽都是一次独立的 MiniMax 音色设计请求。")
                                    # 行3：抽1/3/5 + 试听音色 + voice_id 输入框 + 确认保存
                                    with gr.Row(equal_height=False):
                                        minimax_design_candidate_count = gr.Radio(
                                            choices=[("抽 1 个音", "1"), ("抽 3 个音", "3"), ("抽 5 个音", "5")],
                                            value="1",
                                            show_label=False,
                                            scale=0,
                                            min_width=0,
                                            elem_id="minimax-design-candidate-count",
                                        )
                                        minimax_design_btn = gr.Button(
                                            "试听音色",
                                            variant="primary",
                                            size="sm",
                                            scale=0,
                                            min_width=88,
                                            elem_id="minimax-design-generate-button",
                                        )
                                        minimax_design_voice_id_display = gr.Textbox(
                                            value="",
                                            placeholder="生成后显示候选 voice_id",
                                            show_label=False,
                                            interactive=False,
                                            lines=1,
                                            max_lines=1,
                                            scale=1,
                                            min_width=210,
                                            elem_id="minimax-design-voice-id-display",
                                        )
                                        minimax_design_voice_id_state = gr.State()
                                        minimax_design_save_btn = gr.Button(
                                            "保存到本机音色库",
                                            variant="primary",
                                            size="sm",
                                            scale=0,
                                            min_width=126,
                                            elem_id="minimax-design-save-btn",
                                        )
                                minimax_design_slots_state = gr.State([])
                                minimax_design_slots = gr.Radio(
                                    choices=[],
                                    value=None,
                                    label="音色槽（点击切换右侧试听）",
                                    info="候选音色只保留在当前页面会话中；选中的槽位才会保存到本机音色库，并带入云端配音。",
                                    visible=False,
                                )
                                # 本页所有说明性文字统一收在这里，按编号排列。
                                # 注释 1 讲功能（音色去了哪、怎么才能留住），
                                # 注释 2 讲钱。新增说明按这两类归入，不要再塞回控件行里。
                                gr.Markdown(
                                    "1. 新设计的 voice_id 在确认保存前不会写入音色库；"
                                    "确认保存也只写入本机音色库，不等同于 MiniMax 云端激活——"
                                    "请在 7 天内在“云端配音”完成一次正式生成以保留该音色。\n"
                                    "2. 抽卡要花钱，请量力而行，切勿上头；"
                                    "价格以 MiniMax 中国区官网为准。",
                                    elem_id="minimax-design-notes",
                                )
                    # 右侧仅承载当前云端子页自己的播放器、下载和状态。
                    # 使用普通 Column 而不是 Group，复用本地播放器的轻量布局，
                    # 避免 Group 随左列等高后形成整块灰色底板。
                    with gr.Column(scale=1, min_width=360, elem_id="minimax-result-column"):
                        with gr.Column(elem_id="minimax-dubbing-result-panel"):
                            # 云端上传克隆模块：与「本地配音」右列克隆模块 1:1 同构，
                            # 但独立运行——上传/录音音频 → MiniMax 云端克隆 → 入“已保存音色”。
                            with gr.Column(elem_id="minimax-clone-panel"):
                                minimax_clone_audio = gr.Audio(
                                    sources=["upload", "microphone"],
                                    type="filepath",
                                    label="参考音频（必填 – 上传后用于云端克隆）",
                                    elem_id="minimax-clone-audio",
                                )
                                minimax_clone_name = gr.Textbox(
                                    label="保存音色", placeholder="输入音色名称", show_label=False,
                                    lines=1,
                                    max_lines=1,
                                    elem_id="minimax-clone-name-input",
                                )
                                minimax_clone_btn = gr.Button(
                                    "💎 云端克隆音色", variant="primary", elem_id="minimax-clone-button"
                                )
                                gr.Markdown(
                                    "ℹ️ 该功能会调用API，价格以中国MiniMax官网为准。",
                                    elem_id="minimax-clone-api-note",
                                )
                                minimax_srt_file = gr.File(
                                    label="上传 SRT 字幕文件",
                                    file_types=[".srt"],
                                    visible=True,
                                    elem_id="minimax-srt-file",
                                )
                                gr.Markdown(
                                    "ℹ️ SRT 用于「生成云端音频」的字幕时间轴与文本，不参与「云端克隆音色」。",
                                    elem_id="minimax-srt-note",
                                )
                            minimax_dubbing_audio = gr.Audio(
                                label="MiniMax 云端配音结果",
                                elem_id="minimax-dubbing-audio",
                            )
                            minimax_dubbing_download = gr.DownloadButton(
                                "下载当前云端配音音频",
                                visible=True,
                                interactive=False,
                                variant="primary",
                                size="lg",
                                elem_id="minimax-current-audio-download",
                            )
                            minimax_subtitle_file = gr.File(label="字幕文件下载", visible=False)
                            minimax_status = gr.Markdown(minimax_saved_key_hint, elem_id="minimax-status")
                        with gr.Column(elem_id="minimax-design-result-panel"):
                            minimax_design_audio = gr.Audio(
                                label="MiniMax 云端音色设计试听",
                                elem_id="minimax-design-audio",
                            )
                            minimax_design_download = gr.DownloadButton(
                                "下载当前音色试听",
                                visible=True,
                                interactive=False,
                                variant="primary",
                                size="lg",
                                elem_id="minimax-design-audio-download",
                            )
                            minimax_design_status = gr.Markdown()
                # 子页切换只在浏览器端显示对应的右侧结果面板；不走 Python 队列，
                # 因而不会被云端生成请求阻塞，也不会丢失已挂载的播放器。
                minimax_dubbing_tab.select(
                    fn=None,
                    js="() => { window.__minimaxSetResultPanel?.('dubbing'); }",
                    queue=False,
                    api_name=False,
                )
                minimax_design_tab.select(
                    fn=None,
                    js="() => { window.__minimaxSetResultPanel?.('design'); }",
                    queue=False,
                    api_name=False,
                )
                minimax_insert_pause_btn.click(
                    fn=lambda text_value, pause_value: (text_value, pause_value),
                    inputs=[minimax_text, minimax_pause_seconds],
                    outputs=[minimax_text, minimax_pause_seconds],
                    js="(textValue, pauseValue) => window.__minimaxInsertPauseAtSelection ? window.__minimaxInsertPauseAtSelection(textValue, pauseValue, true) : [textValue, pauseValue]",
                    queue=False,
                    api_name=False,
                )
                for _pause_btn, _pause_value in pause_preset_buttons:
                    _pause_btn.click(
                        fn=lambda text_value, value=_pause_value: (text_value, value),
                        inputs=[minimax_text],
                        outputs=[minimax_text, minimax_pause_seconds],
                        js=f"(textValue) => window.__minimaxInsertPauseAtSelection ? window.__minimaxInsertPauseAtSelection(textValue, {_pause_value}, false) : [textValue, {_pause_value}]",
                        queue=False,
                        api_name=False,
                    )
                minimax_expression.change(
                    fn=lambda text_value, _token: (text_value, ""),
                    inputs=[minimax_text, minimax_expression],
                    outputs=[minimax_text, minimax_expression],
                    js="(textValue, token) => window.__minimaxInsertExpressionAtSelection ? window.__minimaxInsertExpressionAtSelection(textValue, token) : [textValue, '']",
                    queue=False,
                    api_name=False,
                )
                minimax_save_key_btn.click(
                    fn=minimax_save_api_key,
                    inputs=[minimax_api_key],
                    outputs=[minimax_status],
                    api_name=False,
                )

                def _clone_voice_for_ui(key, audio_path, voice_name, preview_text, model_name, prompt_text, current_saved_choices):
                    """上传音频云端克隆：成功后把新 voice_id 加入已保存音色下拉，试听音频送入播放器。"""
                    voice_id, new_saved_choices, demo_path, status = clone_voice_with_upload(
                        key, audio_path, voice_name,
                        preview_text=preview_text,
                        model=model_name,
                        text_validation=prompt_text,
                        output_dir=OUTPUTPUTS,
                    )
                    # 不与页面旧 state 合并：合并只会做加法，会把已删除的音色带回来。
                    # clone_voice_with_upload 返回的已经是写库后重新读盘的完整列表。
                    merged = list(new_saved_choices or [])
                    audio_update = (
                        gr.update(value=demo_path, visible=bool(demo_path))
                        if demo_path
                        else gr.update()
                    )
                    download_update = (
                        gr.update(value=demo_path, visible=True, interactive=True)
                        if demo_path
                        else gr.update()
                    )
                    return (
                        gr.update(choices=merged, value=voice_id),
                        merged,
                        audio_update,
                        download_update,
                        status,
                    )

                minimax_clone_btn.click(
                    fn=_clone_voice_for_ui,
                    inputs=[
                        minimax_api_key,
                        minimax_clone_audio,
                        minimax_clone_name,
                        minimax_clone_preview,
                        minimax_model,
                        minimax_prompt_text,
                        minimax_saved_voice_choices_state,
                    ],
                    outputs=[
                        minimax_saved_voice_id,
                        minimax_saved_voice_choices_state,
                        minimax_dubbing_audio,
                        minimax_dubbing_download,
                        minimax_status,
                    ],
                    show_progress="minimal",
                    show_progress_on=[minimax_dubbing_audio],
                    api_name=False,
                )

                # 上传/更换克隆音频：本地 ASR 自动识别填入「参考音频文本」（与本地配音页一致）
                def _recognize_clone_audio_for_ui(audio_path):
                    if not audio_path:
                        return gr.update(value="")
                    try:
                        demo.get_or_load_asr_model()
                        asr_text = demo.prompt_wav_recognition(audio_path)
                    except Exception as e:
                        logger.warning("云端克隆音频识别失败：%s", e)
                        return gr.update()
                    finally:
                        demo.unload_asr_model()
                    gr.Info("参考音频文本已自动识别填入，如有错请手动修正。")
                    return gr.update(value=asr_text)

                minimax_clone_audio.input(
                    fn=_recognize_clone_audio_for_ui,
                    inputs=[minimax_clone_audio],
                    outputs=[minimax_prompt_text],
                    show_progress="hidden",
                    api_name=False,
                )

                def _voice_dropdown_update(choices, current_voice):
                    """保留当前选择；下拉框本身负责即时筛选名称与 voice_id。"""
                    all_choices = list(choices or [])
                    available_values = {choice[1] for choice in all_choices if len(choice) >= 2}
                    current_voice = (current_voice or "").strip()
                    return gr.update(
                        choices=all_choices,
                        value=current_voice if current_voice in available_values else None,
                    )

                def _merge_minimax_tts_models(*model_lists):
                    merged = []
                    seen = set()
                    for models in model_lists:
                        for model_id in list(models or []):
                            if not isinstance(model_id, str):
                                continue
                            model_id = model_id.strip()
                            if not model_id or model_id in seen:
                                continue
                            seen.add(model_id)
                            merged.append(model_id)
                    return merged

                def _resolve_minimax_voice_id(selected_fetched_voice, selected_saved_voice, manual_voice):
                    return (
                        (manual_voice or "").strip()
                        or (selected_saved_voice or "").strip()
                        or (selected_fetched_voice or "").strip()
                    )

                def _prepare_minimax_result_for_request():
                    return (
                        gr.update(value=None),
                        gr.update(value=None, visible=True, interactive=False),
                        gr.update(value=None, visible=False),
                    )

                def _prepare_design_result_for_request():
                    """生成开始时同时清空 voice_id 状态，避免生成中点击保存旧值。"""
                    return (
                        gr.update(value=None),
                        gr.update(value=None, visible=True, interactive=False),
                        None,
                        gr.update(value="", placeholder="生成后显示候选 voice_id"),
                    )

                def _fetch_available_voices_for_ui(
                    key, current_fetched_voice, current_saved_voice, saved_choices,
                ):
                    all_choices, status = fetch_available_voice_choices(key)
                    # 本机音色库文件是「已保存音色」的唯一真相。删除是通过 Gradio
                    # 接口在另一个会话里执行的，页面自带的 state 仍是删除前的旧
                    # 列表；直接沿用会把刚删掉的音色重新显示出来。这里重新读盘。
                    saved_choices = saved_voice_choices()
                    saved_voice_ids = {
                        choice[1] for choice in saved_choices if len(choice) >= 2
                    }
                    fetched_choices = [
                        choice for choice in all_choices
                        if len(choice) >= 2 and choice[1] not in saved_voice_ids
                    ]
                    fetched_update = _voice_dropdown_update(fetched_choices, current_fetched_voice)
                    saved_update = _voice_dropdown_update(saved_choices, current_saved_voice)
                    return (
                        fetched_choices, fetched_update, saved_update,
                        saved_choices, status,
                    )

                def _fetch_tts_models_for_ui(key, current_model, current_models):
                    fetched_models, status = fetch_available_tts_model_ids(key)
                    merged_models = _merge_minimax_tts_models(current_models, fetched_models)
                    selected_model = current_model if current_model in merged_models else merged_models[0]
                    return (
                        gr.update(choices=merged_models, value=selected_model),
                        merged_models,
                        status,
                        _set_expression_availability(selected_model),
                    )

                def _design_voice_for_ui(key, prompt, preview, watermark, candidate_count):
                    candidates, status = design_voice_candidates_with_saved_key(
                        key, prompt, preview, watermark, OUTPUTPUTS, candidate_count
                    )
                    first_candidate = candidates[0]
                    slot_choices = [
                        (f"音色槽 {index + 1}", str(index))
                        for index in range(len(candidates))
                    ]
                    return (
                        first_candidate["audio_path"],
                        first_candidate["voice_id"],
                        gr.update(value=first_candidate["voice_id"]),
                        gr.update(value=first_candidate["audio_path"], visible=True, interactive=True),
                        candidates,
                        gr.update(choices=slot_choices, value="0", visible=True),
                        status,
                    )

                def _select_design_slot_for_ui(candidates, selected_slot):
                    try:
                        candidate = (candidates or [])[int(selected_slot)]
                    except (TypeError, ValueError, IndexError):
                        raise RuntimeError("未找到该音色槽，请重新抽取候选音色。") from None
                    if not isinstance(candidate, dict):
                        raise RuntimeError("未找到该音色槽，请重新抽取候选音色。")
                    audio_path = candidate.get("audio_path")
                    voice_id = candidate.get("voice_id")
                    if not audio_path or not voice_id:
                        raise RuntimeError("未找到该音色槽，请重新抽取候选音色。")
                    slot_number = int(selected_slot) + 1
                    status = (
                        f"已选中音色槽 {slot_number}；右侧播放器、下载按钮与 voice_id 已同步切换。"
                        "保存时，只会写入当前选中的音色到本机音色库。"
                    )
                    return (
                        audio_path,
                        voice_id,
                        gr.update(value=voice_id),
                        gr.update(value=audio_path, visible=True, interactive=True),
                        status,
                    )

                def _save_designed_voice_for_ui(
                    voice_id, current_fetched_choices, current_saved_choices,
                ):
                    stored_choices, status = save_designed_voice_to_library(voice_id)
                    # 同样以磁盘为准，不与页面旧 state 合并，避免复活已删除的音色。
                    saved_choices = list(stored_choices or [])
                    saved_voice_ids = {
                        choice[1] for choice in saved_choices if len(choice) >= 2
                    }
                    fetched_choices = [
                        choice for choice in list(current_fetched_choices or [])
                        if len(choice) >= 2 and choice[1] not in saved_voice_ids
                    ]
                    return (
                        fetched_choices,
                        saved_choices,
                        _voice_dropdown_update(saved_choices, voice_id),
                        _voice_dropdown_update(fetched_choices, None),
                        status,
                        gr.update(selected="cloud_dubbing"),
                        gr.update(value=""),
                    )

                def _synthesize_for_ui(
                    key, text_value, selected_fetched_voice, selected_saved_voice, manual_voice, model_name, fmt,
                    speed, language, emotion, volume, pitch, bitrate, sample_rate, channel,
                    gen_subtitle=False,
                ):
                    voice = _resolve_minimax_voice_id(
                        selected_fetched_voice, selected_saved_voice, manual_voice
                    )
                    audio_path, status = minimax_synthesize(
                        key, text_value, voice, model_name, fmt, speed, language, OUTPUTPUTS,
                        "", emotion, volume, pitch, bitrate, sample_rate, channel,
                    )
                    subtitle_path = None
                    if gen_subtitle:
                        # 云端生成结果用本地 Whisper 免费生成字幕（SRT 位于音频同目录）
                        try:
                            subtitle_path = demo.generate_subtitle(audio_path)
                            status += "字幕已生成，可在下方下载。"
                        except Exception as e:
                            logger.warning("云端字幕生成失败：%s", e)
                            status += "字幕生成失败，可稍后重试。"
                    return (
                        audio_path,
                        status,
                        gr.update(value=audio_path, visible=True, interactive=True),
                        gr.update(value=subtitle_path, visible=bool(subtitle_path)),
                    )

                def _synthesize_srt_for_ui(
                    key, srt_file, selected_fetched_voice, selected_saved_voice, manual_voice, model_name, fmt,
                    speed, language, emotion, volume, pitch, bitrate, sample_rate, channel,
                ):
                    """SRT 一键配音（云端）：整段字幕文本一次合成（音色最一致、调用最少），
                    按各字幕字符数比例切分后按时间轴铺排；超时顺延避免重叠。"""
                    if not srt_file:
                        raise gr.Error("SRT 模式请先上传 .srt 字幕文件。")
                    entries = parse_srt(srt_file)
                    if not entries:
                        raise gr.Error("SRT 文件解析为空，请检查文件格式。")
                    voice = _resolve_minimax_voice_id(
                        selected_fetched_voice, selected_saved_voice, manual_voice
                    )
                    full_text = _join_srt_texts(e["text"] for e in entries)
                    audio_path, status = minimax_synthesize(
                        key, full_text, voice, model_name, fmt, speed, language, OUTPUTPUTS,
                        "", emotion, volume, pitch, bitrate, sample_rate, channel,
                    )
                    # 读取合成音频，按字符比例切分回填时间轴
                    import soundfile as sf
                    wav, sr = sf.read(audio_path, dtype="float32")
                    total_chars = sum(len((e["text"] or "").strip()) for e in entries) or 1
                    clips = []
                    pos = 0
                    for i, e in enumerate(entries):
                        seg_len = int(len(wav) * len((e["text"] or "").strip()) / total_chars)
                        clips.append((i, np.asarray(wav[pos:pos + seg_len], dtype=np.float32)))
                        pos += seg_len
                    if pos < len(wav) and clips:
                        idx, last = clips[-1]
                        clips[-1] = (idx, np.concatenate([last, np.asarray(wav[pos:], dtype=np.float32)]))

                    total_sec = max(e["end"] for e in entries)
                    _check_timeline_budget(
                        total_sec, sr, channels=(wav.shape[1] if wav.ndim > 1 else 1)
                    )
                    longest = max(len(c) for _, c in clips)
                    track_shape = (int(total_sec * sr) + longest + sr,) + wav.shape[1:]
                    track = np.zeros(track_shape, dtype=np.float32)
                    cursor = 0
                    for idx, clip in clips:
                        start_pos = max(int(entries[idx]["start"] * sr), cursor)
                        end_pos = start_pos + len(clip)
                        if end_pos > len(track):
                            padding_shape = (end_pos - len(track),) + track.shape[1:]
                            track = np.concatenate(
                                [track, np.zeros(padding_shape, dtype=np.float32)]
                            )
                        track[start_pos:end_pos] += clip
                        cursor = end_pos
                    out_path = save_timeline_audio(
                        sr,
                        track,
                        fmt,
                        prefix="minimax_srt",
                        bitrate=bitrate,
                        channel=channel,
                    )
                    return (
                        out_path,
                        status + f"SRT 配音已按时间轴铺排完成，最终格式为 {fmt.upper()}。",
                        gr.update(value=out_path, visible=True, interactive=True),
                        gr.update(),
                    )

                def _set_expression_availability(model_name):
                    resolved = (model_name or "").strip()
                    supported = resolved.startswith("speech-2.8-")
                    return gr.update(
                        value="",
                        interactive=supported,
                        label="语气词（仅 2.8 模型）" if supported else "语气词（2.6 模型不可用）",
                    )

                def _set_bitrate_availability(fmt):
                    is_mp3 = fmt == "mp3"
                    return gr.update(interactive=is_mp3, value=128000 if is_mp3 else None)

                minimax_design_btn.click(
                    fn=_prepare_design_result_for_request,
                    outputs=[
                        minimax_design_audio,
                        minimax_design_download,
                        minimax_design_voice_id_state,
                        minimax_design_voice_id_display,
                    ],
                    api_name=False,
                ).then(
                    fn=_design_voice_for_ui,
                    inputs=[
                        minimax_api_key,
                        minimax_design_prompt,
                        minimax_design_preview_text,
                        minimax_design_watermark,
                        minimax_design_candidate_count,
                    ],
                    outputs=[
                        minimax_design_audio,
                        minimax_design_voice_id_state,
                        minimax_design_voice_id_display,
                        minimax_design_download,
                        minimax_design_slots_state,
                        minimax_design_slots,
                        minimax_design_status,
                    ],
                    api_name=False,
                )
                minimax_design_slots.change(
                    fn=_select_design_slot_for_ui,
                    inputs=[minimax_design_slots_state, minimax_design_slots],
                    outputs=[
                        minimax_design_audio,
                        minimax_design_voice_id_state,
                        minimax_design_voice_id_display,
                        minimax_design_download,
                        minimax_design_status,
                    ],
                    api_name=False,
                )
                minimax_fetch_voices_btn.click(
                    fn=_fetch_available_voices_for_ui,
                    inputs=[
                        minimax_api_key,
                        minimax_fetched_voice_id,
                        minimax_saved_voice_id,
                        minimax_saved_voice_choices_state,
                    ],
                    outputs=[
                        minimax_fetched_voice_choices_state,
                        minimax_fetched_voice_id,
                        minimax_saved_voice_id,
                        minimax_saved_voice_choices_state,
                        minimax_status,
                    ],
                    api_name=False,
                )
                minimax_fetch_models_btn.click(
                    fn=_fetch_tts_models_for_ui,
                    inputs=[minimax_api_key, minimax_model, minimax_model_choices_state],
                    outputs=[
                        minimax_model,
                        minimax_model_choices_state,
                        minimax_status,
                        minimax_expression,
                    ],
                    api_name=False,
                )
                minimax_fetched_voice_id.input(
                    fn=lambda _value: gr.update(value=None),
                    inputs=[minimax_fetched_voice_id],
                    outputs=[minimax_saved_voice_id],
                    queue=False,
                    api_name=False,
                )
                minimax_saved_voice_id.input(
                    fn=lambda _value: gr.update(value=None),
                    inputs=[minimax_saved_voice_id],
                    outputs=[minimax_fetched_voice_id],
                    queue=False,
                    api_name=False,
                )

                def _delete_minimax_saved_voice(selection):
                    """从本机音色库真实删除一条已保存音色，并同步刷新下拉框。

                    只收一个纯文本入参：如果把“已保存音色”下拉也列为输入，
                    Gradio 会先按 choices 校验它的值，前端拿不到 voice_id 时
                    只能传空值，请求会在进入本函数之前就被拒绝。
                    """
                    selection = (selection or "").strip()
                    if not selection:
                        raise gr.Error("未指定要删除的音色。")
                    try:
                        choices, status = minimax_delete_saved_voice(selection)
                    except (RuntimeError, OSError) as exc:
                        raise gr.Error(str(exc)) from None
                    # 删除后不保留旧选择：当前选中项可能正是被删掉的那条。
                    return (
                        gr.update(choices=choices, value=None),
                        choices,
                        gr.update(value=""),
                        status,
                    )

                # 移到屏幕外的 Gradio 控件在 Electron 内嵌页面里收不到合成事件，
                # 所以行内“×”走的是 /gradio_api/call/_delete_minimax_saved_voice。
                # api_name 显式声明，确保该接口一定被暴露；隐藏输入框的 input
                # 事件保留为回退路径，绑定的是同一个后端函数。
                minimax_delete_voice_name.input(
                    fn=_delete_minimax_saved_voice,
                    inputs=[minimax_delete_voice_name],
                    outputs=[
                        minimax_saved_voice_id,
                        minimax_saved_voice_choices_state,
                        minimax_delete_voice_name,
                        minimax_status,
                    ],
                    api_name="_delete_minimax_saved_voice",
                )
                minimax_design_save_btn.click(
                    fn=_save_designed_voice_for_ui,
                    inputs=[
                        minimax_design_voice_id_state,
                        minimax_fetched_voice_choices_state,
                        minimax_saved_voice_choices_state,
                    ],
                    outputs=[
                        minimax_fetched_voice_choices_state,
                        minimax_saved_voice_choices_state,
                        minimax_saved_voice_id,
                        minimax_fetched_voice_id,
                        minimax_status,
                        minimax_cloud_tabs,
                        minimax_manual_voice_id,
                    ],
                    js="() => { window.__minimaxSetResultPanel?.('dubbing'); }",
                    api_name=False,
                )
                def _run_dubbing_for_ui(
                    key, text_value, srt_file, gen_subtitle,
                    selected_fetched_voice, selected_saved_voice, manual_voice, model_name, fmt,
                    speed, language, emotion, volume, pitch, bitrate, sample_rate, channel,
                ):
                    """生成分派：上传了 SRT 文件即走 SRT 模式（整段合成+时间轴铺排），
                    否则普通文本合成（可带字幕生成）。"""
                    srt_mode = bool(srt_file)
                    if srt_mode:
                        return _synthesize_srt_for_ui(
                            key, srt_file, selected_fetched_voice, selected_saved_voice, manual_voice,
                            model_name, fmt, speed, language, emotion,
                            volume, pitch, bitrate, sample_rate, channel,
                        )
                    return _synthesize_for_ui(
                        key, text_value, selected_fetched_voice, selected_saved_voice, manual_voice,
                        model_name, fmt, speed, language, emotion,
                        volume, pitch, bitrate, sample_rate, channel,
                        gen_subtitle=gen_subtitle,
                    )

                minimax_run_btn.click(
                    fn=_prepare_minimax_result_for_request,
                    outputs=[minimax_dubbing_audio, minimax_dubbing_download, minimax_subtitle_file],
                    api_name=False,
                ).then(
                    fn=_run_dubbing_for_ui,
                    inputs=[
                        minimax_api_key, minimax_text, minimax_srt_file,
                        minimax_gen_subtitle_state,
                        minimax_fetched_voice_id, minimax_saved_voice_id, minimax_manual_voice_id,
                        minimax_model, minimax_format, minimax_speed, minimax_language, minimax_emotion,
                        minimax_volume, minimax_pitch, minimax_bitrate, minimax_sample_rate, minimax_channel,
                    ],
                    outputs=[
                        minimax_dubbing_audio, minimax_status,
                        minimax_dubbing_download, minimax_subtitle_file,
                    ],
                    show_progress="minimal",
                    show_progress_on=[minimax_dubbing_audio],
                    api_name=False,
                )
                # 生成字幕 toggle：翻转状态并切换按钮选中样式
                def _toggle_gen_subtitle(current: bool):
                    new_value = not current
                    classes = ["minimax-toggle-btn"] + (["minimax-toggle-on"] if new_value else [])
                    return new_value, gr.update(elem_classes=classes)

                minimax_gen_subtitle_btn.click(
                    fn=_toggle_gen_subtitle,
                    inputs=[minimax_gen_subtitle_state],
                    outputs=[minimax_gen_subtitle_state, minimax_gen_subtitle_btn],
                    queue=False,
                    api_name=False,
                )

                # 上传 SRT 文件：SRT 仍是正式生成的唯一时间轴来源；
                # 「待生成文本」保留在参考文本与试听文本之间，显示只读字幕预览，避免用户误以为该输入框消失。
                def _on_srt_file_change(file_path: str | None, gen_state: bool):
                    if file_path:
                        try:
                            srt_preview = "\n".join(
                                (entry.get("text") or "").strip()
                                for entry in parse_srt(file_path)
                                if (entry.get("text") or "").strip()
                            )
                        except Exception:
                            srt_preview = "（SRT 已上传；字幕内容将在生成时解析。）"
                        return (
                            False,
                            gr.update(elem_classes=["minimax-toggle-btn"]),
                            gr.update(
                                value=srt_preview,
                                visible=True,
                                interactive=False,
                                label=f"{MINIMAX_TEXT_LABEL}｜由 SRT 提取，仅供预览",
                            ),
                        )
                    gen_classes = ["minimax-toggle-btn"] + (["minimax-toggle-on"] if gen_state else [])
                    return (
                        gen_state,
                        gr.update(elem_classes=gen_classes),
                        gr.update(
                            value="",
                            visible=True,
                            interactive=True,
                            label=MINIMAX_TEXT_LABEL,
                        ),
                    )

                for _event_name in ("upload", "clear"):
                    getattr(minimax_srt_file, _event_name)(
                        fn=_on_srt_file_change,
                        inputs=[minimax_srt_file, minimax_gen_subtitle_state],
                        outputs=[
                            minimax_gen_subtitle_state, minimax_gen_subtitle_btn,
                            minimax_text,
                        ],
                        queue=False,
                        api_name=False,
                    )

                minimax_model.change(
                    fn=_set_expression_availability,
                    inputs=[minimax_model],
                    outputs=[minimax_expression],
                    queue=False,
                    api_name=False,
                )
                minimax_format.change(
                    fn=_set_bitrate_availability,
                    inputs=[minimax_format],
                    outputs=[minimax_bitrate],
                    queue=False,
                    api_name=False,
                )

            # ===== Tab 2：生成历史 =====
            with gr.Tab("本地配音历史"):
                clear_history_btn = gr.Button("清空历史记录")

                @gr.render(inputs=[history_state, hist_page_state, hist_expanded_state])
                def _render_history(history, page, expanded):
                    if not history:
                        gr.Markdown("暂无历史记录。生成语音后会自动显示在这里。")
                        return

                    expanded = expanded or []
                    # 倒序：最近生成的放在最前面
                    items = list(reversed(history))
                    total = len(items)
                    total_pages = max(1, (total + HISTORY_PAGE_SIZE - 1) // HISTORY_PAGE_SIZE)
                    page = min(max(0, int(page or 0)), total_pages - 1)
                    start = page * HISTORY_PAGE_SIZE
                    page_items = items[start:start + HISTORY_PAGE_SIZE]

                    gr.Markdown(
                        f"共 {total} 条记录　|　第 {page + 1} / {total_pages} 页"
                        "　（点击「▶ 播放」按需加载音频）",
                        elem_classes=["hist-count"],
                    )

                    for item in page_items:
                        audio_path = item.get("audio")
                        fname = Path(audio_path).name if audio_path else "（未知文件）"
                        is_open = bool(audio_path) and (audio_path in expanded)
                        with gr.Group(elem_classes=["hist-item"]):
                            with gr.Row(equal_height=True, variant="compact"):
                                gr.Markdown(
                                    f"🕒 {item.get('time', '')}　|　📄 {fname}",
                                    elem_classes=["hist-meta"],
                                )
                                toggle_btn = gr.Button(
                                    "收起" if is_open else "▶ 播放",
                                    size="sm", scale=0, min_width=72,
                                    elem_classes=["hist-toggle"],
                                )
                            # 懒加载：只有展开的条目才真正创建 gr.Audio 并加载音频
                            if is_open and audio_path:
                                gr.Audio(
                                    value=audio_path,
                                    show_label=False,
                                    elem_classes=["hist-audio"],
                                )

                        def _toggle(p=audio_path, cur=expanded):
                            s = list(cur or [])
                            if not p:
                                return s
                            if p in s:
                                s.remove(p)
                            else:
                                s.append(p)
                            return s

                        toggle_btn.click(fn=_toggle, outputs=[hist_expanded_state])

                    # 分页控制
                    with gr.Row(variant="compact"):
                        prev_btn = gr.Button(
                            "← 上一页", size="sm", interactive=(page > 0)
                        )
                        next_btn = gr.Button(
                            "下一页 →", size="sm", interactive=(page < total_pages - 1)
                        )
                    prev_btn.click(
                        fn=lambda p=page: max(0, p - 1), outputs=[hist_page_state]
                    )
                    next_btn.click(
                        fn=lambda p=page, mx=total_pages - 1: min(mx, p + 1),
                        outputs=[hist_page_state],
                    )

            # ===== Tab 3：最佳实践 =====
            with gr.Tab("最佳实践"):
                gr.Markdown(
                    "📖 **官网文档：** "
                    "[VoxCPM Cookbook（点击查看）]"
                    "(https://voxcpm.readthedocs.io/zh-cn/latest/cookbook.html)"
                )
                gr.Markdown(_USAGE_INSTRUCTIONS)
                gr.Markdown(_SPEED_NOTICE)
                gr.Markdown(_APIKEY_NOTICE)
                gr.Markdown(_EXAMPLES_FOOTER)

            with gr.Tab("关于"):
                create_about_tab()

        gr.Markdown("", height=50)

        # ---- 事件绑定 ----

        # 上传/更换参考音频：立即加载、识别、卸载 ASR（超长音频自动截取后识别）
        reference_wav.input(
            fn=_load_recognize_unload,
            inputs=[reference_wav],
            outputs=[asr_status, prompt_text, reference_wav],
            show_progress="hidden",
        )

        # 选择预设音色：填入参考音频，并触发识别
        preset_dropdown.input(
            fn=_on_preset_change,
            inputs=[preset_dropdown],
            outputs=[reference_wav],
        ).then(
            fn=_load_recognize_unload,
            inputs=[reference_wav, preset_dropdown],
            outputs=[asr_status, prompt_text, reference_wav],
        )

        # 保存当前音色到 voices 目录，并刷新下拉框（含多角色面板的角色下拉）
        save_voice_btn.click(
            fn=_save_voice,
            inputs=[save_name_box, reference_wav],
            outputs=[preset_dropdown, save_name_box] + [r_ref for _, r_ref, _ in role_components],
        )
        # 行内“×”走的是 /gradio_api/call/_delete_voice。
        # api_name 必须显式声明：两个事件绑的是同一个函数，交给 Gradio 自动命名
        # 会撞车（第二个被改名成 _delete_voice_1 之类），前端按固定路径调用就 404，
        # 表现正是「确认框弹出来了，但音色删不掉」。
        delete_voice_btn.click(
            fn=_delete_voice,
            inputs=[delete_voice_name_box, preset_dropdown],
            outputs=[preset_dropdown, reference_wav] + [r_ref for _, r_ref, _ in role_components],
            api_name="_delete_voice",
        )
        # Electron 内嵌页面的隐藏按钮 click 可能无法桥接到 Gradio；
        # 同时监听隐藏输入框的 input 事件，作为前端删除 API 的兼容回退。
        # 这一条只走会话内路径，不需要也不应再占用一个 API 名字。
        delete_voice_name_box.input(
            fn=_delete_voice,
            inputs=[delete_voice_name_box, preset_dropdown],
            outputs=[preset_dropdown, reference_wav] + [r_ref for _, r_ref, _ in role_components],
            api_name=False,
        )

        # 多角色 / SRT 模式的互斥与控件联动
        multi_role_cb.change(
            fn=_on_multi_role_change,
            inputs=[multi_role_cb, text],
            outputs=[srt_cb, speed_slider, reference_wav, preset_dropdown,
                     multi_role_accordion, text, srt_file],
        )
        srt_cb.change(
            fn=_on_srt_change,
            inputs=[srt_cb],
            outputs=[gen_subtitle_cb, multi_role_cb, text, srt_file,
                     speed_slider, reference_wav, preset_dropdown, multi_role_accordion],
        )

        # 仅清空界面历史记录；outputs 中的音频与字幕文件始终保留。
        def _clear_history_records():
            task_manager.clear_history_records()
            gr.Info("已清空界面历史记录；音频与字幕文件保持不变。")
            return [], 0, []

        clear_history_btn.click(
            fn=_clear_history_records,
            outputs=[history_state, hist_page_state, hist_expanded_state],
        )

        # 清空所有任务 / 取消所有排队任务
        clear_tasks_btn.click(
            fn=lambda: (task_manager.clear_all(),
                        task_manager.snapshot()["summary_md"],
                        task_manager.snapshot()["table_md"])[1:],
            outputs=[queue_summary, queue_table],
        )
        stop_task_btn.click(
            fn=lambda: (task_manager.stop_current(),
                        task_manager.snapshot()["summary_md"],
                        task_manager.snapshot()["table_md"])[1:],
            outputs=[queue_summary, queue_table],
        )

        # 添加到任务队列（队列由后台 worker 顺序执行）
        # 点击瞬间先清掉上一次的结果：queue=False 不排队，立刻生效。
        # 这一步单独拆出来，是为了让后面那个生成器事件全程不碰播放器，
        # 从而保住 show_progress_on 的等待动画（详见 _enqueue_and_wait 的注释）。
        run_btn.click(
            fn=_local_player_cleared,
            outputs=[audio_output, local_dubbing_download, subtitle_output],
            queue=False,
            trigger_mode="multiple",
            api_name=False,
        ).then(
            fn=_enqueue_and_wait,
            inputs=[
                text, local_srt_file, reference_wav,
                prompt_text, cfg_value, DoNormalizeText, DoDenoisePromptAudio,
                dit_steps, speed_slider, gen_subtitle_cb, multi_role_cb, local_srt_file,
                segment_limit_radio,
                *[c for trio in role_components for c in trio],
            ],
            outputs=[
                queue_summary, queue_table,
                audio_output, local_dubbing_download, subtitle_output,
            ],
            api_name="generate",
            # 生成器事件必须进队列。但它全程只是阻塞等一个 Event，不占 GPU、
            # 不做计算，所以放进独立并发桶并取消上限——否则它会吃满全局唯一的
            # 默认并发槽，导致本地长任务期间状态栏、云端页等所有事件一起卡死，
            # 连点提交也会被迫串行。
            concurrency_id="local-task-wait",
            concurrency_limit=None,
            trigger_mode="multiple",
            show_progress="minimal",
            show_progress_on=[audio_output],
        )

        # 定时轮询：刷新队列状态、生成结果、字幕与历史
        poll_timer = gr.Timer(5.0)
        poll_timer.tick(
            fn=_poll,
            # hist_expanded_state 一并传进去：它非空代表历史页里有展开的播放器，
            # 这一轮就不能刷新历史，否则 @gr.render 重建会把声音掐掉。
            inputs=[rev_state, hist_expanded_state],
            outputs=[
                queue_summary, queue_table, history_state, rev_state,
            ],
            queue=False,
        )

    # 供 main() 在 launch() 时使用；挂在对象上是为了不改动
    # create_demo_interface 的返回签名（static_ui_check.py 依赖它返回 interface）。
    interface.yzy_page_css = page_css
    return interface


def main():
    demo = VoxCPMDemo(model_id="./models/VoxCPM2", device="auto")
    interface = create_demo_interface(demo)
    launch_kwargs = {}
    if _LAUNCH_SUPPORTS_CSS:
        launch_kwargs["css"] = getattr(interface, "yzy_page_css", "")
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=server_port,
        show_error=True,
        theme=gr.themes.Ocean(),
        allowed_paths=[str(OUTPUTPUTS.absolute()), str(VOICES.absolute())],
        **launch_kwargs,
    )


if __name__ == "__main__":
    main()
