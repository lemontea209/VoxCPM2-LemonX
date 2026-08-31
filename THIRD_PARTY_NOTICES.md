# 第三方组件声明 / Third-Party Notices

本发行物包含下列第三方软件、模型与运行时组件，各自适用其原始许可证。
本文件为归属与许可声明，不修改、替代或缩减任何第三方许可证授予的权利。

生成日期：2026-08-30

---

## 一、音频处理组件

### FFmpeg

- 版本：`n8.1.2-50-g1a748fe2cd-20260830`
- 构建：BtbN/FFmpeg-Builds，`win64-lgpl-shared` 变体
- 构建脚本：https://github.com/BtbN/FFmpeg-Builds
- 上游源码：https://github.com/FFmpeg/FFmpeg
- 许可证：GNU Lesser General Public License v3
- 许可证全文：`win-unpacked/python/ffmpeg/LICENSE.txt`
- 位置：`win-unpacked/python/ffmpeg/`

本发行物未修改 FFmpeg，以动态链接（shared）方式分发未经改动的官方构建产物。
使用者可自行以兼容版本替换 `win-unpacked/python/ffmpeg/bin/` 下的动态库。
本构建未启用 `--enable-gpl`，不含 libx264、libx265、librubberband 等 GPL-only 组件。

### 音频变速

时间伸缩由 `audiotsm` 的 WSOLA 算法实现。本发行物**不捆绑** Rubber Band 二进制。

---

## 二、模型权重

| 模型 | 提供方 | 许可证 | 许可证全文 |
| --- | --- | --- | --- |
| VoxCPM2 | OpenBMB | Apache-2.0 | `win-unpacked/python/LICENSE` |
| SenseVoiceSmall | Alibaba Group / FunAudioLLM / FunASR | FunASR Model Open Source License Agreement v1.1 | `LICENSES/FunASR-MODEL-LICENSE-v1.1.txt` |
| Whisper medium | OpenAI | MIT | `LICENSES/OpenAI-Whisper-MIT.txt` |
| ZipEnhancer (`speech_zipenhancer_ans_multiloss_16k_base`) | 阿里巴巴达摩院 / ModelScope | Apache-2.0 | 见模型目录 README |

**SenseVoiceSmall 特别说明：** 依 FunASR Model License v1.1 第 2 条，
再分发时必须注明出处及作者信息，并保留模型名称 `SenseVoiceSmall`。
官方模型卡：https://huggingface.co/FunAudioLLM/SenseVoiceSmall

---

## 三、应用运行时

| 组件 | 许可证 | 声明文件 |
| --- | --- | --- |
| Electron | MIT | `win-unpacked/LICENSE.electron.txt` |
| Chromium 及其第三方组件 | 见随附声明 | `win-unpacked/LICENSES.chromium.html` |
| Microsoft Visual C++ Redistributable | Microsoft 可再分发条款 | 未修改分发，见 Microsoft 官方条款 |

---

## 四、Node.js 依赖（107 个，位于 `app.asar`）

| 许可证 | 包数 |
| --- | ---: |
| MIT | 100 |
| ISC | 2 |
| Python-2.0 | 1 |
| BSD-3-Clause | 1 |
| BSD-2-Clause | 1 |
| BlueOak-1.0.0 | 1 |
| MIT OR CC0-1.0 | 1 |

完整清单见本文件第六节。

---

## 五、Python 依赖（154 个，位于 `win-unpacked/python/build_venv/`）

| 许可证 | 包数 |
| --- | ---: |
| MIT | 64 |
| Apache-2.0 | 35 |
| BSD-3-Clause | 25 |
| BSD | 15 |
| ISC | 4 |
| BSD-2-Clause | 3 |
| Python-2.0 | 2 |
| MPL-2.0 | 1 |
| Copyright (c) 2015, matplotlib project | 1 |
| ========================= | 1 |
| License agreement for matplotlib versions 1.3.0 and later | 1 |
| Dual License | 1 |
| LGPL-2.1-or-later | 1 |

### 需特别说明的组件

**`soxr` —— LGPL-2.1-or-later。** 该组件依 LGPL 分发；使用者有权取得其源码并以
兼容版本替换。上游：https://github.com/dofuuz/python-soxr

**`funasr 1.3.9`（MIT）—— 已修改。** 本发行物修改了以下三个文件，
将模块级 `import kaldiio` 改为惰性导入：

```
funasr/utils/load_utils.py
funasr/models/eend/eend_ola_dataloader.py
funasr/utils/compute_det_ctc.py
```

原因：`kaldiio` 附带的是 NTT 评估用途许可协议，不允许再分发，故本发行物
不捆绑该包。`kaldiio` 仅在 `kaldi_ark` 数据类型下被调用，本发行物不使用该
数据类型，功能不受影响。如需该功能，使用者可自行 `pip install kaldiio`。
修改处已在各文件头部标注。

---

## 六、完整依赖清单

### 6.1 Node.js（107）

| 包 | 版本 | 许可证 |
| --- | --- | --- |
| @ant-design/colors | 8.0.0 | MIT |
| @ant-design/cssinjs | 2.0.1 | MIT |
| @ant-design/cssinjs-utils | 2.0.2 | MIT |
| @ant-design/fast-color | 3.0.0 | MIT |
| @ant-design/icons | 6.1.0 | MIT |
| @ant-design/icons-svg | 4.4.2 | MIT |
| @ant-design/react-slick | 2.0.0 | MIT |
| @babel/runtime | 7.28.4 | MIT |
| @electron-toolkit/preload | 3.0.2 | MIT |
| @electron-toolkit/utils | 4.0.0 | MIT |
| @emotion/hash | 0.8.0 | MIT |
| @emotion/unitless | 0.7.5 | MIT |
| @rc-component/async-validator | 5.0.4 | MIT |
| @rc-component/cascader | 1.9.0 | MIT |
| @rc-component/checkbox | 1.0.1 | MIT |
| @rc-component/collapse | 1.1.2 | MIT |
| @rc-component/color-picker | 3.0.3 | MIT |
| @rc-component/context | 2.0.1 | MIT |
| @rc-component/dialog | 1.5.1 | MIT |
| @rc-component/drawer | 1.3.0 | MIT |
| @rc-component/dropdown | 1.0.2 | MIT |
| @rc-component/form | 1.4.0 | MIT |
| @rc-component/image | 1.5.3 | MIT |
| @rc-component/input | 1.1.2 | MIT |
| @rc-component/input-number | 1.6.2 | MIT |
| @rc-component/mentions | 1.6.0 | MIT |
| @rc-component/menu | 1.2.0 | MIT |
| @rc-component/mini-decimal | 1.1.0 | MIT |
| @rc-component/motion | 1.1.6 | MIT |
| @rc-component/mutate-observer | 2.0.1 | MIT |
| @rc-component/notification | 1.2.0 | MIT |
| @rc-component/overflow | 1.0.0 | MIT |
| @rc-component/pagination | 1.2.0 | MIT |
| @rc-component/picker | 1.8.0 | MIT |
| @rc-component/portal | 2.0.1 | MIT |
| @rc-component/progress | 1.0.2 | MIT |
| @rc-component/qrcode | 1.1.1 | MIT |
| @rc-component/rate | 1.0.1 | MIT |
| @rc-component/resize-observer | 1.0.1 | MIT |
| @rc-component/segmented | 1.2.3 | MIT |
| @rc-component/select | 1.3.5 | MIT |
| @rc-component/slider | 1.0.1 | MIT |
| @rc-component/steps | 1.2.2 | MIT |
| @rc-component/switch | 1.0.3 | MIT |
| @rc-component/table | 1.9.0 | MIT |
| @rc-component/tabs | 1.7.0 | MIT |
| @rc-component/textarea | 1.1.2 | MIT |
| @rc-component/tooltip | 1.4.0 | MIT |
| @rc-component/tour | 2.2.1 | MIT |
| @rc-component/tree | 1.1.0 | MIT |
| @rc-component/tree-select | 1.4.0 | MIT |
| @rc-component/trigger | 3.7.1 | MIT |
| @rc-component/upload | 1.1.0 | MIT |
| @rc-component/util | 1.6.0 | MIT |
| @rc-component/virtual-list | 1.0.2 | MIT |
| ajv | 8.17.1 | MIT |
| ajv-formats | 3.0.1 | MIT |
| antd | 6.1.0 | MIT |
| argparse | 2.0.1 | Python-2.0 |
| atomically | 2.1.0 | MIT |
| builder-util-runtime | 9.3.1 | MIT |
| clsx | 2.1.1 | MIT |
| compute-scroll-into-view | 3.1.1 | MIT |
| conf | 15.0.2 | MIT |
| cookie | 1.1.1 | MIT |
| csstype | 3.2.3 | MIT |
| dayjs | 1.11.19 | MIT |
| debounce-fn | 6.0.0 | MIT |
| debug | 4.4.3 | MIT |
| dot-prop | 10.1.0 | MIT |
| electron-store | 11.0.2 | MIT |
| electron-updater | 6.6.2 | MIT |
| env-paths | 3.0.0 | MIT |
| fast-deep-equal | 3.1.3 | MIT |
| fast-uri | 3.1.0 | BSD-3-Clause |
| fs-extra | 10.1.0 | MIT |
| graceful-fs | 4.2.11 | ISC |
| is-mobile | 5.0.0 | MIT |
| js-yaml | 4.1.1 | MIT |
| json-schema-traverse | 1.0.0 | MIT |
| json-schema-typed | 8.0.2 | BSD-2-Clause |
| json2mq | 0.2.0 | MIT |
| jsonfile | 6.2.0 | MIT |
| lazy-val | 1.0.5 | MIT |
| lodash.escaperegexp | 4.1.2 | MIT |
| lodash.isequal | 4.5.0 | MIT |
| mimic-function | 5.0.1 | MIT |
| ms | 2.1.3 | MIT |
| react-is | 18.3.1 | MIT |
| react-router | 7.10.1 | MIT |
| react-router-dom | 7.10.1 | MIT |
| require-from-string | 2.0.2 | MIT |
| sax | 1.4.3 | BlueOak-1.0.0 |
| scroll-into-view-if-needed | 3.1.0 | MIT |
| semver | 7.7.3 | ISC |
| set-cookie-parser | 2.7.2 | MIT |
| string-convert | 0.2.1 | MIT |
| stubborn-fs | 2.0.0 | MIT |
| stubborn-utils | 1.0.2 | MIT |
| stylis | 4.3.6 | MIT |
| tagged-tag | 1.0.0 | MIT |
| throttle-debounce | 5.0.2 | MIT |
| tiny-typed-emitter | 2.1.0 | MIT |
| type-fest | 5.3.1 | (MIT OR CC0-1.0) |
| uint8array-extras | 1.5.0 | MIT |
| universalify | 2.0.1 | MIT |
| when-exit | 2.1.5 | MIT |

### 6.2 Python（154）

| 包 | 版本 | 许可证 |
| --- | --- | --- |
| addict | 2.4.0 | MIT License |
| aiohappyeyeballs | 2.6.2 | PSF-2.0 |
| aiohttp | 3.14.1 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache 2.0 |
| aliyun-python-sdk-core | 2.16.0 | Apache License 2.0 |
| aliyun-python-sdk-kms | 2.16.5 | Apache |
| annotated-doc | 0.0.4 | MIT |
| annotated-types | 0.7.0 | MIT License |
| antlr4-python3-runtime | 4.9.3 | BSD |
| anyascii | 0.3.3 | ISC License (ISCL) |
| anyio | 4.14.0 | MIT |
| argbind | 0.3.9 | MIT License |
| attrs | 26.1.0 | MIT |
| audioread | 3.1.0 | MIT |
| audiotsm | 0.1.2 | MIT |
| brotli | 1.2.0 | MIT |
| build | 1.2.2.post1 | MIT License |
| certifi | 2026.5.20 | MPL-2.0 |
| cffi | 2.0.0 | MIT |
| charset-normalizer | 3.4.7 | MIT |
| click | 8.4.1 | BSD-3-Clause |
| colorama | 0.4.6 | BSD License |
| contourpy | 1.3.3 | BSD 3-Clause License |
| contractions | 0.1.73 | MIT |
| crcmod | 1.7 | MIT |
| cryptography | 49.0.0 | Apache-2.0 OR BSD-3-Clause |
| cycler | 0.12.1 | Copyright (c) 2015, matplotlib project |
| datasets | 3.6.0 | Apache 2.0 |
| decorator | 5.3.1 | BSD-2-Clause |
| dill | 0.3.8 | BSD-3-Clause |
| docstring_parser | 0.18.0 | MIT |
| editdistance | 0.8.1 | MIT |
| einops | 0.8.2 | MIT |
| fastapi | 0.137.1 | MIT |
| filelock | 3.29.4 | MIT |
| fonttools | 4.63.0 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2025.3.0 | BSD 3-Clause License |
| funasr | 1.3.9 | The MIT License |
| gradio | 6.18.0 | Apache-2.0 |
| gradio_client | 2.5.0 | Apache-2.0 |
| groovy | 0.1.2 | MIT License |
| h11 | 0.16.0 | MIT |
| hf-gradio | 0.4.1 | MIT |
| hf-xet | 1.5.1 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| huggingface_hub | 1.19.0 | Apache-2.0 |
| hydra-core | 1.3.3 | MIT |
| idna | 3.18 | BSD-3-Clause |
| inflect | 7.5.0 | MIT License |
| jaconv | 0.5.0 | MIT License |
| jamo | 0.4.1 | http://www.apache.org/licenses/LICENSE-2.0 |
| jieba | 0.42.1 | MIT |
| Jinja2 | 3.1.6 | BSD License |
| jmespath | 0.10.0 | MIT |
| joblib | 1.5.3 | BSD-3-Clause |
| kaldifst | 1.8.0 | Apache-2.0 |
| kiwisolver | 1.5.0 | ========================= |
| lazy-loader | 0.5 | BSD-3-Clause |
| librosa | 0.11.0 | ISC |
| llvmlite | 0.47.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception |
| markdown-it-py | 4.2.0 | MIT License |
| MarkupSafe | 3.0.3 | BSD-3-Clause |
| matplotlib | 3.11.0 | License agreement for matplotlib versions 1.3.0 and later |
| mdurl | 0.1.2 | MIT License |
| modelscope | 1.37.1 | Apache-2.0 |
| more-itertools | 11.1.0 | MIT |
| mpmath | 1.3.0 | BSD |
| msgpack | 1.2.0 | Apache-2.0 |
| multidict | 6.7.1 | Apache License 2.0 |
| multiprocess | 0.70.16 | BSD-3-Clause |
| narwhals | 2.22.1 | MIT |
| networkx | 3.6.1 | BSD-3-Clause |
| numba | 0.65.1 | BSD |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| nvidia-ml-py | 13.610.43 | BSD |
| omegaconf | 2.3.1 | BSD License |
| openai-whisper | 20250625 | MIT |
| orjson | 3.11.9 | MPL-2.0 AND (Apache-2.0 OR MIT) |
| oss2 | 2.19.1 | MIT License |
| packaging | 25.0 | Apache Software License; BSD License |
| pandas | 3.0.3 | BSD 3-Clause License |
| pillow | 12.2.0 | MIT-CMU |
| pip | 25.1.1 | MIT |
| platformdirs | 4.10.0 | MIT |
| pooch | 1.9.0 | BSD-3-Clause |
| propcache | 0.5.2 | Apache-2.0 |
| protobuf | 7.35.1 | 3-Clause BSD License |
| psutil | 7.2.2 | BSD-3-Clause |
| pyahocorasick | 2.3.1 | BSD-3-Clause and Public-Domain |
| pyarrow | 24.0.0 | Apache-2.0 |
| pycparser | 3.0 | BSD-3-Clause |
| pycryptodome | 3.23.0 | BSD, Public Domain |
| pydantic | 2.13.4 | MIT |
| pydantic_core | 2.46.4 | MIT |
| pydub | 0.25.1 | MIT |
| Pygments | 2.20.0 | BSD-2-Clause |
| pynndescent | 0.6.0 | BSD-2-Clause |
| pyparsing | 3.3.2 | MIT |
| pyproject_hooks | 1.2.0 | MIT License |
| pyrubberband | 0.4.0 | ISC |
| python-dateutil | 2.9.0.post0 | Dual License |
| python-multipart | 0.0.32 | Apache-2.0 |
| pytz | 2026.2 | MIT |
| PyYAML | 6.0.3 | MIT |
| regex | 2026.5.9 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| rich | 15.0.0 | MIT |
| safehttpx | 0.1.7 | MIT License |
| safetensors | 0.8.0 | Apache Software License |
| scikit-learn | 1.9.0 | BSD-3-Clause |
| scipy | 1.17.1 | BSD-3-Clause |
| semantic-version | 2.10.0 | BSD |
| sentencepiece | 0.2.1 | Apache-2.0 (per upstream; no licence file in wheel) |
| setuptools | 79.0.1 | MIT |
| shellingham | 1.5.4 | ISC License |
| simplejson | 4.1.1 | MIT OR AFL-2.1 |
| six | 1.17.0 | MIT |
| sortedcontainers | 2.4.0 | Apache 2.0 |
| soundfile | 0.14.0 | BSD 3-Clause License |
| soxr | 1.1.0 | LGPL-2.1-or-later |
| spaces | 0.50.4 | Apache-2.0 |
| sqlite_bro | 0.13.1 | MIT License |
| starlette | 1.3.1 | BSD-3-Clause |
| sv-ttk | 2.6.0 | MIT |
| sympy | 1.14.0 | BSD |
| tensorboardX | 2.6.5 | MIT |
| textsearch | 0.0.24 | MIT |
| threadpoolctl | 3.6.0 | BSD-3-Clause |
| tiktoken | 0.13.0 | MIT License |
| tokenizers | 0.22.2 | Apache Software License |
| tomlkit | 0.14.0 | MIT |
| torch | 2.8.0+cu129 | BSD-3-Clause |
| torch-complex | 0.4.4 | Apache Software License |
| torchaudio | 2.8.0+cu129 | BSD License |
| torchcodec | 0.14.0 | BSD-3-Clause |
| torchvision | 0.23.0+cu129 | BSD |
| tqdm | 4.68.2 | MPL-2.0 AND MIT |
| transformers | 5.12.1 | Apache 2.0 License |
| typeguard | 4.5.2 | MIT |
| typer | 0.25.1 | MIT |
| typing-inspection | 0.4.2 | MIT |
| typing_extensions | 4.15.0 | PSF-2.0 |
| tzdata | 2026.2 | Apache-2.0 |
| umap-learn | 0.5.12 | BSD |
| urllib3 | 2.7.0 | MIT |
| uvicorn | 0.49.0 | BSD-3-Clause |
| voxcpm | 2.0.3.post8+g856d2fc2a | Apache-2.0 |
| wetext | 0.1.4 | Apache-2.0 |
| wheel | 0.45.1 | MIT License |
| winpython | 16.5.20250614 | MIT License |
| xxhash | 3.7.0 | BSD |
| yarl | 1.24.2 | Apache-2.0 |

---

## 七、声明范围

本文件依据发布副本的实际文件内容自动生成并人工复核。若你发现遗漏或错误的
归属信息，请通过仓库 Issue 告知，我们将及时更正。

本文件不构成律师法律意见。
