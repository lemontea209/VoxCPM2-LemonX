# NOTICE

**本修改版版本：v2.15.0**

## 项目来源

本整合包基于 [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) 构建，
并在余子越制作的 YZY 启动器与便携整合包基础上进行了二次修改。

本项目与 OpenBMB、ModelBest、阿里巴巴集团、FunAudioLLM / FunASR、OpenAI、
FFmpeg、Electron 及其他第三方项目**不存在隶属或背书关系**。

## 作者与授权

- **原始模型与推理代码：** VoxCPM，版权归 OpenBMB 所有，依 Apache License 2.0 授权。
  许可证全文见 `win-unpacked/python/LICENSE`。
- **YZY 启动器与原整合包：** 余子越（[哔哩哔哩 余子越Talk](https://space.bilibili.com/3493266750179909)）。
- **本修改版：** Lemon-X（[哔哩哔哩 Lemon-X](https://space.bilibili.com/3690974001761138)）。

余子越已单独授权 Lemon-X 将本二次修改整合包发布到 GitHub。

该授权是 Lemon-X 发布本版本所依据的**专项许可**。除非另有明确许可证文件，
它**不表示**余子越的独立代码已按 Apache License 2.0 授权，也**不自动**向第三方
授予更广泛的修改、再分发、商用或再许可权利。若你希望在本整合包基础上继续
分发或修改余子越创作的部分，请自行取得其许可。

## 本版所做的修改

音色管理、任务队列并发处理、长文本段间响度对齐、明暗主题适配、云端交互
与稳定性修复，以及使用本地模型文件的路径调整。详见发布说明与 `CHANGES.md`。

依 Apache License 2.0 第 4(b) 条，被修改的上游文件已在文件头部标注修改声明：

- `win-unpacked/python/src/voxcpm/core.py`
- `win-unpacked/python/src/voxcpm/zipenhancer.py`

上游示例入口 `app.py` 已移除，改用整合包自有入口。

## 许可范围（重要）

**本发行物包含多种许可证，并非所有文件统一适用 Apache License 2.0。**

各第三方项目、模型权重、运行时及依赖组件分别适用其各自的原始许可证，
完整清单见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)，
许可证全文见 `LICENSES/` 目录及各组件所在目录。

由 Lemon-X 与余子越享有权利的整合层，不得用于未经许可的商业盈利贩卖。
**本声明不限制或修改第三方许可证已经授予的权利**——各第三方组件（包括
FFmpeg、模型权重及全部依赖库）仍分别适用其原始许可证，其中多数允许商业使用。

OpenBMB 的 VoxCPM2 代码与权重可商用，**不等于**本整合包整体可商用。

## 关于 API 密钥的处理

本整合包的云端语音功能需要用户自行提供 MiniMax API Key。该密钥：

- **不随整合包分发**，包内不含任何密钥；
- 仅在用户明确点击「安全保存」后才写入磁盘；
- 落盘内容为 Windows DPAPI 加密后的密文，而非明文，仅当前 Windows 账户可解密；
- 存放于用户目录 `%LOCALAPPDATA%\YZYLauncher\VoxCPM2\`，不在整合包目录内。

## 无担保

本发行物按「现状（AS IS）」提供。各组件的担保排除、责任限制及使用条件
以其各自许可证和适用法律为准。本 NOTICE 不修改、替代或缩减任何第三方许可证。
