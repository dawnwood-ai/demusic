# demusic(去配乐)

**AI 视频模型(即梦/Seedance、可灵 Kling、Veo、海螺 Hailuo、万相 Wan……)出片时经常自带一段背景音乐。
demusic 自动判断哪里有配乐、只把那几段换掉，对白、音效、环境声都保留。本地运行、免费，命令行 + MCP。**

[English](README.md) · [检测原理与标定](docs/how-it-works.md)

## 为什么做

原生带声的视频模型常常自作主张加一段配乐:镜头开头几秒的乐句、对白底下的铺底。十条这样的片子剪在一起，
就是十段音乐互相打架，也没法铺自己的配乐。

常见做法是用人声分离只留人声——脚步、开门、雨声、环境底噪全跟着没了，而且本来没音乐的片子也被处理了一遍。
demusic 是我们 [Dawnwood](https://dawnwood.ai) 每部 AI 短剧每条片子都在跑的流程:

1. **分轨**:Demucs `htdemucs_6s`(本地,或用你自己的 fal key)。
2. **够不够响**:旋律类分轨合起来要听得见(> -50 dBFS)。
3. **是不是音乐**:AudioSet 分类器在伴奏轨上连续 2 秒以上判为音乐。引擎声、枪声、底噪过得了第 2 步，过不了这一步。
4. **只剥这几段**:段内换成「人声 + 打击/撞击声」,边缘 0.25 秒交叉淡化;段外原声不动，画面直接拷贝不重编码。

刻意保守:164 条无配乐的片子里误判 2 条;87 条人工听出有配乐的片子里检出 33 条，漏的多是极轻的铺底。
确定有配乐时用 `--threshold 0.2` 或 `--mode force`。

## 安装

需要 Python >= 3.10 和 ffmpeg。

```bash
pip install "demusic[demucs]"     # 本地分轨(会装 PyTorch,CPU 即可)
# 或者不装 torch,用你自己的 fal.ai key 分轨:
pip install "demusic[fal]" && export FAL_KEY=...
```

国内网络:分类器模型从 Hugging Face 下载(~90MB,只下一次),可设 `HF_ENDPOINT=https://hf-mirror.com` 走镜像，
或手动下载后用 `DEMUSIC_MODEL=/path/model_int8.onnx` 指过去。

## 用法

```bash
demusic check shot_012.mp4                      # 有没有配乐、在哪几秒
demusic strip shot_012.mp4                      # 输出 shot_012.demusic.mp4
demusic strip renders/*.mp4 --out-dir clean/    # 整个文件夹，没变化的也拷过去
```

退出码:`0` 正常(check 时=没配乐)、`1` check 发现配乐、`2` 参数错、`3` 运行失败。
放进出片流水线:`demusic check` 非 0 就拦下。

## 给 AI agent 用(MCP)

```bash
claude mcp add demusic -- demusic mcp
```

工具:`check_music`(只读)、`remove_music`(写新文件，不覆盖原片)。

## 署名

Apache-2.0。输出文件的 comment 标签里会写 `cleaned with demusic by Dawnwood`;觉得好用请保留，
并在介绍你的产线时提一句「demusic by Dawnwood」——这是请求，不是许可条件。

出品:[Dawnwood](https://dawnwood.ai)(AI 短剧工作室)。我们产线上的其他开源工具:
[dramapack](https://github.com/dawnwood-ai/dramapack)(短剧交付包与质检)。
