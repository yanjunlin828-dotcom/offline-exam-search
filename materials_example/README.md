# materials_example

这是一份**真实的示例资料**（连同已经构建好的索引 `index_example.pkl`），让你不用先准备自己的笔记也能直接跑起来体验检索效果，或者用来回归测试改动有没有把检索效果搞坏。

- `final/` —— 示例的 `.md` / `.ipynb` 笔记原文。
- `index_example.pkl` —— 用 `python build_index.py materials_example -o materials_example/index_example.pkl` 对上面这批文件构建出的索引，文件名和正式的 `index.pkl` 不一样，是为了避免被 `.gitignore` 里那条 `index.pkl` 规则误排除。

> 注意：项目正式使用时，你自己的笔记应该放在仓库根目录的 `materials/`（已在 `.gitignore` 里排除，不会被提交），生成的索引也叫 `index.pkl`。`materials_example/` 只是用于演示和测试，跟你日常用的 `materials/` 互不影响、互不覆盖。

## 直接体验示例效果

不需要自己准备任何笔记，直接用仓库里现成的索引：

```bash
python server.py --index materials_example/index_example.pkl --aliases aliases.txt --materials materials_example --port 8000
```

然后浏览器打开 `http://127.0.0.1:8000/`，可以搜 "ADMM"、"范数"、"一致性 ADMM 通信模式" 之类的词试试效果。

## 想换成你自己的笔记测试

1. 把 `materials_example/final/` 下的文件换成（或者新建一个目录放）你自己的 `.md`/`.ipynb`。
2. 重新构建索引：
   ```bash
   python build_index.py materials_example -o materials_example/index_example.pkl
   ```
3. 重启第一步那条 `server.py` 命令（Ctrl+C 关掉旧进程再重新跑）。

## 用来做回归测试

如果你改了 `tokenizer.py`/`parser.py`/`ranking.py` 之类的核心逻辑，想确认有没有把检索效果搞坏，可以：

```bash
python build_index.py materials_example -o materials_example/index_example.pkl
python search.py "ADMM 定义" --index materials_example/index_example.pkl
python search.py "一致性 admm 分布式" --index materials_example/index_example.pkl
```

对比改动前后的排序和分数，比凭感觉判断靠谱。
