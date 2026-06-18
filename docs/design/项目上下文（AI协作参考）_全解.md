# 项目上下文文档（AI 协作参考）

> 本文件供 AI 助手在新对话开始时快速建立完整上下文。每次继续开发前请先阅读本文件、`CLAUDE.md`（硬约束）、`P1实施规格_代码实现.md`（P1 规格）。

---

## 一、项目定位

**离线本地资料检索引擎**，目标场景：上机考试（无网络、环境不可控）。  
用户考前将 `.md` / `.ipynb` 笔记预处理成倒排索引，考中在浏览器里按关键词检索，能区分"概念定义"与"代码实现"，支持中文两字概念（如"范数""凸性"），支持同义词别名扩展。

**关键约束（不可违背）**：
1. 只用 Python 标准库，零 pip 运行依赖
2. 不依赖 SQLite FTS5 / trigram 等编译期特性
3. 前端 JS（`marked.js`、`highlight.js`）本地打包，禁止 CDN

---

## 二、当前实现状态

### 已完成（P0 + P1 逻辑层）

| 文件 | 功能 | 状态 |
|------|------|------|
| `tokenizer.py` | 中英文分词（ASCII → 小写词、CJK → 字符 bigram） | ✓ 完成，17 测试通过 |
| `parser.py` | Markdown 围栏状态机 + ipynb JSON 解析 → chunk 列表 | ✓ 完成，18 测试通过 |
| `build_index.py` | 构建倒排索引并序列化到 `index.pkl` | ✓ 完成 |
| `ranking.py` | BM25 评分 + 意图识别 + 别名展开 | ✓ 完成，36 测试通过 |
| `search.py` | 命令行检索接口（P0） | ✓ 完成 |
| `aliases.txt` | 同义词定义（4 组） | ✓ 完成 |
| `web/index.html` | 前端搜索 UI | ✓ 完成（约 420 行） |
| `web/viewer.html` | 备用查看器 | ✓ 存在 |
| `web/marked.min.js` | Markdown 渲染库（本地） | ✓ 存在 |
| `web/highlight.min.js` | 代码高亮库（本地） | ✓ 存在 |
| `web/highlight-theme.css` | 高亮主题 | ✓ 存在 |

### **未完成（待实现）**

| 文件 | 功能 | 参考规格 |
|------|------|---------|
| **`server.py`** | HTTP Web 服务，提供 `/`、`/search`、`/web/` 路由 | `P1实施规格_代码实现.md` §4 |

`server.py` 是唯一缺失的核心模块，完成后项目即可完整运行。

---

## 三、模块职责速查

### `tokenizer.py`（77 行）
**索引与查询的唯一分词源，两侧必须共用相同配置。**

```python
tokenize(text, enable_unigram=False, cjk_ext=False) -> list[str]
```

- ASCII 连续字母数字 → 1 个小写 token
- 连续 CJK 字符 → 滑动 bigram（"范数学" → ["范数", "数学"]）
- 单个 CJK 字 → 原字输出；`enable_unigram=True` 时同时输出每个单字
- 其余字符（标点、空白）→ 跳过，不报错

### `parser.py`（276 行）
**解析文件为 chunk 列表，不执行分词。**

```python
parse_markdown(text, file) -> list[dict]
parse_notebook(text, file) -> list[dict]
```

每个 chunk：
```python
{
    'file': '凸优化综述（ADMM·范数·凸性）_全解.md',
    'heading': 'ADMM 定义',
    'heading_path': 'ADMM > 定义',
    'text': '正文内容...',
    'ctype': 'prose' | 'code',
    'position': 0   # 在同一节内的顺序（0 最近标题）
}
```

关键：**先判围栏状态，再判标题**——代码块内的 `# 注释` 不识别为标题。

### `build_index.py`（150 行）
**一次性构建管道，输出 `index.pkl`。**

```bash
python build_index.py <materials_dir> [-o index.pkl]
```

index.pkl 结构：
```python
{
    'version': 1,
    'config': {'enable_unigram': False, 'cjk_ext': False},
    'chunks': [chunk_with_tf, ...],   # 含 id, tf, dl, heading_tokens
    'inverted': {token: [(chunk_id, count), ...]},
    'df': {token: doc_frequency},
    'N': total_chunks,
    'avgdl': average_chunk_length
}
```

### `ranking.py`（267 行）
**检索核心，供 `search.py` 和 `server.py` 共用。**

```python
# 主检索函数
results, intent = search(index, query, aliases, type_filter="all", topk=20)
# results: list[dict]，含 {file, heading_path, ctype, score, text}，按 score 降序
# intent:  "prose" | "code" | None

# 辅助函数
intent, cleaned_query = detect_intent(query)
matched_phrases = expand_aliases(topic_token_set, query, alias_groups, cfg)
alias_groups = load_aliases("aliases.txt")  # list[list[str]]
score = bm25(index, chunk, qtokens)
```

评分公式：
```
final = BM25(chunk, score_tokens)
      + 2.0 * (heading_hits / n_topics)   # 关键词出现在标题
      + 0.5 * (1 / (1 + position))         # 越近标题得分越高
      + 1.5 * (1 if intent and ctype==intent else 0)  # 意图类型匹配
```

### `search.py`（89 行）
命令行接口，用法：
```bash
python search.py "ADMM 定义" [--type {all,prose,code}] [--topk 10] [--index index.pkl]
```

### `server.py`（待实现）
完整规格见 `P1实施规格_代码实现.md` §4，关键接口：

```
GET /                                   → web/index.html
GET /search?q=<query>&type=<all|prose|code>  → JSON 响应
GET /web/<filename>                     → 静态文件（含路径穿越防护）
```

启动方式：
```bash
python server.py [--index index.pkl] [--aliases aliases.txt] [--port 8000]
```

---

## 四、数据流总览

```
构建阶段（一次性）
  materials/*.md + *.ipynb
    → parser.py: 解析为 chunk 列表
    → tokenizer.py: 对 text + heading 分词
    → build_index.py: 构建倒排索引
    → index.pkl（磁盘）

查询阶段（每次搜索，纯内存）
  用户输入 query
    → detect_intent() → intent + cleaned_query
    → tokenize(cleaned_query) → topic_tokens
    → expand_aliases() → alias_phrase_tokens
    → inverted index 取并集 → 候选 chunk IDs
    → BM25 + 加分项 → 排序
    → (results, intent) 返回给 search.py / server.py
    → 展示给用户
```

---

## 五、关键不变量（容易破坏的地方）

1. **分词一致性**：`build_index.py` 和 `search()` 必须用完全相同的 `tokenize()` 配置（`enable_unigram`、`cjk_ext`）。索引与查询配置不同 → 静默查不到。

2. **围栏先于标题**：`parser.py` 在判断是否为标题行之前，必须先检查是否处于代码围栏内。

3. **ipynb 四要点**：
   - `cell["source"]` 兼容 `str` 和 `list[str]`（需 join）
   - 只索引 `source`，完全忽略 `outputs`
   - markdown cell 也要解析标题
   - 标题状态跨 cell 保持，围栏状态每 cell 重置

4. **别名 AND 语义**：别名短语的所有 token 必须全部出现在同一 chunk 才算命中（交集，不是并集）。防止含单词 "method" 的无关块被 "alternating direction method of multipliers" 误召。

5. **BM25 IDF 公式**：`ln(1 + (N - df + 0.5) / (df + 0.5))`（Lucene 式），防止常见词出现负 IDF。

6. **HTML 转义顺序**：先转义（`&` `<` `>` `"`），再包 `<mark>` 标签。顺序反了会导致 XSS 或渲染错误。

7. **intent 是软偏好**：意图识别只影响评分（W_TYPE 加分）和前端默认开关，不硬过滤结果。

8. **`search()` 返回 tuple**：`results, intent = search(...)` ——不要当 list 直接用。

---

## 六、测试策略

```bash
python -m pytest tests/ -v
```

测试文件：
- `tests/test_tokenizer.py`：17 个，覆盖 bigram、边界、特殊字符
- `tests/test_parser.py`：18 个，覆盖围栏状态机、标题路径、ipynb 格式
- `tests/test_ranking.py`：36 个，覆盖 BM25、意图识别、别名展开、gold 查询

**黄金测试用例**（任何改动后必须通过）：

| 查询 | 期望 |
|------|------|
| `ADMM 定义` | prose 块排第 1，`intent == 'prose'` |
| `ADMM 实现` | code 块优先，`intent == 'code'` |
| `范数` / `凸性` | 能召回（bigram 验证） |
| `交替方向乘子法` | 与 `ADMM` 召回相同文件（别名验证） |
| `ADMM 定义` 时查含单词 `method` 的无关块 | 不出现（AND 语义验证） |
| `a AND "b*` 等特殊字符 | 不报错 |
| 代码块内 `# 定义 xxx` | 不识别为标题 |

---

## 七、实现 server.py 的完整参考

（完整规格在 `P1实施规格_代码实现.md` §4，以下为执行时要点）

```python
# 全局只加载一次（线程安全，只读）
_INDEX   = pickle.load(open(args.index, 'rb'))
_ALIASES = ranking.load_aliases(args.aliases)
_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

# 路径穿越防护（必须用 realpath，不能用 normpath）
web_dir_real = os.path.realpath(_WEB_DIR)
requested_real = os.path.realpath(os.path.join(_WEB_DIR, rel_path))
if not requested_real.startswith(web_dir_real + os.sep):
    send_404()

# 中文 URL 解码（urllib 自动处理 utf-8，无需手动）
from urllib.parse import parse_qs, urlparse
params = parse_qs(urlparse(self.path).query)
q = params.get('q', [''])[0]

# JSON 响应（中文不转义 unicode）
body = json.dumps(resp, ensure_ascii=False).encode('utf-8')

# MIME 类型
MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8',
}
```

响应 JSON 结构：
```json
{
  "query": "原始查询字符串",
  "intent": "prose",
  "results": [
    {
      "file": "materials/凸优化综述（ADMM·范数·凸性）_全解.md",
      "heading_path": "优化方法综述 > ADMM > 定义",
      "ctype": "prose",
      "score": 12.34,
      "text": "ADMM 是交替方向乘子法..."
    }
  ]
}
```

---

## 八、前端高亮规则（web/index.html 已实现）

**prose 块**（严格顺序）：
1. HTML 转义：`&` → `&amp;`，`<` → `&lt;`，`>` → `&gt;`，`"` → `&quot;`
2. 关键词包 `<mark>`（在转义后的字符串上操作）
3. 赋给 `elem.innerHTML`

**code 块**：
1. `hljs.highlightAuto(text).value`（已是 HTML）
2. 用 `<pre><code class="hljs">` 包裹
3. 不叠加 `<mark>`

---

## 九、aliases.txt 格式

```
# 注释行（井号开头）
ADMM=交替方向乘子法=alternating direction method of multipliers
范数=norm
凸性=convexity
近端梯度法=proximal gradient method
```

每行一组等价写法，`=` 分隔，空行忽略。

---

## 十、快速启动

```bash
# 1. 构建索引
python build_index.py materials/ -o index.pkl

# 2. 命令行搜索（P0，无需 server.py）
python search.py "ADMM 定义"
python search.py "范数" --type prose --topk 5

# 3. Web 界面（P1，需先实现 server.py）
python server.py --index index.pkl --port 8000
# 浏览器打开 http://localhost:8000

# 4. 运行测试
python -m pytest tests/ -v
```

---

## 十一、明确不在范围内

- 语义 / 向量检索
- SQLite FTS5 加速（版本不可控，即使做也必须能降级到纯 Python 路径）
- 完整 CommonMark 解析（围栏正确即可）
- 单字中文查询（已知不支持，需显式开启 `enable_unigram=True`）
