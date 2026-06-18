"""
parser.py 单测。
运行：python tests/test_parser.py（在 exam-search/ 目录下执行）
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parser import parse_markdown, parse_notebook


# ─── Markdown 测试 ────────────────────────────────────────────────────────────

def test_code_comment_not_heading():
    """代码块里的 # 注释行不得被当作标题。"""
    md = (
        "## 优化方法\n"
        "\n"
        "```python\n"
        "# 定义 ADMM\n"
        "x = 1\n"
        "```\n"
        "\n"
        "这是散文。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    code_chunks = [c for c in chunks if c['ctype'] == 'code']
    prose_chunks = [c for c in chunks if c['ctype'] == 'prose']

    assert len(code_chunks) == 1, f"期望 1 个 code chunk，实际 {len(code_chunks)}"
    assert '# 定义 ADMM' in code_chunks[0]['text']
    assert code_chunks[0]['heading_path'] == '优化方法'

    # 不能有以"定义 ADMM"为标题的块
    all_paths = [c['heading_path'] for c in chunks]
    assert not any('定义 ADMM' in p for p in all_paths), f"发现误识别的标题路径：{all_paths}"

    assert len(prose_chunks) >= 1
    assert any('这是散文' in c['text'] for c in prose_chunks)


def test_tilde_fence_not_closed_by_backtick():
    """~~~ 围栏不能被 ``` 闭合。"""
    md = (
        "~~~python\n"
        "# 里面\n"
        "```\n"
        "还在围栏里\n"
        "~~~\n"
        "围栏外\n"
    )
    chunks = parse_markdown(md, 'test.md')
    code_chunks = [c for c in chunks if c['ctype'] == 'code']
    prose_chunks = [c for c in chunks if c['ctype'] == 'prose']

    assert len(code_chunks) == 1
    assert '```' in code_chunks[0]['text']
    assert '还在围栏里' in code_chunks[0]['text']
    # 围栏外的散文应该出现
    assert any('围栏外' in c['text'] for c in prose_chunks)


def test_two_backticks_in_fence_not_close():
    """代码块内两个反引号不误判为闭合。"""
    md = (
        "```python\n"
        "x = ``some code``\n"
        "still inside\n"
        "```\n"
    )
    chunks = parse_markdown(md, 'test.md')
    code_chunks = [c for c in chunks if c['ctype'] == 'code']
    assert len(code_chunks) == 1
    assert 'still inside' in code_chunks[0]['text']
    assert '``some code``' in code_chunks[0]['text']


def test_longer_fence_needs_longer_close():
    """4 个反引号开的围栏需要 4 个反引号才能关闭，3 个不算。"""
    md = (
        "````python\n"
        "```\n"
        "still inside\n"
        "````\n"
        "outside\n"
    )
    chunks = parse_markdown(md, 'test.md')
    code_chunks = [c for c in chunks if c['ctype'] == 'code']
    assert len(code_chunks) == 1
    assert '```' in code_chunks[0]['text']
    assert 'still inside' in code_chunks[0]['text']


def test_multi_level_heading_path():
    """多级标题的 heading_path 不串层。"""
    md = (
        "# 一\n"
        "\n"
        "## 1.1\n"
        "\n"
        "### 名词\n"
        "\n"
        "这是名词的定义。\n"
        "\n"
        "## 1.2\n"
        "\n"
        "另一小节。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    prose_chunks = [c for c in chunks if c['ctype'] == 'prose']

    # 名词下的块
    under_noun = [c for c in prose_chunks if '名词' in c['heading_path']]
    assert len(under_noun) >= 1
    assert under_noun[0]['heading_path'] == '一 > 1.1 > 名词'

    # 1.2 节不包含"名词"
    under_12 = [c for c in prose_chunks if '1.2' in c['heading_path']]
    assert len(under_12) >= 1
    assert '名词' not in under_12[0]['heading_path']
    assert under_12[0]['heading_path'] == '一 > 1.2'


def test_same_level_heading_replaces():
    """同级新标题正确替换旧标题（不叠加）。"""
    md = (
        "## 第一节\n"
        "\n"
        "内容一。\n"
        "\n"
        "## 第二节\n"
        "\n"
        "内容二。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    prose_chunks = [c for c in chunks if c['ctype'] == 'prose']

    paths = [c['heading_path'] for c in prose_chunks]
    assert '第一节' in paths
    assert '第二节' in paths
    # 第二节的路径不应含第一节
    second = [c for c in prose_chunks if c['heading_path'] == '第二节']
    assert len(second) >= 1


def test_position_resets_on_heading():
    """遇到新标题时 position 重置为 0。"""
    md = (
        "## 第一节\n"
        "\n"
        "散文一。\n"
        "\n"
        "散文二。\n"
        "\n"
        "## 第二节\n"
        "\n"
        "散文三。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    # 第二节下的第一块 position 应为 0
    second = [c for c in chunks if '第二节' in c['heading_path']]
    assert second[0]['position'] == 0


def test_prose_position_increments():
    """同一节内 position 依次递增。"""
    md = (
        "## 节\n"
        "\n"
        "段一。\n"
        "\n"
        "```\n"
        "code\n"
        "```\n"
        "\n"
        "段二。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    positions = [c['position'] for c in chunks]
    assert positions == [0, 1, 2]


def test_empty_chunks_discarded():
    """空块（只有空行/空格的块）不产生 chunk。"""
    md = (
        "## 节\n"
        "\n"
        "\n"
        "\n"
        "有内容的段落。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    assert len(chunks) == 1
    assert chunks[0]['text'] == '有内容的段落。'


def test_no_heading_before_first_heading():
    """标题前的内容 heading_path 为空字符串。"""
    md = (
        "前导散文。\n"
        "\n"
        "## 第一节\n"
        "\n"
        "节内容。\n"
    )
    chunks = parse_markdown(md, 'test.md')
    preamble = [c for c in chunks if c['heading_path'] == '']
    assert len(preamble) >= 1
    assert '前导散文' in preamble[0]['text']


# ─── ipynb 测试 ───────────────────────────────────────────────────────────────

def test_ipynb_source_as_list():
    """source 为 list[str] 时能正确拼接。"""
    nb = {
        'nbformat': 4,
        'cells': [
            {
                'cell_type': 'markdown',
                'source': ['## 标题\n', '\n', '散文内容。']
            }
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'test.ipynb')
    prose = [c for c in chunks if c['ctype'] == 'prose']
    assert len(prose) >= 1
    assert '散文内容' in prose[0]['text']


def test_ipynb_source_as_str():
    """source 为 str 时能正确处理。"""
    nb = {
        'nbformat': 4,
        'cells': [
            {
                'cell_type': 'code',
                'source': 'x = 1\ny = 2',
                'outputs': []
            }
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'test.ipynb')
    assert len(chunks) == 1
    assert 'x = 1' in chunks[0]['text']
    assert chunks[0]['ctype'] == 'code'


def test_ipynb_outputs_ignored():
    """code cell 的 outputs 不计入 chunk text。"""
    nb = {
        'nbformat': 4,
        'cells': [
            {
                'cell_type': 'code',
                'source': 'print("hello")',
                'outputs': [{'output_type': 'stream', 'text': 'OUTPUT_SENTINEL_XYZ'}]
            }
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'test.ipynb')
    assert len(chunks) == 1
    assert 'OUTPUT_SENTINEL_XYZ' not in chunks[0]['text']


def test_ipynb_heading_carries_across_cells():
    """markdown cell 的标题状态跨 cell 延续。"""
    nb = {
        'nbformat': 4,
        'cells': [
            {
                'cell_type': 'markdown',
                'source': '## ADMM\n\n这是ADMM的介绍。'
            },
            {
                'cell_type': 'code',
                'source': 'def admm(): pass'
            },
            {
                'cell_type': 'markdown',
                'source': '继续在ADMM章节下的内容。'
            }
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'test.ipynb')
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk['heading_path'] == 'ADMM', \
            f"期望 heading_path='ADMM'，实际 '{chunk['heading_path']}' in chunk: {chunk}"


def test_ipynb_raw_cell_skipped():
    """raw cell 被跳过，不产生 chunk。"""
    nb = {
        'nbformat': 4,
        'cells': [
            {'cell_type': 'raw', 'source': '不应出现的内容'},
            {'cell_type': 'code', 'source': 'x = 1'}
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'test.ipynb')
    assert len(chunks) == 1
    assert '不应出现' not in chunks[0]['text']


def test_ipynb_invalid_json_returns_empty():
    """无效 JSON 返回空列表并不抛异常。"""
    chunks = parse_notebook('this is not json {{ }}', 'bad.ipynb')
    assert chunks == []


def test_ipynb_old_format_worksheets():
    """老格式 worksheets 能正确解析。"""
    nb = {
        'nbformat': 3,
        'worksheets': [
            {
                'cells': [
                    {
                        'cell_type': 'code',
                        'source': 'print("hello")',
                        'outputs': []
                    }
                ]
            }
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'old.ipynb')
    assert len(chunks) >= 1
    assert 'print' in chunks[0]['text']


def test_ipynb_markdown_cell_heading_parsed():
    """ipynb markdown cell 里的标题被解析，产生正确 heading_path。"""
    nb = {
        'nbformat': 4,
        'cells': [
            {
                'cell_type': 'markdown',
                'source': '# 第一章\n\n## 第一节\n\n散文。'
            }
        ]
    }
    chunks = parse_notebook(json.dumps(nb), 'test.ipynb')
    prose = [c for c in chunks if c['ctype'] == 'prose']
    assert len(prose) >= 1
    assert prose[0]['heading_path'] == '第一章 > 第一节'


# ─── 运行 ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    tests = [
        test_code_comment_not_heading,
        test_tilde_fence_not_closed_by_backtick,
        test_two_backticks_in_fence_not_close,
        test_longer_fence_needs_longer_close,
        test_multi_level_heading_path,
        test_same_level_heading_replaces,
        test_position_resets_on_heading,
        test_prose_position_increments,
        test_empty_chunks_discarded,
        test_no_heading_before_first_heading,
        test_ipynb_source_as_list,
        test_ipynb_source_as_str,
        test_ipynb_outputs_ignored,
        test_ipynb_heading_carries_across_cells,
        test_ipynb_raw_cell_skipped,
        test_ipynb_invalid_json_returns_empty,
        test_ipynb_old_format_worksheets,
        test_ipynb_markdown_cell_heading_parsed,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(0 if failed == 0 else 1)
