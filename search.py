"""
命令行检索入口（P0）。

用法：
    python search.py "<query>" [--type {all,prose,code}] [--topk N] [--index index.pkl]
"""

import sys
import os
import pickle
import argparse

# Windows 终端 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranking

_PREVIEW_LINES = 5  # 每条结果显示的正文行数


def _format_result(i, result):
    """格式化单条结果为可打印字符串列表。"""
    lines = []
    lines.append(
        f"[{i}] {result['heading_path']}  "
        f"({result['file']} | {result['ctype']} | score={result['score']:.4f})"
    )
    text_lines = result['text'].splitlines()
    for line in text_lines[:_PREVIEW_LINES]:
        lines.append(f"    {line}")
    if len(text_lines) > _PREVIEW_LINES:
        lines.append(f"    ... ({len(text_lines) - _PREVIEW_LINES} 行省略)")
    return lines


def main():
    parser = argparse.ArgumentParser(description='离线资料检索引擎 CLI')
    parser.add_argument('query', help='检索查询词')
    parser.add_argument(
        '--type', choices=['all', 'prose', 'code'], default='all',
        dest='type_filter', metavar='TYPE',
        help='块类型过滤：all（默认）/ prose（只看概念）/ code（只看代码）'
    )
    parser.add_argument('--topk', type=int, default=10,
                        help='最多返回结果数（默认 10）')
    parser.add_argument('--index', default='index.pkl',
                        help='索引文件路径（默认 index.pkl）')
    args = parser.parse_args()

    if not os.path.exists(args.index):
        print(
            f"错误：索引文件 {args.index!r} 不存在。\n"
            f"请先运行：python build_index.py <materials_dir>"
        )
        sys.exit(1)

    with open(args.index, 'rb') as f:
        index = pickle.load(f)

    results, intent = ranking.search(
        index, args.query,
        aliases=None,
        type_filter=args.type_filter,
        topk=args.topk,
    )

    if not results:
        print(f"未找到与 {args.query!r} 相关的结果。")
        return

    intent_hint = f"，意图: {intent}" if intent else ""
    print(
        f"共 {len(results)} 条结果"
        f"（查询: {args.query!r}，类型: {args.type_filter}{intent_hint}）\n"
    )
    for i, r in enumerate(results, 1):
        for line in _format_result(i, r):
            print(line)
        print()


if __name__ == '__main__':
    main()
