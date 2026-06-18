"""
构建期：遍历资料目录 → 解析 → 分词 → 建倒排索引 → 序列化到 index.pkl。

用法：
    python build_index.py <materials_dir> [-o index.pkl]
"""

import sys
import os
import pickle
import argparse
from collections import Counter

# 保证无论从哪个目录运行，都能找到同目录的本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import parser as doc_parser
from tokenizer import tokenize

# 查询期读取同一份 config，保证索引/查询切词一致（决策 G）
_DEFAULT_CONFIG = {"enable_unigram": False, "cjk_ext": False}


def _collect_files(materials_dir):
    """递归收集 .md/.ipynb，按相对路径排序，返回 [(rel_path, abs_path)]。"""
    materials_dir = os.path.abspath(materials_dir)
    entries = []
    for dirpath, dirs, filenames in os.walk(materials_dir):
        dirs[:] = [d for d in dirs if d != '.ipynb_checkpoints']
        for filename in filenames:
            if filename.endswith('.md') or filename.endswith('.ipynb'):
                abs_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(abs_path, materials_dir).replace('\\', '/')
                entries.append((rel_path, abs_path))
    entries.sort(key=lambda x: x[0])
    return entries, materials_dir


def _read_file(abs_path):
    """读取文件文本，失败时返回 None 并打印警告。"""
    try:
        with open(abs_path, encoding='utf-8') as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        print(f"[警告] 跳过文件 {abs_path!r}: {e}")
        return None


def _make_chunk(raw, chunk_id, cfg):
    """给 parser 产生的 chunk 雏形填充 tf/dl/heading_tokens/id。

    Returns:
        完整 chunk dict，或 None（若 dl == 0 则丢弃）。
    """
    text_tokens = tokenize(raw['text'], **cfg)
    heading_tokens_list = tokenize(raw['heading'], **cfg)
    combined = text_tokens + heading_tokens_list
    tf = dict(Counter(combined))
    dl = sum(tf.values())
    if dl == 0:
        return None
    chunk = dict(raw)
    chunk['id'] = chunk_id
    chunk['tf'] = tf
    chunk['dl'] = dl
    chunk['heading_tokens'] = set(heading_tokens_list)
    return chunk


def _build_inverted(chunks):
    """从 chunk 列表构建倒排索引和 df 表。"""
    inverted = {}
    for chunk in chunks:
        chunk_id = chunk['id']
        for token, count in chunk['tf'].items():
            if token not in inverted:
                inverted[token] = []
            inverted[token].append((chunk_id, count))
    df = {token: len(postings) for token, postings in inverted.items()}
    return inverted, df


def build(materials_dir, out_path="index.pkl"):
    """遍历资料目录，构建并保存倒排索引。

    Args:
        materials_dir: 资料目录路径。
        out_path: 输出 index.pkl 路径。
    """
    entries, materials_dir = _collect_files(materials_dir)
    cfg = dict(_DEFAULT_CONFIG)

    raw_chunks_all = []
    file_count = 0

    for rel_path, abs_path in entries:
        text = _read_file(abs_path)
        if text is None:
            continue
        if abs_path.endswith('.ipynb'):
            raw_chunks = doc_parser.parse_notebook(text, rel_path)
        else:
            raw_chunks = doc_parser.parse_markdown(text, rel_path)
        raw_chunks_all.extend(raw_chunks)
        file_count += 1

    # 分词、过滤空块、分配最终 id
    chunks = []
    for raw in raw_chunks_all:
        chunk = _make_chunk(raw, len(chunks), cfg)
        if chunk is not None:
            chunks.append(chunk)

    inverted, df = _build_inverted(chunks)

    N = len(chunks)
    avgdl = sum(c['dl'] for c in chunks) / N if N > 0 else 0.0

    index = {
        'version': 1,
        'config': cfg,
        'chunks': chunks,
        'inverted': inverted,
        'df': df,
        'N': N,
        'avgdl': avgdl,
    }

    with open(out_path, 'wb') as f:
        pickle.dump(index, f)

    print(f"文件数:      {file_count}")
    print(f"块数:        {N}")
    print(f"token 种类:  {len(inverted)}")
    print(f"avgdl:       {avgdl:.2f}")
    print(f"索引已写入   {out_path!r}")


def main():
    parser = argparse.ArgumentParser(description='构建离线检索索引')
    parser.add_argument('materials_dir', help='资料目录路径')
    parser.add_argument('-o', '--output', default='index.pkl',
                        dest='out_path', help='输出索引文件路径（默认 index.pkl）')
    args = parser.parse_args()
    build(args.materials_dir, args.out_path)


if __name__ == '__main__':
    main()
