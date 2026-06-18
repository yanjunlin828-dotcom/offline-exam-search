"""
Markdown 围栏状态机 + ipynb JSON 解析。
只切块、不分词；返回 chunk 雏形（不含 tf/dl/id）。
"""

import json
import re

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)')


# ─── 围栏检测 ─────────────────────────────────────────────────────────────────

def _is_fence_open(line):
    """若该行是开围栏行，返回 (fence_char, fence_len)，否则返回 None。"""
    stripped = line.lstrip()
    for fc in ('`', '~'):
        if not stripped.startswith(fc + fc + fc):
            continue
        count = 0
        while count < len(stripped) and stripped[count] == fc:
            count += 1
        return (fc, count)
    return None


def _is_fence_close(line, fence_char, fence_len):
    """闭围栏：同种字符、长度 >= 开围栏、整行仅该字符（允许首尾空白）。"""
    stripped = line.strip()
    if not stripped:
        return False
    return (
        len(set(stripped)) == 1 and
        stripped[0] == fence_char and
        len(stripped) >= fence_len
    )


# ─── 标题栈 ───────────────────────────────────────────────────────────────────

def _update_heading_stack(stack, level, text):
    """弹出 level >= 当前 level 的条目，再压入新标题。"""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))


def _heading_path(stack):
    return ' > '.join(t for _, t in stack)


def _heading_levels(stack):
    """heading_path 每一段对应的真实 Markdown 标题级别（# 的个数）。"""
    return [lvl for lvl, _ in stack]


def _current_heading(stack):
    return stack[-1][1] if stack else ''


# ─── 散文行 flush ──────────────────────────────────────────────────────────────

def _flush_prose_lines(prose_lines, file, heading_stack, position):
    """按空行分段，返回 (新 chunks 列表, 新 position)。不修改 prose_lines 本身。"""
    new_chunks = []
    current_para = []

    def emit_para():
        nonlocal position
        block = '\n'.join(current_para).strip()
        if block:
            new_chunks.append({
                'file': file,
                'heading': _current_heading(heading_stack),
                'heading_path': _heading_path(heading_stack),
                'heading_levels': _heading_levels(heading_stack),
                'text': block,
                'ctype': 'prose',
                'position': position,
            })
            position += 1

    for line in prose_lines:
        if line.strip() == '':
            if current_para:
                emit_para()
                current_para = []
        else:
            current_para.append(line)
    if current_para:
        emit_para()

    return new_chunks, position


# ─── 共享 Markdown 状态机 ──────────────────────────────────────────────────────

def _make_md_state():
    return {
        'in_fence': False,
        'fence_char': '',
        'fence_len': 0,
        'fence_lines': [],
        'heading_stack': [],   # list of (level, str)
        'prose_lines': [],
        'position': 0,
    }


def _flush_fence(state, file, chunks):
    """提交当前围栏内容为 code chunk（如内容非空），并重置 fence_lines。"""
    block = '\n'.join(state['fence_lines'])
    if block.strip():
        chunks.append({
            'file': file,
            'heading': _current_heading(state['heading_stack']),
            'heading_path': _heading_path(state['heading_stack']),
            'heading_levels': _heading_levels(state['heading_stack']),
            'text': block,
            'ctype': 'code',
            'position': state['position'],
        })
        state['position'] += 1
    state['fence_lines'] = []


def _process_lines(lines, file, state, chunks):
    """逐行处理，更新 state，将新 chunk 追加到 chunks。"""
    for line in lines:
        if state['in_fence']:
            if _is_fence_close(line, state['fence_char'], state['fence_len']):
                state['in_fence'] = False
                _flush_fence(state, file, chunks)
            else:
                state['fence_lines'].append(line)
        else:
            fence_info = _is_fence_open(line)
            if fence_info is not None:
                new_chunks, state['position'] = _flush_prose_lines(
                    state['prose_lines'], file, state['heading_stack'], state['position'])
                chunks.extend(new_chunks)
                state['prose_lines'] = []
                state['in_fence'] = True
                state['fence_char'], state['fence_len'] = fence_info
                state['fence_lines'] = []
            else:
                m = _HEADING_RE.match(line)
                if m:
                    new_chunks, state['position'] = _flush_prose_lines(
                        state['prose_lines'], file, state['heading_stack'], state['position'])
                    chunks.extend(new_chunks)
                    state['prose_lines'] = []
                    level = len(m.group(1))
                    heading_text = m.group(2).strip()
                    _update_heading_stack(state['heading_stack'], level, heading_text)
                    state['position'] = 0
                else:
                    state['prose_lines'].append(line)


def _finalize_md(state, file, chunks):
    """所有行处理完毕后：flush 剩余 prose；未闭合的 fence 当代码块处理。"""
    if not state['in_fence']:
        new_chunks, state['position'] = _flush_prose_lines(
            state['prose_lines'], file, state['heading_stack'], state['position'])
        chunks.extend(new_chunks)
        state['prose_lines'] = []
    else:
        if state['fence_lines']:
            _flush_fence(state, file, chunks)
        state['in_fence'] = False


# ─── 公开接口 ──────────────────────────────────────────────────────────────────

def parse_markdown(text, file):
    """把 Markdown 文本切成 chunk 雏形列表。

    Args:
        text: Markdown 字符串。
        file: 文件相对路径（用 / 分隔）。

    Returns:
        list[dict]，每条含 file/heading/heading_path/text/ctype/position。
        不含 tf/dl/id（由 build_index 填）。
    """
    state = _make_md_state()
    chunks = []
    _process_lines(text.splitlines(), file, state, chunks)
    _finalize_md(state, file, chunks)
    return chunks


def _join_source(source):
    """source 兼容 str / list[str]，统一返回 str。"""
    if isinstance(source, list):
        return ''.join(source)
    return source if isinstance(source, str) else ''


def _parse_notebook_cells(cells, file):
    """处理 cell 列表，标题状态全程跨 cell 共享，围栏状态在 cell 边界重置。"""
    chunks = []
    state = _make_md_state()

    for cell in cells:
        cell_type = cell.get('cell_type', '')
        source = _join_source(cell.get('source', ''))

        if cell_type == 'markdown':
            # 围栏不跨 cell：若上一 cell 未闭合，视为代码块处理后重置
            if state['in_fence']:
                if state['fence_lines']:
                    _flush_fence(state, file, chunks)
                state['in_fence'] = False
                state['fence_char'] = ''
                state['fence_len'] = 0

            _process_lines(source.splitlines(), file, state, chunks)

            # cell 结束时 flush prose，但保留标题状态供后续 cell 使用
            if not state['in_fence']:
                new_chunks, state['position'] = _flush_prose_lines(
                    state['prose_lines'], file, state['heading_stack'], state['position'])
                chunks.extend(new_chunks)
                state['prose_lines'] = []
            else:
                # cell 内未闭合的 fence 当代码处理，重置
                if state['fence_lines']:
                    _flush_fence(state, file, chunks)
                state['in_fence'] = False

        elif cell_type == 'code':
            # code cell 前先 flush 任何积压的 prose
            if state['prose_lines']:
                new_chunks, state['position'] = _flush_prose_lines(
                    state['prose_lines'], file, state['heading_stack'], state['position'])
                chunks.extend(new_chunks)
                state['prose_lines'] = []

            if source.strip():
                chunks.append({
                    'file': file,
                    'heading': _current_heading(state['heading_stack']),
                    'heading_path': _heading_path(state['heading_stack']),
                    'text': source,
                    'ctype': 'code',
                    'position': state['position'],
                })
                state['position'] += 1
        # raw cell: 跳过

    return chunks


def parse_notebook(text, file):
    """解析 .ipynb（JSON）文本，返回 chunk 雏形列表。

    Args:
        text: .ipynb 文件内容字符串。
        file: 文件相对路径。

    Returns:
        list[dict]（同 parse_markdown），解析失败时返回空列表并打印警告。
    """
    try:
        nb = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[警告] 跳过无效 JSON 文件 {file!r}: {e}")
        return []

    if 'cells' in nb:
        cells = nb['cells']
    elif 'worksheets' in nb:
        cells = []
        for ws in nb.get('worksheets', []):
            cells.extend(ws.get('cells', []))
    else:
        print(f"[警告] {file!r} 无 cells 字段，跳过")
        return []

    return _parse_notebook_cells(cells, file)
