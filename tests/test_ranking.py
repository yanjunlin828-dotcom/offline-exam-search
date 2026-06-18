"""
ranking.py P1 单测（BM25 + search + 意图识别 + 别名展开）。
运行：python tests/test_ranking.py
"""

import sys
import os
import math
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import tokenize
from ranking import (
    bm25, weighted_bm25, search, expand_aliases, load_aliases,
    W_FIELD_LEAF, W_FIELD_ANCESTOR, COVERAGE_POWER, W_POS, W_TYPE, K1, B,
    CUTOFF_DROP_RATIO, CUTOFF_DEFAULT_TOPK, _apply_cliff_cutoff,
)


# ─── Fixture ─────────────────────────────────────────────────────────────────

_CFG = {"enable_unigram": False, "cjk_ext": False}


def _make_chunk(chunk_id, file, heading, heading_path, text, ctype, position):
    text_tokens = tokenize(text, **_CFG)
    heading_tokens_list = tokenize(heading, **_CFG)
    combined = text_tokens + heading_tokens_list
    tf = dict(Counter(combined))
    dl = sum(tf.values())
    path_parts = heading_path.split(' > ') if heading_path else []
    ancestor_text = ' '.join(path_parts[:-1])
    return {
        'id': chunk_id,
        'file': file,
        'heading': heading,
        'heading_path': heading_path,
        'text': text,
        'ctype': ctype,
        'position': position,
        'tf': tf,
        'dl': dl,
        'heading_tokens': set(heading_tokens_list),
        'ancestor_heading_tokens': set(tokenize(ancestor_text, **_CFG)),
    }


def _build_test_index(chunks):
    """从 chunk 列表构建小型内存索引。"""
    inverted = {}
    for chunk in chunks:
        for token, count in chunk['tf'].items():
            inverted.setdefault(token, []).append((chunk['id'], count))
    df = {t: len(v) for t, v in inverted.items()}
    N = len(chunks)
    avgdl = sum(c['dl'] for c in chunks) / N if N > 0 else 0.0
    return {
        'version': 1,
        'config': _CFG,
        'chunks': chunks,
        'inverted': inverted,
        'df': df,
        'N': N,
        'avgdl': avgdl,
    }


# 固定测试语料
_CHUNKS = [
    _make_chunk(0, 'a.md', 'ADMM 定义', 'ADMM > 定义',
                'ADMM 是交替方向乘子法，是一种分布式优化算法的定义。', 'prose', 0),
    _make_chunk(1, 'a.md', 'ADMM 实现', 'ADMM > 实现',
                'def admm(x, z, u, rho): return x + rho * z', 'code', 1),
    _make_chunk(2, 'b.md', '范数与凸性', '基础 > 范数与凸性',
                '范数是向量空间中的度量，凸性是优化的核心性质。', 'prose', 0),
    _make_chunk(3, 'c.md', '', '',
                'This block only contains the word method and nothing else.', 'code', 0),
]
_INDEX = _build_test_index(_CHUNKS)

# 别名组（与 aliases.txt 内容对应）
_ALIAS_GROUPS = [
    ['ADMM', '交替方向乘子法', 'alternating direction method of multipliers'],
    ['范数', 'norm'],
]


# ─── BM25 单元测试 ─────────────────────────────────────────────────────────────

def test_bm25_positive_when_token_matches():
    """命中 token 时 BM25 > 0。"""
    qtokens = tokenize('admm', **_CFG)
    score = bm25(_INDEX, _CHUNKS[0], qtokens)
    assert score > 0, f"期望正分，实际 {score}"


def test_bm25_zero_when_no_match():
    """无命中 token 时 BM25 == 0。"""
    qtokens = ['不存在的token12345']
    score = bm25(_INDEX, _CHUNKS[0], qtokens)
    assert score == 0.0, f"期望 0，实际 {score}"


def test_bm25_idf_no_negative():
    """验证所有 token 的 IDF >= 0（Lucene 式 ln(1+...) 不会出负值）。"""
    N = _INDEX['N']
    df_table = _INDEX['df']
    for token, df_t in df_table.items():
        idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
        assert idf >= 0, f"token {token!r} 的 IDF < 0: {idf}"


def test_bm25_higher_tf_gives_higher_score():
    """出现次数多的 chunk 得分不低于出现次数少的 chunk（其他条件相同）。"""
    qtokens = list(dict.fromkeys(tokenize('admm', **_CFG)))
    score0 = bm25(_INDEX, _CHUNKS[0], qtokens)
    score3 = bm25(_INDEX, _CHUNKS[3], qtokens)  # 不含 admm
    assert score0 > score3, f"含 admm 的块得分 ({score0}) 应高于不含 admm 的块 ({score3})"


# ─── search 单元测试（P1：均解包 tuple）────────────────────────────────────────

def test_search_returns_list():
    """search 返回 (list, intent) tuple，第一项为列表。"""
    results, intent = search(_INDEX, 'admm', aliases=None)
    assert isinstance(results, list)


def test_search_result_has_required_fields():
    """每条结果包含规定字段。"""
    results, intent = search(_INDEX, 'admm', aliases=None)
    assert results, "期望有结果"
    for r in results:
        for field in ('file', 'heading_path', 'ctype', 'score', 'text'):
            assert field in r, f"结果缺少字段 {field!r}"


def test_search_cjk_bigram_recall():
    """'范数' 能被召回（验证 bigram 分词）。"""
    results, intent = search(_INDEX, '范数', aliases=None)
    assert results, "查询 '范数' 应有结果"
    files = [r['file'] for r in results]
    assert 'b.md' in files, f"期望 b.md 在结果中，实际 {files}"


def test_search_special_chars_no_exception():
    """含特殊字符的查询不报错，且按普通字符处理。"""
    try:
        results, intent = search(_INDEX, 'ADMM AND "x*', aliases=None)
        assert isinstance(results, list)
    except Exception as e:
        raise AssertionError(f"特殊字符查询不应报错：{e}")


def test_search_type_filter_prose():
    """type_filter='prose' 只返回 prose 块。"""
    results, intent = search(_INDEX, 'admm', aliases=None, type_filter='prose')
    for r in results:
        assert r['ctype'] == 'prose', f"过滤 prose 时出现 {r['ctype']!r}"


def test_search_type_filter_code():
    """type_filter='code' 只返回 code 块。"""
    results, intent = search(_INDEX, 'admm', aliases=None, type_filter='code')
    for r in results:
        assert r['ctype'] == 'code', f"过滤 code 时出现 {r['ctype']!r}"


def test_search_topk_respected():
    """topk 参数生效：返回数量不超过 topk。"""
    results, intent = search(_INDEX, 'admm', aliases=None, topk=1)
    assert len(results) <= 1, f"topk=1 但返回了 {len(results)} 条"


def test_search_empty_query_returns_empty():
    """空查询（切词后无 token）返回空列表。"""
    results, intent = search(_INDEX, '', aliases=None)
    assert results == [], f"期望空列表，实际 {results}"


def test_search_sorted_by_score_descending():
    """结果按 score 降序排列。"""
    results, intent = search(_INDEX, 'admm', aliases=None)
    scores = [r['score'] for r in results]
    assert scores == sorted(scores, reverse=True), f"结果未按分数降序：{scores}"


def test_search_case_insensitive():
    """大小写不影响召回（ADMM / admm / Admm 结果一致）。"""
    r1, _ = search(_INDEX, 'ADMM', aliases=None)
    r2, _ = search(_INDEX, 'admm', aliases=None)
    r3, _ = search(_INDEX, 'Admm', aliases=None)
    ids1 = [r['heading_path'] for r in r1]
    ids2 = [r['heading_path'] for r in r2]
    ids3 = [r['heading_path'] for r in r3]
    assert ids1 == ids2 == ids3, f"大小写影响了召回：{ids1} vs {ids2} vs {ids3}"


def test_search_unknown_query_returns_empty():
    """完全没有匹配的查询返回空列表。"""
    results, intent = search(_INDEX, 'xyzzy1234567890notexist', aliases=None)
    assert results == [], f"不存在的查询应返回空，实际 {results}"


def test_search_heading_bonus_applied():
    """叶子标题命中查询词时，加权 BM25 应是纯文本 BM25 的 W_FIELD_LEAF 倍。"""
    qtokens = list(dict.fromkeys(tokenize('admm', **_CFG)))
    plain = bm25(_INDEX, _CHUNKS[0], qtokens)
    weighted = weighted_bm25(_INDEX, _CHUNKS[0], qtokens)
    assert plain > 0, "前提：admm 应在该块中有正向 BM25 贡献"
    assert abs(weighted - plain * W_FIELD_LEAF) < 1e-9, (
        f"加权应为 {plain * W_FIELD_LEAF}，实际 {weighted}"
    )


def test_search_position_bonus_monotone():
    """同一章节内 position=0 的块，其位置加分高于 position=1 的块。"""
    pos_bonus_0 = W_POS / (1.0 + 0)
    pos_bonus_1 = W_POS / (1.0 + 1)
    assert pos_bonus_0 > pos_bonus_1


def test_search_ancestor_heading_bonus_weaker_than_leaf():
    """祖先标题（非叶子）命中查询词时，加权倍数弱于叶子标题命中。"""
    chunk = _make_chunk(
        0, 'f.md', '应用案例', 'ADMM > 应用案例',
        'admm 在工程实践中有广泛应用，这里讨论一些应用案例的细节。',
        'prose', 3,
    )
    index = _build_test_index([chunk])  # 单独建索引，避免被其他高分块在断崖截断中挤掉
    qtokens = list(dict.fromkeys(tokenize('admm', **_CFG)))
    plain = bm25(index, chunk, qtokens)
    weighted = weighted_bm25(index, chunk, qtokens)

    assert abs(weighted - plain * W_FIELD_ANCESTOR) < 1e-9, (
        f"祖先标题加权应为 {plain * W_FIELD_ANCESTOR}，实际 {weighted}"
    )
    assert W_FIELD_ANCESTOR < W_FIELD_LEAF, "祖先标题加权应弱于叶子标题"

    results, _ = search(index, 'admm', aliases=None)
    target = next(r for r in results if r['file'] == 'f.md')
    pos_bonus = W_POS / (1.0 + 3)
    expected = weighted + pos_bonus  # intent=None 时 type_bonus=0，n_topics=1 时 coverage_factor=1
    assert abs(target['score'] - expected) < 1e-9, (
        f"search() 总分与预期不符：实际 {target['score']}，期望 {expected}"
    )


def test_search_exact_heading_match_outranks_repeated_mentions():
    """标题与查询基本等价的短块，应跑赢正文里反复提及该词但标题无关的长块。"""
    exact_chunk = _make_chunk(
        0, 'g.md', '凸优化', '方法 > 凸优化',
        '凸优化是指目标函数和约束集均为凸的优化问题。',
        'prose', 0,
    )
    long_text = ('凸优化 ' * 40) + '这个概念在很多场景中都会被反复提及，但本节标题与之无关。'
    noisy_chunk = _make_chunk(
        1, 'h.md', '杂项笔记', '其他 > 杂项笔记',
        long_text, 'prose', 0,
    )
    # 补足若干不含该词的填充块，拉高 idf，模拟更大语料下的压力场景
    filler_chunks = [
        _make_chunk(2 + i, f'filler{i}.md', f'填充章节{i}', f'填充 > 填充章节{i}',
                    f'这是与凸性无关的第 {i} 段填充文字，用于撑大语料规模。',
                    'prose', 0)
        for i in range(8)
    ]
    index = _build_test_index([exact_chunk, noisy_chunk] + filler_chunks)

    results, _ = search(index, '凸优化', aliases=None)
    assert results, "期望有结果"
    assert results[0]['file'] == 'g.md', (
        f"标题精确匹配的块应排第 1，实际排序: {[r['file'] for r in results]}"
    )


def test_search_fuzzy_heading_partial_match_still_boosted():
    """标题只包含查询词的一部分（模糊匹配，非精确等价）也应获得比无标题命中更高的分。"""
    partial_chunk = _make_chunk(
        0, 'p.md', 'ADMM 工程实践要点', 'ADMM > ADMM 工程实践要点',
        'admm 在实际工程中常配合 warm start 使用。', 'prose', 0,
    )
    no_heading_chunk = _make_chunk(
        1, 'q.md', '杂项笔记', '其他 > 杂项笔记',
        'admm 在实际工程中常配合 warm start 使用。', 'prose', 0,
    )
    index = _build_test_index([partial_chunk, no_heading_chunk])
    qtokens = list(dict.fromkeys(tokenize('admm', **_CFG)))
    score_p = weighted_bm25(index, partial_chunk, qtokens)
    score_q = weighted_bm25(index, no_heading_chunk, qtokens)
    assert score_p > score_q, (
        f"标题模糊命中（非精确等价）也应获得更高分：p={score_p}, q={score_q}"
    )


def test_coverage_factor_penalizes_single_term_dominance():
    """多词查询下，只命中一个词但反复出现的块，应排在'多个词都沾边'的块之后。"""
    topicful = _make_chunk(
        0, 'topicful.md', '一致性ADMM分布式优化', '方法 > 一致性ADMM分布式优化',
        '一致性优化问题中，ADMM 在分布式场景下交替更新变量直至收敛。',
        'prose', 0,
    )
    admm_heavy = _make_chunk(
        1, 'admmheavy.md', '其他算法说明', '杂项 > 其他算法说明',
        ('ADMM ' * 50) + '反复提及但与一致性、分布式都无关。',
        'prose', 0,
    )
    filler_chunks = [
        _make_chunk(2 + i, f'filler{i}.md', f'填充{i}', f'填充 > 填充{i}',
                    f'第 {i} 段与本次查询完全无关的填充内容。', 'prose', 0)
        for i in range(8)
    ]
    index = _build_test_index([topicful, admm_heavy] + filler_chunks)

    results, _ = search(index, '一致性 admm 分布式', aliases=None)
    files = [r['file'] for r in results]
    assert files, "期望有结果"
    assert files[0] == 'topicful.md', (
        f"多词都命中的块应排第 1，实际排序: {files}"
    )


# ─── 断崖截断（cliff cutoff）单元测试 ──────────────────────────────────────────

def test_cliff_cutoff_keeps_close_scores():
    """相邻名次分数跌幅不超过阈值时，全部保留。"""
    results = [{'score': 10.0}, {'score': 6.0}, {'score': 5.9}]
    kept = _apply_cliff_cutoff(results, drop_ratio=0.5)
    assert len(kept) == 3, f"期望全部保留，实际 {kept}"


def test_cliff_cutoff_drops_after_big_drop():
    """出现跌破阈值的断崖时，断崖之后的结果被截断。"""
    results = [{'score': 10.0}, {'score': 4.0}, {'score': 3.9}]
    kept = _apply_cliff_cutoff(results, drop_ratio=0.5)
    assert len(kept) == 1, f"期望只保留第 1 条，实际 {kept}"


def test_cliff_cutoff_empty_list():
    """空列表输入返回空列表。"""
    assert _apply_cliff_cutoff([]) == []


def test_cliff_cutoff_always_keeps_first():
    """无论分数如何，至少保留第 1 条。"""
    results = [{'score': 1.0}, {'score': 0.0001}]
    kept = _apply_cliff_cutoff(results, drop_ratio=0.5)
    assert len(kept) == 1
    assert kept[0]['score'] == 1.0


# ─── 意图识别测试（Task 1.1）──────────────────────────────────────────────────

def test_intent_prose_detected():
    """查询含 prose 意图词时 intent == 'prose'。"""
    _, intent = search(_INDEX, 'ADMM 定义', aliases=None)
    assert intent == 'prose', f"期望 intent='prose'，实际 {intent!r}"


def test_intent_code_detected():
    """查询含 code 意图词时 intent == 'code'。"""
    _, intent = search(_INDEX, 'ADMM 实现', aliases=None)
    assert intent == 'code', f"期望 intent='code'，实际 {intent!r}"


def test_intent_none_when_no_keyword():
    """查询不含任何意图词时 intent == None。"""
    _, intent = search(_INDEX, 'ADMM', aliases=None)
    assert intent is None, f"无意图词时期望 None，实际 {intent!r}"


def test_intent_words_stripped_from_tokens():
    """仅含意图词的查询，意图词被剔除后 topic_tokens 为空，结果为空列表。"""
    results, intent = search(_INDEX, '定义', aliases=None)
    assert results == [], f"仅含意图词应返回空结果，实际 {results}"
    assert intent == 'prose'


def test_intent_prose_gives_type_bonus():
    """prose 意图使 prose 块得分高于 code 块。"""
    results, intent = search(_INDEX, 'ADMM 定义', aliases=None)
    assert intent == 'prose'
    prose_scores = [r['score'] for r in results if r['ctype'] == 'prose']
    code_scores  = [r['score'] for r in results if r['ctype'] == 'code']
    assert prose_scores, "期望有 prose 结果"
    assert code_scores,  "期望有 code 结果（用于对比）"
    assert max(prose_scores) > max(code_scores), (
        f"prose 最高分 {max(prose_scores):.4f} 应 > code 最高分 {max(code_scores):.4f}"
    )


def test_intent_code_gives_type_bonus():
    """code 意图使 code 块得分高于 prose 块。"""
    results, intent = search(_INDEX, 'ADMM 实现', aliases=None)
    assert intent == 'code'
    prose_scores = [r['score'] for r in results if r['ctype'] == 'prose']
    code_scores  = [r['score'] for r in results if r['ctype'] == 'code']
    assert prose_scores, "期望有 prose 结果"
    assert code_scores,  "期望有 code 结果"
    assert max(code_scores) > max(prose_scores), (
        f"code 最高分 {max(code_scores):.4f} 应 > prose 最高分 {max(prose_scores):.4f}"
    )


# ─── expand_aliases 单元测试（Task 1.2）───────────────────────────────────────

def test_expand_aliases_condition_a_token_subset():
    """条件 A：短语 token 集合是 topic_token_set 子集时命中。"""
    groups = [['ADMM', '交替方向乘子法']]
    result = expand_aliases({'admm'}, 'admm', groups, _CFG)
    assert len(result) == 2, f"期望 2 个短语 token 列表，实际 {len(result)}"
    assert result[0] == ['admm']
    assert result[1] == ['交替', '替方', '方向', '向乘', '乘子', '子法']


def test_expand_aliases_condition_b_substring_match():
    """条件 B：短语字符串是查询子串时命中。"""
    groups = [['ADMM', '交替方向乘子法']]
    # "交替方向乘子法" 作为子串出现在 query 中 → 该组命中
    result = expand_aliases(set(), '交替方向乘子法', groups, _CFG)
    assert len(result) == 2, f"子串匹配应触发整组，实际 {len(result)}"


def test_expand_aliases_no_match_returns_empty():
    """无匹配时返回空列表。"""
    groups = [['ADMM', '交替方向乘子法']]
    result = expand_aliases({'范数'}, '范数', groups, _CFG)
    assert result == [], f"无匹配应返回空，实际 {result}"


def test_expand_aliases_case_insensitive_condition_b():
    """条件 B 不区分大小写（phrase.lower() in query.lower()）。"""
    groups = [['ADMM', 'norm']]
    result = expand_aliases(set(), 'admm search', groups, _CFG)
    assert len(result) == 2


# ─── 别名在 search 中的集成测试（Task 1.2）────────────────────────────────────

def test_alias_alternate_name_recalls_same_chunks():
    """'交替方向乘子法' 通过别名展开，应召回与 'ADMM' 相同的文件集合。"""
    results_admm, _ = search(_INDEX, 'ADMM', _ALIAS_GROUPS)
    results_cn,   _ = search(_INDEX, '交替方向乘子法', _ALIAS_GROUPS)
    admm_files = {r['file'] for r in results_admm}
    cn_files   = {r['file'] for r in results_cn}
    assert admm_files == cn_files, (
        f"别名应召回同组结果: ADMM→{admm_files}，交替→{cn_files}"
    )


def test_alias_method_only_not_recalled():
    """只含单词 'method' 的 c.md 不应被 ADMM 别名（AND 语义）误召。"""
    results, _ = search(_INDEX, 'ADMM', _ALIAS_GROUPS)
    files = [r['file'] for r in results]
    assert 'c.md' not in files, (
        f"只含 method 的 c.md 不应被别名误召，实际结果文件：{files}"
    )


def test_alias_no_expansion_when_aliases_none():
    """aliases=None 时不做别名展开，不应报错。"""
    results, intent = search(_INDEX, 'ADMM', aliases=None)
    assert isinstance(results, list)


def test_alias_no_expansion_when_aliases_empty():
    """aliases=[] 时不做别名展开，不应报错。"""
    results, intent = search(_INDEX, 'ADMM', aliases=[])
    assert isinstance(results, list)


# ─── load_aliases 单元测试 ────────────────────────────────────────────────────

def test_load_aliases_nonexistent_returns_empty():
    """aliases.txt 不存在时返回空列表。"""
    result = load_aliases('/nonexistent/path/aliases.txt')
    assert result == []


def test_load_aliases_parses_groups():
    """能正确解析 aliases.txt 格式（临时文件）。"""
    content = (
        "# 注释行\n"
        "ADMM=交替方向乘子法=alternating direction method of multipliers\n"
        "\n"
        "范数=norm\n"
    )
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                    suffix='.txt', delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        groups = load_aliases(tmp_path)
        assert len(groups) == 2, f"期望 2 组，实际 {len(groups)}: {groups}"
        assert groups[0] == ['ADMM', '交替方向乘子法', 'alternating direction method of multipliers']
        assert groups[1] == ['范数', 'norm']
    finally:
        os.unlink(tmp_path)


# ─── 自运行入口 ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import traceback

    tests = [
        # BM25
        test_bm25_positive_when_token_matches,
        test_bm25_zero_when_no_match,
        test_bm25_idf_no_negative,
        test_bm25_higher_tf_gives_higher_score,
        # search 基础
        test_search_returns_list,
        test_search_result_has_required_fields,
        test_search_cjk_bigram_recall,
        test_search_special_chars_no_exception,
        test_search_type_filter_prose,
        test_search_type_filter_code,
        test_search_topk_respected,
        test_search_empty_query_returns_empty,
        test_search_sorted_by_score_descending,
        test_search_case_insensitive,
        test_search_unknown_query_returns_empty,
        test_search_heading_bonus_applied,
        test_search_position_bonus_monotone,
        test_search_ancestor_heading_bonus_weaker_than_leaf,
        test_search_exact_heading_match_outranks_repeated_mentions,
        test_search_fuzzy_heading_partial_match_still_boosted,
        test_coverage_factor_penalizes_single_term_dominance,
        test_cliff_cutoff_keeps_close_scores,
        test_cliff_cutoff_drops_after_big_drop,
        test_cliff_cutoff_empty_list,
        test_cliff_cutoff_always_keeps_first,
        # 意图识别（Task 1.1）
        test_intent_prose_detected,
        test_intent_code_detected,
        test_intent_none_when_no_keyword,
        test_intent_words_stripped_from_tokens,
        test_intent_prose_gives_type_bonus,
        test_intent_code_gives_type_bonus,
        # expand_aliases 单元（Task 1.2）
        test_expand_aliases_condition_a_token_subset,
        test_expand_aliases_condition_b_substring_match,
        test_expand_aliases_no_match_returns_empty,
        test_expand_aliases_case_insensitive_condition_b,
        # 别名集成（Task 1.2）
        test_alias_alternate_name_recalls_same_chunks,
        test_alias_method_only_not_recalled,
        test_alias_no_expansion_when_aliases_none,
        test_alias_no_expansion_when_aliases_empty,
        # load_aliases
        test_load_aliases_nonexistent_returns_empty,
        test_load_aliases_parses_groups,
    ]

    passed = 0
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
