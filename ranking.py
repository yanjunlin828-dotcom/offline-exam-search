"""
相关度计算：BM25 + 标题加分 + 位置加分 + 意图加分（P1）+ 别名展开（P1）。
"""

import math
import os
from tokenizer import tokenize

W_HEAD = 2.0   # 标题命中加分权重
W_TYPE = 1.5   # 类型匹配加分权重
W_POS  = 0.5   # 位置加分权重（越靠近标题越高）
K1 = 1.5       # BM25 词频饱和参数
B  = 0.75      # BM25 文档长度归一化参数


def bm25(index, chunk, qtokens):
    """计算单个 chunk 对查询 token 列表的 BM25 分数。

    使用 Lucene 式 IDF：ln(1 + (N - df + 0.5) / (df + 0.5))，避免常见词负 IDF。

    Args:
        index: 完整索引 dict。
        chunk: 单个 chunk dict（含 tf / dl）。
        qtokens: 查询 token 列表（调用方去重后传入）。

    Returns:
        float BM25 分数（>= 0）。
    """
    N = index['N']
    avgdl = index['avgdl']
    df_table = index['df']
    dl = chunk['dl']
    tf_dict = chunk['tf']

    if avgdl == 0:
        return 0.0

    score = 0.0
    for t in qtokens:
        tf = tf_dict.get(t, 0)
        if tf == 0:
            continue
        df_t = df_table.get(t, 0)
        idf = math.log(1.0 + (N - df_t + 0.5) / (df_t + 0.5))
        score += idf * tf * (K1 + 1) / (tf + K1 * (1.0 - B + B * dl / avgdl))

    return score


def detect_intent(query):
    """识别查询意图，剔除意图词后返回清洁查询串。

    意图词按子串匹配；命中后从查询串中删除，以免污染 topic token 提取。
    prose 优先：同时含两类词时，按 prose 处理（罕见边界情况）。

    Args:
        query: 原始查询字符串。

    Returns:
        (intent, cleaned_query)：intent in {"prose", "code", None}。
    """
    PROSE_WORDS = ['定义', '概念', '是什么', '含义', '介绍']
    CODE_WORDS  = ['实现', '代码', '怎么写', '例子', '用法']

    cleaned = query
    intent = None

    for word in PROSE_WORDS:
        if word in cleaned:
            cleaned = cleaned.replace(word, '')
            intent = 'prose'

    if intent is None:
        for word in CODE_WORDS:
            if word in cleaned:
                cleaned = cleaned.replace(word, '')
                intent = 'code'

    return intent, cleaned.strip()


def expand_aliases(topic_token_set, query, alias_groups, cfg):
    """找出与查询匹配的别名组，返回各组内所有短语的 token 列表。

    命中规则（对每个 alias_group，任一短语满足即整组命中）：
      A: set(tokenize(phrase, **cfg)) ⊆ topic_token_set
      B: phrase.lower() in query.lower()

    命中后返回该组每个短语的 token 列表（跳过 tokenize 结果为空的短语）。

    Args:
        topic_token_set: set[str] — cleaned_query 切词后的 token 集合。
        query: str — cleaned_query 字符串（用于子串匹配）。
        alias_groups: list[list[str]] — load_aliases 返回的原始短语组。
        cfg: dict — tokenize 关键字参数。

    Returns:
        list[list[str]] — 每个元素是一个短语的 token 列表。
    """
    result = []
    query_lower = query.lower()

    for group in alias_groups:
        group_matched = False
        for phrase in group:
            phrase_tokens = tokenize(phrase, **cfg)
            # 条件 A：短语的所有 token 均在 topic 集合中
            if phrase_tokens and set(phrase_tokens) <= topic_token_set:
                group_matched = True
                break
            # 条件 B：短语字符串（原始）出现在查询中
            if phrase.lower() in query_lower:
                group_matched = True
                break

        if not group_matched:
            continue

        for phrase in group:
            phrase_tokens = tokenize(phrase, **cfg)
            if phrase_tokens:
                result.append(phrase_tokens)

    return result


def load_aliases(path):
    """解析 aliases.txt，返回 list[list[str]]（每组是若干原始短语字符串）。

    Args:
        path: aliases.txt 路径。

    Returns:
        list[list[str]]，文件不存在时返回空列表。
    """
    if not os.path.exists(path):
        return []

    groups = []
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split('=') if p.strip()]
                if parts:
                    groups.append(parts)
    except OSError as e:
        print(f"[警告] 读取 aliases.txt 失败: {e}")
    return groups


def search(index, query, aliases, type_filter="all", topk=20):
    """检索并按相关度返回 topk 结果。

    打分 = BM25(score_tokens) + W_HEAD*(标题命中率) + W_POS*(1/(1+pos)) + W_TYPE*(意图匹配)
    score_tokens = topic_tokens ∪ 该块完整命中的别名短语 token（部分命中不计分）。

    Args:
        index: pickle.load 得到的索引 dict。
        query: 原始查询字符串（含特殊字符也不报错）。
        aliases: 别名组列表（load_aliases 结果；None 或 [] 则跳过别名展开）。
        type_filter: "all" | "prose" | "code"。
        topk: 最多返回条数。

    Returns:
        (list[dict], intent)：
          list[dict] 每条含 file/heading_path/ctype/score/text，按 score 降序；
          intent 为 "prose" | "code" | None。
    """
    cfg = index['config']

    # 1. 意图识别：剔除意图词，得到 cleaned_query
    intent, cleaned_query = detect_intent(query)

    # 2. 对 cleaned_query 切词并保序去重
    raw_tokens = tokenize(cleaned_query, **cfg)
    topic_tokens = list(dict.fromkeys(raw_tokens))

    if not topic_tokens:
        return [], intent

    # 3. 基础候选：topic_tokens 倒排表并集
    inverted = index['inverted']
    candidate_ids = set()
    for t in topic_tokens:
        if t in inverted:
            for cid, _ in inverted[t]:
                candidate_ids.add(cid)

    # 4. 别名扩展候选（AND 语义：每个短语的 token 取交集）
    # alias_hits[chunk_id] = list of phrase_token_lists（该块完整命中的短语）
    alias_hits = {}
    if aliases:
        expanded = expand_aliases(set(topic_tokens), cleaned_query, aliases, cfg)
        for phrase_tokens in expanded:
            if not phrase_tokens:
                continue
            # 对该短语的每个 token 取倒排表交集，只有"全中"的块才进入候选
            phrase_cids = None
            for t in phrase_tokens:
                if t not in inverted:
                    phrase_cids = set()
                    break
                ids_for_t = {cid for cid, _ in inverted[t]}
                if phrase_cids is None:
                    phrase_cids = ids_for_t
                else:
                    phrase_cids &= ids_for_t
            if phrase_cids:
                candidate_ids |= phrase_cids
                for cid in phrase_cids:
                    if cid not in alias_hits:
                        alias_hits[cid] = []
                    alias_hits[cid].append(phrase_tokens)

    if not candidate_ids:
        return [], intent

    # 5. 打分
    chunks = index['chunks']
    n_topics = len(topic_tokens)
    results = []

    for chunk_id in candidate_ids:
        chunk = chunks[chunk_id]

        if type_filter in ('prose', 'code') and chunk['ctype'] != type_filter:
            continue

        # score_tokens = topic_tokens + 完整命中的别名短语 token（去重）
        # 当 intent 明确时，别名 token 只计入同类型块的打分；
        # 否则别名会给含完整别名的 prose 块带来 20+ BM25 增益，彻底压制 W_TYPE=1.5 的意图加分。
        # intent=None 时行为不变（alias_token_eligible 恒 True）。
        alias_token_eligible = (intent is None or chunk['ctype'] == intent)
        score_token_set = set(topic_tokens)
        if alias_token_eligible and chunk_id in alias_hits:
            for phrase_toks in alias_hits[chunk_id]:
                for t in phrase_toks:
                    score_token_set.add(t)
        score_tokens = list(score_token_set)

        bm25_score = bm25(index, chunk, score_tokens)

        # 标题加分：用 topic_tokens 衡量，别名 token 不膨胀此加分
        heading_hit = sum(1 for t in topic_tokens if t in chunk['heading_tokens'])
        head_bonus = W_HEAD * (heading_hit / n_topics)

        pos_bonus = W_POS * (1.0 / (1.0 + chunk['position']))

        # 意图加分：intent 与 ctype 匹配才加分（软偏好，不硬过滤）
        type_bonus = W_TYPE * (1 if intent and chunk['ctype'] == intent else 0)

        final = bm25_score + head_bonus + pos_bonus + type_bonus

        results.append({
            'file': chunk['file'],
            'heading_path': chunk['heading_path'],
            'ctype': chunk['ctype'],
            'score': final,
            'text': chunk['text'],
            'matched_tokens': score_tokens,
        })

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:topk], intent
