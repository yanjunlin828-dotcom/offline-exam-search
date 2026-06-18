"""
唯一真源分词器。索引构建与查询期均只调此模块，不得在别处重写切词逻辑。
"""


def _is_ascii_alnum(c):
    return ('a' <= c <= 'z') or ('0' <= c <= '9')


def _is_cjk(c, cjk_ext=False):
    cp = ord(c)
    if 0x4E00 <= cp <= 0x9FFF:
        return True
    if cjk_ext and 0x3400 <= cp <= 0x4DBF:
        return True
    return False


def tokenize(text, enable_unigram=False, cjk_ext=False):
    """文本 -> token 列表。

    规则：
    1. text.lower()（只影响英文/数字，中文无副作用）。
    2. 顺序扫描，按字符类别分段：
       - 连续 [a-z0-9]+ -> 一个英文词 token。
       - 连续 CJK（默认 U+4E00–U+9FFF；cjk_ext=True 再含 U+3400–U+4DBF）->
         输出重叠 bigram（如 范数学 -> 范数, 数学）；段长为 1 时直接输出该单字；
         enable_unigram=True 时额外为每个汉字补一个单字 token。
       - 其余字符（标点/空白/全角符号/下划线/连字符等）-> 分隔符，丢弃。
    3. 类别切换处断开（ADMM算法 -> ["admm", "算法"]）。

    Args:
        text: 待分词文本。
        enable_unigram: True 时额外为 CJK 多字段的每个汉字补单字 token。
        cjk_ext: True 时扩展 CJK 范围含 U+3400–U+4DBF。

    Returns:
        token 列表（str）。
    """
    text = text.lower()
    tokens = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        if _is_ascii_alnum(c):
            j = i + 1
            while j < n and _is_ascii_alnum(text[j]):
                j += 1
            tokens.append(text[i:j])
            i = j

        elif _is_cjk(c, cjk_ext):
            j = i + 1
            while j < n and _is_cjk(text[j], cjk_ext):
                j += 1
            seg = text[i:j]
            seg_len = len(seg)

            if seg_len == 1:
                tokens.append(seg)
            else:
                for k in range(seg_len - 1):
                    tokens.append(seg[k:k + 2])
                if enable_unigram:
                    for k in range(seg_len):
                        tokens.append(seg[k])

            i = j

        else:
            i += 1

    return tokens
