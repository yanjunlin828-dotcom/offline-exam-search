"""
tokenizer.py 单测。覆盖计划 §6 Task 0.1 全部验收用例。
运行：python -m pytest tests/test_tokenizer.py -v
      或 python tests/test_tokenizer.py
"""
import sys
import os

# 把 exam-search 根加进 sys.path，使 import tokenizer 不依赖安装
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import tokenize


# ── 基础验收用例（计划 §6 Task 0.1 逐条列出）────────────────────────────────

def test_cjk_bigram_overlapping():
    """范数学 -> ['范数', '数学']，重叠 bigram。"""
    assert tokenize("范数学") == ["范数", "数学"]


def test_mixed_ascii_cjk_boundary():
    """ADMM算法 -> ['admm', '算法']，类别切换处断开。"""
    assert tokenize("ADMM算法") == ["admm", "算法"]


def test_single_cjk_char():
    """单字 CJK 段 -> 输出该单字。"""
    assert tokenize("凸") == ["凸"]


def test_special_chars_no_exception():
    """含特殊字符的查询（'ADMM AND "x*'）不报错，and/x 作为普通词。"""
    result = tokenize('ADMM AND "x*')
    assert "admm" in result
    assert "and" in result
    assert "x" in result


def test_hyphen_as_separator():
    """L1-norm 中 '-' 作为分隔符，输出 ['l1', 'norm']。"""
    assert tokenize("L1-norm") == ["l1", "norm"]


# ── 额外边界用例 ─────────────────────────────────────────────────────────────

def test_lowercase_normalization():
    """大写字母全部折叠成小写。"""
    assert tokenize("Hello") == ["hello"]
    assert tokenize("BM25") == ["bm25"]


def test_empty_string():
    """空字符串返回空列表，不报错。"""
    assert tokenize("") == []


def test_only_separators():
    """全部是标点/空格 -> 空列表。"""
    assert tokenize("  ,.!@#$%^&*()") == []


def test_cjk_two_char_bigram():
    """两字 CJK 段 -> 一个 bigram（与单字不同的路径）。"""
    assert tokenize("范数") == ["范数"]


def test_longer_cjk_bigram():
    """四字 CJK -> 三个重叠 bigram。"""
    assert tokenize("交替方向") == ["交替", "替方", "方向"]


def test_enable_unigram_multi_char():
    """enable_unigram=True 时多字 CJK 段额外输出每个单字。"""
    result = tokenize("范数学", enable_unigram=True)
    # bigrams
    assert "范数" in result
    assert "数学" in result
    # extra unigrams
    assert "范" in result
    assert "数" in result
    assert "学" in result


def test_enable_unigram_single_char():
    """单字 CJK 段，enable_unigram 不产生重复。"""
    assert tokenize("凸", enable_unigram=True) == ["凸"]


def test_cjk_ext_range():
    """cjk_ext=True 时 U+3400–U+4DBF 范围也被识别为 CJK。"""
    # U+3400 = '㐀'
    char_ext = "㐀"
    assert tokenize(char_ext, cjk_ext=True) == [char_ext]
    # 默认不识别
    assert tokenize(char_ext, cjk_ext=False) == []


def test_mixed_full_pipeline():
    """中英混写综合用例：交替方向乘子法 ADMM。"""
    result = tokenize("交替方向乘子法 ADMM")
    assert "交替" in result
    assert "替方" in result
    assert "方向" in result
    assert "向乘" in result
    assert "乘子" in result
    assert "子法" in result
    assert "admm" in result


def test_underscore_as_separator():
    """下划线作为分隔符，不合并到词中。"""
    assert tokenize("hello_world") == ["hello", "world"]


def test_numbers_merged_with_letters():
    """数字与字母连续合并为一个 token。"""
    assert tokenize("l2") == ["l2"]
    assert tokenize("bm25") == ["bm25"]


def test_fullwidth_punctuation_as_separator():
    """全角标点（如 U+FF0C 全角逗号）作为分隔符。"""
    assert tokenize("范数，凸性") == ["范数", "凸性"]


# ── 简单自运行入口 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    tests = [
        test_cjk_bigram_overlapping,
        test_mixed_ascii_cjk_boundary,
        test_single_cjk_char,
        test_special_chars_no_exception,
        test_hyphen_as_separator,
        test_lowercase_normalization,
        test_empty_string,
        test_only_separators,
        test_cjk_two_char_bigram,
        test_longer_cjk_bigram,
        test_enable_unigram_multi_char,
        test_enable_unigram_single_char,
        test_cjk_ext_range,
        test_mixed_full_pipeline,
        test_underscore_as_separator,
        test_numbers_merged_with_letters,
        test_fullwidth_punctuation_as_separator,
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
