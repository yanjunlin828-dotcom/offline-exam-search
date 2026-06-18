/**
 * 共享 Markdown / 公式 / 关键词高亮渲染逻辑。
 * 被 index.html（搜索结果预览）和 viewer.html（全文查看）共用，避免重复实现。
 */
(function () {
  'use strict';

  // 占位符前缀：只含字母数字，不含 markdown 特殊字符，避免被 marked 处理
  var _MKEY = 'XMATH' + Date.now().toString(36).toUpperCase();
  var _markedReady = false;

  function escapeHtml(s) {
    return s
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // marked v5+ 删除了 highlight 选项，必须用 renderer 方式接 hljs
  function configureMarked() {
    if (!window.marked || _markedReady) return;
    _markedReady = true;
    marked.use({
      gfm: true,
      breaks: false,
      renderer: {
        code: function (code, infostring) {
          var lang = (infostring || '').split(/\s/)[0];
          if (window.hljs) {
            var out = (lang && hljs.getLanguage(lang))
              ? hljs.highlight(code, { language: lang }).value
              : hljs.highlightAuto(code).value;
            return '<pre><code class="hljs">' + out + '</code></pre>';
          }
          return '<pre><code>' + escapeHtml(code) + '</code></pre>';
        }
      }
    });
  }

  // marked 会把 $x_1$ 里的 _ 转成 <em>，破坏 LaTeX；用占位符绕过这个问题
  function protectMath(src) {
    var blocks = [];
    src = src.replace(/\$\$([\s\S]*?)\$\$/g, function (_, tex) {
      return _MKEY + (blocks.push({ tex: tex, display: true }) - 1) + 'E';
    });
    src = src.replace(/\\\[([\s\S]*?)\\\]/g, function (_, tex) {
      return _MKEY + (blocks.push({ tex: tex, display: true }) - 1) + 'E';
    });
    src = src.replace(/\\\(([\s\S]*?)\\\)/g, function (_, tex) {
      return _MKEY + (blocks.push({ tex: tex, display: false }) - 1) + 'E';
    });
    src = src.replace(/\$([^\n$]+?)\$/g, function (_, tex) {
      return _MKEY + (blocks.push({ tex: tex, display: false }) - 1) + 'E';
    });
    return { src: src, blocks: blocks };
  }

  function restoreMath(html, blocks) {
    return html.replace(new RegExp(_MKEY + '(\\d+)E', 'g'), function (_, i) {
      var b = blocks[+i];
      if (!window.katex) {
        return escapeHtml(b.display ? '$$' + b.tex + '$$' : '$' + b.tex + '$');
      }
      try {
        return katex.renderToString(b.tex, { displayMode: b.display, throwOnError: false });
      } catch (e) {
        return '<span style="color:#c00" title="' + escapeHtml(String(e)) + '">'
             + escapeHtml(b.display ? '$$' + b.tex + '$$' : '$' + b.tex + '$') + '</span>';
      }
    });
  }

  function renderMarkdownToHtml(text) {
    configureMarked();
    var math = protectMath(text);
    var html = window.marked ? marked.parse(math.src) : '<pre>' + escapeHtml(text) + '</pre>';
    return restoreMath(html, math.blocks);
  }

  function walkText(node, fn) {
    if (node.nodeType === 3) {
      var result = fn(node);
      if (result) node.parentNode.replaceChild(result, node);
      return;
    }
    if (node.nodeName === 'CODE' || node.nodeName === 'PRE' ||
        node.nodeName === 'SCRIPT' || node.nodeName === 'STYLE') return;
    Array.from(node.childNodes).forEach(function (c) { walkText(c, fn); });
  }

  // 在已渲染的 DOM 上按文本节点查找并包裹 <mark>，不触碰 CODE/PRE 内部，
  // 避免在渲染后的 HTML 标签里做字符串替换破坏结构。
  function highlightTerms(el, words) {
    if (!words || !words.length) return;
    walkText(el, function (node) {
      var text = node.nodeValue;
      var lower = text.toLowerCase();
      var changed = false;
      words.forEach(function (w) {
        if (w && lower.indexOf(w.toLowerCase()) !== -1) changed = true;
      });
      if (!changed) return null;
      var frag = document.createDocumentFragment();
      var remaining = text;
      while (remaining.length > 0) {
        var bestIdx = -1; var bestWord = '';
        words.forEach(function (w) {
          if (!w) return;
          var idx = remaining.toLowerCase().indexOf(w.toLowerCase());
          if (idx !== -1 && (bestIdx === -1 || idx < bestIdx)) {
            bestIdx = idx; bestWord = w;
          }
        });
        if (bestIdx === -1) { frag.appendChild(document.createTextNode(remaining)); break; }
        if (bestIdx > 0) frag.appendChild(document.createTextNode(remaining.slice(0, bestIdx)));
        var mark = document.createElement('mark');
        mark.textContent = remaining.slice(bestIdx, bestIdx + bestWord.length);
        frag.appendChild(mark);
        remaining = remaining.slice(bestIdx + bestWord.length);
      }
      return frag;
    });
  }

  // ── 关键词标黄开关（跨 index.html / viewer.html 共享，存 localStorage） ──────
  // 不重新渲染内容，只靠 CSS class 控制 <mark> 的视觉样式，切换是瞬时的，
  // 且对已经渲染好的旧结果同样生效。

  var HIGHLIGHT_STORAGE_KEY = 'examSearchHighlightEnabled';

  function isHighlightEnabled() {
    var v;
    try { v = localStorage.getItem(HIGHLIGHT_STORAGE_KEY); } catch (e) { v = null; }
    return v !== '0';  // 默认开启
  }

  function applyHighlightState(enabled) {
    document.documentElement.classList.toggle('no-highlight', !enabled);
  }

  function setHighlightEnabled(enabled) {
    try { localStorage.setItem(HIGHLIGHT_STORAGE_KEY, enabled ? '1' : '0'); } catch (e) { /* 隐私模式等场景忽略 */ }
    applyHighlightState(enabled);
  }

  function updateToggleLabel(btn, enabled) {
    btn.textContent = enabled ? '🟡 关键词标黄' : '⚪ 关键词标黄';
    btn.classList.toggle('active', enabled);
  }

  // 绑定一个按钮元素：恢复上次状态、显示当前状态、点击切换
  function initHighlightToggle(btn) {
    if (!btn) return;
    var enabled = isHighlightEnabled();
    updateToggleLabel(btn, enabled);
    btn.addEventListener('click', function () {
      enabled = !enabled;
      setHighlightEnabled(enabled);
      updateToggleLabel(btn, enabled);
    });
  }

  // 脚本加载时立刻应用一次，避免内容渲染后才切到"关闭"状态产生闪烁
  applyHighlightState(isHighlightEnabled());

  window.MDRender = {
    escapeHtml: escapeHtml,
    configureMarked: configureMarked,
    renderMarkdownToHtml: renderMarkdownToHtml,
    highlightTerms: highlightTerms,
    initHighlightToggle: initHighlightToggle
  };
})();
