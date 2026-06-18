"""
本地检索服务器（P1）。

用法：
    python server.py [--index index.pkl] [--aliases aliases.txt] [--materials <dir>] [--port 8000]
"""

import sys
import os
import json
import pickle
import argparse
from urllib.parse import parse_qs, urlparse

# Windows 终端 UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ranking

try:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
except ImportError:
    import socketserver
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True

_INDEX        = None
_ALIASES      = None
_WEB_DIR      = None
_MATERIALS_DIR = None   # 原始资料目录，用于 /raw 路由

MIME_TYPES = {
    '.html':  'text/html; charset=utf-8',
    '.js':    'application/javascript; charset=utf-8',
    '.css':   'text/css; charset=utf-8',
    '.json':  'application/json; charset=utf-8',
    '.md':    'text/plain; charset=utf-8',
    '.ipynb': 'application/json; charset=utf-8',
    '.woff2': 'font/woff2',
    '.woff':  'font/woff',
    '.ttf':   'font/ttf',
}
DEFAULT_MIME = 'application/octet-stream'


class _ReuseServer(ThreadingHTTPServer):
    allow_reuse_address = True


class SearchHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[{self.address_string()}] {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == '/':
            self._serve_index()
        elif path == '/search':
            self._serve_search(parsed.query)
        elif path == '/raw':
            self._serve_raw(parsed.query)
        elif path == '/view':
            self._serve_viewer(parsed.query)
        elif path.startswith('/web/'):
            self._serve_static(path[len('/web/'):])
        else:
            self._send_404()

    # ── route handlers ────────────────────────────────────────────────────────

    def _serve_index(self):
        self._send_file(os.path.join(_WEB_DIR, 'index.html'),
                        'text/html; charset=utf-8')

    def _serve_viewer(self, query_string):
        """GET /view?path=class/1.md  →  viewer.html（客户端渲染）"""
        self._send_file(os.path.join(_WEB_DIR, 'viewer.html'),
                        'text/html; charset=utf-8')

    def _serve_raw(self, query_string):
        """GET /raw?path=class/1.md  →  原始文件文本，供 viewer.html 获取"""
        if _MATERIALS_DIR is None:
            self._send_error(503, '未指定资料目录（--materials），无法提供原文')
            return

        params   = parse_qs(query_string)
        rel_path = params.get('path', [''])[0]
        if not rel_path:
            self._send_404()
            return

        mat_dir_real   = os.path.realpath(_MATERIALS_DIR)
        requested_real = os.path.realpath(os.path.join(_MATERIALS_DIR, rel_path))

        if not (requested_real == mat_dir_real or
                requested_real.startswith(mat_dir_real + os.sep)):
            self._send_404()
            return

        ext  = os.path.splitext(rel_path)[1].lower()
        mime = MIME_TYPES.get(ext, 'text/plain; charset=utf-8')
        self._send_file(requested_real, mime)

    def _serve_search(self, query_string):
        params = parse_qs(query_string)
        q           = params.get('q',    [''])[0].strip()
        type_filter = params.get('type', ['all'])[0]
        if type_filter not in ('all', 'prose', 'code'):
            type_filter = 'all'

        if not q:
            results_list, intent = [], None
        else:
            try:
                results_list, intent = ranking.search(
                    _INDEX, q, _ALIASES, type_filter=type_filter
                )
            except Exception as exc:
                print(f"[错误] 检索失败: {exc}")
                results_list, intent = [], None

        body = json.dumps(
            {'query': q, 'intent': intent, 'results': results_list},
            ensure_ascii=False
        ).encode('utf-8')

        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel_path):
        if not rel_path:
            self._send_404()
            return

        web_dir_real   = os.path.realpath(_WEB_DIR)
        requested_real = os.path.realpath(os.path.join(_WEB_DIR, rel_path))

        if not (requested_real == web_dir_real or
                requested_real.startswith(web_dir_real + os.sep)):
            self._send_404()
            return

        ext  = os.path.splitext(rel_path)[1].lower()
        mime = MIME_TYPES.get(ext, DEFAULT_MIME)
        self._send_file(requested_real, mime)

    # ── low-level helpers ─────────────────────────────────────────────────────

    def _send_file(self, abs_path, mime):
        if not os.path.isfile(abs_path):
            self._send_404()
            return
        try:
            with open(abs_path, 'rb') as f:
                data = f.read()
        except OSError as exc:
            print(f"[错误] 读取文件失败: {abs_path!r}: {exc}")
            self._send_404()
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_404(self):
        body = b'Not Found'
        self.send_response(404)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, code, msg):
        body = msg.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    global _INDEX, _ALIASES, _WEB_DIR, _MATERIALS_DIR

    ap = argparse.ArgumentParser(description='离线检索本地 Web 服务器')
    ap.add_argument('--index',     default='index.pkl',
                    help='索引文件路径（默认 index.pkl）')
    ap.add_argument('--aliases',   default='aliases.txt',
                    help='别名文件路径（默认 aliases.txt）')
    ap.add_argument('--materials', default=None,
                    help='原始资料目录（用于"查看全文"；不填则禁用全文功能）')
    ap.add_argument('--port',      type=int, default=8000,
                    help='监听端口（默认 8000）')
    args = ap.parse_args()

    if not os.path.exists(args.index):
        print(f"错误：索引文件 {args.index!r} 不存在。\n请先运行：python build_index.py <materials_dir>")
        sys.exit(1)

    with open(args.index, 'rb') as f:
        _INDEX = pickle.load(f)

    _ALIASES = ranking.load_aliases(args.aliases)
    _WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')

    if args.materials:
        _MATERIALS_DIR = os.path.realpath(args.materials)
        print(f"资料目录：{_MATERIALS_DIR}")
    else:
        print("资料目录：未指定（查看全文功能不可用，加 --materials <dir> 启用）")

    print(f"索引：{args.index}（{_INDEX['N']} 块，{len(_INDEX['inverted'])} token 种类）")
    print(f"别名组：{len(_ALIASES)} 组")
    print(f"服务已启动 → http://127.0.0.1:{args.port}/")
    print("Ctrl+C 退出")

    server = _ReuseServer(('127.0.0.1', args.port), SearchHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
