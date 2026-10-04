"""Serve only public website assets on localhost, with a useful 404 page."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from functools import partial
from urllib.parse import urlsplit, unquote
import mimetypes

ROOT = Path(__file__).resolve().parents[1]


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Always show current local edits during review.
        self.send_header('Cache-Control', 'no-store, max-age=0')
        super().end_headers()

    def allowed(self):
        path = unquote(urlsplit(self.path).path)
        if any(part.startswith('.') for part in Path(path).parts) or path.startswith(('/data/', '/scripts/')):
            self.send_error(404)
            return False
        resolved = (ROOT / path.lstrip('/')).resolve()
        if not resolved.is_relative_to(ROOT):
            self.send_error(404)
            return False
        return True

    def do_HEAD(self):
        if self.allowed():
            super().do_HEAD()

    def do_GET(self):
        if not self.allowed():
            return
        resolved = (ROOT / unquote(urlsplit(self.path).path).lstrip('/')).resolve()
        if not resolved.exists():
            body = (ROOT / '404.html').read_bytes()
            self.send_response(404)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def list_directory(self, path):
        self.send_error(404)
        return None


if __name__ == '__main__':
    mimetypes.add_type('application/pdf', '.pdf')
    mimetypes.add_type('image/webp', '.webp')
    server = ThreadingHTTPServer(('127.0.0.1', 8765), partial(PreviewHandler, directory=str(ROOT)))
    print('Preview: http://127.0.0.1:8765', flush=True)
    server.serve_forever()
