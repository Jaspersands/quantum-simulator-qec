# Ported from fusion_blossom 0.2.13's helper.py (Yue Wu; MIT, see LICENSE here), modified for
# stabilizer_qec: the website is served by the standard library's HTTP server (fusion_blossom
# embedded bottle), and its files are read from this package's ``visualizer`` folder.
"""The visualizer website, its server, and helpers for viewing a code or rendering a snapshot."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

fb = None

_WEBSITE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizer")
_WEBSITE_FILES = ["gui3d.js", "index.js", "patches.js", "primal.js", "cmd.js", "mocker.js",
                  "index.html", "partition-profile.html", "icon.svg", "package.json", "package-lock.json"]


def _website():
    files = {}
    for name in _WEBSITE_FILES:
        with open(os.path.join(_WEBSITE_FOLDER, name), "r", encoding="utf-8") as f:
            files[name] = f.read()
    return files


visualizer_website = _website()


def register(module):
    """called by fusion_blossom package at import, to set up global variables"""
    global fb
    fb = module


_MIME = {"html": "text/html; charset=utf8", "js": "text/javascript; charset=utf8",
         "json": "application/json; charset=utf8", "svg": "image/svg+xml"}


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, data_folder, quiet, **kwargs):
        self.data_folder = data_folder
        self.quiet = quiet
        super().__init__(*args, directory=data_folder, **kwargs)

    def log_message(self, format, *args):
        if not self.quiet:
            super().log_message(format, *args)

    def _send(self, body, mime):
        self.send_response(200)
        if mime is not None:
            self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        from urllib.parse import unquote
        path = unquote(path).lstrip("/")
        if path.startswith("data/"):
            root = os.path.realpath(self.data_folder)
            target = os.path.realpath(os.path.join(root, path[len("data/"):]))
            if os.path.commonpath([root, target]) != root or not os.path.isfile(target):
                self.send_error(404)
                return
            with open(target, "rb") as f:
                self._send(f.read(), _MIME.get(target.rsplit(".", 1)[-1]))
            return
        if path == "":
            path = "index.html"
        if path in visualizer_website:
            self._send(visualizer_website[path].encode("utf-8"), _MIME.get(path.rsplit(".", 1)[-1]))
            return
        self.send_error(404)


class _Server:
    """The server ``serve`` returns: ``srv`` is the HTTP server, shut down by ``srv.shutdown()``."""

    def __init__(self, host, port, data_folder, quiet):
        self.srv = ThreadingHTTPServer((host, port), partial(_Handler, data_folder=data_folder, quiet=quiet))


def serve(host='localhost', port=51665, data_folder=".", return_server=False, quiet=True):
    """start a server to host the visualizer websites locally"""
    server = _Server(host, port, data_folder, quiet)

    def run_server():
        try:
            server.srv.serve_forever()
        finally:
            server.srv.server_close()
    if return_server:
        return server, run_server
    run_server()


def open_visualizer(filename, host='localhost', port=51666, data_folder=".", open_browser=True, quiet=True):
    """open the website directly after starting the browser"""
    server, run_server = serve(host, port, data_folder, return_server=True, quiet=quiet)
    threading.Thread(target=run_server).start()
    if open_browser:
        def open_browser():
            webbrowser.open(f"http://{host}:{port}/?filename={filename}")
        threading.Timer(0.5, open_browser).start()  # wait 500ms for the server to start
    print("Hit ENTER to exit server.")
    input()
    server.srv.shutdown()
    time.sleep(0.3)


def peek_code(code, host='localhost', port=51667, data_folder=".", open_browser=True, quiet=True):
    """open the website to have a look at the code object"""
    positions = code.get_positions()
    initializer = code.get_initializer()
    solver = fb.SolverSerial(initializer)
    solver.solve(fb.SyndromePattern())
    visualize_filename = fb.static_visualize_data_filename()
    visualizer = fb.Visualizer(filepath=os.path.join(data_folder, visualize_filename), positions=positions)
    solver.perfect_matching(visualizer)
    open_visualizer(visualize_filename, host=host, port=port, data_folder=data_folder, open_browser=open_browser)


def run_command_get_stdout(command, no_stdout=True, use_tmp_out=False, stderr_to_stdout=False, cwd=None):
    env = os.environ.copy()
    stdout = subprocess.PIPE
    if use_tmp_out:
        out_file = tempfile.NamedTemporaryFile(delete=False)
        out_filename = out_file.name
        stdout = out_file
    if no_stdout:
        stdout = sys.stdout
    process = subprocess.Popen(command, universal_newlines=True, env=env, stdout=stdout,
                               stderr=(stdout if stderr_to_stdout else sys.stderr), cwd=cwd)
    stdout, _ = process.communicate()
    if use_tmp_out:
        out_file.flush()
        out_file.close()
        with open(out_filename, "r", encoding="utf8") as f:
            stdout = f.read()
        os.remove(out_filename)
    return stdout, process.returncode


def local_render_visualizer(filename, image_filename="rendered", renderer_folder="./local_renderer", width=1024, height=1024,
                            data_folder=".", snapshot_idx=0, patch_script=None):
    """render a visualizer image just like a browser does, but locally and save to file.
    note that we'll run node.js in `renderer_folder`, which generates a lot of files in `renderer_folder`/node_modules.
    this is necessary because some dependencies require local environment to compile native code."""
    if not os.path.exists(renderer_folder):
        os.makedirs(renderer_folder)
    for website_filename in visualizer_website:
        filepath = os.path.join(renderer_folder, website_filename)
        # always replace the files because the content may be outdated in a new version
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(visualizer_website[website_filename])
    if not os.path.exists(os.path.join(renderer_folder, "node_modules")):
        run_command_get_stdout(["npm", "install"], cwd=renderer_folder)
    if patch_script is not None:
        patch_filename = f"{filename}.patch.js"
        with open(os.path.join(renderer_folder, patch_filename), "w", encoding="utf8") as f:
            f.write(patch_script)
    renderer_data_folder = os.path.join(renderer_folder, "data")
    if not os.path.exists(renderer_data_folder):
        os.makedirs(renderer_data_folder)
    shutil.copy2(os.path.join(data_folder, filename), renderer_data_folder)
    domain = "http://localhost"  # doesn't matter, won't use in the local renderer
    url = f"{domain}?filename={filename}&snapshot_idx={snapshot_idx}"
    if patch_script is not None:
        url += f"&patch_url=./{patch_filename}"
    commands = ["node", "index.js", url, f"{width}", f"{height}", image_filename]
    print(f"[run] {commands}")
    run_command_get_stdout(commands, cwd=renderer_folder)
