# Ported from fusion_blossom 0.2.13's helper.py (Yue Wu; MIT, see LICENSE here), modified for
# stabilizer_qec: the website's files are read from this package's ``visualizer`` folder, and the
# Bottle fusion_blossom embeds is the submodule ``helper.bottle`` (unchanged), not installed as the
# modules ``bottle`` and ``helper``.
"""The visualizer website, its server (on ``helper.bottle``, the Bottle 0.14-dev web framework
fusion_blossom embeds), and helpers for viewing a code or rendering a snapshot.

The package import loads this helper and Bottle, without registering top-level ``bottle`` or
``helper`` modules. ``serve`` binds only when its runner starts; routes use Bottle's process-wide
default app and the latest call's data folder. Data paths are resolved within that folder (this
check is not a defense against concurrent filesystem replacement). Bottle supplies data MIME
types: JSON is application/json and extensionless files default to text/html. ``quiet`` suppresses
request logs, not every server error or malformed HTTP syntax message."""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import webbrowser

# As fusion_blossom's helper module, this one gets its Bottle when the package is imported.
from . import bottle  # noqa: F401

fb = None

_WEBSITE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "visualizer")
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


def serve(host='localhost', port=51665, data_folder=".", return_server=False, quiet=True):
    """Host the visualizer locally; with return_server, return its unbound adapter and runner."""
    # As fusion_blossom's, the routes go on Bottle's default app: the latest call's data_folder is
    # the one every server of the process serves.
    from .bottle import WSGIRefServer, route, abort, run, response, request, static_file

    data_root = os.path.realpath(os.path.abspath(data_folder))

    def guess_mime(filename):
        if filename.endswith("html"):
            response.content_type = 'text/html; charset=utf8'
        elif filename.endswith("js"):
            response.content_type = 'text/javascript; charset=utf8'
        elif filename.endswith("json"):
            response.content_type = 'application/json; charset=utf8'

    @route('/data/<filename:path>')
    def send_data(filename):
        try:
            resolved = os.path.realpath(os.path.join(data_root, filename.strip("/\\")))
            if os.path.commonpath((data_root, resolved)) != data_root:
                abort(403, "Access denied.")
        except (ValueError, OSError):
            abort(403, "Access denied.")
        # Bottle opens a GET body before discovering an unsatisfiable Range. A HEAD preflight
        # preserves its conditional/Range decisions without opening that body on refused requests.
        if request.method == "GET" and request.environ.get("HTTP_RANGE"):
            request.environ["REQUEST_METHOD"] = "HEAD"
            try:
                checked = static_file(filename, root=data_root)
            finally:
                request.environ["REQUEST_METHOD"] = "GET"
            if checked.status_code not in (200, 206):
                return checked
        return static_file(filename, root=data_root)

    @route('/')
    @route('/<filename:path>')
    def send_root(filename=""):
        if filename == "":
            filename = "index.html"
        if filename in visualizer_website:
            guess_mime(filename)
            return visualizer_website[filename]
        abort(404)
    server = WSGIRefServer(host=host, port=port)

    def run_server():
        app = bottle.default_app()

        def validated_app(environ, start_response):
            # WSGI carries the original path bytes through Latin-1. Reject invalid UTF-8 before
            # Bottle's error renderer attempts to interpolate a surrogate-containing URL.
            try:
                environ.get("PATH_INFO", "").encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                body = b"400 Bad Request\n"
                start_response("400 Bad Request", [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))])
                return [] if environ.get("REQUEST_METHOD") == "HEAD" else [body]
            return app(environ, start_response)

        run(app=validated_app, server=server, quiet=quiet)
    if return_server:
        return server, run_server
    run_server()


def open_visualizer(filename, host='localhost', port=51666, data_folder=".", open_browser=True, quiet=True):
    """Run the visualizer until ENTER (or an input interruption), closing this server's worker.

    Bind errors are raised before prompting or opening a browser. Browser startup waits for binding.
    """
    server, run_server = serve(host, port, data_folder, return_server=True, quiet=quiet)
    from wsgiref.simple_server import WSGIServer

    serving = threading.Event()

    class VisualizerServer(WSGIServer):
        def service_actions(self):
            # Called inside BaseServer's serving loop, after its shutdown event was initialized.
            serving.set()
            super().service_actions()

    server.options["server_class"] = VisualizerServer
    errors = []

    def worker():
        try:
            run_server()
        except BaseException as error:
            errors.append(error)
        finally:
            if hasattr(server, "srv"):
                server.srv.server_close()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def wait_for_start():
        while thread.is_alive() and not serving.is_set():
            thread.join(0.01)

    try:
        wait_for_start()
        if errors:
            raise errors[0]
        if not serving.is_set():
            raise RuntimeError("The visualizer server did not initialize")
        if open_browser:
            webbrowser.open(f"http://{host}:{server.port}/?filename={filename}")
        print("Hit ENTER to exit server.")
        input()
    finally:
        # Also handle interruption during startup; never access srv when binding failed.
        wait_for_start()
        if thread.is_alive() and serving.is_set():
            server.srv.shutdown()
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("The visualizer server worker did not stop")


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
