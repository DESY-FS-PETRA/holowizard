"""The Neuroglancer client: the prebuilt one of the ``neuroglancer`` Python package (its Python integration build),
served with a small in-browser stand-in for its Python server and a patch for the scale bars."""

import importlib.util
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

CLIENT_TOKEN = re.compile(r"[A-Za-z0-9_-]{1,64}")
# Added to the client's index.html, before its scripts: the client's "Python server" (state updates pushed over
# an EventSource, state changes and actions posted back) is answered in the browser. A real EventSource would
# keep one of the browser's few connections per server open for as long as the viewer is open.
CLIENT_SHIM = """<script>
(() => {
  class SilentEventSource extends EventTarget {
    constructor(url) {
      super();
      this.url = String(url); this.readyState = 1; this.onopen = this.onmessage = this.onerror = null;
      setTimeout(() => { if (this.readyState === 1 && this.onopen) this.onopen(new Event('open')); }, 0);
    }
    close() { this.readyState = 2; }
  }
  window.EventSource = SilentEventSource;
  const realFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (/\\/neuroglancer\\/(state|action)\\//.test(url)) {
      let g = null;
      try { g = JSON.parse(init && init.body).g; } catch (e) { /* not a state update */ }
      return Promise.resolve(new Response(JSON.stringify({ g: String(g) }), { headers: { 'Content-Type': 'application/json' } }));
    }
    return realFetch(input, init);
  };
})();
</script>"""

# Patch of the client's main script: in the 4-panel layouts, Neuroglancer shows the scale bar only in the XY panel
# (the XZ and YZ panels get a fixed "off"); they get the viewer's scale bar setting like XY. Matches the minified
# code of the installed version; if a version differs, nothing is replaced and the client works as shipped.
SCALE_BAR_PATTERN = re.compile(
    r"(\w+)=\{\.\.\.(\w+)\((\w+)\),showScaleBar:\3\.showScaleBar\},"
    r"(\w+)=\{\.\.\.\2\(\3\),showScaleBar:new \w+\(!1,!1\)\}"
)
CLIENT_PATCH_VERSION = "1"  # in the script URL, so browsers do not use an unpatched cached copy


def client_dir() -> Optional[Path]:
    """Folder of the prebuilt Neuroglancer client in the installed ``neuroglancer`` package, or None."""
    spec = importlib.util.find_spec("neuroglancer")  # finds the package without importing it
    if spec is None or not spec.submodule_search_locations:
        return None
    folder = Path(list(spec.submodule_search_locations)[0]) / "static" / "client"
    return folder if (folder / "index.html").is_file() else None


def client_index_html(path: Path) -> str:
    """The client's index.html with the server shim, loading the patched main script."""
    html = path.read_text().replace("<head>", "<head>" + CLIENT_SHIM, 1)
    return re.sub(r'(src="main\.[^"]+\.js)"', rf'\1?patch={CLIENT_PATCH_VERSION}"', html)


def is_main_script(name: str) -> bool:
    return name.startswith("main.") and name.endswith(".js")


@lru_cache(maxsize=2)
def patched_main_script(path: str, mtime_ns: int) -> bytes:
    """The client's main script with scale bars in all slice panels; ``mtime_ns`` is only part of the cache key."""
    text = Path(path).read_text()
    text = SCALE_BAR_PATTERN.sub(
        lambda m: (
            f"{m[1]}={{...{m[2]}({m[3]}),showScaleBar:{m[3]}.showScaleBar}},"
            f"{m[4]}={{...{m[2]}({m[3]}),showScaleBar:{m[3]}.showScaleBar}}"
        ),
        text,
    )
    return text.encode()
