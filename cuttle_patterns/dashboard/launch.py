"""Launch the Phase 7 embedding explorer as a live Bokeh server.

Runs a `bokeh.server.server.Server` programmatically (rather than shelling out to the
`bokeh serve` CLI, unlike `cuttle train`/`cuttle predict`'s subprocess wrappers around
BEAST's own CLI) so an extra static-file route can be registered for frame images —
giving genuine hover-triggered image tooltips (the browser requests `/images/...` by URL
and caches it, no per-hover round trip) instead of a click-to-view workaround.
"""

import argparse
from functools import partial
from pathlib import Path

from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.server.server import Server
from tornado.web import StaticFileHandler

from cuttle_patterns import paths
from cuttle_patterns.dashboard.app import make_document

DEFAULT_PORT = 5006

# Bokeh's default (300s) is tuned for short-lived web-app sessions; this is a local
# analysis tool a user is expected to leave open, hover/study a point for a while, or
# come back to after a break, with no server round trip in between. A short expiration
# there means the websocket token goes stale, and the client's silent reconnect attempt
# fails server-side with "Token is expired" — logged as an error even though the page
# transparently reloads and keeps working. A generous expiration (1 day) avoids that.
SESSION_TOKEN_EXPIRATION_SECONDS = 24 * 60 * 60


def run_server(results_dir: Path, port: int = DEFAULT_PORT, show: bool = True) -> None:
    """Start the embedding explorer and block until interrupted.

    Args:
        results_dir: the resolved results directory (holds `beast_models/` and
            `beast_frames/`).
        port: port to serve the app and its static image route on.
        show: whether to open a browser tab pointed at the app once it's up.
    """
    application = Application(FunctionHandler(partial(make_document, results_dir=results_dir)))
    images_root = results_dir / paths.BEAST_FRAMES_RELPATH
    extra_patterns = [(r'/images/(.*)', StaticFileHandler, {'path': str(images_root)})]

    server = Server(
        {'/': application},
        port=port,
        extra_patterns=extra_patterns,
        allow_websocket_origin=[f'localhost:{port}', f'127.0.0.1:{port}'],
        session_token_expiration=SESSION_TOKEN_EXPIRATION_SECONDS,
    )
    server.start()
    if show:
        server.io_loop.add_callback(server.show, '/')
    server.io_loop.start()


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and start the embedding explorer.

    Args:
        argv: command-line arguments (defaults to `sys.argv[1:]`); lets this module be
            run directly (`python -m cuttle_patterns.dashboard.launch`) for debugging
            outside the `cuttle serve` CLI.
    """
    parser = argparse.ArgumentParser(description='Launch the cuttle pattern explorer.')
    parser.add_argument('--results-dir', type=Path, required=True, metavar='PATH')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--no-show', action='store_true', help='do not open a browser tab')
    args = parser.parse_args(argv)

    run_server(args.results_dir, port=args.port, show=not args.no_show)


if __name__ == '__main__':
    main()
