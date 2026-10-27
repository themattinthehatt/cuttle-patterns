"""Draw photometric reference boxes (tank interior, white spout, black drain) on full frames.

Opens one full-frame image per session (a `Day{n}_Tank{m}` directory of frames) and lets you
drag a box for each region. The boxes are saved as pixel coordinates `[x0, y0, x1, y1]` in the
frame's own resolution, keyed by session name:

    {
      "Day1_Tank1": {
        "frame": "00462.jpg",
        "frame_size": [771, 434],
        "regions": {"tank": [x0, y0, x1, y1], "white_spout": [...], "black_drain": null}
      }
    }

A region set to `null` means "looked, not visible in this session"; a missing region means
"not annotated yet". Keys while the window is focused:

    1 / 2 / 3   choose the active region: tank / white_spout / black_drain
    drag        draw the active region's box (redraw to replace it)
    x           mark the active region as not visible
    d           delete the active region's box
    c           copy all boxes from the previous session (when the camera did not move)
    f           show the next sampled frame (e.g. when a fish covers a spout)
    b           toggle the fish-free temporal-median background
    n / p       next / previous session (saves first)
    s / q       save / save and quit

Tip: leave the matplotlib toolbar's zoom and pan buttons off while drawing, since they take
over the mouse.

Example:
    python scratch/photometry/annotate_boxes.py /media/mattw/CUTTLE/raw_data_samples
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image

logger = logging.getLogger(__name__)

REGIONS = ('tank', 'white_spout', 'black_drain')
REGION_COLORS = {'tank': 'tab:green', 'white_spout': 'tab:red', 'black_drain': 'tab:blue'}
REGION_KEYS = {'1': 'tank', '2': 'white_spout', '3': 'black_drain'}
DEFAULT_OUTPUT = Path(__file__).parent / 'boxes.json'
DEFAULT_FRAMES_SUBDIR = 'Downsampled_frames'
FRAME_GLOB = '*.jpg'
N_FRAMES_MEDIAN = 60


@dataclass
class Session:
    """One `Day{n}_Tank{m}` recording and its sampled full frames.

    Attributes:
        name: session name with whitespace removed, e.g. `Day1_Tank4`
        frames: sorted frame paths
        idx_frame: index into `frames` of the frame currently shown
        background: cached fish-free temporal-median frame, computed on first request
    """

    name: str
    frames: list[Path]
    idx_frame: int = 0
    background: np.ndarray | None = field(default=None, repr=False)

    @property
    def frame(self) -> Path:
        """Path of the frame currently shown."""
        return self.frames[self.idx_frame]

    def load_frame(self) -> np.ndarray:
        """Load the current frame as an RGB uint8 array."""
        return np.asarray(Image.open(self.frame).convert('RGB'))

    def load_background(self) -> np.ndarray:
        """Return the per-pixel temporal median over evenly spaced frames (fish removed)."""
        if self.background is None:
            idx = np.linspace(0, len(self.frames) - 1, min(N_FRAMES_MEDIAN, len(self.frames)))
            stack = [
                np.asarray(Image.open(self.frames[int(i)]).convert('RGB'))
                for i in idx
            ]
            self.background = np.median(np.stack(stack), axis=0).astype(np.uint8)
        return self.background


def discover_sessions(root: Path, frames_subdir: str = DEFAULT_FRAMES_SUBDIR) -> list[Session]:
    """Find `{root}/{session}/{frames_subdir}/*.jpg` and build one `Session` per directory.

    Args:
        root: directory containing one subdirectory per session
        frames_subdir: subdirectory of each session that holds the frames

    Returns:
        Sessions sorted by (day, tank) when the name parses, otherwise by name.

    Raises:
        FileNotFoundError: if no session directory contains any frames
    """
    sessions = []
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        frames = sorted((session_dir / frames_subdir).glob(FRAME_GLOB))
        if frames:
            sessions.append(Session(name=re.sub(r'\s+', '', session_dir.name), frames=frames))
    if not sessions:
        raise FileNotFoundError(f'no {frames_subdir}/{FRAME_GLOB} frames found under {root}')

    def sort_key(session: Session) -> tuple[int, int, str]:
        match = re.match(r'Day(\d+)_Tank(\d+)', session.name)
        return (int(match[1]), int(match[2]), session.name) if match else (10**6, 0, session.name)

    return sorted(sessions, key=sort_key)


def load_boxes(path: Path) -> dict[str, dict]:
    """Load saved annotations, or an empty dict if the file does not exist yet."""
    return json.loads(path.read_text()) if path.exists() else {}


def save_boxes(path: Path, boxes: dict[str, dict]) -> None:
    """Write annotations as sorted, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(boxes, indent=2, sort_keys=True) + '\n')


def normalize_box(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: int,
    height: int,
) -> list[int]:
    """Order a dragged rectangle's corners, round to pixels, and clamp to the image.

    Args:
        x0: x of the press point
        y0: y of the press point
        x1: x of the release point
        y1: y of the release point
        width: image width in pixels
        height: image height in pixels

    Returns:
        `[x0, y0, x1, y1]` with x0 < x1 and y0 < y1, inside `[0, width] x [0, height]`.
    """
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return [
        int(round(min(max(xa, 0), width))),
        int(round(min(max(ya, 0), height))),
        int(round(min(max(xb, 0), width))),
        int(round(min(max(yb, 0), height))),
    ]


class BoxAnnotator:
    """Interactive matplotlib window for drawing the reference boxes of each session."""

    def __init__(self, sessions: list[Session], output: Path) -> None:
        """Set up the figure, load any existing annotations, and show the first session.

        Args:
            sessions: sessions to annotate, in display order
            output: JSON file to read existing boxes from and save to
        """
        self.sessions = sessions
        self.output = output
        self.boxes = load_boxes(output)
        self.idx_session = 0
        self.region = REGIONS[0]
        self.show_background = False

        self.fig, self.ax = plt.subplots(figsize=(13, 8))
        self.selector = RectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords='pixels',
            interactive=False,
        )
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('close_event', lambda event: self.save())
        self.redraw()

    @property
    def session(self) -> Session:
        """The session currently shown."""
        return self.sessions[self.idx_session]

    def entry(self) -> dict:
        """Return (creating if needed) the saved entry of the current session."""
        entry = self.boxes.setdefault(self.session.name, {'regions': {}})
        entry['frame'] = self.session.frame.name
        return entry

    def redraw(self) -> None:
        """Draw the current frame, the saved boxes, and the status text."""
        image = (
            self.session.load_background() if self.show_background else self.session.load_frame()
        )
        height, width = image.shape[:2]
        self.ax.clear()
        self.ax.imshow(image)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        regions = self.boxes.get(self.session.name, {}).get('regions', {})
        lines = []
        for name in REGIONS:
            box = regions.get(name, 'unset')
            marker = '>' if name == self.region else ' '
            key = next(k for k, v in REGION_KEYS.items() if v == name)
            lines.append(f'{marker} [{key}] {name}: {"not visible" if box is None else box}')
            if isinstance(box, list):
                self.ax.add_patch(Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    edgecolor=REGION_COLORS[name],
                    linewidth=2.5 if name == self.region else 1.5,
                ))
        view = 'median background' if self.show_background else self.session.frame.name
        self.ax.set_title(
            f'{self.session.name}  ({self.idx_session + 1}/{len(self.sessions)})  '
            f'{view}  {width}x{height}'
        )
        self.ax.text(
            0.01,
            0.02,
            '\n'.join(lines),
            transform=self.ax.transAxes,
            va='bottom',
            family='monospace',
            fontsize=9,
            color='white',
            bbox={'facecolor': 'black', 'alpha': 0.6, 'pad': 4},
        )
        self.fig.canvas.draw_idle()

    def on_select(self, press, release) -> None:
        """Store the dragged rectangle as the active region's box.

        Args:
            press: mouse press event, in data (pixel) coordinates
            release: mouse release event, in data (pixel) coordinates
        """
        image = self.session.load_frame()
        height, width = image.shape[:2]
        box = normalize_box(press.xdata, press.ydata, release.xdata, release.ydata, width, height)
        entry = self.entry()
        entry['frame_size'] = [width, height]
        entry['regions'][self.region] = box
        self.redraw()

    def on_key(self, event) -> None:
        """Dispatch a key press (see the module docstring for the bindings)."""
        key = event.key
        if key in REGION_KEYS:
            self.region = REGION_KEYS[key]
        elif key == 'x':
            self.entry()['regions'][self.region] = None
        elif key == 'd':
            self.entry()['regions'].pop(self.region, None)
        elif key == 'c':
            self.copy_previous()
        elif key == 'f':
            session = self.session
            session.idx_frame = (session.idx_frame + 1) % len(session.frames)
        elif key == 'b':
            self.show_background = not self.show_background
        elif key in ('n', 'right'):
            self.move(1)
        elif key in ('p', 'left'):
            self.move(-1)
        elif key == 's':
            self.save()
        elif key == 'q':
            self.save()
            plt.close(self.fig)
            return
        else:
            return
        self.redraw()

    def copy_previous(self) -> None:
        """Copy every box of the previous session into the current one."""
        if self.idx_session == 0:
            return
        previous = self.boxes.get(self.sessions[self.idx_session - 1].name)
        if previous is None:
            return
        entry = self.entry()
        entry['frame_size'] = previous.get('frame_size')
        entry['regions'] = dict(previous['regions'])

    def move(self, step: int) -> None:
        """Save, then move to another session (clamped to the list)."""
        self.save()
        self.idx_session = min(max(self.idx_session + step, 0), len(self.sessions) - 1)

    def save(self) -> None:
        """Write all annotations to the output file."""
        save_boxes(self.output, self.boxes)
        logger.info(f'saved {len(self.boxes)} sessions to {self.output}')


def main() -> None:
    """Parse arguments and run the annotation window."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('root', type=Path, help='directory with one subdirectory per session')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='JSON file to write')
    parser.add_argument(
        '--frames-subdir',
        default=DEFAULT_FRAMES_SUBDIR,
        help='subdirectory of each session holding the frames',
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    sessions = discover_sessions(args.root, args.frames_subdir)
    logger.info(f'found {len(sessions)} sessions: {", ".join(s.name for s in sessions)}')
    annotator = BoxAnnotator(sessions, args.output)
    plt.show()
    del annotator


if __name__ == '__main__':
    main()
