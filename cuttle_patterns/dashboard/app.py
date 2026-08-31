"""Bokeh document: widgets, scatter plot, and callbacks for the embedding explorer.

Registered as the Bokeh application handler by `cuttle_patterns.dashboard.launch`. Data
loading/validation logic lives in `cuttle_patterns.dashboard.data`; this module is just the
widget wiring on top of it, and isn't unit tested (no real Bokeh server/browser in CI) —
see the "Verification" section of the Phase 7 plan for manual checks.
"""

from pathlib import Path

import pandas as pd
from bokeh.document import Document
from bokeh.layouts import column, row
from bokeh.models import CheckboxGroup, ColumnDataSource, CustomJS, Div, HoverTool, Select
from bokeh.palettes import Category20_20, Viridis256
from bokeh.plotting import figure

from cuttle_patterns import paths
from cuttle_patterns.dashboard import data

BASE_SOURCE_COLUMNS = ('umap_x', 'umap_y', 'video_name', 'frame_number', 'day', 'tank', 'role')
MAX_HOVER_TOOLTIPS = 3
HOVER_DEBOUNCE_MS = 150

# Used only for exactly-2-category columns (e.g. `role`): Category20's own first two
# colors are two shades of blue, easy to blend together at a glance or when zoomed out.
# For higher-cardinality columns (e.g. `video_name`) Category20 is kept as-is instead —
# its light/dark pairing is useful there, since alphabetically adjacent categories (a
# session's resident/intruder video pair) end up as the same hue at different lightness.
BINARY_CATEGORICAL_COLORS = ('#762a83', '#1b7837')  # purple, green

# Bokeh's native tooltips="<html>" would instantiate a fresh <img> per hit on every
# mousemove hit-test, which at ~100k+ points meant sweeping the cursor across the cloud
# could fire thousands of distinct image loads/decodes a minute — enough to crash the
# browser tab over a long session. tooltips=None + this callback instead reuse a single
# DOM node and only touch it (loading new images) after the cursor rests for
# HOVER_DEBOUNCE_MS, capping the image-load rate independent of how fast the mouse moves.
HOVER_CALLBACK_JS = """
const indices = cb_data.index.indices;

if (!window.__cuttleMouseListenerAdded) {
    window.__cuttleMouseListenerAdded = true;
    window.addEventListener('mousemove', (e) => {
        window.__cuttleMouseX = e.clientX;
        window.__cuttleMouseY = e.clientY;
    });
}

if (indices.length === 0) {
    clearTimeout(window.__cuttleHoverTimer);
    window.__cuttleHoverKey = null;
    const el = document.getElementById('cuttle-hover-tooltip');
    if (el) el.style.display = 'none';
    return;
}

const key = indices.slice(0, limit).join(',');
if (key === window.__cuttleHoverKey) {
    return;
}

clearTimeout(window.__cuttleHoverTimer);
window.__cuttleHoverTimer = setTimeout(() => {
    window.__cuttleHoverKey = key;

    let el = document.getElementById('cuttle-hover-tooltip');
    if (!el) {
        el = document.createElement('div');
        el.id = 'cuttle-hover-tooltip';
        el.style.position = 'fixed';
        el.style.pointerEvents = 'none';
        el.style.zIndex = '10000';
        el.style.background = 'white';
        el.style.border = '1px solid #ccc';
        el.style.borderRadius = '4px';
        el.style.padding = '4px';
        el.style.boxShadow = '0 2px 6px rgba(0,0,0,0.3)';
        document.body.appendChild(el);
    }

    const rows = source.data;
    let html = '';
    for (const i of indices.slice(0, limit)) {
        html += `
            <div style="max-width:220px; margin-bottom:6px;">
                <img src="/images/${rows['image_relpath'][i]}" style="width:100%; height:auto;">
                <div><b>video:</b> ${rows['video_name'][i]}</div>
                <div><b>frame:</b> ${rows['frame_number'][i]}</div>
                <div>
                    <b>day / tank / role:</b>
                    ${rows['day'][i]} / ${rows['tank'][i]} / ${rows['role'][i]}
                </div>
            </div>`;
    }
    el.innerHTML = html;
    el.style.display = 'block';
    el.style.left = (window.__cuttleMouseX + 15) + 'px';
    el.style.top = (window.__cuttleMouseY + 15) + 'px';
}, debounce_ms);
"""


def _categorical_colors(series: pd.Series) -> list[str]:
    """Map a column's values to hex colors from a repeating discrete palette."""
    categories = sorted(series.astype(str).unique())
    palette = BINARY_CATEGORICAL_COLORS if len(categories) == 2 else Category20_20
    color_by_category = {
        category: palette[i % len(palette)] for i, category in enumerate(categories)
    }
    return [color_by_category[value] for value in series.astype(str)]


def _continuous_colors(series: pd.Series) -> list[str]:
    """Map a column's values to hex colors via a linearly-normalized Viridis palette."""
    values = series.astype(float)
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return [Viridis256[0]] * len(values)
    normalized = ((values - vmin) / (vmax - vmin) * (len(Viridis256) - 1)).round().astype(int)
    return [Viridis256[i] for i in normalized]


def _compute_colors(df: pd.DataFrame, column_name: str) -> list[str]:
    """Compute one hex color per row for the given color-by column."""
    if data.is_categorical_column(df, column_name):
        return _categorical_colors(df[column_name])
    return _continuous_colors(df[column_name])


def _empty_source_data() -> dict[str, list]:
    return {col: [] for col in (*BASE_SOURCE_COLUMNS, 'image_relpath', 'color')}


def make_document(doc: Document, results_dir: Path) -> None:
    """Build the Phase 7 embedding explorer document.

    Args:
        doc: the Bokeh `Document` to populate.
        results_dir: the resolved results directory (holds `beast_models/`).
    """
    state: dict = {'df': None, 'cluster_paths': {}, 'reduce_paths': {}}

    source = ColumnDataSource(data=_empty_source_data())

    model_select = Select(title='Model', options=data.list_model_names(results_dir), value='')
    reduce_select = Select(title='Reduction', options=[], value='')
    cluster_label = Div(text='<b>Cluster attributes</b>')
    cluster_checkbox = CheckboxGroup(labels=[], active=[])
    color_select = Select(title='Color by', options=[], value='')
    error_div = Div(text='', styles={'color': 'red'})

    plot = figure(
        output_backend='webgl',
        tools='pan,wheel_zoom,box_zoom,reset',
        sizing_mode='stretch_both',
        x_axis_label='umap_x',
        y_axis_label='umap_y',
    )
    plot.scatter(
        'umap_x', 'umap_y', source=source, color='color', size=4, alpha=0.6, line_color=None,
    )
    hover_callback = CustomJS(
        args=dict(source=source, limit=MAX_HOVER_TOOLTIPS, debounce_ms=HOVER_DEBOUNCE_MS),
        code=HOVER_CALLBACK_JS,
    )
    plot.add_tools(HoverTool(tooltips=None, callback=hover_callback, limit=MAX_HOVER_TOOLTIPS))

    def _reset_selection() -> None:
        state['df'] = None
        source.data = _empty_source_data()
        color_select.options = []
        color_select.value = ''
        error_div.text = ''

    def _refresh_source() -> None:
        df = state['df']
        if df is None or df.empty:
            source.data = _empty_source_data()
            color_select.options = []
            color_select.value = ''
            return

        options = data.colorable_columns(df)
        color_column = color_select.value if color_select.value in options else (
            options[0] if options else ''
        )

        # update source.data (and color_select.options) before color_select.value, since
        # setting .value can synchronously trigger on_color_change, which patches
        # source.data['color'] assuming it already has the new dataset's length
        new_data = {col: df[col].to_numpy() for col in df.columns}
        new_data['color'] = (
            _compute_colors(df, color_column) if color_column else ['#888888'] * len(df)
        )
        source.data = new_data
        color_select.options = options
        color_select.value = color_column

    def on_model_change(attr: str, old: str, new: str) -> None:
        _reset_selection()
        reduce_select.options = []
        reduce_select.value = ''
        cluster_checkbox.labels = []
        cluster_checkbox.active = []
        state['reduce_paths'] = {}
        state['cluster_paths'] = {}
        if not new:
            return

        model_dir = results_dir / paths.BEAST_MODELS_RELPATH / new
        state['reduce_paths'] = {p.name: p for p in data.list_reduce_paths(model_dir)}
        state['cluster_paths'] = {p.name: p for p in data.list_cluster_paths(model_dir)}
        reduce_select.options = sorted(state['reduce_paths'])
        cluster_checkbox.labels = sorted(state['cluster_paths'])

    def on_reduce_change(attr: str, old: str, new: str) -> None:
        cluster_checkbox.active = []
        if not new:
            _reset_selection()
            return
        state['df'] = data.load_reduce_dataframe(state['reduce_paths'][new])
        error_div.text = ''
        _refresh_source()

    def on_cluster_toggle(attr: str, old: list[int], new: list[int]) -> None:
        if state['df'] is None:
            return

        labels = cluster_checkbox.labels
        newly_checked = [labels[i] for i in new if i not in old]
        newly_unchecked = [labels[i] for i in old if i not in new]

        for label in newly_unchecked:
            column_name = Path(label).stem
            if column_name in state['df'].columns:
                state['df'] = state['df'].drop(columns=[column_name])

        for label in newly_checked:
            cluster_path = state['cluster_paths'][label]
            try:
                state['df'] = data.attach_cluster_column(state['df'], cluster_path)
            except ValueError as e:
                error_div.text = f'<b>Error:</b> {e}'
                idx = labels.index(label)
                cluster_checkbox.active = [i for i in cluster_checkbox.active if i != idx]
                continue
            error_div.text = ''

        _refresh_source()

    def on_color_change(attr: str, old: str, new: str) -> None:
        df = state['df']
        if df is None or df.empty or not new:
            return
        colors = _compute_colors(df, new)
        source.patch({'color': [(slice(0, len(colors)), colors)]})

    model_select.on_change('value', on_model_change)
    reduce_select.on_change('value', on_reduce_change)
    cluster_checkbox.on_change('active', on_cluster_toggle)
    color_select.on_change('value', on_color_change)

    controls = column(
        model_select, reduce_select, cluster_label, cluster_checkbox, color_select, error_div,
        width=300,
    )
    doc.add_root(row(controls, plot, sizing_mode='stretch_both'))
    doc.title = 'Cuttle Pattern Explorer'
