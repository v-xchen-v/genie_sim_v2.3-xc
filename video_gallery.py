"""
Usage:
python -m pip install gradio
python video_gallery.py
Then open http://localhost:7860 in your browser.
Or http://<your-server-ip>:7860 if you run on a remote server.
"""

import os
import glob
import time
import math
from typing import List, Tuple

import pandas as pd
import gradio as gr


def scan_tree(root_dir: str) -> pd.DataFrame:
    """
    Expect structure: <root>/<port>/<iter>/<task>/*.mp4
    Returns a DataFrame with columns:
    ['port', 'iter', 'task', 'filename', 'filepath', 'size_mb', 'mtime']
    """
    pattern = os.path.join(root_dir, "*", "*", "*", "*.mp4")
    paths = glob.glob(pattern, recursive=False)

    rows = []
    for p in paths:
        try:
            # Extract levels
            filename = os.path.basename(p)
            task_dir = os.path.dirname(p)
            iter_dir = os.path.dirname(task_dir)
            port_dir = os.path.dirname(iter_dir)

            task = os.path.basename(task_dir)
            iter_name = os.path.basename(iter_dir)
            port = os.path.basename(port_dir)

            stat = os.stat(p)
            size_mb = round(stat.st_size / (1024 * 1024), 2)
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

            rows.append(
                {
                    "port": port,
                    "iter": iter_name,
                    "task": task,
                    "filename": filename,
                    "filepath": p,
                    "size_mb": size_mb,
                    "mtime": mtime,
                }
            )
        except Exception:
            # Skip anything malformed
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort most recent first
        df = df.sort_values(by=["mtime", "port", "iter", "task", "filename"], ascending=[False, True, True, True, True])
        df = df.reset_index(drop=True)
    return df


def unique_sorted(values: List[str]) -> List[str]:
    return sorted(list(set(values)))


def compute_pages(total_items: int, per_page: int) -> int:
    if total_items == 0:
        return 1
    return max(1, math.ceil(total_items / max(1, per_page)))


def filter_df(df: pd.DataFrame, ports, iters, tasks):
    if df is None or df.empty:
        return df
    sub = df.copy()
    if ports:
        sub = sub[sub["port"].isin(ports)]
    if iters:
        sub = sub[sub["iter"].isin(iters)]
    if tasks:
        sub = sub[sub["task"].isin(tasks)]
    sub = sub.reset_index(drop=True)
    return sub


def page_slice(df: pd.DataFrame, page: int, per_page: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    page = max(1, page)
    start = (page - 1) * per_page
    end = start + per_page
    return df.iloc[start:end].reset_index(drop=True)


def df_to_gallery_items(df_page: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Gradio Gallery accepts a list of (path, caption). It can render mp4 files.
    """
    if df_page is None or df_page.empty:
        return []
    items = []
    for _, r in df_page.iterrows():
        label = f"{r['port']}/{r['iter']}/{r['task']}/{r['filename']}"
        items.append((r["filepath"], label))
    return items


def do_scan(root_dir: str):
    if not root_dir or not os.path.isdir(root_dir):
        empty = pd.DataFrame(columns=["port", "iter", "task", "filename", "filepath", "size_mb", "mtime"])
        return empty, [], [], [], gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), 1, "No root or invalid directory."
    df = scan_tree(root_dir)
    if df.empty:
        return df, [], [], [], gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), 1, "No MP4 files found."
    ports = unique_sorted(df["port"].tolist())
    iters = unique_sorted(df["iter"].tolist())
    tasks = unique_sorted(df["task"].tolist())
    total_pages = compute_pages(len(df), 12)
    return (
        df,
        ports,
        iters,
        tasks,
        gr.update(choices=ports, value=[]),
        gr.update(choices=iters, value=[]),
        gr.update(choices=tasks, value=[]),
        total_pages,
        f"Found {len(df)} videos.",
    )


def refresh_view(df, ports, iters, tasks, page, per_page):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return pd.DataFrame(), [], 1, "No data."
    sub = filter_df(df, ports, iters, tasks)
    total_pages = compute_pages(len(sub), per_page)
    # clamp page
    page = min(max(1, page), total_pages)
    df_page = page_slice(sub, page, per_page)
    gallery_items = df_to_gallery_items(df_page)
    # For the table, show a compact subset
    table_cols = ["port", "iter", "task", "filename", "size_mb", "mtime", "filepath"]
    table = df_page[table_cols] if not df_page.empty else pd.DataFrame(columns=table_cols)
    info = f"{len(sub)} videos match filters · Page {page}/{total_pages}"
    return table, gallery_items, total_pages, info


def pick_video(table_df, selected_row):
    if table_df is None or len(table_df) == 0:
        return None, "No selection."
    try:
        idx = int(selected_row)
    except Exception:
        return None, "Enter a valid row index (0-based on current page)."
    if idx < 0 or idx >= len(table_df):
        return None, f"Row index out of range (0..{len(table_df)-1})."
    path = table_df.iloc[idx]["filepath"]
    label = f"{table_df.iloc[idx]['port']}/{table_df.iloc[idx]['iter']}/{table_df.iloc[idx]['task']}/{table_df.iloc[idx]['filename']}"
    return path, f"Selected: {label}"


with gr.Blocks(title="Video Tree Browser") as demo:
    gr.Markdown("# 🎥 Video Tree Browser\nExplore videos under `<ROOT>/<PORT>/<ITER>/<TASK>/*.mp4`")

    with gr.Row():
        root_dir = gr.Textbox(label="Root directory", value="/home/xichen6/Documents/repos/genie_sim_v2.3/genie_sim_v2.3-xc/AgiBot-World-Submission/CogACT/video_recordings", scale=4)
        scan_btn = gr.Button("Scan", variant="primary", scale=1)

    info = gr.Markdown()

    # State to hold the full dataframe
    df_state = gr.State(None)

    with gr.Row():
        port_filter = gr.Dropdown(label="Filter: port", choices=[], multiselect=True)
        iter_filter = gr.Dropdown(label="Filter: iter", choices=[], multiselect=True)
        task_filter = gr.Dropdown(label="Filter: task", choices=[], multiselect=True)

    with gr.Row():
        per_page = gr.Slider(4, 48, value=12, step=4, label="Videos per page")
        page = gr.Number(value=1, label="Page (1-based)", precision=0)
        total_pages = gr.Number(value=1, label="Total pages", interactive=False)

    with gr.Tab("Gallery"):
        gallery = gr.Gallery(label="Videos (click to play)", columns=3, height=600, allow_preview=True)
    with gr.Tab("Table + Player"):
        table = gr.Dataframe(
            headers=["port", "iter", "task", "filename", "size_mb", "mtime", "filepath"],
            datatype=["str"] * 7,
            interactive=False,
            wrap=True,
            max_height=360,   # ✅ replace height with max_height
        )

        with gr.Row():
            row_idx = gr.Textbox(label="Row index (0..N on current page)", value="0")
            load_btn = gr.Button("Load Selected")
        player = gr.Video(label="Preview")
        selection_info = gr.Markdown()

    # Wire up actions
    def _scan_click(root):
        df, ports, iters, tasks, up1, up2, up3, total_p, msg = do_scan(root)
        return df, up1, up2, up3, total_p, msg

    scan_btn.click(
        _scan_click,
        inputs=[root_dir],
        outputs=[df_state, port_filter, iter_filter, task_filter, total_pages, info],
    ).then(
        refresh_view,
        inputs=[df_state, port_filter, iter_filter, task_filter, page, per_page],
        outputs=[table, gallery, total_pages, info],
    )

    # Any change in filters/pagination refreshes the view
    for ctl in [port_filter, iter_filter, task_filter, page, per_page]:
        ctl.change(
            refresh_view,
            inputs=[df_state, port_filter, iter_filter, task_filter, page, per_page],
            outputs=[table, gallery, total_pages, info],
        )

    # Load selected row to the single-player
    load_btn.click(
        pick_video,
        inputs=[table, row_idx],
        outputs=[player, selection_info],
    )

if __name__ == "__main__":
    # You can change server_name/port if needed.
    demo.launch(server_name="0.0.0.0", server_port=7860)
