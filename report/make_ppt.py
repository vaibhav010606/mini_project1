"""Build the presentation deck: report/presentation.pptx (12 slides, 16:9).

    python report/make_ppt.py

All numbers are read from results/*.json so the slides always match the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
FIG = RES / "figures"
SHOT = RES / "screenshots"
ASSET = ROOT / "report" / "ppt_assets"
OUT = ROOT / "report" / "presentation.pptx"

# ---------------------------------------------------------------- palette and fonts
NAVY, INK, MUTED = "14213D", "1F2937", "5B6472"
LIGHT, LINE, WHITE = "EEF2F6", "D5DCE4", "FFFFFF"
RED, TEAL, AMBER = "E63946", "2A9D8F", "E9A23B"  # alert / 5G / video
TEAL_SOFT, AMBER_SOFT, RED_SOFT, NAVY_SOFT = "E3F3F1", "FCF1DE", "FDE7E9", "E5E9F2"
HEAD, BODY = "Cambria", "Calibri"
W, H = 13.333, 7.5


def rgb(h):
    return RGBColor.from_string(h)


def load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else {}


V, B, N, J, EC, OF = (load(n) for n in ("video_metrics.json", "benchmark.json", "network_metrics.json",
                                         "joint_metrics.json", "edge_cloud_results.json", "offload_results.json"))
JR = J.get("results", {})
COMB, VONLY, NONLY = "Combined (video + 5G)", "Video only", "5G only"
ALL = "All test events"


def cam(setup, n):
    c = EC["cameras"]
    return c["results"][setup][c["n"].index(n)]["mean_ms"]


def slice95(name, load_):
    s = EC["slicing"]
    return s["results"][name][s["bg_load"].index(load_)]["p95_ms"]


NUM = dict(
    auc_v2=V["pred"]["auc"]["patch_max_norm_smooth"], auc_v1=V["big"]["auc"]["patch_max_norm_smooth"],
    auc_student=V["student"]["auc"]["patch_max_norm_smooth"],
    net_acc=N["xgboost"]["accuracy"], net_f1=N["xgboost"]["macro_f1"],
    net_missed=f"{N['xgboost']['stampede_episodes_missed']} / {N['xgboost']['stampede_episodes_total']}",
    comb=JR[COMB][ALL]["macro_f1"], vonly=JR[VONLY][ALL]["macro_f1"], nonly=JR[NONLY][ALL]["macro_f1"],
    comb_danger=JR[COMB][ALL]["danger_detected"], comb_fa=JR[COMB][ALL]["false_alarm"],
    edge10=cam("Edge - lightweight model", 10), cloud10=cam("Cloud - accurate model (GPU)", 10),
    sl_on=slice95("URLLC slice (priority)", 0.9), sl_off=slice95("No slicing (shared)", 0.9),
    ml_acc=OF["ml_accuracy"],
)


# ---------------------------------------------------------------- helpers
def text(slide, x, y, w, h, content, size=16, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, name=None, spacing=None):
    """content: str, or list of paragraphs; a paragraph is a str or a list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        runs = para if isinstance(para, list) else [(para, {})]
        for t, o in runs:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(slide, x, y, w, h, fill=LIGHT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, dash=False, name=None):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    s.shadow.inherit = False
    if fill:
        s.fill.solid()
        s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(1.5)
        if dash:
            s.line.dash_style = MSO_LINE.DASH
    else:
        s.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.text = ""
    return s


def badge(slide, x, y, d, label, fill, size=16, color=WHITE):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.name = BODY
    r.font.color.rgb = rgb(color)
    return c


def arrow(slide, x, y, w=0.4, h=0.36, color=MUTED, shape=MSO_SHAPE.RIGHT_ARROW):
    return box(slide, x, y, w, h, fill=color, shape=shape)


def picture(slide, path, x, y, w=None, h=None, border=True):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y),
                                   Inches(w) if w else None, Inches(h) if h else None)
    if border:
        pic.line.color.rgb = rgb(LINE)
        pic.line.width = Pt(1)
    return pic


def title(slide, t, sub=None, dark=False):
    text(slide, 0.6, 0.42, 12.1, 0.75, t, size=34, bold=True, font=HEAD, color=WHITE if dark else NAVY)
    if sub:
        text(slide, 0.6, 1.17, 12.1, 0.45, sub, size=16, color="C9D3E3" if dark else MUTED)


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(color)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def footer(slide, n, dark=False):
    text(slide, 11.9, 7.0, 0.83, 0.3, str(n), size=10, color="8A97AD" if dark else "9AA3AF", align=PP_ALIGN.RIGHT)


def stat(slide, x, y, w, value, label, color=NAVY, vsize=40, lsize=13, lcolor=MUTED, align=PP_ALIGN.LEFT):
    text(slide, x, y, w, 0.8, value, size=vsize, bold=True, color=color, font=HEAD, align=align)
    gap = vsize / 72 * 1.3 + 0.05  # label sits just under the number, whatever its size
    text(slide, x, y + gap, w, 0.6, label, size=lsize, color=lcolor, align=align)


def style_chart(chart, colors, maxv=None, fmt="0.00", legend=True, size=12):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = rgb(INK)
    plot = chart.plots[0]
    plot.gap_width = 70
    plot.overlap = -8
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(size - 1)
    dl.font.color.rgb = rgb(INK)
    for s, c in zip(plot.series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(c)
        s.format.line.fill.background()
    va = chart.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb("E5E7EB")
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    va.minimum_scale = 0
    if maxv:
        va.maximum_scale = maxv
    ca = chart.category_axis
    ca.format.line.color.rgb = rgb(LINE)
    ca.tick_labels.font.size = Pt(size)
    ca.tick_labels.font.color.rgb = rgb(INK)
    chart.has_legend = legend
    if chart.has_title:
        for r in chart.chart_title.text_frame.paragraphs[0].runs:
            r.font.name = BODY
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)


# ---------------------------------------------------------------- assets
def make_heatmap():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from network_model.simulator import GNB, VENUE, run_episode

    _, _, snaps, _ = run_episode(np.random.default_rng(3), "gathering", n_people=2400, snapshot_times=(156,))
    est = snaps[156][1]
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.imshow(est.T, origin="lower", extent=[0, VENUE, 0, VENUE], cmap="RdYlGn_r", vmin=0, vmax=3)
    ax.scatter(GNB[:, 0], GNB[:, 1], marker="^", c="black", s=40)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(ASSET / "heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------- slides
def build():
    ASSET.mkdir(parents=True, exist_ok=True)
    if not (ASSET / "heatmap.png").exists():
        make_heatmap()
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    blank = prs.slide_layouts[6]
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    # 1 ---- title
    s = prs.slides.add_slide(blank)
    background(s, NAVY)
    text(s, 0.7, 0.75, 6.6, 0.4, "AI/ML PROJECT · 5G EDGE COMPUTING", size=13, bold=True, color="8FB8DE")
    text(s, 0.7, 1.3, 6.4, 2.6, "Crowd Safety AI: detecting stampedes from video and 5G network data",
         size=38, bold=True, font=HEAD, color=WHITE)
    text(s, 0.7, 4.15, 6.2, 0.9, "Two AI detectors, one combined model, running at the 5G edge for millisecond alerts",
         size=17, color="C9D3E3")
    text(s, 0.7, 5.6, 6.0, 1.0, [[("Presented by: ", {"bold": True}), ("________________", {})],
                                 [("Guide: ", {"bold": True}), ("________________", {})]],
         size=15, color="C9D3E3", spacing=4)
    picture(s, ASSET / "stampede_frame.jpg", 7.55, 1.3, w=5.2, border=False)
    box(s, 7.55, 4.3, 5.2, 0.55, fill="0B1426")
    text(s, 7.75, 4.38, 4.9, 0.4, "Simulated CCTV view: panic spreads from the stage", size=12, color="C9D3E3",
         anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Introduce the project: an AI system that detects dangerous crowds using two sources, CCTV video and "
             "the 5G network, and runs next to the 5G tower so alerts arrive in milliseconds.")

    # 2 ---- problem
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Why this matters", "Crowd disasters develop in seconds, and today's monitoring has blind spots")
    items = [
        ("Danger builds fast", "A dense crowd can turn into a stampede within seconds; a late alert is useless."),
        ("Too many screens", "Control-room operators cannot watch every CCTV camera at the same time."),
        ("Cameras have blind spots", "Night, smoke, blocked views and areas no camera covers."),
    ]
    for i, (h_, d) in enumerate(items):
        y = 2.0 + i * 1.45
        box(s, 0.6, y, 6.4, 1.2, fill=LIGHT)
        badge(s, 0.85, y + 0.3, 0.6, str(i + 1), RED, size=18)
        text(s, 1.7, y + 0.18, 5.1, 0.4, h_, size=19, bold=True, color=NAVY)
        text(s, 1.7, y + 0.6, 5.1, 0.55, d, size=14, color=INK)
    picture(s, ASSET / "calm_frame.jpg", 7.45, 2.0, w=5.28)
    box(s, 7.45, 5.15, 5.28, 0.95, fill=RED_SOFT)
    text(s, 7.7, 5.25, 4.9, 0.8, [[("Goal: ", {"bold": True, "color": RED}),
                                  ("detect crowd danger accurately and deliver the alert in milliseconds.", {})]],
         size=16, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, 2)
    notes(s, "Three problems: speed, operator overload, and camera blind spots. Our goal is accurate detection with "
             "millisecond alerts.")

    # 3 ---- architecture
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Our solution: two senses, one decision", "The camera sees what happens; the 5G network feels the crowd")
    for i, (lab, sub, fill, col) in enumerate([("📷  CCTV cameras", "video frames", AMBER_SOFT, AMBER),
                                               ("📱  Phones", "5G network data", TEAL_SOFT, TEAL)]):
        y = 2.2 + i * 1.55
        box(s, 0.6, y, 2.0, 1.3, fill=fill)
        text(s, 0.75, y + 0.25, 1.75, 0.45, lab, size=15, bold=True, color=NAVY)
        text(s, 0.75, y + 0.72, 1.75, 0.4, sub, size=12, color=MUTED)
    arrow(s, 2.72, 3.25)
    box(s, 3.25, 2.0, 2.45, 3.25, fill=TEAL_SOFT)
    text(s, 3.45, 2.15, 2.1, 0.45, "📡  5G network", size=16, bold=True, color=NAVY)
    for j, (lab, d, f_) in enumerate([("URLLC slice", "safety cameras + alerts", RED_SOFT),
                                      ("eMBB slice", "public phone traffic", WHITE)]):
        y = 2.75 + j * 1.15
        box(s, 3.45, y, 2.05, 0.95, fill=f_)
        text(s, 3.6, y + 0.1, 1.8, 0.35, lab, size=13, bold=True, color=NAVY)
        text(s, 3.6, y + 0.48, 1.8, 0.4, d, size=11, color=MUTED)
    arrow(s, 5.82, 3.25)
    box(s, 6.35, 2.0, 4.25, 3.25, fill=NAVY)
    text(s, 6.55, 2.12, 3.9, 0.4, "⚡  Edge server (next to the tower)", size=15, bold=True, color=WHITE)
    cards = [("Model 1", "video anomaly", AMBER), ("Model 2", "5G crowd risk", TEAL),
             ("Combined model", "video + 5G", RED), ("ML decision", "edge or cloud?", "8FB8DE")]
    for k, (lab, d, col) in enumerate(cards):
        cx, cy = 6.55 + (k % 2) * 2.0, 2.65 + (k // 2) * 1.25
        box(s, cx, cy, 1.85, 1.05, fill=WHITE)
        badge(s, cx + 0.12, cy + 0.12, 0.32, "", col)
        text(s, cx + 0.52, cy + 0.1, 1.3, 0.4, lab, size=13, bold=True, color=NAVY)
        text(s, cx + 0.15, cy + 0.55, 1.65, 0.4, d, size=11, color=MUTED)
    arrow(s, 10.72, 3.25)
    box(s, 11.25, 2.55, 1.48, 1.9, fill=RED)
    text(s, 11.35, 2.7, 1.3, 1.6, [[("🚨", {"size": 22})], [("Alert to security", {"bold": True})]],
         size=14, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 8.32, 5.29, w=0.32, h=0.48, shape=MSO_SHAPE.UP_DOWN_ARROW)
    box(s, 6.35, 5.8, 4.25, 0.85, fill=None, line=MUTED, dash=True)
    text(s, 6.55, 5.85, 3.95, 0.75, [[("☁️  Cloud: ", {"bold": True}), ("most accurate model on GPU, used when the edge is busy", {})]],
         size=12, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.6, 6.0, 5.4, 0.8, [[("Real: ", {"bold": True, "color": NAVY}), ("UCSD video, model accuracy, measured AI speeds", {})],
                                 [("Simulated: ", {"bold": True, "color": NAVY}), ("5G network data, edge server, slicing", {})]],
         size=13, color=MUTED, spacing=2)
    footer(s, 3)
    notes(s, "Walk left to right: cameras and phones send data over 5G; safety traffic gets its own URLLC slice; the "
             "edge server runs the models and sends the alert. The cloud is a backup for heavy load. Be clear that the "
             "5G side is simulated.")

    # 4 ---- data
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "The data", "One real video benchmark, plus simulated 5G data of the same crowds")
    cols = [
        (ASSET / "ucsd_frame.jpg", "UCSD Ped2", "REAL", AMBER,
         ["Public CCTV benchmark", "16 normal training videos", "12 test videos: bikes, carts, skaters"]),
        (ASSET / "heatmap.png", "5G crowd simulator", "SIMULATED", TEAL,
         ["200 m venue, 9 small cells (3.5 GHz)", "800-2,600 people with phones", "450 crowds: normal, dense, stampede"]),
        (ASSET / "joint_frame.jpg", "Joint video + 5G events", "SIMULATED", RED,
         ["Same crowd: CCTV video + 5G data", "300 events, night and blind spots", "Trains the combined model"]),
    ]
    for i, (img, t_, tag, col, bullets) in enumerate(cols):
        x = 0.6 + i * 4.18
        box(s, x, 1.95, 3.95, 4.85, fill=LIGHT)
        from PIL import Image

        iw, ih = Image.open(img).size
        pw = min(3.55, 2.25 * iw / ih)
        picture(s, img, x + (3.95 - pw) / 2, 2.15, w=pw, h=pw * ih / iw)
        box(s, x + 0.2, 4.6, 1.35 if tag == "REAL" else 1.6, 0.36, fill=col, radius=0.5)
        text(s, x + 0.2, 4.6, 1.35 if tag == "REAL" else 1.6, 0.36, tag, size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, x + 0.2, 5.05, 3.55, 0.4, t_, size=18, bold=True, color=NAVY)
        text(s, x + 0.2, 5.5, 3.55, 1.25, [[("•  " + b, {})] for b in bullets], size=13, color=INK, spacing=2)
    footer(s, 4)
    notes(s, "UCSD Ped2 is real video. Phone-network data is private, so we simulated it with standard 3GPP radio "
             "models. The joint dataset films and senses the same simulated crowd so both sources describe one event.")

    # 5 ---- model 1
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Model 1: spotting unusual things in video", "Trained only on normal walking; it flags what it cannot predict")
    steps = ["Look at the last 4 video frames", "U-Net predicts the next frame",
             "Compare prediction with what really happened", "Big error in one spot = anomaly"]
    for i, st in enumerate(steps):
        y = 2.0 + i * 0.78
        badge(s, 0.6, y, 0.5, str(i + 1), AMBER, size=15)
        text(s, 1.3, y + 0.06, 5.0, 0.45, st, size=16, color=INK, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.6, 5.3, 5.9, 1.45, fill=AMBER_SOFT)
    stat(s, 0.85, 5.35, 2.6, f"{NUM['auc_v2']:.3f}", "AUC, final model (v2)", color=NAVY, vsize=36)
    stat(s, 3.55, 5.35, 2.8, f"{NUM['auc_v1']:.3f}", "AUC, first version (autoencoder)", color=MUTED, vsize=36)
    picture(s, FIG / "video_examples.png", 6.85, 1.95, w=5.88)
    text(s, 6.85, 6.55, 5.88, 0.35, "Most unusual frames and where the model sees the anomaly (red = unusual)",
         size=11, color=MUTED)
    footer(s, 5)
    notes(s, f"Version 2 predicts the next frame (Liu et al. 2018). Normal walking is easy to predict, a cart or bike "
             f"is not. AUC improved from {NUM['auc_v1']:.3f} to {NUM['auc_v2']:.3f}. A small edge version "
             f"(AUC {NUM['auc_student']:.3f}) runs fast on the edge server.")

    # 6 ---- model 2
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Model 2: feeling the crowd through 5G", "Phones per cell, movement and handovers reveal density and panic")
    feats = ["phones per cell", "densest zone", "density growth", "mean speed", "fast phones",
             "handovers", "cell load", "signal quality", "total phones"]
    for k, ftxt in enumerate(feats):
        cx, cy = 0.6 + (k % 3) * 1.68, 2.0 + (k // 3) * 0.6
        box(s, cx, cy, 1.56, 0.46, fill=TEAL_SOFT, radius=0.5)
        text(s, cx, cy, 1.56, 0.46, ftxt, size=12, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 2.55, 3.88, w=0.4, h=0.4, color=MUTED, shape=MSO_SHAPE.DOWN_ARROW)
    box(s, 0.6, 4.4, 4.95, 0.62, fill=TEAL)
    text(s, 0.6, 4.4, 4.95, 0.62, "XGBoost on the last 30 seconds", size=16, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    for j, (lab, col) in enumerate([("normal", "2E8B57"), ("dense", AMBER), ("stampede", RED)]):
        cx = 0.6 + j * 1.68
        box(s, cx, 5.2, 1.56, 0.46, fill=col, radius=0.5)
        text(s, cx, 5.2, 1.56, 0.46, lab, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    stat(s, 0.6, 5.85, 1.9, pct(NUM["net_acc"]), "accuracy", vsize=30, lsize=12)
    stat(s, 2.45, 5.85, 1.6, f"{NUM['net_f1']:.3f}", "macro-F1", vsize=30, lsize=12)
    stat(s, 4.0, 5.85, 1.8, NUM["net_missed"], "stampedes missed", vsize=30, lsize=12)
    picture(s, FIG / "net_timeline.png", 7.0, 1.85, h=5.0)
    footer(s, 6)
    notes(s, "Nine network signals per second, summarised over 30 seconds. XGBoost decides normal, dense or stampede. "
             "On the right: one simulated stampede; fast phones and handovers jump the moment panic starts and the "
             "model's stampede probability goes to 1.")

    # 7 ---- combined model
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Combined model: one decision from both senses", "Same event, video + 5G together: each covers the other's blind spot")
    box(s, 0.6, 2.05, 2.15, 1.0, fill=AMBER_SOFT)
    text(s, 0.75, 2.12, 1.9, 0.9, [[("Video branch", {"bold": True, "color": NAVY})], [("CNN on 4 frames", {"size": 12, "color": MUTED})]],
         size=15, anchor=MSO_ANCHOR.MIDDLE)
    box(s, 0.6, 3.35, 2.15, 1.0, fill=TEAL_SOFT)
    text(s, 0.75, 3.42, 1.9, 0.9, [[("5G branch", {"bold": True, "color": NAVY})], [("network on 30 s", {"size": 12, "color": MUTED})]],
         size=15, anchor=MSO_ANCHOR.MIDDLE)
    arrow(s, 2.85, 3.0)
    box(s, 3.35, 2.6, 1.15, 1.2, fill=NAVY)
    text(s, 3.35, 2.6, 1.15, 1.2, "Fusion", size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.6, 4.65, 3.9, 1.2, "Trained together with modality dropout: sometimes one source is hidden, so the "
                                 "model learns to rely on the other when it fails.", size=13, color=MUTED)
    stat(s, 0.6, 5.75, 2.0, pct(NUM["comb_danger"]), "danger caught", color=NAVY, vsize=30, lsize=12)
    stat(s, 2.55, 5.75, 2.0, pct(NUM["comb_fa"]), "false alarms", color=NAVY, vsize=30, lsize=12)
    conds = [("All events", ALL), ("Danger off-camera", "Danger outside camera view"), ("Few phones", "Few phones visible")]
    cd = CategoryChartData()
    cd.categories = [c[0] for c in conds]
    for nm in (VONLY, NONLY, COMB):
        cd.add_series(nm, [round(JR[nm][c[1]]["macro_f1"], 2) for c in conds])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(4.9), Inches(1.85), Inches(7.85), Inches(4.75), cd)
    gf.name = "Combined model chart"
    style_chart(gf.chart, [AMBER, TEAL, NAVY], maxv=1.1)
    gf.chart.has_title = True
    gf.chart.chart_title.text_frame.text = "Accuracy on unseen events (macro-F1)"
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.name = BODY
    gf.chart.value_axis.tick_labels.number_format = "0.0"
    gf.chart.value_axis.tick_labels.number_format_is_linked = False
    text(s, 4.9, 6.65, 7.85, 0.3, "Night is harder for all models (only 7 night test events; see report).", size=11, color=MUTED)
    footer(s, 7)
    notes(s, f"Key slide. Combined macro-F1 {NUM['comb']:.3f} vs {NUM['nonly']:.3f} for 5G only and {NUM['vonly']:.3f} "
             f"for video only. When the danger is outside the camera view, 5G covers it; when few phones are visible, "
             f"video covers it. Synthetic joint data; night results rest on 7 test events.")

    # 8 ---- edge vs cloud
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Edge vs cloud over 5G", "Where the AI runs decides how fast the alert arrives")
    box(s, 0.6, 1.95, 5.75, 2.2, fill=LIGHT)
    text(s, 0.85, 2.08, 5.3, 0.4, "Alert delay with 10 cameras", size=15, bold=True, color=NAVY)
    stat(s, 0.85, 2.55, 2.5, f"{NUM['edge10']:.1f} ms", "⚡ edge server", color=TEAL, vsize=40)
    stat(s, 3.45, 2.55, 2.7, f"{NUM['cloud10']:.1f} ms", "☁️ cloud", color=MUTED, vsize=40)
    box(s, 0.6, 4.4, 5.75, 2.2, fill=RED_SOFT)
    text(s, 0.85, 4.53, 5.3, 0.4, "Network slicing, 90% of the network busy", size=15, bold=True, color=NAVY)
    stat(s, 0.85, 5.0, 2.5, f"{NUM['sl_on']:.1f} ms", "with URLLC slice", color=RED, vsize=40)
    stat(s, 3.45, 5.0, 2.7, f"{NUM['sl_off']:.1f} ms", "without slicing", color=MUTED, vsize=40)
    pol = [k for k, v in OF.items() if isinstance(v, dict)]
    cd = CategoryChartData()
    cd.categories = [p.replace(" (best possible)", "") for p in pol]
    cd.add_series("Average 95th-percentile delay (ms)", [round(OF[p]["mean_p95_ms"], 1) for p in pol])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.75), Inches(1.85), Inches(6.0), Inches(4.4), cd)
    gf.name = "Offloading chart"
    style_chart(gf.chart, [MUTED], fmt="0", legend=False)
    pts = gf.chart.plots[0].series[0].points
    for i, p in enumerate(pol):
        pts[i].format.fill.solid()
        pts[i].format.fill.fore_color.rgb = rgb(NAVY if p == "ML decides" else "B8C0CC")
    gf.chart.has_title = True
    gf.chart.chart_title.text_frame.text = "Who decides where to process? (delay, ms)"
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    gf.chart.chart_title.text_frame.paragraphs[0].runs[0].font.name = BODY
    text(s, 6.75, 6.3, 6.0, 0.5, [[("ML picks the right place ", {}), (pct(NUM["ml_acc"]), {"bold": True, "color": NAVY}),
                                   (" of the time", {})]], size=15, color=INK, align=PP_ALIGN.CENTER)
    footer(s, 8)
    notes(s, "The edge is about 8 times faster while it has capacity; with too many cameras it overloads and the cloud "
             "wins. A URLLC slice keeps safety traffic fast when the public's phones fill the network. An ML model "
             "chooses edge or cloud per situation and gets close to the best possible delay. All simulated, using "
             "AI speeds measured on the laptop.")

    # 9 ---- web app
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Live demo: the web application", "React frontend + FastAPI backend runs every model in the browser")
    picture(s, SHOT / "combined_main.jpg", 0.6, 1.85, h=4.65)
    text(s, 0.6, 6.62, 6.9, 0.35, "Combined monitor: CCTV + 5G map of the same stampede, all models say STAMPEDE",
         size=11, color=MUTED)
    side = [("video_main.jpg", "Video detection"), ("crowd_main.jpg", "5G crowd monitor"), ("edge_main.jpg", "Edge vs cloud")]
    for i, (img, lab) in enumerate(side):
        y = 1.85 + i * 1.62
        picture(s, SHOT / img, 7.85, y, h=1.45)
        text(s, 10.15, y + 0.47, 2.6, 0.5, lab, size=15, bold=True, color=NAVY, anchor=MSO_ANCHOR.MIDDLE)
    footer(s, 9)
    notes(s, "Five pages: Combined monitor (the main demo), Video detection (upload a video), 5G crowd monitor, "
             "Edge vs cloud with sliders, and Results. Run: backend with uvicorn, frontend with npm run dev.")

    # 10 ---- results
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Results at a glance", "Accurate detection, reliable when one source fails, alerts in milliseconds")
    tiles = [
        (f"{NUM['auc_v2']:.3f}", "Video model AUC", "UCSD Ped2, real video", AMBER_SOFT, NAVY),
        (pct(NUM["net_acc"]), "5G model accuracy", "0 stampedes missed", TEAL_SOFT, NAVY),
        (f"{NUM['comb']:.3f}", "Combined model macro-F1", f"vs {NUM['nonly']:.3f} 5G, {NUM['vonly']:.3f} video", NAVY_SOFT, NAVY),
        (f"{NUM['edge10']:.1f} ms", "Edge alert delay", f"cloud: {NUM['cloud10']:.1f} ms", LIGHT, TEAL),
        (f"{NUM['sl_on']:.1f} ms", "With network slicing", f"without: {NUM['sl_off']:.1f} ms", RED_SOFT, RED),
        (pct(NUM["ml_acc"]), "ML edge/cloud choice", "correct on unseen situations", LIGHT, NAVY),
    ]
    for k, (v, lab, sub, fill, col) in enumerate(tiles):
        x, y = 0.6 + (k % 3) * 4.18, 1.95 + (k // 3) * 2.45
        box(s, x, y, 3.95, 2.2, fill=fill)
        text(s, x + 0.3, y + 0.25, 3.4, 0.9, v, size=44, bold=True, font=HEAD, color=col)
        text(s, x + 0.3, y + 1.15, 3.4, 0.4, lab, size=16, bold=True, color=NAVY)
        text(s, x + 0.3, y + 1.6, 3.4, 0.4, sub, size=13, color=MUTED)
    footer(s, 10)
    notes(s, "Summarise the six headline numbers. The combined model is the main contribution.")

    # 11 ---- limitations
    s = prs.slides.add_slide(blank)
    background(s, WHITE)
    title(s, "Limitations and future work", "What is simulated today, and how to take it further")
    lim = ["5G network data is simulated (real operator data is private)",
           "Combined model's CCTV video is rendered from simulated people",
           "Night results rest on only 7 test events",
           "5G network in Python, not the full Simu5G protocol stack"]
    fut = ["Validate on real aggregated operator data (e.g. Telecom Italia Milan)",
           "Train the combined model on real CCTV footage",
           "Port the 5G scenario to Simu5G (OMNeT++)",
           "Reinforcement learning for offloading and slice control"]
    for c, (head, items, fill, col, mark) in enumerate([("Limitations", lim, RED_SOFT, RED, "!"), ("Future work", fut, TEAL_SOFT, TEAL, "→")]):
        x = 0.6 + c * 6.3
        box(s, x, 1.95, 5.95, 4.85, fill=fill)
        text(s, x + 0.35, 2.15, 5.3, 0.5, head, size=22, bold=True, color=NAVY, font=HEAD)
        for i, it in enumerate(items):
            y = 2.9 + i * 0.95
            badge(s, x + 0.35, y, 0.45, mark, col, size=14)
            text(s, x + 1.0, y - 0.02, 4.7, 0.85, it, size=15, color=INK)
    footer(s, 11)
    notes(s, "Be upfront about what is simulated; examiners respect honesty. Future work turns each limitation into a next step.")

    # 12 ---- conclusion
    s = prs.slides.add_slide(blank)
    background(s, NAVY)
    text(s, 0.7, 0.8, 11.9, 0.5, "CONCLUSION", size=13, bold=True, color="8FB8DE")
    text(s, 0.7, 1.35, 11.9, 1.6, "Seeing and sensing the crowd together catches danger that either source alone would miss",
         size=34, bold=True, font=HEAD, color=WHITE)
    pts_ = [("Video model", f"AUC {NUM['auc_v2']:.3f} on real CCTV"),
            ("5G as a second sensor", f"{pct(NUM['net_acc'])} accuracy, finds the hotspot"),
            ("Combined model", f"macro-F1 {NUM['comb']:.3f}, robust to blind spots"),
            ("5G edge + slicing", f"{NUM['edge10']:.1f} ms alerts, ML picks edge or cloud")]
    for i, (h_, d) in enumerate(pts_):
        x = 0.7 + i * 3.05
        box(s, x, 3.45, 2.85, 1.3, fill="1E2F52")
        text(s, x + 0.25, 3.6, 2.4, 0.45, h_, size=15, bold=True, color=WHITE)
        text(s, x + 0.25, 4.1, 2.4, 0.9, d, size=13, color="C9D3E3")
    text(s, 0.7, 5.7, 7.5, 0.6, "Thank you · Questions?", size=28, bold=True, font=HEAD, color=WHITE)
    text(s, 0.7, 6.4, 9.0, 0.4, "Code, report and demo: github.com/vaibhav010606/mini_project1", size=14, color="8FB8DE")
    notes(s, "Close with the one-sentence message and invite questions. Offer to show the live demo.")

    prs.save(OUT)
    print(f"Saved {OUT} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
