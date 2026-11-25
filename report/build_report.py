"""Build report.pdf and slides.pdf from the saved results (HTML -> PDF with headless Edge/Chrome)."""
from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RESULTS  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = "../results/figures"


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else {}


V = load("video_metrics.json")
B = load("benchmark.json")
N = load("network_metrics.json")
FU = load("fusion_results.json")
JM = load("joint_metrics.json")
JN = {"video": "Video only", "net": "5G only", "fusion": "Combined (video + 5G)"}
JC = ["All test events", "Daytime, danger in view, phones normal", "Night (dark video)",
      "Danger outside camera view", "Few phones visible"]


def jr(model, cond, key):
    return JM.get("results", {}).get(JN[model], {}).get(cond, {}).get(key)
EC = load("edge_cloud_results.json")
OF = load("offload_results.json")
DET = load("detect/Test004_ae_alerts.json")


def f(x, nd=3, pct=False):
    if x is None:
        return "-"
    return f"{x * 100:.{max(0, nd - 2)}f}%" if pct else f"{x:.{nd}f}"


def auc(m):
    return V.get(m, {}).get("auc", {}).get("patch_max_norm_smooth")


VNAME = {"pred": "v2: future-frame prediction U-Net (final, cloud)", "big": "v1: autoencoder, big",
         "student": "v1: autoencoder, student (edge)"}
AUC_COLS = {"pred": ("global_psnr_raw", "patch_max", "patch_max_norm_smooth"),
            "big": ("global_mse", "patch_max", "patch_max_norm_smooth"),
            "student": ("global_mse", "patch_max", "patch_max_norm_smooth")}


def best_auc():
    return auc("pred") or auc("big")


def bm(m, k):
    return B.get("models", {}).get(m, {}).get(k)


def cam_delay(setup, n):
    r = EC.get("cameras", {})
    if not r:
        return None
    return r["results"][setup][r["n"].index(n)]["mean_ms"]


def slice_p95(name, load_):
    r = EC.get("slicing", {})
    if not r:
        return None
    return r["results"][name][r["bg_load"].index(load_)]["p95_ms"]


def ms(x):
    if x is None:
        return "-"
    return f"{x / 1000:.1f} s (overloaded)" if x >= 1000 else f"{x:.1f} ms"


CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1f2328; }
h1 { font-size: 22pt; margin: 0 0 6px; } h2 { font-size: 15pt; margin: 22px 0 8px; border-bottom: 2px solid #2f6feb; padding-bottom: 3px; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 14px 0 5px; page-break-after: avoid; }
table { border-collapse: collapse; margin: 8px 0 12px; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #d0d7de; padding: 4px 8px; text-align: left; font-size: 9.5pt; vertical-align: top; }
th { background: #f6f8fa; }
pre { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px; padding: 8px 10px; font-family: Consolas, monospace; font-size: 8.6pt; white-space: pre; page-break-inside: avoid; }
code { font-family: Consolas, monospace; background: #f6f8fa; padding: 0 3px; font-size: 9.5pt; }
figure { margin: 10px 0 14px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 330px; }
figcaption { font-size: 9pt; color: #57606a; margin-top: 3px; }
.title-page { height: 250mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; }
.sub { font-size: 13pt; color: #57606a; margin-bottom: 30px; }
.box { border: 1px solid #d0d7de; border-left: 4px solid #2f6feb; background: #f6f8fa; padding: 8px 12px; margin: 10px 0; page-break-inside: avoid; }
.warn { border-left-color: #d4a72c; }
.kpi { display: flex; gap: 10px; margin: 10px 0; } .kpi div { flex: 1; border: 1px solid #d0d7de; border-radius: 6px; padding: 8px; text-align: center; }
.kpi b { display: block; font-size: 16pt; color: #2f6feb; } .kpi span { font-size: 8.5pt; color: #57606a; }
.pb { page-break-before: always; }
"""

ARCH = """📷 Cameras                                         📱 Public's phones
   │ frames                                           │ (network data collected
   ▼                                                  ▼  by the gNB automatically)
════════════════ 5G RAN (gNB small cells) with NETWORK SLICING ════════════════
   🚨 URLLC slice: safety cameras + alerts   │   👥 eMBB / best effort: public traffic
══════════════════════════════════════════════════════════════════════════════
   │                                                  │
   ▼                                                  ▼
┌──────────────────────── ⚡ MEC EDGE SERVER (next to gNB) ────────────────────────┐
│  🧠 Model 1: video autoencoder         🧠 Model 2: network crowd-risk (XGBoost)  │
│     (lightweight student)                 + positioning-based density heatmap   │
│                  └──── 🧠 Combined model (trained on both) ────┘                │
│                                     ▼                                           │
│              🚨 Alert: "Stampede risk at zone X"  → security staff              │
│  🧠 ML offloading decision: process here, or send to cloud when edge is busy     │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │ 5G core → internet (20-100 ms RTT)
                                       ▼
                ☁️ CLOUD: big model on GPU, storage, retraining"""


def report_html():
    today = dt.date.today().strftime("%d %B %Y")
    nb, ns = bm("big", "params"), bm("student", "params")
    ratio = (nb / ns) if nb and ns else None
    xg, ls, ru = N.get("xgboost", {}), N.get("lstm", {}), N.get("rule", {})
    E1, E2, CL = "Edge - lightweight model", "Edge - big model (CPU)", "Cloud - accurate model (GPU)"
    SL, NS = "URLLC slice (priority)", "No slicing (shared)"
    fusion_rows = "".join(
        f"<tr><td>{k}</td><td>{f(v['auc'])}</td><td>{f(v['tpr_at_5pct_fpr'])}</td><td>{f(v['caught_when_camera_blind'])}</td></tr>"
        for k, v in FU.items() if k != "setup")
    off_rows = "".join(
        f"<tr><td>{k}</td><td>{v['mean_p95_ms']:.1f} ms</td><td>{f(v['miss_rate'], 3, True)}</td></tr>"
        for k, v in OF.items() if isinstance(v, dict))
    cam_rows = "".join(
        f"<tr><td>{n}</td><td>{ms(cam_delay(E1, n))}</td><td>{ms(cam_delay(E2, n))}</td><td>{ms(cam_delay(CL, n))}</td></tr>"
        for n in (1, 10, 20, 40, 60, 80)) if EC else ""

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Crowd Anomaly Detection over 5G - Report</title>
<style>{CSS}</style></head><body>

<div class="title-page">
<h1>Multimodal Crowd Anomaly and Stampede-Risk Detection using Video AI and 5G Network Data with Edge Computing</h1>
<div class="sub">Project report</div>
<table style="width:70%">
<tr><th>Submitted by</th><td>&nbsp;</td></tr>
<tr><th>Guide</th><td>&nbsp;</td></tr>
<tr><th>Department / Institute</th><td>&nbsp;</td></tr>
<tr><th>Date</th><td>{today}</td></tr>
</table>
</div>

<h2>Abstract</h2>
<p>Crowd disasters such as stampedes develop in seconds, so detection must be both accurate and fast. This project builds an
AI/ML system that detects crowd anomalies from two independent sources and runs at the 5G network edge. <b>Model 1</b> learns
from normal CCTV video only (UCSD Ped2). A first version, a convolutional autoencoder, reaches a frame-level AUC of {f(auc('big'))}; the final
version, a <b>future-frame prediction U-Net</b> that flags motion it cannot predict, raises this to <b>AUC {f(best_auc())}</b>.
A <b>{f(ratio, 0) if ratio else '-'}× smaller student model</b> trained by knowledge distillation reaches AUC {f(auc('student'))} and is designed for the edge server. <b>Model 2</b> detects dangerous crowding and stampedes from 5G
network observations (phones per cell, handovers, network-estimated density and speed) produced by a 5G crowd simulator; XGBoost reaches
<b>{f(xg.get('accuracy'), 3, True)} accuracy (macro-F1 {f(xg.get('macro_f1'))})</b>. To detect danger from <b>both sources at once</b>,
a joint synthetic dataset was built in which the <i>same</i> simulated crowd event is filmed by a CCTV camera and sensed by the 5G network.
A <b>combined model</b> (video CNN + 5G network branch, trained together) reaches macro-F1 <b>{f(jr('fusion', JC[0], 'macro_f1'))}</b> on unseen
events versus {f(jr('net', JC[0], 'macro_f1'))} (5G only) and {f(jr('video', JC[0], 'macro_f1'))} (video only), detects
{f(jr('fusion', JC[0], 'danger_detected'), 3, True)} of dangerous moments with {f(jr('fusion', JC[0], 'false_alarm'), 3, True)} false alarms, and stays
reliable when one source fails: danger outside the camera's view ({f(jr('fusion', JC[3], 'macro_f1'))} vs {f(jr('video', JC[3], 'macro_f1'))} for video only)
or few phones visible to the operator ({f(jr('fusion', JC[4], 'macro_f1'))} vs {f(jr('net', JC[4], 'macro_f1'))} for 5G only). A 5G MEC simulation shows that edge processing delivers alerts in about
{ms(cam_delay(E1, 10))} versus {ms(cam_delay(CL, 10))} from the cloud, that a <b>URLLC network slice</b> keeps safety traffic fast even when the
crowd's own phones load the uplink, and that an <b>ML offloading model</b> choosing edge or cloud per situation gets close to the best possible delay.</p>

<div class="kpi">
<div><b>{f(best_auc())}</b><span>Video AUC (UCSD Ped2)</span></div>
<div><b>{f(xg.get('macro_f1'))}</b><span>Network model macro-F1</span></div>
<div><b>{f(jr('fusion', JC[0], 'macro_f1'))}</b><span>Combined video + 5G model (macro-F1)</span></div>
<div><b>{ms(cam_delay(E1, 10))}</b><span>Edge alert delay (10 cams)</span></div>
</div>

<h2>1. Introduction</h2>
<p>Large gatherings (festivals, pilgrimages, stadiums, railway stations) carry a risk of crowd crushes and stampedes. CCTV operators
cannot watch every screen, and an alert that arrives seconds late is of little use. Two technologies help:</p>
<ul>
<li><b>AI/ML video analytics</b> can watch every camera continuously and flag unusual behaviour.</li>
<li><b>5G networks</b> connect cameras wirelessly, can sense the crowd itself (every phone connects to a cell), offer
<b>Multi-access Edge Computing (MEC)</b> servers right next to the base station, and support <b>network slicing</b> to give safety traffic a protected lane.</li>
</ul>

<h2>2. Problem statement and objectives</h2>
<div class="box"><b>Problem statement.</b> Edge-AI crowd anomaly detection: train a detection model on public crowd-video datasets, and use a
5G edge-computing (MEC) model to compare processing the video at the edge versus in the cloud over 5G.</div>
<p><b>Objectives</b></p>
<ol>
<li>Train a video anomaly detector on a public dataset and evaluate it with the standard frame-level AUC.</li>
<li>Build a lightweight version suitable for an edge server and measure its accuracy/speed trade-off.</li>
<li>Use 5G network data as a second, independent crowd sensor (density, mass movement, location of hotspots).</li>
<li>Detect crowd danger from video and 5G data <b>of the same event</b> with one combined model, robust when either source fails.</li>
<li>Compare edge vs cloud processing over 5G, including network slicing, and let an ML model decide where to process.</li>
</ol>

<h2>3. System architecture</h2>
<pre>{ARCH}</pre>
<p>AI/ML is used in five places: (1) video anomaly detection, (2) knowledge distillation for the edge model, (3) crowd-risk
classification from network data, (4) the combined video + 5G model, and (5) the edge/cloud offloading decision. 5G plays three roles: <b>sensor</b>
(network data), <b>transport</b> (sliced uplink/downlink) and <b>compute location</b> (MEC edge).</p>
<div class="box"><b>What is implemented end to end.</b> The detection path (one crowd event → CCTV video + 5G data → combined model → one
verdict every second) runs end to end on simulated events (Section 5.5) and is demonstrated live in the web app's <i>Combined monitor</i>.
The 5G radio network, the edge server and network slicing are simulated in Python; no physical 5G equipment is used.</div>

<h2 class="pb">4. Datasets</h2>
<table>
<tr><th>Dataset</th><th>Used for</th><th>Details</th></tr>
<tr><td>UCSD Ped2 (Mahadevan et al., 2010)</td><td>Model 1 training and testing</td><td>16 normal training videos (2,550 frames), 12 test videos (2,010 frames, 1,648 anomalous: bikes, skaters, carts). Official frame-level labels.</td></tr>
<tr><td>Simulated 5G crowd dataset (this project)</td><td>Model 2</td><td>450 simulated episodes × 180 s (normal, gathering, stampede), 22,950 windows of 30 s × 9 network features.</td></tr>
<tr><td><b>Joint video + 5G dataset</b> (this project)</td><td>Combined model</td><td>300 simulated events × 180 s. Each event gives a
96×96 overhead CCTV video (1 frame/s, camera covering 80 m × 80 m of the venue) <i>and</i> the 5G network features of the same people.
{JM.get('test_events', {}).get('total', '-')} test events. Hard cases: night video (20%), few phones visible (20%), hotspot outside the camera view (~25%).</td></tr>
<tr><td>Simulated 5G MEC conditions (this project)</td><td>Offloading model</td><td>500 random network/load conditions, each simulated for edge and cloud.</td></tr>
</table>
<div class="box warn">Real operator data (per-cell phone counts, positions) is private, so the network side is simulated with 3GPP-style models.
The Telecom Italia Milan dataset (Barlacchi et al., 2015) is a public source of real aggregated mobile activity that could validate this part in future work.</div>

<h2>5. Methodology</h2>
<h3>5.1 Model 1: video anomaly detection</h3>
<p><b>v1: convolutional autoencoder (reconstruction).</b></p>
<ul>
<li>Input: 5 consecutive grayscale frames, 128×128, stacked as channels, so the model learns both appearance and motion.</li>
<li>Encoder: 4 strided convolutions (64-128-256-256 channels) to an 8×8 bottleneck; mirrored transposed-convolution decoder.</li>
<li>Training: only <b>normal</b> videos, mean-squared reconstruction loss, Adam, cosine learning-rate schedule, 60 epochs, horizontal flips.</li>
<li>Scoring: squared error map → 16×16 patches → score = worst patch (anomalies are small objects) → per-video normalisation and temporal smoothing.</li>
</ul>
<p><b>v2 (final): future-frame prediction U-Net</b>, after Liu et al. (2018). An autoencoder can learn to reconstruct even unusual
objects well, so v2 instead <i>predicts</i> the next frame from the 4 previous ones. Normal walking is predictable; a bike, skater or cart is not.</p>
<ul>
<li>Input: 4 consecutive grayscale frames at <b>256×256</b> (twice v1's resolution, so small objects keep their detail); output: the 5th frame.</li>
<li>U-Net: 4 encoder levels (32-64-128-256 channels), 512-channel bottleneck, decoder with skip connections ({f(bm('pred', 'params') / 1e6, 1) if bm('pred', 'params') else '-'} M parameters).</li>
<li>Loss: mean-squared intensity error + image-gradient loss (keeps predicted edges sharp); 30 epochs, mixed precision on the GPU.</li>
<li>Scoring: the same rule as v1 (worst 32×32 patch of the prediction error, per-video normalisation, temporal smoothing), fixed before testing.</li>
</ul>
<h3>5.2 Lightweight edge model (knowledge distillation)</h3>
<p>The student has the same structure with 16-32-64-64 channels ({ns:,} parameters vs {nb:,}). It is trained to reconstruct the input
<i>and</i> to imitate the big model's output (loss = ½·MSE(input) + ½·MSE(teacher)).</p>
<h3>5.3 5G crowd simulator</h3>
<ul>
<li>200 m × 200 m venue, 3×3 grid of 5G small cells at 3.5 GHz, 3GPP UMi path loss, 4 dB shadowing, 2 dB fast fading, 3 dB handover hysteresis.</li>
<li>800-2,600 people; 5% carry no phone, 10% carry two. Movement slows down in dense zones.</li>
<li>Scenarios: <b>normal</b> (wandering, safe attractions, harmless rushes such as people hurrying to a train), <b>gathering</b> (crowd packs into a hotspot),
<b>stampede</b> (gathering, then panic: people flee at 2.2-5 m/s, spreading outward).</li>
<li>Network-side features per second: phones in venue, phones on busiest cell, densest-zone density from network positioning (5 m error),
its 10 s growth, mean phone speed, share of fast phones, handover rate, busiest-cell load, mean SINR. Only zone counts are used: no individual tracking.</li>
<li>Ground truth: <i>dense</i> = any 10 m zone above 2 people/m²; <i>stampede</i> = panic in progress.</li>
</ul>
<h3>5.4 Model 2: crowd-risk detection from network data</h3>
<p>Input is the last 30 s of features. Three detectors: a threshold rule (tuned on training data), <b>XGBoost</b> on 63 window statistics
(last, mean, max, min, std, slope), and a 2-layer <b>LSTM</b> on the raw sequence. Train/test split is by episode.</p>
<h3>5.5 Combined video + 5G model</h3>
<p><b>Joint data.</b> No public dataset contains CCTV video and 5G network data of the same crowd, so both are generated from <i>one</i>
simulation. The 5G crowd simulator (5.3) moves the people; their phones produce the network features, and a fixed overhead CCTV camera
covering an 80 m × 80 m part of the venue is rendered from the same positions (96×96 grayscale, one frame per second, paving texture,
camera noise). Three hard cases make sure neither source is enough on its own:</p>
<ul>
<li><b>Night:</b> the video is dark (30% contrast) with strong noise, so people are barely visible.</li>
<li><b>Danger outside the camera view:</b> the hotspot is placed anywhere in the venue, so the camera sometimes cannot see it.</li>
<li><b>Few phones visible:</b> 60% of people are invisible to the operator (no phone, phone off, other network).</li>
</ul>
<p><b>Model.</b> For every second t the input is the last 4 CCTV frames and the last 30 s of 5G features.</p>
<ul>
<li><b>Video branch:</b> CNN (4 conv blocks, 32-64-128-128 channels) on the frames plus their frame differences (motion), with
per-clip contrast normalisation.</li>
<li><b>5G branch:</b> MLP on the 63 window statistics used by Model 2.</li>
<li><b>Fusion:</b> the two embeddings are concatenated and a classification head outputs normal / dense / stampede. The whole network
is trained end to end with class weights and <i>modality dropout</i> (15% of the time one branch is hidden), so it learns to rely on the
other source when one fails.</li>
</ul>
<p>For a fair comparison, a video-only and a 5G-only model with the same branches are trained on exactly the same samples.
Events are split 70/10/20 into train/validation/test, so no event appears in both training and testing.</p>
<p><b>Earlier baseline (score-level fusion).</b> Before the joint dataset existed, the outputs of Model 1 (on UCSD video) and Model 2
(on simulated 5G data) were paired artificially, with an incident visible to the camera with probability 0.75 and to the network with
probability 0.80, and combined with max, average or logistic regression. It is kept for reference in 6.4 but is superseded by the combined
model, which uses video and 5G data of the same events.</p>
<h3>5.6 Edge vs cloud over 5G, with network slicing</h3>
<p>A discrete-event simulation sends camera frames ({f(B.get('frame_kb', {}).get('ped2_native_360x240'), 1)} KB JPEG, 10 fps) over a 250 Mbps cell uplink shared with the
public's phone traffic. <b>With slicing</b>, safety cameras use a URLLC slice with strict scheduling priority; <b>without slicing</b> all traffic shares one queue.
Edge = MEC host 1 ms from the gNB, 2 workers running the student model at the <b>measured</b> CPU speed. Cloud = 20-100 ms RTT, GPU running the most accurate model (v2 U-Net) at the measured GPU speed.</p>
<h3>5.7 ML offloading decision</h3>
<p>500 random conditions (1-80 cameras, 5-15 fps, 0-95% public load, slicing on/off, 0-80% edge busy with other apps, 20-100 ms cloud RTT) are
simulated for both paths. XGBoost learns which path gives the lower 95th-percentile delay.</p>

<h2 class="pb">6. Results</h2>
<h3>6.1 Video anomaly detection (Model 1)</h3>
<table><tr><th>Model</th><th>AUC (whole frame)</th><th>AUC (worst patch)</th><th>AUC (final score)</th></tr>
{''.join(f"<tr><td>{VNAME.get(m, m)}</td><td>{f(V[m]['auc'][AUC_COLS[m][0]])}</td><td>{f(V[m]['auc'][AUC_COLS[m][1]])}</td><td><b>{f(V[m]['auc'][AUC_COLS[m][2]])}</b></td></tr>" for m in ('pred', 'big', 'student') if m in V)}
</table>
<p>Predicting the next frame instead of reconstructing the current one raises the AUC from {f(auc('big'))} to <b>{f(best_auc())}</b>.
Scoring by the worst patch instead of the whole frame also helps a lot, because anomalies (a bike, a cart) occupy a small part of the frame.
For reference, the classic convolutional autoencoder of Hasan et al. (2016) reports about 0.85-0.90 on Ped2, Liu et al. (2018) report 0.95 with
future-frame prediction plus optical flow and adversarial training, and the newest methods exceed 0.97.</p>
<figure><img src="{FIG}/video_roc.png"><figcaption>ROC curves on UCSD Ped2 test set.</figcaption></figure>
<figure><img src="{FIG}/video_score_curves.png"><figcaption>Anomaly score over time; red = ground-truth anomalous frames.</figcaption></figure>
<figure><img src="{FIG}/video_examples.png"><figcaption>Most anomalous frames and where the prediction error is concentrated (v2 model).</figcaption></figure>

<h3>6.2 Lightweight edge model and speed</h3>
<table><tr><th>Model</th><th>Parameters</th><th>Size</th><th>AUC</th><th>Edge (CPU, {B.get('edge_threads', 4)} threads)</th><th>Cloud (GPU)</th></tr>
{''.join(f"<tr><td>{VNAME.get(m, m)}</td><td>{r['params']:,}</td><td>{r['size_mb']:.1f} MB</td><td>{f(r['auc'])}</td><td>{r['cpu_ms']:.2f} ms</td><td>{f(r['gpu_ms'], 2)} ms</td></tr>" for m, r in B.get('models', {}).items())}
</table>
<figure><img src="{FIG}/video_benchmark.png"><figcaption>Accuracy, size and speed of the big (cloud) and student (edge) models. Measured on {B.get('gpu') or 'GPU'} and the laptop CPU.</figcaption></figure>

<h3>6.3 Crowd-risk detection from 5G network data (Model 2)</h3>
<table><tr><th>Detector</th><th>Accuracy</th><th>Macro-F1</th><th>Stampede detection delay (median)</th><th>Stampedes missed</th></tr>
{''.join(f"<tr><td>{k}</td><td>{f(N[k]['accuracy'], 3, True)}</td><td>{f(N[k]['macro_f1'])}</td><td>{N[k]['stampede_delay_median_s']} s</td><td>{N[k]['stampede_episodes_missed']}/{N[k]['stampede_episodes_total']}</td></tr>" for k in ('rule', 'xgboost', 'lstm') if k in N)}
</table>
<figure><img src="{FIG}/net_model_comparison.png"><figcaption>Rule baseline vs XGBoost vs LSTM.</figcaption></figure>
<figure><img src="{FIG}/net_confusion_xgb.png" style="max-height:260px"> <img src="{FIG}/net_confusion_lstm.png" style="max-height:260px"><figcaption>Confusion matrices (test episodes).</figcaption></figure>
<figure><img src="{FIG}/net_feature_importance.png"><figcaption>Which network signals matter most.</figcaption></figure>
<figure><img src="{FIG}/net_heatmaps.png" style="max-height:420px"><figcaption>Crowd density heatmaps: ground truth (top) and estimated from 5G positioning (bottom). Triangles = 5G cells.</figcaption></figure>
<figure><img src="{FIG}/net_timeline.png" style="max-height:420px"><figcaption>One simulated stampede seen through network data, with the model's risk output.</figcaption></figure>

<h3>6.4 Combined video + 5G model</h3>
<p>Macro-F1 (normal / dense / stampede) on {JM.get('test_events', {}).get('total', '-')} unseen test events, split by condition:</p>
<table><tr><th>Test condition</th><th>Video only</th><th>5G only</th><th>Combined</th></tr>
{''.join(f"<tr><td>{c}</td><td>{f(jr('video', c, 'macro_f1'))}</td><td>{f(jr('net', c, 'macro_f1'))}</td><td><b>{f(jr('fusion', c, 'macro_f1'))}</b></td></tr>" for c in JC)}
</table>
<table><tr><th>All test events</th><th>Video only</th><th>5G only</th><th>Combined</th></tr>
<tr><td>Dangerous moments detected</td>{''.join(f"<td>{f(jr(m, JC[0], 'danger_detected'), 3, True)}</td>" for m in ('video', 'net', 'fusion'))}</tr>
<tr><td>False alarms</td>{''.join(f"<td>{f(jr(m, JC[0], 'false_alarm'), 3, True)}</td>" for m in ('video', 'net', 'fusion'))}</tr>
<tr><td>Stampede detection delay (median)</td>{''.join(f"<td>{JM.get('stampede_delay', {}).get(JN[m], {}).get('median_s', '-')} s</td>" for m in ('video', 'net', 'fusion'))}</tr>
<tr><td>Stampedes missed</td>{''.join(f"<td>{JM.get('stampede_delay', {}).get(JN[m], {}).get('missed', '-')} / {JM.get('stampede_delay', {}).get(JN[m], {}).get('total', '-')}</td>" for m in ('video', 'net', 'fusion'))}</tr>
</table>
<figure><img src="{FIG}/joint_by_condition.png"><figcaption>Each source has a blind spot; the combined model covers it.</figcaption></figure>
<p>The combined model is the best in every condition. When the danger is outside the camera view, video alone drops to
{f(jr('video', JC[3], 'macro_f1'))} and the 5G branch keeps the combined model at {f(jr('fusion', JC[3], 'macro_f1'))}; when few phones are visible,
5G alone drops to {f(jr('net', JC[4], 'macro_f1'))} and the video branch keeps the combined model at {f(jr('fusion', JC[4], 'macro_f1'))}.
<b>Night</b> is the weakest condition ({f(jr('fusion', JC[2], 'macro_f1'))}), but this group has only {JM.get('test_events', {}).get('night', '-')} test events,
so its macro-F1 is unreliable. On those events the combined model still detects {f(jr('fusion', JC[2], 'danger_detected'), 3, True)} of dangerous moments
with {f(jr('fusion', JC[2], 'false_alarm'), 3, True)} false alarms (video only: {f(jr('video', JC[2], 'danger_detected'), 3, True)} and
{f(jr('video', JC[2], 'false_alarm'), 3, True)}). What it finds hard at night is telling <i>dense</i> from <i>stampede</i>.</p>
<figure><img src="{FIG}/joint_video_preview.png" style="max-height:230px"><figcaption>Synthetic CCTV frames rendered from the same crowd as the 5G data.</figcaption></figure>
<figure><img src="{FIG}/joint_confusion.png" style="max-height:250px"><figcaption>Combined model on test events.</figcaption></figure>
<p><b>Earlier baseline: score-level fusion of separately trained models</b> (artificial pairing, see 5.5):</p>
<table><tr><th>Method</th><th>AUC</th><th>Detection at 5% false alarms</th><th>Caught when camera blind</th></tr>{fusion_rows}</table>

<h3>6.5 Edge vs cloud over 5G</h3>
<table><tr><th>Cameras</th><th>Edge, lightweight model</th><th>Edge, big model on CPU</th><th>Cloud, big model on GPU</th></tr>{cam_rows}</table>
<figure><img src="{FIG}/edge_vs_cloud_cameras.png"><figcaption>Mean alert delay and deadline misses (100 ms) as cameras are added (50% public load, slicing on).</figcaption></figure>
<p>The edge is several times faster while it has spare capacity; the big model on the edge CPU saturates after a few cameras, which is why the
lightweight model is needed. Beyond the edge server's capacity, the cloud becomes the better choice.</p>
<h3>6.6 Network slicing</h3>
<p>With 20 cameras: 95th-percentile delay at 50% public load is {ms(slice_p95(SL, 0.5))} with a URLLC slice vs {ms(slice_p95(NS, 0.5))} without;
at 90% load {ms(slice_p95(SL, 0.9))} vs {ms(slice_p95(NS, 0.9))}.</p>
<figure><img src="{FIG}/slicing_effect.png"><figcaption>A URLLC slice protects safety traffic when the crowd's phones load the uplink.</figcaption></figure>
<h3>6.7 ML offloading decision</h3>
<p>Decision accuracy on unseen conditions: <b>{f(OF.get('ml_accuracy'), 3, True)}</b>.</p>
<table><tr><th>Policy</th><th>Average 95th-pct delay</th><th>Conditions with p95 &gt; 100 ms</th></tr>{off_rows}</table>
<figure><img src="{FIG}/offload_policies.png"><figcaption>Letting ML choose edge or cloud comes close to the best possible (oracle) choice.</figcaption></figure>
<figure><img src="{FIG}/offload_importance.png" style="max-height:230px"><figcaption>Inputs that drive the decision.</figcaption></figure>

<h2 class="pb">7. Web app and demo tool</h2>
<p>A React + FastAPI web app runs everything live: <b>Combined monitor</b> (one event, CCTV video and 5G density map side by side, verdicts
of the video-only, 5G-only and combined models every second, with night and few-phones switches), <b>Video detection</b> (upload a video),
<b>5G crowd monitor</b>, <b>Edge vs cloud</b> (sliders for the 5G simulation and ML decision) and <b>Results</b>.</p>
<h3>Command-line tool</h3>
<p><code>python detect.py --video &lt;file or folder&gt;</code> produces an annotated video (red border + "ANOMALY"), a score graph and a JSON list of alert times.</p>
<ul>
<li><b>ae mode</b>: the trained autoencoder; accurate on UCSD-style scenes. A threshold is calibrated on normal training clips.</li>
<li><b>motion mode</b>: optical-flow crowd-motion detector for any fixed camera; learns normal motion from the first seconds and alerts on sudden strong motion (running, scattering).</li>
</ul>
<p>Example on UCSD Ped2 test video 004 (a cart enters at frame 31 = 3.0 s and stays until the end): the tool raises an alert for
{DET.get('alert_intervals_s', '-')} seconds.</p>
<figure><img src="../results/detect/Test004_ae_scores.png"><figcaption>detect.py output on Test004 (autoencoder mode).</figcaption></figure>

<h2>8. Discussion and limitations</h2>
<ul>
<li><b>Scene-specific "normal".</b> The autoencoder learns normality for one camera view; each new camera needs its own normal footage (minutes to hours).</li>
<li><b>Simulated network data.</b> Model 2 is trained and tested on simulated 5G data. Results show the idea works under realistic noise but must be validated on real operator data.</li>
<li><b>Synthetic joint data.</b> The combined model is trained and tested on simulated events: the CCTV video is rendered from simulated
people (overhead dots on paving, not real footage), so its video branch is separate from the UCSD-trained Model 1. Results show that
combining the two sources helps, but real paired CCTV + operator data would be needed to confirm the numbers.</li>
<li><b>Night results</b> rest on only {JM.get('test_events', {}).get('night', '-')} test events; more events are needed for a reliable estimate.</li>
<li><b>Python 5G simulation instead of Simu5G.</b> The edge/cloud study uses a discrete-event simulation built on the same concepts as Simu5G
(gNB, UEs, MEC host, slicing). Porting the scenario to OMNeT++/Simu5G would add a full protocol stack (HARQ, scheduling details, core network).</li>
<li><b>Edge hardware.</b> The edge server is represented by a laptop CPU limited to 4 threads; a real MEC host may have a small GPU.</li>
<li><b>Privacy.</b> Only aggregate zone counts are used; individuals are never tracked. Real deployment must follow telecom data-protection rules.</li>
</ul>

<h2>9. Future work</h2>
<ul>
<li>Port the 5G scenario to <b>Simu5G</b> (OMNeT++), using the same parameters and measured processing times.</li>
<li>Validate the network model on real aggregated data such as the Telecom Italia Milan dataset.</li>
<li>Stronger video models (future-frame prediction, memory-augmented autoencoders) and more datasets (CUHK Avenue, ShanghaiTech, UMN).</li>
<li>Reinforcement learning for offloading and dynamic slice resource allocation.</li>
<li>Split computing: run the first layers on the camera and send compact features over 5G.</li>
</ul>

<h2>10. Conclusion</h2>
<p>The project shows a complete AI/ML pipeline for crowd safety over 5G. A video autoencoder detects anomalies on a public benchmark,
a distilled model makes edge deployment practical, 5G network data acts as a second crowd sensor that also locates dangerous zones, a combined
model that sees the same event through the camera and the 5G network stays reliable when either source fails, and edge processing with a URLLC slice and ML-based offloading delivers alerts within real-time limits.</p>

<h2>References</h2>
<ol style="font-size:9.5pt">
<li>V. Mahadevan, W. Li, V. Bhalodia, N. Vasconcelos. Anomaly detection in crowded scenes. CVPR 2010. (UCSD dataset)</li>
<li>M. Hasan, J. Choi, J. Neumann, A. Roy-Chowdhury, L. Davis. Learning temporal regularity in video sequences. CVPR 2016.</li>
<li>W. Liu, W. Luo, D. Lian, S. Gao. Future frame prediction for anomaly detection: a new baseline. CVPR 2018.</li>
<li>O. Ronneberger, P. Fischer, T. Brox. U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.</li>
<li>G. Hinton, O. Vinyals, J. Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.</li>
<li>T. Chen, C. Guestrin. XGBoost: a scalable tree boosting system. KDD 2016.</li>
<li>S. Hochreiter, J. Schmidhuber. Long short-term memory. Neural Computation, 1997.</li>
<li>G. Farnebäck. Two-frame motion estimation based on polynomial expansion. SCIA 2003.</li>
<li>G. Nardini, D. Sabella, G. Stea, P. Thakkar, A. Virdis. Simu5G: an OMNeT++ library for end-to-end performance evaluation of 5G networks. IEEE Access, 2020.</li>
<li>ETSI GS MEC 003: Multi-access Edge Computing; Framework and reference architecture.</li>
<li>3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz.</li>
<li>3GPP TS 38.305: NG-RAN; Stage 2 functional specification of UE positioning.</li>
<li>G. Barlacchi et al. A multi-source dataset of urban life in the city of Milan and the Province of Trentino. Scientific Data, 2015.</li>
</ol>

<h2>Appendix: how to run</h2>
<pre>python run_all.py                 # full pipeline (about 45-50 min with GPU)
python run_all.py --skip-train    # reuse trained models
python detect.py --video my_crowd_video.mp4</pre>
</body></html>"""


SLIDE_CSS = """
@page { size: 297mm 167mm; margin: 0; }
body { margin: 0; font-family: 'Segoe UI', Arial, sans-serif; color: #1f2328; }
.s { width: 297mm; height: 167mm; box-sizing: border-box; padding: 12mm 16mm; page-break-after: always; position: relative; overflow: hidden; }
.s h1 { font-size: 26pt; margin: 0 0 6mm; color: #0b3d91; } .s h2 { font-size: 20pt; margin: 0 0 5mm; color: #0b3d91; border-bottom: 2px solid #2f6feb; padding-bottom: 2mm; }
.s li { font-size: 13pt; margin: 2mm 0; } .s p { font-size: 13pt; }
.cols { display: flex; gap: 8mm; align-items: flex-start; } .cols > div { flex: 1; }
img { max-width: 100%; max-height: 105mm; } pre { font-size: 7.6pt; background: #f6f8fa; padding: 3mm; border-radius: 3mm; }
.big { font-size: 30pt; font-weight: 700; color: #2f6feb; } .lbl { font-size: 10pt; color: #57606a; }
.kpis { display: flex; gap: 6mm; margin-top: 8mm; } .kpis > div { flex: 1; border: 1px solid #d0d7de; border-radius: 3mm; padding: 4mm; text-align: center; }
.title { background: linear-gradient(135deg, #0b3d91, #2f6feb); color: white; display: flex; flex-direction: column; justify-content: center; }
.title h1 { color: white; font-size: 28pt; } .title p { font-size: 14pt; opacity: 0.9; }
.foot { position: absolute; bottom: 5mm; right: 10mm; font-size: 9pt; color: #8c959f; }
table { border-collapse: collapse; } td, th { border: 1px solid #d0d7de; padding: 1.5mm 3mm; font-size: 11pt; } th { background: #f6f8fa; }
.center { text-align: center; } .take { font-size: 14pt; text-align: center; margin-top: 3mm; }
.arch { display: flex; gap: 6mm; align-items: stretch; } .flow { flex: 3; display: flex; flex-direction: column; gap: 1.5mm; }
.row { display: flex; gap: 4mm; } .row > div { flex: 1; }
.node { border: 2px solid #2f6feb; border-radius: 3mm; padding: 2.5mm 4mm; font-size: 13pt; background: #eef4ff; text-align: center; }
.arrow { text-align: center; font-size: 15pt; color: #57606a; line-height: 1.1; }
.band { border: 2px dashed #8250df; background: #f6f0ff; border-radius: 3mm; padding: 2mm 3mm; }
.band .hd, .edge .hd { font-size: 12pt; font-weight: 700; margin-bottom: 2mm; text-align: center; }
.urllc { border-color: #cf222e; background: #fff0f0; } .embb { border-color: #8c959f; background: #f6f8fa; }
.edge { border: 2.5px solid #1a7f37; background: #effaf1; border-radius: 3mm; padding: 2.5mm 3mm; }
.edge .node { background: white; border-color: #1a7f37; font-size: 12pt; }
.alert { border-color: #cf222e; background: #fff0f0; font-weight: 700; }
.cloudcol { flex: 1.1; display: flex; flex-direction: column; justify-content: flex-end; }
.cloud { border: 2.5px solid #8250df; background: #f6f0ff; border-radius: 3mm; padding: 4mm; font-size: 12.5pt; text-align: center; }
.kpis.top { margin: 0 0 3mm; } .kpis.top > div { padding: 2mm; } .kpis.top .big { font-size: 24pt; }
"""

ARCH_SLIDE = """<div class="arch"><div class="flow">
<div class="row"><div class="node">📷 CCTV cameras<br><small>video</small></div><div class="node">📱 People's phones<br><small>5G network data</small></div></div>
<div class="arrow">▼</div>
<div class="band"><div class="hd">📡 5G network with slicing</div>
<div class="row"><div class="node urllc">🚨 URLLC slice: safety cameras + alerts</div><div class="node embb">👥 eMBB: public phone traffic</div></div></div>
<div class="arrow">▼</div>
<div class="edge"><div class="hd">⚡ MEC edge server (next to the 5G tower)</div>
<div class="row"><div class="node">🧠 Model 1<br>video anomaly</div><div class="node">🧠 Model 2<br>network crowd risk</div>
<div class="node">🧠 Combined model<br>(video + 5G)</div><div class="node alert">🚨 Alert to<br>security staff</div></div>
<div class="arrow" style="font-size:11.5pt;margin-top:2mm">🧠 ML decides: process here at the edge, or send to the cloud ▶</div></div>
</div>
<div class="cloudcol"><div class="cloud">☁️ <b>Cloud</b><br><br>v2 most accurate model (GPU)<br><br>storage, retraining<br><br><small>20-100 ms away</small></div></div>
</div>"""


def slides_html():
    xg = N.get("xgboost", {})
    E1, CL = "Edge - lightweight model", "Cloud - accurate model (GPU)"
    slides = [
        f"""<div class="s title"><h1>Multimodal Crowd Anomaly &amp; Stampede-Risk Detection<br>using Video AI and 5G Network Data with Edge Computing</h1>
<p>AI/ML project · {dt.date.today().strftime('%B %Y')}</p></div>""",
        """<div class="s"><h2>Problem</h2><ul>
<li>Crowd crushes and stampedes develop in <b>seconds</b></li>
<li>Operators cannot watch every CCTV screen; late alerts are useless</li>
<li>Goal: detect anomalies <b>accurately</b> and deliver alerts <b>fast</b></li>
<li>Problem statement: train an anomaly detector on public crowd video and compare <b>edge vs cloud</b> processing over <b>5G (MEC)</b></li>
</ul></div>""",
        f"""<div class="s"><h2>Our approach: two detectors + 5G edge</h2>{ARCH_SLIDE}</div>""",
        """<div class="s"><h2>Where AI/ML is used</h2><table>
<tr><th>#</th><th>AI/ML part</th><th>Technique</th></tr>
<tr><td>1</td><td>Video anomaly detection</td><td>Future-frame prediction U-Net (v1: autoencoder)</td></tr>
<tr><td>2</td><td>Edge-ready model</td><td>Knowledge distillation (16× fewer parameters)</td></tr>
<tr><td>3</td><td>Crowd risk from 5G data</td><td>XGBoost / LSTM on network time series</td></tr>
<tr><td>4</td><td>Detect from both at once</td><td>Combined model: video CNN + 5G branch, trained together on the same events</td></tr>
<tr><td>5</td><td>Edge or cloud?</td><td>XGBoost offloading decision</td></tr></table>
<p>5G roles: <b>sensor</b> (network data) · <b>transport</b> (URLLC slice) · <b>compute</b> (MEC edge)</p></div>""",
        f"""<div class="s"><h2>Model 1: video anomaly detection (UCSD Ped2)</h2><div class="cols">
<div><ul><li>Trained only on <b>normal</b> video</li><li>v2 predicts the next frame: what it cannot predict = anomaly</li><li>Worst-patch scoring for small objects</li></ul>
<div class="kpis"><div><div class="big">{f(auc('big'))}</div><div class="lbl">v1 autoencoder</div></div><div><div class="big">{f(best_auc())}</div><div class="lbl">v2 prediction (final)</div></div><div><div class="big">{f(auc('student'))}</div><div class="lbl">edge model</div></div></div></div>
<div><img src="{FIG}/video_examples.png"></div></div></div>""",
        f"""<div class="s"><h2>Accuracy vs speed: cloud model (v2) vs edge model</h2><div class="center"><img src="{FIG}/video_benchmark.png" style="max-height:105mm"></div>
<p class="take">v2 is the most accurate but takes {ms(bm('pred', 'cpu_ms'))} per frame on a CPU, so it runs in the <b>cloud</b> (GPU: {ms(bm('pred', 'gpu_ms'))}); the small student model runs at the <b>edge</b>.</p></div>""",
        f"""<div class="s"><h2>Model 2: crowd risk from 5G network data</h2><div class="cols">
<div><ul><li>Simulated venue with 9 small cells, 800-2,600 people with phones</li><li>Phones per cell, handovers, density and speed from 5G positioning</li><li>Normal / dense / stampede</li></ul>
<div class="kpis"><div><div class="big">{f(xg.get('accuracy'), 3, True)}</div><div class="lbl">XGBoost accuracy</div></div><div><div class="big">{f(xg.get('macro_f1'))}</div><div class="lbl">macro-F1</div></div></div></div>
<div><img src="{FIG}/net_timeline.png"></div></div></div>""",
        f"""<div class="s"><h2>Where is the danger? Density heatmap from 5G</h2><img src="{FIG}/net_heatmaps.png" style="max-height:125mm"></div>""",
        f"""<div class="s"><h2>Combined model: same event, video + 5G</h2>
<div class="kpis top"><div><div class="big">{f(jr('video', JC[0], 'macro_f1'))}</div><div class="lbl">video only (macro-F1)</div></div><div><div class="big">{f(jr('net', JC[0], 'macro_f1'))}</div><div class="lbl">5G only</div></div><div><div class="big">{f(jr('fusion', JC[0], 'macro_f1'))}</div><div class="lbl">combined</div></div></div>
<div class="center"><img src="{FIG}/joint_by_condition.png" style="max-height:78mm"></div>
<p class="take">Camera can't see the danger → 5G covers it · few phones visible → video covers it · combined catches {f(jr('fusion', JC[0], 'danger_detected'), 3, True)} of danger with {f(jr('fusion', JC[0], 'false_alarm'), 3, True)} false alarms.</p></div>""",
        f"""<div class="s"><h2>How the joint data is made</h2><div class="cols">
<div><ul><li>One simulated crowd event drives <b>both</b> sensors</li><li>📷 CCTV camera films an 80 m × 80 m part of the venue</li>
<li>📡 5G network senses the same people's phones everywhere</li><li>Hard cases: <b>night</b>, danger <b>outside the camera view</b>, <b>few phones</b> visible</li>
<li>300 events · video CNN + 5G branch trained together</li></ul></div>
<div><img src="{FIG}/joint_video_preview.png"></div></div></div>""",
        f"""<div class="s"><h2>Edge vs cloud over 5G</h2><img src="{FIG}/edge_vs_cloud_cameras.png" style="max-height:105mm">
<p>10 cameras: edge <b>{ms(cam_delay(E1, 10))}</b> vs cloud <b>{ms(cam_delay(CL, 10))}</b></p></div>""",
        f"""<div class="s"><h2>5G network slicing protects safety traffic</h2><div class="center"><img src="{FIG}/slicing_effect.png" style="max-height:100mm"></div>
<p class="take">20 cameras, 90% of the uplink used by the public: 95th-percentile delay <b>{ms(slice_p95('URLLC slice (priority)', 0.9))}</b> with a URLLC slice vs <b>{ms(slice_p95('No slicing (shared)', 0.9))}</b> without.</p></div>""",
        f"""<div class="s"><h2>ML decides: edge or cloud?</h2><div class="center"><img src="{FIG}/offload_policies.png" style="max-height:100mm"></div>
<p class="take">ML picks the right place in <b>{f(OF.get('ml_accuracy'), 3, True)}</b> of unseen situations, close to the best possible delay.</p></div>""",
        """<div class="s"><h2>Limitations and future work</h2><ul>
<li>5G network data is simulated; validate on real aggregated operator data (e.g. Telecom Italia Milan)</li>
<li>Edge/cloud study in Python; port the scenario to <b>Simu5G</b> (OMNeT++)</li>
<li>Autoencoder learns "normal" per camera view</li>
<li>Next: stronger video models, more datasets, RL-based offloading and slice control</li>
</ul></div>""",
        """<div class="s title"><h1>Conclusion</h1><p>Two independent AI detectors (video + 5G network), fused for reliability,<br>
running at the 5G edge with a protected URLLC slice and ML-based offloading,<br>deliver accurate crowd-safety alerts in real time.</p><p>Thank you</p></div>""",
    ]
    body = "".join(slides)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Slides</title><style>{SLIDE_CSS}</style></head><body>{body}</body></html>"


def find_browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"]:
        if Path(p).exists():
            return p
    return shutil.which("msedge") or shutil.which("chrome") or shutil.which("chromium")


def to_pdf(html_path: Path, pdf_path: Path):
    b = find_browser()
    if not b:
        print(f"No Edge/Chrome found; open {html_path} and print to PDF manually.")
        return
    subprocess.run([b, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print(f"Saved {pdf_path} ({pdf_path.stat().st_size / 1e6:.1f} MB)" if pdf_path.exists() else f"PDF failed: {pdf_path}")


def main():
    r = HERE / "report.html"
    s = HERE / "slides.html"
    from report.full_report import full_report_html  # full explanatory report (with website screenshots)

    r.write_text(full_report_html(), encoding="utf-8")
    s.write_text(slides_html(), encoding="utf-8")
    to_pdf(r, HERE / "report.pdf")
    to_pdf(s, HERE / "slides.pdf")


if __name__ == "__main__":
    main()
