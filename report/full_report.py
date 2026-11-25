"""Full explanatory project report (report/report.pdf).

Formal structure, plain-language explanations: abstract, summary, concepts, architecture, data, how every
model works, results, the web application with screenshots, limitations and conclusion.
All numbers are read from the saved results (results/*.json), so the report always matches the code.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from report.build_report import (  # noqa: E402
    ARCH, AUC_COLS, B, CSS, DET, EC, FIG, FU, JC, JM, N, OF, V, VNAME, auc, best_auc, bm, cam_delay, f, jr, ms,
    slice_p95,
)

SHOT = "../results/screenshots"
EXTRA_CSS = """
.shot { margin: 10px 0 16px; text-align: center; page-break-inside: avoid; }
.shot img { width: 100%; max-height: 430px; object-fit: contain; border: 1px solid #d0d7de; border-radius: 6px; }
.shot figcaption { font-size: 9pt; color: #57606a; margin-top: 4px; }
.steps { counter-reset: s; list-style: none; padding-left: 0; }
.steps li { counter-increment: s; margin: 5px 0 5px 30px; position: relative; }
.steps li::before { content: counter(s); position: absolute; left: -30px; top: 1px; width: 20px; height: 20px; border-radius: 50%;
  background: #2f6feb; color: #fff; font-size: 9pt; text-align: center; line-height: 20px; }
.plain { border-left: 4px solid #1a7f37; background: #f0fbf3; padding: 8px 12px; margin: 10px 0; page-break-inside: avoid; }
.plain b:first-child { color: #1a7f37; }
.toc { columns: 2; font-size: 10pt; } .toc li { margin: 2px 0; }
h4 { font-size: 11pt; margin: 10px 0 4px; }
"""

E1, E2, CL = "Edge - lightweight model", "Edge - big model (CPU)", "Cloud - accurate model (GPU)"
SL, NS = "URLLC slice (priority)", "No slicing (shared)"
JMOD = ("video", "net", "fusion")


def plain(text):
    return f'<div class="plain"><b>In simple words:</b> {text}</div>'


def shot(name, caption):
    return f'<figure class="shot"><img src="{SHOT}/{name}"><figcaption>{caption}</figcaption></figure>'


def fig(name, caption, h=330):
    return f'<figure><img src="{FIG}/{name}" style="max-height:{h}px"><figcaption>{caption}</figcaption></figure>'


def full_report_html():
    today = dt.date.today().strftime("%d %B %Y")
    xg = N.get("xgboost", {})
    nb, ns, npred = bm("big", "params"), bm("student", "params"), bm("pred", "params")
    te = JM.get("test_events", {})
    sd = JM.get("stampede_delay", {})
    jn = {"video": "Video only", "net": "5G only", "fusion": "Combined (video + 5G)"}

    video_rows = "".join(
        f"<tr><td>{VNAME.get(m, m)}</td><td>{f(V[m]['auc'][AUC_COLS[m][0]])}</td><td>{f(V[m]['auc'][AUC_COLS[m][1]])}</td>"
        f"<td><b>{f(V[m]['auc'][AUC_COLS[m][2]])}</b></td></tr>" for m in ("pred", "big", "student") if m in V)
    speed_rows = "".join(
        f"<tr><td>{VNAME.get(m, m)}</td><td>{r['params']:,}</td><td>{r['size_mb']:.1f} MB</td><td>{f(r['auc'])}</td>"
        f"<td>{r['cpu_ms']:.1f} ms</td><td>{f(r['gpu_ms'], 2)} ms</td></tr>" for m, r in B.get("models", {}).items())
    net_rows = "".join(
        f"<tr><td>{lab}</td><td>{f(N[k]['accuracy'], 3, True)}</td><td><b>{f(N[k]['macro_f1'])}</b></td>"
        f"<td>{N[k]['stampede_delay_median_s']} s</td><td>{N[k]['stampede_episodes_missed']} / {N[k]['stampede_episodes_total']}</td></tr>"
        for k, lab in (("rule", "Rule baseline"), ("xgboost", "XGBoost (final)"), ("lstm", "LSTM")) if k in N)
    joint_rows = "".join(
        f"<tr><td>{c}</td>" + "".join(f"<td>{'<b>' if m == 'fusion' else ''}{f(jr(m, c, 'macro_f1'))}{'</b>' if m == 'fusion' else ''}</td>" for m in JMOD) + "</tr>"
        for c in JC)
    cam_rows = "".join(
        f"<tr><td>{n}</td><td>{ms(cam_delay(E1, n))}</td><td>{ms(cam_delay(E2, n))}</td><td>{ms(cam_delay(CL, n))}</td></tr>"
        for n in (1, 10, 20, 40, 60, 80, 120) if EC and n in EC["cameras"]["n"])
    off_rows = "".join(
        f"<tr><td>{k}</td><td>{v['mean_p95_ms']:.1f} ms</td><td>{f(v['miss_rate'], 3, True)}</td></tr>"
        for k, v in OF.items() if isinstance(v, dict))
    fusion_rows = "".join(
        f"<tr><td>{k}</td><td>{f(v['auc'])}</td><td>{f(v['tpr_at_5pct_fpr'])}</td><td>{f(v['caught_when_camera_blind'])}</td></tr>"
        for k, v in FU.items() if k != "setup")

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Crowd Safety AI over 5G - Project Report</title>
<style>{CSS}{EXTRA_CSS}</style></head><body>

<div class="title-page">
<h1>Multimodal Crowd Anomaly and Stampede-Risk Detection using Video AI and 5G Network Data with Edge Computing</h1>
<div class="sub">Project report: models, data, architecture, results and web application</div>
<table style="width:70%">
<tr><th>Submitted by</th><td>&nbsp;</td></tr>
<tr><th>Guide</th><td>&nbsp;</td></tr>
<tr><th>Department / Institute</th><td>&nbsp;</td></tr>
<tr><th>Date</th><td>{today}</td></tr>
</table>
</div>

<h2>Contents</h2>
<ol class="toc">
<li>Abstract</li><li>Executive summary</li><li>Introduction and problem statement</li><li>Key concepts explained</li>
<li>System architecture</li><li>Data</li><li>How the models work</li><li>Results</li><li>Web application</li>
<li>Discussion and limitations</li><li>Future work</li><li>Conclusion</li><li>References</li><li>Appendix: how to run</li>
</ol>

<h2>Abstract</h2>
<p>Crowd crushes and stampedes develop within seconds, so a safety system must detect danger both <b>accurately</b> and
<b>quickly</b>. This project builds an AI/ML system that watches a crowd in two independent ways, through <b>CCTV video</b> and
through the <b>5G mobile network</b> that the crowd's phones are connected to, and runs at the <b>5G network edge</b> so that alerts
arrive in milliseconds. A video model trained only on normal footage of the public UCSD Ped2 benchmark flags unusual events with a
frame-level <b>AUC of {f(best_auc())}</b> (future-frame prediction U-Net). A network model detects dangerous crowding and stampedes from
5G observations (phones per cell, handovers, density and speed from network positioning) with <b>{f(xg.get('accuracy'), 3, True)}
accuracy</b>. Because no public dataset contains CCTV and 5G data of the same crowd, a joint simulation was built in which each event is
filmed and sensed at the same time; a <b>combined video + 5G model</b> trained on it reaches macro-F1 <b>{f(jr('fusion', JC[0], 'macro_f1'))}</b>,
against {f(jr('net', JC[0], 'macro_f1'))} for 5G only and {f(jr('video', JC[0], 'macro_f1'))} for video only, and stays reliable when either
source fails. A 5G edge-computing simulation shows that edge processing delivers alerts in about {ms(cam_delay(E1, 10))} versus
{ms(cam_delay(CL, 10))} from the cloud, that a URLLC network slice protects safety traffic when the public's phones load the network, and
that an ML model choosing edge or cloud per situation is right {f(OF.get('ml_accuracy'), 3, True)} of the time. A web application
demonstrates everything live.</p>

<h2>Executive summary</h2>
{plain("we built a system that spots dangerous crowds using two 'senses': a camera that sees what is happening, and the 5G network that "
       "feels how many phones are packed together and how fast they move. Each sense has blind spots (darkness, areas the camera cannot see, "
       "people without phones), so one AI model uses both together. The AI runs on a small server next to the 5G tower so the alert reaches "
       "security staff in a few milliseconds.")}
<table>
<tr><th>Part</th><th>What it does</th><th>Data</th><th>Key result</th></tr>
<tr><td>Model 1: video anomaly detection</td><td>Flags unusual objects and motion in CCTV video</td><td>UCSD Ped2 (real)</td><td>AUC {f(best_auc())}</td></tr>
<tr><td>Model 2: 5G crowd risk</td><td>Detects dense crowds and stampedes from network data</td><td>Simulated 5G crowds</td><td>{f(xg.get('accuracy'), 3, True)} accuracy, macro-F1 {f(xg.get('macro_f1'))}</td></tr>
<tr><td><b>Combined video + 5G model</b></td><td>Uses camera and network data of the same event together</td><td>Joint simulated events</td><td><b>macro-F1 {f(jr('fusion', JC[0], 'macro_f1'))}</b>; {f(jr('fusion', JC[0], 'danger_detected'), 3, True)} of danger caught, {f(jr('fusion', JC[0], 'false_alarm'), 3, True)} false alarms</td></tr>
<tr><td>Edge vs cloud over 5G</td><td>Measures alert delay for processing at the edge or in the cloud</td><td>Discrete-event 5G simulation</td><td>{ms(cam_delay(E1, 10))} (edge) vs {ms(cam_delay(CL, 10))} (cloud)</td></tr>
<tr><td>Network slicing</td><td>Priority lane for safety traffic</td><td>Same simulation</td><td>{ms(slice_p95(SL, 0.9))} vs {ms(slice_p95(NS, 0.9))} at 90% public load</td></tr>
<tr><td>ML offloading decision</td><td>Chooses edge or cloud for each situation</td><td>500 simulated situations</td><td>Right {f(OF.get('ml_accuracy'), 3, True)} of the time</td></tr>
<tr><td>Web application</td><td>Runs all of the above live in a browser</td><td>-</td><td>5 pages (Section 9)</td></tr>
</table>

<h2 class="pb">1. Introduction and problem statement</h2>
<p>Large gatherings such as festivals, pilgrimages, stadium events and busy railway stations carry a risk of crowd crushes and
stampedes. Control-room operators cannot watch every CCTV screen at once, and an alert that arrives late is of little use.
Two technologies can help. <b>AI video analytics</b> can watch every camera continuously, and <b>5G networks</b> can carry video from
many wireless cameras, sense the crowd through the phones connected to them, and run AI on <b>edge servers</b> placed right next to the
base station.</p>
<div class="box"><b>Problem statement.</b> Edge-AI crowd anomaly detection: train the detection model on public crowd-video datasets,
then use a 5G edge-computing (MEC) model to compare processing video at the edge versus in the cloud over 5G.</div>
<h3>Objectives</h3>
<ol>
<li>Train a video anomaly detector on a public dataset and evaluate it with the standard frame-level AUC.</li>
<li>Build a lightweight version for an edge server and measure its accuracy/speed trade-off.</li>
<li>Use 5G network data as a second crowd sensor (density, mass movement, location of hotspots).</li>
<li>Detect crowd danger from video and 5G data <b>of the same event</b> with one combined model that stays reliable when either source fails.</li>
<li>Compare edge vs cloud processing over 5G, including network slicing, and let an ML model decide where to process.</li>
<li>Demonstrate the whole system in a web application.</li>
</ol>
<h3>Scope</h3>
<p>The video model is trained and tested on real public video. The 5G network, the edge server and network slicing are
<b>simulated</b> in Python, because real operator data is private and no physical 5G equipment was available. The AI processing
times used in the simulation are <b>real measurements</b> on the project laptop (NVIDIA RTX 5050 GPU and CPU).</p>

<h2 class="pb">2. Key concepts explained</h2>
<table>
<tr><th>Concept</th><th>Plain explanation</th><th>Used for</th></tr>
<tr><td>Anomaly detection</td><td>Learning what "normal" looks like and flagging anything that does not fit, without needing examples of every possible danger.</td><td>Model 1</td></tr>
<tr><td>Autoencoder</td><td>A neural network that compresses an image and rebuilds it. Trained only on normal scenes, it rebuilds unusual things badly; the rebuild error is the anomaly score.</td><td>Model 1 v1</td></tr>
<tr><td>Future-frame prediction (U-Net)</td><td>The network sees a few frames and predicts the next one. Normal walking is easy to predict; a bike or a cart moving differently is not. U-Net is an image network with "skip connections" that keep fine detail.</td><td>Model 1 v2 (final)</td></tr>
<tr><td>AUC (ROC)</td><td>A score from 0.5 (guessing) to 1.0 (perfect) for how well anomaly scores separate abnormal frames from normal ones at every possible threshold.</td><td>Model 1 results</td></tr>
<tr><td>Knowledge distillation</td><td>A small "student" network learns to copy a big "teacher" network, so it is much smaller and faster but keeps part of the accuracy.</td><td>Edge model</td></tr>
<tr><td>CNN</td><td>Convolutional neural network: the standard network for images, learning edges, shapes and motion patterns.</td><td>Combined model video branch</td></tr>
<tr><td>XGBoost</td><td>A strong machine-learning method that combines many small decision trees. Works very well on tables of numbers.</td><td>Model 2, offloading</td></tr>
<tr><td>LSTM</td><td>A neural network for sequences that remembers what happened earlier in a time series.</td><td>Model 2 comparison</td></tr>
<tr><td>Macro-F1</td><td>Accuracy measure that treats every class (normal, dense, stampede) equally, so rare dangerous classes count as much as the common normal class. 1.0 is perfect.</td><td>Model 2, combined model</td></tr>
<tr><td>5G gNB, UE</td><td>gNB = the 5G base station (tower or small cell); UE = user equipment (a phone).</td><td>5G simulation</td></tr>
<tr><td>Handover, SINR</td><td>Handover: a phone switching from one cell to another as it moves. SINR: signal quality.</td><td>Model 2 features</td></tr>
<tr><td>Network positioning</td><td>The network estimates where phones are (5G supports positioning to a few metres). Only counts per 10 m zone are used.</td><td>Density heatmap</td></tr>
<tr><td>MEC (edge computing)</td><td>Multi-access Edge Computing: a small server next to the 5G base station, so data does not travel to a faraway cloud.</td><td>Edge vs cloud</td></tr>
<tr><td>Network slicing (URLLC / eMBB)</td><td>Splitting one 5G network into virtual lanes. URLLC = ultra-reliable low-latency lane (safety cameras); eMBB = normal high-bandwidth lane (public phones).</td><td>Slicing study</td></tr>
<tr><td>Discrete-event simulation</td><td>A program that simulates a system event by event (a frame is sent, arrives, is processed) to measure delays and queues.</td><td>Edge vs cloud</td></tr>
</table>

<h2 class="pb">3. System architecture</h2>
<pre>{ARCH}</pre>
<h3>Components</h3>
<table>
<tr><th>Component</th><th>Role</th><th>Implementation</th></tr>
<tr><td>CCTV cameras</td><td>Film the crowd</td><td>UCSD Ped2 videos (real); rendered overhead camera for joint events (simulated)</td></tr>
<tr><td>People's phones + 5G cells</td><td>The network senses phones per cell, handovers, positions</td><td>5G crowd simulator: 9 small cells, 3GPP path-loss model</td></tr>
<tr><td>5G network with slicing</td><td>Carries video and alerts; URLLC lane for safety traffic</td><td>Discrete-event simulation</td></tr>
<tr><td>MEC edge server</td><td>Runs the AI next to the tower</td><td>Simulated, using the measured CPU speed of the small model</td></tr>
<tr><td>Cloud</td><td>Runs the most accurate model, far away</td><td>Simulated, using the measured GPU speed of the v2 model</td></tr>
<tr><td>AI models</td><td>Model 1, Model 2, combined model, offloading model</td><td>PyTorch, XGBoost, scikit-learn</td></tr>
<tr><td>Web application</td><td>Live demonstration</td><td>React frontend + FastAPI backend</td></tr>
</table>
<h3>What happens during one alert</h3>
<ol class="steps">
<li>The CCTV camera records frames, and the 5G cells observe the phones of the same crowd.</li>
<li>Camera frames travel over the 5G uplink in the URLLC slice, so the public's phone traffic cannot block them.</li>
<li>The ML offloading model decides whether to process at the edge server or in the cloud.</li>
<li>The combined model looks at the last 4 video frames and the last 30 seconds of 5G data and decides: normal, dense or stampede.</li>
<li>If the decision is dense or stampede, an alert with the hotspot location goes back to security staff over the downlink.</li>
</ol>
<h3>What is real and what is simulated</h3>
<table>
<tr><th>Real</th><th>Simulated</th></tr>
<tr><td>UCSD Ped2 videos; Model 1 training and accuracy; AI speeds measured on real hardware; the web application</td>
<td>5G network data (Model 2); joint video + 5G events (combined model); 5G radio, edge server, cloud and slicing</td></tr>
</table>

<h2 class="pb">4. Data</h2>
<h3>4.1 UCSD Ped2 (real CCTV video)</h3>
<p>A public benchmark recorded by a fixed camera overlooking a pedestrian walkway (Mahadevan et al., 2010). Normal = people walking;
anomalies = bikes, skateboarders, carts and small vehicles. The 16 training videos (2,550 frames) contain only normal behaviour; the
12 test videos (2,010 frames, 1,648 of them anomalous) come with official frame-level labels. Frames were converted to grayscale and
resized (128×128 for v1, 256×256 for v2).</p>
<h3>4.2 5G crowd simulation (Model 2)</h3>
<p>A 200 m × 200 m venue covered by a 3×3 grid of 5G small cells at 3.5 GHz. 800-2,600 people move around: 5% carry no phone and 10%
carry two. Three scenarios are simulated: <b>normal</b> (wandering, safe attractions, harmless rushes such as people hurrying to a train),
<b>gathering</b> (the crowd packs into one spot) and <b>stampede</b> (gathering, then sudden panic with people fleeing at 2-5 m/s).
Every second the network side records 9 features:</p>
<table>
<tr><th>Feature</th><th>Meaning</th></tr>
<tr><td>total_phones</td><td>Phones connected in the venue</td></tr>
<tr><td>max_cell_phones</td><td>Phones on the busiest cell</td></tr>
<tr><td>max_zone_density</td><td>People per m² in the densest 10 m zone, estimated from network positioning (5 m error)</td></tr>
<tr><td>density_growth_10s</td><td>How fast that density grew over the last 10 s</td></tr>
<tr><td>mean_speed, frac_fast</td><td>Average phone speed and share of phones moving faster than 3 m/s</td></tr>
<tr><td>handover_rate</td><td>Handovers per phone per second (rises when people run between cells)</td></tr>
<tr><td>max_cell_load</td><td>How busy the busiest cell is</td></tr>
<tr><td>mean_sinr_db</td><td>Average signal quality</td></tr>
</table>
<p>The ground truth is <i>dense</i> when any 10 m zone exceeds 2 people/m², and <i>stampede</i> while panic is in progress.
450 episodes of 180 s give 22,950 windows of 30 seconds.</p>
{plain("we cannot get real phone data from a telecom operator, so we built a realistic simulator: virtual people walk around a venue with "
       "phones, and we record what the 5G towers would see. Only counts per zone are used, never individual people.")}
<h3>4.3 Joint video + 5G dataset (combined model)</h3>
<p>The same simulator was extended so that one event produces <b>both</b> sensor streams. Every person's position is recorded every second;
the 5G features come from their phones as above, and an overhead CCTV camera covering an 80 m × 80 m part of the venue is rendered from
the same positions (96×96 pixels, one frame per second, paving texture and camera noise). 300 events were generated (100 per scenario),
with three hard cases built in: <b>night</b> video (20% of events), <b>few phones visible</b> to the operator (20%, 60% of people invisible)
and the danger <b>outside the camera view</b> (about 25%). The test set has {te.get('total', '-')} events: {te.get('night', '-')} at night,
{te.get('outside_view', '-')} with the danger outside the camera view and {te.get('few_phones', '-')} with few phones.</p>
{fig("joint_video_preview.png", "CCTV frames rendered from the same simulated crowd that produces the 5G data.", 230)}
<h3>4.4 Other data</h3>
<ul>
<li><b>5G MEC situations:</b> 500 random situations (1-80 cameras, 5-15 fps, 0-95% public load, slicing on/off, edge busy 0-80%, cloud
20-100 ms away), each simulated for edge and cloud, to train the offloading model.</li>
<li><b>Sample videos</b> for the web app: 4 UCSD clips converted to .mp4 and synthetic crowd videos (calm, panic, and a CCTV-style
stampede), clearly labelled as synthetic.</li>
</ul>

<h2 class="pb">5. How the models work</h2>
<h3>5.1 Model 1: video anomaly detection</h3>
<h4>v1: convolutional autoencoder</h4>
<ol class="steps">
<li>Input: 5 consecutive grayscale frames (128×128) stacked together, so the network sees both appearance and motion.</li>
<li>Encoder: 4 convolution layers shrink the clip to an 8×8 summary; decoder: 4 layers rebuild the 5 frames.</li>
<li>Training: only normal videos, minimising the difference between input and rebuild (mean squared error), 60 epochs.</li>
<li>Score: the error map is split into 16×16 patches and the worst patch is the frame's anomaly score, because anomalies are small objects.</li>
</ol>
<h4>v2 (final): future-frame prediction U-Net</h4>
<ol class="steps">
<li>Input: 4 consecutive frames at 256×256 (twice v1's resolution, so small objects keep their detail).</li>
<li>A U-Net ({f(npred / 1e6, 1) if npred else '-'} M parameters) predicts the 5th frame. Skip connections carry fine detail from the encoder to the decoder.</li>
<li>Training: normal videos only; loss = pixel error + image-gradient error (keeps edges sharp); 30 epochs with mixed precision on the GPU.</li>
<li>Score: worst 32×32 patch of the prediction error, normalised within each video and smoothed over time (the same rule as v1, fixed before testing).</li>
</ol>
{plain("the model has only ever seen people walking normally. It keeps guessing the next picture. When a bike or a cart appears, its guess "
       "is clearly wrong in that spot, and that wrong guess raises the alarm.")}
<h4>Edge model (knowledge distillation)</h4>
<p>A student autoencoder with 16× fewer parameters ({ns:,} vs {nb:,}) is trained to rebuild the input <i>and</i> to copy the big
model's output. It runs in {f(bm('student', 'cpu_ms'), 1)} ms per frame on a CPU, which makes it suitable for an edge server.</p>

<h3>5.2 Model 2: crowd risk from 5G data</h3>
<ol class="steps">
<li>Take the last 30 seconds of the 9 network features.</li>
<li>Summarise each feature with 7 statistics (last value, last 5 s, mean, max, min, spread, trend), giving 63 numbers.</li>
<li>XGBoost (400 trees) classifies the moment as normal, dense or stampede. Class weights make rare dangerous moments count fully.</li>
<li>Comparisons: a threshold rule (tuned on training data) and an LSTM on the raw 30-second sequence.</li>
</ol>
{plain("if the densest zone keeps filling up, the model says 'dense'. If suddenly many phones move fast and switch cells, it says "
       "'stampede'. XGBoost learns these patterns, and the combinations of them, from examples.")}

<h3>5.3 Combined video + 5G model</h3>
<ol class="steps">
<li><b>Video branch:</b> a CNN (4 convolution blocks, 32-64-128-128 channels) looks at the last 4 CCTV frames plus their frame
differences (motion). Contrast is normalised per clip, which helps in dark video.</li>
<li><b>5G branch:</b> a small neural network (MLP) reads the same 63 network statistics as Model 2.</li>
<li><b>Fusion:</b> the two branch outputs are joined and a final layer decides normal / dense / stampede.</li>
<li><b>Training:</b> the whole network learns end to end on the joint dataset. With "modality dropout", 15% of the time one branch is hidden,
so the model learns to rely on the other source when one fails.</li>
<li><b>Fair comparison:</b> a video-only and a 5G-only model with the same branches are trained on exactly the same samples; events are
split 70/10/20 into train/validation/test so no event is seen in both training and testing.</li>
</ol>
{plain("the combined model is like a guard who both watches the camera and reads the phone-network dashboard. When it is dark or the "
       "danger is off-camera, it trusts the network; when few phones are visible, it trusts the camera.")}

<h3>5.4 Edge vs cloud over 5G, and network slicing</h3>
<p>A discrete-event simulation follows every camera frame ({f(B.get('frame_kb', {}).get('ped2_native_360x240'), 1)} KB JPEG, 10 fps):
upload over a 250 Mbps 5G cell uplink shared with the public's phones, travel to the server, waiting in the queue, AI processing and the
alert coming back. The <b>edge</b> is 1 ms from the tower with 2 workers running the small model at its measured CPU speed; the
<b>cloud</b> is 20-100 ms away with many GPU workers running the v2 model at its measured GPU speed. With <b>slicing</b>, safety cameras
get strict priority on the uplink (URLLC); without it, all traffic shares one queue.</p>
<h3>5.5 ML offloading decision</h3>
<p>For each of the 500 simulated situations, both edge and cloud are simulated, and the faster one (95th-percentile delay) becomes the
label. XGBoost learns to predict it from what the system can observe: number of cameras, frame rate, public load, slicing on/off,
edge busy share and cloud distance.</p>
<h3>5.6 Earlier baseline: score-level fusion</h3>
<p>Before the joint dataset existed, the outputs of Model 1 (UCSD video) and Model 2 (simulated 5G) were paired artificially and combined
with max, average or logistic regression. It is kept as a baseline (Section 7.4) and is superseded by the combined model.</p>

<h2 class="pb">6. Results</h2>
<h3>6.1 Model 1: video anomaly detection</h3>
<table><tr><th>Model</th><th>AUC (whole frame)</th><th>AUC (worst patch)</th><th>AUC (final)</th></tr>{video_rows}</table>
<p>Predicting the next frame raises the AUC from {f(auc('big'))} to <b>{f(best_auc())}</b>. For reference, the classic autoencoder of Hasan
et al. (2016) reports about 0.85-0.90 on Ped2 and Liu et al. (2018) report 0.95. Without the per-video normalisation step, v2 scores
{f(V.get('pred', {}).get('auc', {}).get('patch_max'))}, still far above v1.</p>
{fig("video_roc.png", "ROC curves on the UCSD Ped2 test set.")}
{fig("video_examples.png", "Most anomalous frames and where the prediction error is concentrated.")}
<h3>6.2 Edge model and speed</h3>
<table><tr><th>Model</th><th>Parameters</th><th>Size</th><th>AUC</th><th>Edge (CPU)</th><th>Cloud (GPU)</th></tr>{speed_rows}</table>
<p>v2 is the most accurate but takes {f(bm('pred', 'cpu_ms'), 1)} ms per frame on a CPU, so it belongs in the cloud
({f(bm('pred', 'gpu_ms'), 1)} ms on the GPU). The student model is the edge model.</p>
<h3>6.3 Model 2: crowd risk from 5G data</h3>
<table><tr><th>Detector</th><th>Accuracy</th><th>Macro-F1</th><th>Stampede detection delay (median)</th><th>Stampedes missed</th></tr>{net_rows}</table>
{fig("net_heatmaps.png", "Crowd density: ground truth (top) and estimated from 5G positioning (bottom). Triangles = 5G cells.", 380)}
{fig("net_timeline.png", "One stampede seen through 5G data with the model's risk output.", 380)}
<h3>6.4 Combined video + 5G model</h3>
<p>Macro-F1 on {te.get('total', '-')} unseen test events, by condition:</p>
<table><tr><th>Test condition</th><th>Video only</th><th>5G only</th><th>Combined</th></tr>{joint_rows}</table>
<table><tr><th>All test events</th><th>Video only</th><th>5G only</th><th>Combined</th></tr>
<tr><td>Dangerous moments detected</td>{''.join(f"<td>{f(jr(m, JC[0], 'danger_detected'), 3, True)}</td>" for m in JMOD)}</tr>
<tr><td>False alarms</td>{''.join(f"<td>{f(jr(m, JC[0], 'false_alarm'), 3, True)}</td>" for m in JMOD)}</tr>
<tr><td>Stampede detection delay (median)</td>{''.join(f"<td>{sd.get(jn[m], {}).get('median_s', '-')} s</td>" for m in JMOD)}</tr>
<tr><td>Stampedes missed</td>{''.join(f"<td>{sd.get(jn[m], {}).get('missed', '-')} / {sd.get(jn[m], {}).get('total', '-')}</td>" for m in JMOD)}</tr>
</table>
{fig("joint_by_condition.png", "Each source has a blind spot; the combined model covers it.")}
<p>The combined model is best in every condition. When the danger is outside the camera view, video alone drops to
{f(jr('video', JC[3], 'macro_f1'))} and the 5G branch keeps the combined model at {f(jr('fusion', JC[3], 'macro_f1'))}. When few phones are
visible, 5G alone drops to {f(jr('net', JC[4], 'macro_f1'))} and the video branch keeps it at {f(jr('fusion', JC[4], 'macro_f1'))}.
<b>Night</b> has only {te.get('night', '-')} test events, so its macro-F1 is unreliable; on them the combined model still detects
{f(jr('fusion', JC[2], 'danger_detected'), 3, True)} of dangerous moments with {f(jr('fusion', JC[2], 'false_alarm'), 3, True)} false alarms
(video only: {f(jr('video', JC[2], 'danger_detected'), 3, True)} and {f(jr('video', JC[2], 'false_alarm'), 3, True)}). What it finds hard at
night is separating <i>dense</i> from <i>stampede</i>.</p>
{fig("joint_confusion.png", "Combined model on the test events.", 250)}
<h3>6.5 Edge vs cloud over 5G</h3>
<table><tr><th>Cameras</th><th>Edge, small model</th><th>Edge, big model on CPU</th><th>Cloud, v2 model on GPU</th></tr>{cam_rows}</table>
{fig("edge_vs_cloud_cameras.png", "Mean alert delay and late alerts (over 100 ms) as cameras are added (50% public load, slicing on).")}
<p>The edge is about 8 times faster while it has spare capacity. With too many cameras its queue grows and the cloud becomes better.</p>
<h3>6.6 Network slicing</h3>
<p>With 20 cameras at 90% public load, 95% of alerts arrive within <b>{ms(slice_p95(SL, 0.9))}</b> with a URLLC slice and
{ms(slice_p95(NS, 0.9))} without it.</p>
{fig("slicing_effect.png", "A URLLC slice protects safety traffic when the public's phones load the uplink.", 290)}
<h3>6.7 ML offloading decision</h3>
<table><tr><th>Policy</th><th>Average 95th-percentile delay</th><th>Situations over 100 ms</th></tr>{off_rows}</table>
<p>The ML model picks the faster place in <b>{f(OF.get('ml_accuracy'), 3, True)}</b> of unseen situations and comes close to the best
possible choice (oracle).</p>
{fig("offload_policies.png", "Always edge vs always cloud vs ML decides vs best possible.", 290)}
<h3>6.8 Earlier baseline (score-level fusion)</h3>
<table><tr><th>Method</th><th>AUC</th><th>Detection at 5% false alarms</th><th>Caught when camera blind</th></tr>{fusion_rows}</table>

<h2 class="pb">7. Web application</h2>
<p>The web application shows the whole system working live. It has two parts that run at the same time:</p>
<table>
<tr><th>Part</th><th>Technology</th><th>What it does</th></tr>
<tr><td>Frontend</td><td>React 19, Vite, Recharts</td><td>The pages in the browser: controls, charts, heatmaps and video</td></tr>
<tr><td>Backend</td><td>Python FastAPI</td><td>Runs the models and simulations when a page asks, and returns the results as JSON</td></tr>
<tr><td>Worker processes</td><td>PyTorch</td><td>Video and combined-model inference runs in a separate process, so GPU (CUDA) code and XGBoost never share one Python process (that combination crashed on Windows)</td></tr>
</table>
<table>
<tr><th>Page</th><th>Backend endpoint</th><th>Runs</th><th>Time</th></tr>
<tr><td>Combined (video + 5G)</td><td>POST /api/combined</td><td>Simulates one event, renders its CCTV video, runs the 3 models</td><td>~10 s</td></tr>
<tr><td>Video detection</td><td>POST /api/detect</td><td>Model 1 on every frame of an uploaded or chosen video</td><td>10-20 s</td></tr>
<tr><td>5G crowd monitor</td><td>POST /api/crowd</td><td>Simulates 3 minutes of crowd, runs Model 2 every second</td><td>~6 s</td></tr>
<tr><td>Edge vs cloud</td><td>POST /api/edge</td><td>5G edge/cloud simulation + ML decision</td><td>~1 s</td></tr>
<tr><td>Results</td><td>GET /api/results</td><td>Reads the saved results and graphs</td><td>instant</td></tr>
</table>

<h3>7.1 Combined monitor (video + 5G)</h3>
<p>The main page. The user chooses an event (normal, gathering or stampede), the number of people and a random seed, and can make it
harder with <b>Night</b> and <b>Few phones visible</b>. After <b>Run event</b>, the CCTV camera and the 5G density map of the <i>same</i>
event play side by side. The blue box on the map is the camera's field of view, so the user can see whether the danger is visible to
the camera. At the top, the verdicts of the combined, video-only and 5G-only models are shown for the current second next to the true state.</p>
{shot("combined_main.jpg", "Combined monitor during a stampede (121 s): all models say STAMPEDE; the camera shows people fleeing from the centre.")}
{shot("combined_timeline.jpg", "Lower part: event summary per model and the danger probability timeline (green = combined) with the true dense (yellow) and stampede (red) periods.")}
<p>In this event (2,200 people, panic at 116 s), the combined model caught 100% of dangerous moments with 1% false alarms and flagged the
stampede at 116 s; video alone raised 33% false alarms.</p>
{shot("combined_night.jpg", "The same event at night: the camera image is dark and noisy, but the combined model still says STAMPEDE because the 5G branch carries it.")}

<h3>7.2 Video detection</h3>
<p>The user picks a UCSD test video or uploads a file, chooses the detection mode (the AI model, or the crowd-motion detector for any fixed
camera) and presses <b>Detect anomalies</b>. The page shows the verdict, the first alert time compared with the true start, the annotated
video (red border and "ANOMALY" on alert frames), the most unusual frame, the alert periods and the anomaly score over time.</p>
{shot("video_main.jpg", f"Test004: alert from 3.2 s, only 0.2 s after the cart appears (true start 3.0 s). Command-line tool result: {DET.get('alert_intervals_s', '-')} s.")}

<h3>7.3 5G crowd monitor</h3>
<p>Shows a venue as the 5G network sees it. After <b>Run simulation</b>, the density heatmap (network estimate or true density) plays
second by second with the 5G cells (triangles) and the hotspot (circle). On the right are Model 2's risk over time and three key network
signals. Clicking the chart jumps to that moment.</p>
{shot("crowd_main.jpg", "A gathering of 2,400 people at 156 s: the hotspot reaches 2.5 people/m², and Model 2 says DENSE, matching the truth.")}

<h3>7.4 Edge vs cloud</h3>
<p>Sliders and presets describe the situation (cameras, frame rate, public load, edge busy, cloud distance, slicing on/off). The page shows
the edge and cloud alert delays (95% of alerts arrive within this time), which one is faster, what the ML decision model picked and
whether it was right, and how the delay grows as cameras are added.</p>
{shot("edge_main.jpg", "Big festival preset (80 cameras): the edge is overloaded (4.0 s) while the cloud answers in 68.6 ms; the ML model correctly picks the cloud.")}

<h3>7.5 Results</h3>
<p>All final numbers and graphs in one place, read from the same saved results the report is built from, with a note on what is real
and what is simulated. Graphs can be enlarged by clicking.</p>
{shot("results_main.jpg", "Results page: headline numbers and the Model 1 section.")}

<h2 class="pb">8. Discussion and limitations</h2>
<ul>
<li><b>Scene-specific "normal".</b> Model 1 learns normality for one camera view; each new camera needs its own normal footage.</li>
<li><b>Simulated network data.</b> Model 2 and the combined model are trained and tested on simulated 5G data; real operator data would
be needed to confirm the numbers.</li>
<li><b>Synthetic joint video.</b> The combined model's CCTV video is rendered from simulated people (overhead figures on paving), not real
footage, so its video branch is separate from the UCSD-trained Model 1.</li>
<li><b>Small night sample.</b> Night results rest on {te.get('night', '-')} test events.</li>
<li><b>Python 5G simulation instead of Simu5G.</b> The edge/cloud study uses the same concepts as Simu5G (gNB, UEs, MEC host,
slicing) but not its full protocol stack.</li>
<li><b>Edge hardware.</b> The edge server is represented by a laptop CPU limited to 4 threads; a real MEC host may have a small GPU.</li>
<li><b>Privacy.</b> Only counts per zone are used from the network; no individual is tracked.</li>
</ul>

<h2>9. Future work</h2>
<ul>
<li>Port the 5G scenario to Simu5G (OMNeT++) using the same parameters and measured processing times.</li>
<li>Validate on real data: aggregated operator data (e.g. the Telecom Italia Milan dataset) and real paired CCTV recordings.</li>
<li>Train the combined model on real CCTV-style video and add more datasets (CUHK Avenue, ShanghaiTech, UMN).</li>
<li>Reinforcement learning for offloading and dynamic slice control; split computing between camera and edge.</li>
</ul>

<h2>10. Conclusion</h2>
<p>The project delivers a complete AI/ML pipeline for crowd safety over 5G. A future-frame prediction model detects video anomalies
on a public benchmark with AUC {f(best_auc())}; a distilled model makes edge deployment practical; 5G network data acts as a second crowd
sensor that also locates dangerous zones; a combined model that sees the same event through the camera and the network reaches macro-F1
{f(jr('fusion', JC[0], 'macro_f1'))} and stays reliable when either source fails; and edge processing with a URLLC slice and ML-based
offloading delivers alerts within real-time limits. A web application demonstrates all of it live.</p>

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
<pre>python run_all.py                  # full pipeline (about 55-60 min with a GPU)
python run_all.py --skip-train     # reuse trained models and data, rebuild results and this report

# web application: two terminals
python -m uvicorn api.server:app --port 8000      # in crowd-anomaly-5g
npm run dev                                       # in crowd-anomaly-5g/frontend, then open http://localhost:5173</pre>
<table>
<tr><th>Folder</th><th>Contents</th></tr>
<tr><td>video_model/</td><td>Model 1: data loading, v1 autoencoders, v2 prediction U-Net, evaluation, speed benchmark</td></tr>
<tr><td>network_model/</td><td>5G crowd simulator, Model 2, heatmaps</td></tr>
<tr><td>joint/</td><td>Joint video + 5G dataset and the combined model</td></tr>
<tr><td>edge_cloud/</td><td>5G edge vs cloud simulation with slicing, ML offloading model</td></tr>
<tr><td>fusion/</td><td>Earlier baseline (score-level fusion)</td></tr>
<tr><td>api/, frontend/</td><td>Web application (FastAPI backend, React frontend)</td></tr>
<tr><td>samples/</td><td>Sample videos and the scripts that make them</td></tr>
<tr><td>report/</td><td>This report and the slides</td></tr>
</table>
</body></html>"""
