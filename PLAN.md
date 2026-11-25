# One-Day Build Plan: Multimodal Crowd Anomaly Detection over 5G

**Title:** Multimodal Crowd Anomaly and Stampede-Risk Detection using Video AI and 5G Network Data with Edge Computing

**Deadline:** submission tomorrow. The plan below is scoped so every part produces a real, working result in about one working day.

---

## What will be finished by submission

| # | Part | Field | Output |
|---|------|-------|--------|
| 1 | **Model 1: Video anomaly detection** (convolutional autoencoder, UCSD Ped2) | AI/ML | AUC score, ROC curve, anomaly-score plots, sample frames |
| 2 | **Lightweight edge model** (smaller student model) + speed benchmark on CPU (edge) vs GPU (cloud) | AI/ML | Accuracy vs speed vs size table |
| 3 | **5G network crowd simulator** (people = phones moving across a venue covered by several 5G cells) | 5G | Synthetic network dataset: phones per cell, handovers, SINR, growth rate |
| 4 | **Model 2: Network crowd-risk model** (XGBoost + LSTM) + zone density heatmap | AI/ML | Accuracy/F1, confusion matrix, heatmap image |
| 5 | **Fusion** of video score + network score | AI/ML | Fusion vs video-only vs network-only comparison |
| 6 | **Edge vs cloud over 5G with network slicing** (URLLC vs eMBB), discrete-event simulation in Python | 5G | Delay vs number of cameras graphs |
| 7 | **ML edge/cloud offloading decision** | AI/ML + 5G | "Always edge vs always cloud vs ML decides" graph |
| 8 | **Report (PDF) + presentation slides** | - | Submission files |

### Honest scope note: Simu5G
Installing OMNeT++ + INET + Simu5G and building custom MEC apps takes days, not hours (it needs a full C++ build; WSL/Linux is not set up on this PC). For tomorrow, the 5G network, MEC edge and slicing are simulated **in Python** using the same concepts and parameters Simu5G uses (gNB cells, UEs, SINR, handovers, URLLC/eMBB slices, MEC host vs remote cloud). The report states this clearly and lists **porting the scenario to Simu5G as future work**. Ask your guide if this is acceptable; most guides accept a Python simulation for a first version.

---

## Hour-by-hour plan

| Time | Phase | Tasks | Done when |
|------|-------|-------|-----------|
| **0:00–0:30** | 1. Setup | Project folders, download UCSD Ped2 (706 MB), check GPU | Dataset extracted, `python check_env.py` passes |
| **0:30–2:30** | 2. Model 1 (video) | Load frames → train autoencoder on normal frames (GPU, ~20–30 min) → anomaly scores → AUC | AUC printed, ROC + score plots saved |
| **2:30–3:15** | 3. Lightweight model | Train small student model, benchmark CPU vs GPU, compare | Benchmark table saved (CSV + PNG) |
| **3:15–4:30** | 4. Network simulator | Venue grid + 5G cells, people moving in 3 scenarios (normal / gathering / stampede), log per-cell features | `network_data.csv` generated |
| **4:30–5:30** | 5. Model 2 (network) | Train XGBoost + LSTM on network features, heatmap per zone | Accuracy/F1 + confusion matrix + heatmap saved |
| **5:30–6:15** | 6. Fusion | Pair video scores with network scenarios, compare fusion vs single models | Comparison graph saved |
| **6:15–7:30** | 7. Edge vs cloud + slicing + ML offloading | Discrete-event simulation of cameras → gNB → MEC / cloud with URLLC/eMBB slices; train offloading classifier | Delay graphs + ML-decision graph saved |
| **7:30–9:00** | 8. Report + slides | Generate PDF report with all graphs, results tables, architecture diagram; slides | `report.pdf` + slides ready |
| **9:00–9:30** | 9. Final check | Re-run everything from scratch with one command, read through report | `python run_all.py` works end-to-end |

---

## Folder structure

```
crowd-anomaly-5g/
├── PLAN.md                 ← this file
├── README.md               ← how to run
├── run_all.py              ← runs every step in order
├── data/
│   └── UCSD_Anomaly_Dataset.v1p2/   ← Ped2 frames + ground truth
├── video_model/            ← Model 1 (traditional / video)
│   ├── dataset.py
│   ├── models.py           ← big autoencoder + lightweight student
│   ├── train.py
│   ├── evaluate.py         ← anomaly scores + AUC
│   └── benchmark.py        ← CPU (edge) vs GPU (cloud) speed
├── network_model/          ← Model 2 (5G network data)
│   ├── simulator.py        ← people/phones moving across 5G cells
│   ├── train.py            ← XGBoost + LSTM crowd-risk model
│   └── heatmap.py          ← zone density map
├── fusion/
│   └── fuse.py             ← combines video + network scores
├── edge_cloud/
│   ├── sim_5g_mec.py       ← cameras → gNB → MEC edge / cloud, URLLC/eMBB slicing
│   └── offload_model.py    ← ML decides edge vs cloud
├── results/                ← all graphs, tables, saved models
└── report/
    ├── build_report.py
    └── report.pdf
```

---

## Key design decisions

**Model 1 (video).** Convolutional autoencoder on 128×128 grayscale frames, trained only on normal training videos. Anomaly score = per-frame reconstruction error, smoothed over time and normalized per video. Evaluated with frame-level AUC against the UCSD Ped2 ground truth. Target AUC: about 0.80–0.90 (state of the art is higher, but this is a solid first version).

**Lightweight model.** Same architecture with fewer channels (student), trained the same way. Compared on AUC, parameter count, file size, and time per frame on CPU (stand-in for the edge server) and GPU (stand-in for the cloud).

**Network simulator.** A venue of about 200 m × 200 m split into a grid of zones, covered by several 5G small cells. Each person carries a phone (with realistic noise: some have none, some have two). Three scenarios: normal walking, gradual gathering at a hotspot, and stampede (sudden fast movement away from a point). Per cell, per second: connected phones, change rate, handovers, average SINR, average speed. Only zone-level counts are used, never individual tracks (privacy).

**Model 2 (network).** Input: last 30 seconds of per-cell features. Output: normal / dense / stampede risk. XGBoost on summary features (main), LSTM on raw sequences (comparison).

**Fusion.** Late fusion: weighted combination of the normalized video score and the network risk score, plus a small logistic-regression fusion learned on paired data. Shown against video-only and network-only.

**Edge vs cloud + slicing.** Discrete-event simulation: N cameras send frames at a set rate over a 5G uplink shared with background public traffic. Two slices: URLLC (reserved capacity, priority) for safety traffic, eMBB / best effort for everything else. Edge path: gNB → MEC host (1–2 ms transport, slower CPU model). Cloud path: gNB → core → internet → cloud (20–60 ms transport, fast GPU model). Processing times come from the real benchmark in Phase 3. Metrics: end-to-end delay (mean and 95th percentile), throughput, server queueing.

**ML offloading.** Dataset built from many simulation runs (number of cameras, edge load, background traffic, SINR, slicing on/off → which path was faster). XGBoost classifier picks edge or cloud per moment. Compared with always-edge and always-cloud.

---

## Risks and fallbacks

| Risk | Fallback |
|------|----------|
| Dataset download slow or link down | Use a mirror, or switch to CUHK Avenue |
| Autoencoder AUC is low | Add temporal smoothing and per-video normalization (usually +5–10 AUC points); report honestly |
| Running out of time | Phases 1, 2, 4, 5 and 8 are the core. Phases 3, 6 and 7 can be shortened |
| Guide insists on Simu5G | Present the Python results now; the same scenario parameters can be ported to Simu5G later |
