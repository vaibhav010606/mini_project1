"""Run the whole project end to end.

    python run_all.py            # everything (about 55-60 min on an RTX laptop GPU)
    python run_all.py --skip-train   # reuse trained models, regenerate results/figures/report
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Load UCSD Ped2", ["video_model/dataset.py"], False),
    ("Train big video model (cloud)", ["video_model/train.py", "--model", "big"], True),
    ("Train student video model (edge)", ["video_model/train.py", "--model", "student", "--distill"], True),
    ("Evaluate video models", ["video_model/evaluate.py"], False),
    ("Train v2 future-frame prediction model", ["video_model/predict_model.py", "train", "--epochs", "30"], True),
    ("Evaluate v2 prediction model", ["video_model/predict_model.py", "eval"], False),
    ("Benchmark edge vs cloud speed", ["video_model/benchmark.py"], False),
    ("Simulate 5G crowd network data", ["network_model/simulator.py"], True),
    ("Train network crowd-risk models", ["network_model/train.py"], True),
    ("Network heatmaps and timeline", ["network_model/heatmap.py"], False),
    ("Fuse video + network (earlier baseline)", ["fusion/fuse.py"], False),
    ("Generate joint video + 5G data (same events)", ["joint/generate.py"], True),
    ("Train combined video + 5G model", ["joint/train.py"], True),
    ("5G edge vs cloud + slicing simulation", ["edge_cloud/sim_5g_mec.py"], False),
    ("ML offloading decision", ["edge_cloud/offload_model.py"], False),
    ("Demo: detect.py on a test video", ["detect.py", "--video",
                                         "data/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test/Test004", "--mode", "ae"], False),
    ("Build report + slides", ["report/build_report.py"], False),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for name, cmd, is_train in STEPS:
        if args.skip_train and is_train:
            print(f"--- skip: {name}")
            continue
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"Step failed: {name}")
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. See results/ and report/.")


if __name__ == "__main__":
    main()
