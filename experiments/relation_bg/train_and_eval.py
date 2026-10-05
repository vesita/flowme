import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.engine import MultiTaskEngine
from dtseek.tasks.plugin import all_tasks, probe_units_of
from dtseek.tasks.probe import run_probe

out_card = ROOT / "experiments" / "relation_bg" / "relation_repaired.pt"
log_path = ROOT / "logs" / "relation_retrain.log"

cmd = [
    "uv", "run", "python", "-u", "training/train_task_card.py",
    "--card", "relation",
    "--base", "checkpoints/base_encoder.pt",
    "--out", str(out_card),
    "--epochs", "12",
    "--steps-per-epoch", "150",
    "--samples", "14000",
    "--seed", "42",
]
print("Running command:", " ".join(cmd))
res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
with open(log_path, "w", encoding="utf-8") as f:
    f.write(res.stdout + "\n" + res.stderr)

print("Exit code:", res.returncode)
if res.returncode != 0:
    print("FAILED:\n", res.stderr)
    sys.exit(res.returncode)

data = {}
for line in res.stdout.splitlines():
    if line.startswith("AB_METRICS"):
        data = json.loads(line[len("AB_METRICS"):].strip())
        print("Train metrics:", json.dumps(data["metrics"], indent=2))

engine = MultiTaskEngine(base_path=str(ROOT / "checkpoints" / "base_encoder.pt"))
engine.attach(str(out_card))

units = probe_units_of(all_tasks()["relation"])
probe_res = run_probe(engine, "relation", units)

print("=== Probe Results ===")
print("Units:", probe_res["n_units"], "Cases:", probe_res["n_cases"])
print(f"class_acc: {probe_res['class_acc']*100:.2f}%")
print(f"span_acc: {probe_res['span_acc']*100:.2f}%")
if "pair_exact" in probe_res:
    print(f"pair_exact: {probe_res['pair_exact']*100:.2f}%")

with open(ROOT / "experiments" / "relation_bg" / "probe_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "train_metrics": data.get("metrics"),
        "probe": {
            "class_acc": probe_res["class_acc"],
            "span_acc": probe_res["span_acc"],
            "pair_exact": probe_res.get("pair_exact"),
            "n_units": probe_res["n_units"],
            "n_cases": probe_res["n_cases"],
        },
    }, f, indent=2, ensure_ascii=False)
