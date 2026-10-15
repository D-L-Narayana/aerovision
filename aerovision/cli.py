import argparse, json
import numpy as np
from PIL import Image
from .pipeline import AeroVision
from pathlib import Path

def main():
    ap = argparse.ArgumentParser("aerovision")
    ap.add_argument("--rgb", help="aerial RGB image")
    ap.add_argument("--thermal", help=".npz of radiometric frames (°C)"); ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--probe", default=str(Path(__file__).resolve().parent.parent / "web" / "probe_head.json"))
    a = ap.parse_args()
    av = AeroVision(probe_json=Path(a.probe))
    rgb = Image.open(a.rgb) if a.rgb else None
    fc = np.load(a.thermal)["frames"][a.frame] if a.thermal else None
    print(json.dumps(av.inspect(rgb, fc).model_dump(), indent=2))

if __name__ == "__main__":
    main()
