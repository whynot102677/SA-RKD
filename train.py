"""Run SA-RKD with a YOLO detection dataset or the bundled synthetic example."""

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from src.trainer import SARKDTrainer

ROOT = Path(__file__).resolve().parent
KD_IDS = ("A1", "A2", "A3", "A4")


def train(experiment, weights, options, teacher=None):
    model = YOLO(str(weights))
    if experiment in KD_IDS:
        config = yaml.safe_load((ROOT / "configs/distillation.yaml").read_text(encoding="utf-8"))
        SARKDTrainer.kd_options = dict(config, variant=experiment)
        SARKDTrainer.teacher_path = teacher
        model.train(trainer=SARKDTrainer, name=experiment, **options)
    else:
        model.train(name=experiment, **options)

    checkpoint = model.trainer.best
    if not checkpoint.is_file():
        raise RuntimeError(f"No best checkpoint was saved for {experiment}: {checkpoint}")
    print(f"{experiment} checkpoint: {checkpoint}")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", choices=("T0", "A0", *KD_IDS), default="A3")
    parser.add_argument("--smoke", action="store_true", help="Run a one-epoch CPU demo on synthetic images")
    parser.add_argument("--data", type=Path, help="YOLO dataset YAML")
    parser.add_argument("--weights", help="Student or teacher initialization, according to --id")
    parser.add_argument("--teacher", type=Path, help="Best checkpoint from T0")
    parser.add_argument("--device", default="0", help="Single GPU index or cpu")
    parser.add_argument("--project", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if not args.smoke:
        if args.data is None or not args.data.is_file():
            parser.error("--data must point to an existing YOLO dataset YAML")
        if args.id in KD_IDS and (args.teacher is None or not args.teacher.is_file()):
            parser.error("--teacher must point to an existing T0 checkpoint")
    elif any((args.data, args.weights, args.teacher)):
        parser.error("--smoke uses the bundled dataset and creates its own checkpoints")

    torch.set_num_threads(4)
    options = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8"))
    options.update(project=str(args.project.resolve()), exist_ok=False)
    if args.smoke:
        options.update(
            data=str(ROOT / "dataset/example.yaml"), device="cpu", imgsz=128,
            epochs=1, batch=2, workers=0, amp=False, mosaic=0.0,
            close_mosaic=0, plots=False,
        )
        teacher = train("T0", "yolo11s.yaml", options) if args.id in KD_IDS else None
        model_yaml = "yolo11s.yaml" if args.id == "T0" else "yolo11n.yaml"
        train(args.id, model_yaml, options, teacher)
        return

    options.update(
        data=str(args.data.resolve()), device=args.device, batch=16, workers=8,
        amp=args.device.lower() != "cpu",
    )
    weights = args.weights or ("yolo11s.pt" if args.id == "T0" else "yolo11n.pt")
    train(args.id, weights, options, args.teacher.resolve() if args.teacher else None)


if __name__ == "__main__":
    main()
