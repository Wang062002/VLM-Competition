"""Check ORena FOCUS dataset loading and optional video sample generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from focus import FocusConfig, set_config
from focus.data.base_dataset import FocusDataset
from focus.data.video_dataset import FocusVideoDataset
from focus.enums import DatasetSplit, Track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="/data/focus", help="FOCUS root data directory.")
    parser.add_argument("--dataset", default="heico", help="FOCUS dataset name.")
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--track", default="segment", choices=["frame", "segment", "procedure"])
    parser.add_argument("--make-video-sample", action="store_true")
    parser.add_argument("--no-overlay", action="store_true", help="Use original videos instead of overlay videos.")
    parser.add_argument("--video-stride", type=int, default=25)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    set_config(FocusConfig(root_dir=str(root_dir)))

    dataset = FocusDataset(
        dataset=args.dataset,
        split=DatasetSplit(args.split),
        track=Track(args.track),
    )
    print(dataset)
    print(f"Sample count: {len(dataset)}")
    print(f"Unique videos: {len(dataset.video_ids())}")

    request, reference = dataset[0]
    print("First qID:", request.qID)
    print("First videoID:", request.videoID)
    print("First time window:", request.start_time, request.end_time)
    print("First question:", request.question)
    print("First reference format:", reference.format.type)
    print("First reference answer:", reference.answer)

    if args.make_video_sample:
        video_dataset = FocusVideoDataset(
            dataset,
            stride=args.video_stride,
            use_overlay=not args.no_overlay,
            resolution=(args.width, args.height),
        )
        sample = video_dataset[0]
        try:
            size = sample.video_path.stat().st_size
            print("Generated clip:", sample.video_path)
            print("Generated clip size:", size)
            print("Clip fps:", sample.fps)
            if size <= 0:
                raise RuntimeError("Generated clip is empty.")
        finally:
            sample.video_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
