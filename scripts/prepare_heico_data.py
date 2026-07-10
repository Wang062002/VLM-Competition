"""Prepare HeiCo data for the ORena FOCUS SEGMENT baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from focus import FocusConfig, download, set_config
from focus.preprocessing import FrameExtractorPreprocessor, VideoTimestampOverlayPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="/data/focus", help="FOCUS root data directory.")
    parser.add_argument("--dataset", default="heico", help="FOCUS dataset name.")
    parser.add_argument("--max-workers", type=int, default=4, help="Preprocessing worker count.")
    parser.add_argument("--skip-overlay", action="store_true", help="Do not create timestamp overlay videos.")
    parser.add_argument("--skip-frames", action="store_true", help="Do not extract JPEG frames.")
    parser.add_argument(
        "--overlay-frames",
        action="store_true",
        help="If extracting frames, also extract frames from overlay videos into frames_overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    set_config(FocusConfig(root_dir=str(root_dir)))
    print(f"FOCUS root: {root_dir}")
    print(f"Dataset: {args.dataset}")

    print("Downloading dataset videos if needed...")
    download(args.dataset)

    if not args.skip_overlay:
        print("Creating timestamp overlay videos...")
        VideoTimestampOverlayPreprocessor().process(
            dataset=args.dataset,
            max_workers=args.max_workers,
        )
    else:
        print("Skipping timestamp overlay videos.")

    if not args.skip_frames:
        print("Extracting original video frames...")
        FrameExtractorPreprocessor(stride=1).process(
            dataset=args.dataset,
            max_workers=args.max_workers,
        )
        if args.overlay_frames:
            print("Extracting overlay video frames into frames_overlay...")
            set_config(FocusConfig(root_dir=str(root_dir), frames_folder="frames_overlay"))
            FrameExtractorPreprocessor(stride=1, use_overlay=True).process(
                dataset=args.dataset,
                max_workers=args.max_workers,
            )
            set_config(FocusConfig(root_dir=str(root_dir)))
    else:
        print("Skipping frame extraction.")

    print("Data preparation complete.")


if __name__ == "__main__":
    main()
