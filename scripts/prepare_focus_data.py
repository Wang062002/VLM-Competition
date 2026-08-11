"""Download and preprocess a FOCUS dataset.

This is the generic version of `prepare_heico_data.py`; keep the old script for
backward compatibility, but use this one for HeiCo + LapChole workflows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from focus import FocusConfig, download, set_config
from focus.preprocessing import FrameExtractorPreprocessor, VideoTimestampOverlayPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="/mnt/data/jiali_wang/focus")
    parser.add_argument("--dataset", required=True, help="FOCUS dataset name, e.g. heico or lapchole.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument("--skip-frames", action="store_true")
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

    if not args.skip_download:
        print("Downloading dataset videos if needed...")
        download(args.dataset)
    else:
        print("Skipping dataset download.")

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
