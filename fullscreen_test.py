#!/usr/bin/env python3
"""
fullscreen_test.py

rpg-based screen test for the behavior Raspberry Pi.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw

SCREEN_RESOLUTION = (1024, 600)
SCREEN_BACKGROUND_GRAY = 127
SCREEN_COLORMODE = 16
REFRESH_RATE_HZ = 60


def make_test_frame(source_path):
    canvas = Image.new("RGB", SCREEN_RESOLUTION, (SCREEN_BACKGROUND_GRAY,) * 3)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((SCREEN_RESOLUTION[0] - 140, 0, SCREEN_RESOLUTION[0] - 1, 140), fill=(255, 255, 255))
    draw.rectangle((0, 0, 220, 220), fill=(0, 0, 255))
    draw.rectangle((220, 0, 440, 220), fill=(255, 0, 0))
    draw.rectangle((0, 220, 220, 440), fill=(0, 255, 0))
    with source_path.open("wb") as handle:
        handle.write(canvas.tobytes())
    return source_path


def main():
    try:
        import rpg
    except ImportError as exc:
        raise RuntimeError("The rpg package is not installed.") from exc

    print("Starting rpg fullscreen test.")
    print("Screen resolution: %s" % (SCREEN_RESOLUTION,))
    print("Display environment: DISPLAY=%s" % os.environ.get("DISPLAY"))

    with rpg.Screen(SCREEN_RESOLUTION, background=SCREEN_BACKGROUND_GRAY, colormode=SCREEN_COLORMODE) as screen:
        screen.display_greyscale(SCREEN_BACKGROUND_GRAY)
        time.sleep(1.0)
        screen.display_greyscale(255)
        time.sleep(1.0)
        screen.display_greyscale(0)
        time.sleep(1.0)
        screen.display_greyscale(SCREEN_BACKGROUND_GRAY)
        time.sleep(1.0)

        with tempfile.TemporaryDirectory(prefix="vstim_rpg_test_") as tempdir:
            tempdir_path = Path(tempdir)
            source_path = tempdir_path / "test.rgb.raw"
            raw_path = tempdir_path / "test.raw"
            make_test_frame(source_path)
            rpg.convert_raw(
                str(source_path),
                str(raw_path),
                1,
                SCREEN_RESOLUTION[0],
                SCREEN_RESOLUTION[1],
                REFRESH_RATE_HZ,
                SCREEN_COLORMODE,
            )
            raw = screen.load_raw(str(raw_path))
            screen.display_raw(raw)
            time.sleep(1.0)

        screen.display_greyscale(SCREEN_BACKGROUND_GRAY)
        time.sleep(1.0)

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise
