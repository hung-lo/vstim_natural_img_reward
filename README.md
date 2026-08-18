
# Raspberry Pi Stringer Natural-Image Visual Stimulus Script

This repo now uses the lab's `rpg` framebuffer path again, not pygame.
That matters for the headless behavior Pi, because the screen is controlled
directly from the Pi framebuffer rather than through a desktop/X11 session.

Current entrypoints:

```text
run_stringer_vstim.py
run_stringer_vstim_cam.py
run_stringer_reward_conditioning.py
```

What the runtime does:

- asks for mouse ID and optional session notes
- asks for number of images and repeats, using the current defaults when you just press Enter
- confirms the session settings and estimated playback time before starting
- chooses a reproducible subset of Stringer center-crop PNGs
- bakes session-specific `rpg` raw files at startup
- displays each image fullscreen for a fixed duration
- shows gray ITI between images
- prints a terminal progress line with percent and ETA during playback
- optionally bakes a photodiode patch into the frames
- logs display-request timestamps for `stim_on` and `iti_on` while the photodiode remains the ground truth for physical onset
- with the camera wrapper, keeps the monitor on the ITI-style gray frame while the camera is started and confirmed, records a pre-stimulus baseline while stimulus raws are prepared, then holds a black post-stimulus screen until you confirm the camera stop/fetch step
- logs planned sequence, request timestamps, baseline metadata, and session metadata

## Default timing

The current default stimulus timing is:

- image on-screen time: `0.5` seconds
- ITI time: uniform `3.0–4.5` seconds after image offset
- initial gray screen: `3.0` seconds
- final gray screen: `3.0` seconds

## Expected paths on the Pi

The script looks for the PNG folder in these places:

```text
~/vstim_natural_img_reward/stringer_natimg2800_center_crop_png
~/stringer_natimg2800_center_crop_png
./stringer_natimg2800_center_crop_png
```

Session output goes to:

```text
/mnt/hd/<mouse_id>_<YYYYMMDDThhmmssZ>_vstim_natural/
```

## Dependencies

The runtime expects:

- `rpg` installed from the SjulsonLab rpg repository
- `Pillow`
- `gpiozero`
- `RPi.GPIO` on the Pi if GPIO output is enabled
- `ffmpeg` on box 151 if you want automatic `.h264` to `.mp4` conversion

Example rpg install on the Pi:

```bash
cd ~
git clone https://github.com/SjulsonLab/rpg
cd rpg
sudo pip3 install .
```

Or, if the repo is already present somewhere else, install it from that checkout.

Install the Python packages used by this repo:

```bash
pip3 install Pillow
pip3 install gpiozero
pip3 install RPi.GPIO
```

## Running it

Run the script from the behavior Pi itself, ideally from a local TTY or a plain
SSH shell on the behavior Pi. Do not use X-forwarded sessions for this.

```bash
cd ~/vstim_natural_img_reward
python3 run_stringer_vstim.py
```

If the basic framebuffer path looks good, try the smoke test:

```bash
python3 fullscreen_test.py
```

If you want camera recording plus automatic file transfer and conversion, use:

```bash
python3 run_stringer_vstim_cam.py
```

That wrapper now asks for a pre-stimulus camera baseline in minutes, then lets
you type `y` and press Enter to start early after the camera process is alive
and its session-specific `.h264` file is confirmed to be growing. The baseline
clock starts from that output-growth confirmation. After the stimulus sequence
finishes, it leaves the screen black until camera stop is verified by both the
tracked PID and the session-specific acquisition-process check. The PID file is
removed only after that verification, and the screen remains black during file
transfer or while a failed transfer is being resolved.

Keep `run_stringer_vstim.py` around as the plain no-camera runner and as the
baseline path if you want to debug the display flow independently.

## Photodiode patch

The photodiode patch is built into the session raw files. The current default
patch is smaller than the first pass version. To adjust it, set:

```python
ENABLE_PHOTODIODE_PATCH = True
```

The helper functions already bake the patch into the session raw files.

## Notes on the display backend

The older pygame approach was a dead end for the headless behavior Pi.
The rpg path is the one the lab code already uses for framebuffer display,
and it is the right place to put this stimulus runner.

## Reward-conditioning runner

The new open-loop reward task lives in `run_stringer_reward_conditioning.py`.
It uses the same `rpg` framebuffer path, but adds fixed-probability reward
delivery for a selected subset of images.

Launch it from the repo root:

```bash
cd ~/vstim_natural_img_reward
python3 run_stringer_reward_conditioning.py
```

Before the first hardware run, copy the template config and fill in the
already calibrated solenoid pulse width from the working Go/NoGo setup:

```bash
cp reward_conditioning_config.example.json reward_conditioning_config.json
```

Do not guess `reward_pulse_on_sec`. Use the calibrated
`session_info["solenoid_blink_duration"]` value from the working box.

Persistent assignments are saved under `/mnt/hd/vstim_reward_assignments/`:

- `_global_reward_conditioning_14_image_panel.json` for the cohort-wide image panel
- `<mouse_id>_reward_conditioning_assignment.json` for the per-mouse role mapping

Session output is written under:

```text
/mnt/hd/<mouse_id>_<YYYYMMDDThhmmssZ>_vstim_natural_reward_conditioning/
```

The main files there are:

- `<session>_event_log.csv`
- `<session>_planned_sequence.csv`
- `<session>_trial_summary.csv`
- `<session>_image_assignment.csv`
- `<session>_plan_summary.csv`
- `<session>_metadata.json`
- `<session>_session_qc.json`
- `raw_cache/` with the baked RPG raws

Recommended test sequence:

```bash
python3 test_reward_conditioning_protocol.py
python3 test_reward_conditioning_runtime.py
python3 smoke_test_reward_gpio.py
```

After that, run one simulated session with `--simulate-gpio` before touching
the valve or lick hardware. On the Pi, confirm the GPIO pins are wired as
BCM19 for reward, BCM25 for suction, and BCM26 for lick detection.

For conditioned-image trials, reward or omission occurs at 1.0 s after image
onset, gray begins at 1.5 s, and suction begins at 3.5 s after onset on both
rewarded and omitted conditioned-cue trials. Suction and reward are open-loop
and never depend on licking. Lick onset and offset events remain in the main
event log and are also exported to `<session>_lick_events.csv`.
