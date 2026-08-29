
# Raspberry Pi Stringer Natural-Image Reward Conditioning

This repository contains the open-loop natural-image reward-conditioning task
using the lab's `rpg` framebuffer path. The screen is controlled directly from
the headless behavior Pi rather than through a desktop/X11 session.

The supported experiment entrypoint in this repository is:

```text
run_stringer_reward_conditioning.py
```

`run_stringer_vstim.py` and `run_stringer_vstim_cam.py` are retained only for
imports used by the reward runner and refuse direct execution. Natural-image-only
experiments are maintained at <https://github.com/hung-lo/vstim_natural>.

The supported runner:

- asks for the mouse/session settings and shows a final setup summary
- uses the fixed 14-image assignment and precomputed 90% reward schedule
- displays each stimulus as a 1.0 s plus 0.5 s RPG sequence
- triggers reward at the 1.0 s boundary, independent of licking
- applies scheduled suction at its open-loop onset on rewarded and omitted cues
- records licks, display timing, QC, camera archival state, and completion status
- reports live WAITING/PRE/TASK/POST status with realized-ITI ETA

## Default timing

The current default stimulus timing is:

- image on-screen time: `1.5` seconds (`1.0 + 0.5` RPG segments)
- ITI time: configured realized uniform sequence
- PRE and POST gray backgrounds: configured independently

## Expected paths on the Pi

The script looks for the PNG folder in these places:

```text
~/vstim_natural_img_reward/stringer_natimg2800_center_crop_png
~/stringer_natimg2800_center_crop_png
./stringer_natimg2800_center_crop_png
```

Session output goes to:

```text
/mnt/hd/<mouse_id>_<YYYYMMDDThhmmssZ>_vstim_natural_reward_conditioning/
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
python3 run_stringer_reward_conditioning.py
```

Before hardware use, run the GPIO-only safety smoke test:

```bash
python3 fullscreen_test.py
```

Camera recording, transfer, conversion, and recovery are selected through the
reward-conditioning runner's CLI or interactive prompts.

## Photodiode patch

The photodiode patch is built into the session raw files. The current default
patch is smaller than the first pass version. To adjust it, set:

```python
ENABLE_PHOTODIODE_PATCH = True
```

The helper functions already bake the patch into the session raw files.

### Box151 display-timing calibration

The Box151 profile is an optional, hardware-specific RPG timing calibration.
It was measured with a 20 kS/s Intan recording using ADC1 for the display
photodiode and ADC3 for IRIG-H, across 50 no-mouse trials at 60 Hz. It is not
a universal monitor or refresh-rate correction. The calibration uses 64 RPG
refreshes before the reward boundary and 24 after it, with a `0.095` s
stimulus-onset compensation for suction scheduling. Intan/photodiode timing
remains the physical ground truth.

When the calibration fields are null, the runner falls back to nominal 60/30
refreshes at the configured 60 Hz base rate and zero compensation. To use the
Box151 profile, set these fields in the local, uncommitted
`reward_conditioning_config.json` (after copying the example):

```json
"display_timing_calibration_id": "box151_photodiode_20ksps_50trial_60hz_v1",
"display_timing_calibration_refresh_rate_hz": 60.0,
"stim_segment1_refreshes": 64,
"stim_segment2_refreshes": 24,
"stimulus_onset_compensation_sec": 0.095
```

The behavioral targets remain a 1.0 s reward boundary, 1.5 s planned
stimulus, and the configured physical suction delay. The calibration changes
only the programmed raw refresh counts and the software suction target; the
trial summary records both the software-aligned and compensation-adjusted
suction delays.

## Notes on the display backend

The older pygame approach was a dead end for the headless behavior Pi.
The rpg path is the one the lab code already uses for framebuffer display,
and it is the right place to put this stimulus runner.

## Reward-conditioning runner (supported entrypoint)

The new open-loop reward task lives in `run_stringer_reward_conditioning.py`.
It uses the same `rpg` framebuffer path, but adds fixed-probability reward
delivery for a selected subset of images.

Launch it from the repo root:

```bash
cd ~/vstim_natural_img_reward
python3 run_stringer_reward_conditioning.py
```

This repository supports the reward-conditioning runner above. The copied
natural-image-only runners are retained only as importable compatibility
modules and refuse direct execution. For the natural-image-only protocol, use
the maintained repository at <https://github.com/hung-lo/vstim_natural>.

Before the first hardware run, copy the template config and fill in the
already calibrated solenoid pulse width from the working Go/NoGo setup:

```bash
cp reward_conditioning_config.example.json reward_conditioning_config.json
```

Do not guess `reward_pulse_on_sec`. Use the calibrated
`session_info["solenoid_blink_duration"]` value from the working box.

Reward is triggered at 1.0 s after image onset, and the visual image always
ends at 1.5 s. A calibrated reward train longer than the remaining 0.5 s may
continue into the gray ITI, but it must finish before the scheduled suction
onset. For example, the Box151 configuration of 6 pulses with 0.100 s on and
0.010 s between pulses produces a 0.650 s train that nominally ends at 1.650 s;
these pulse values are a Box151 example, not universal defaults. Physical
timing should be confirmed from Intan/photodiode/valve recordings.

Persistent assignments are saved under `/mnt/hd/vstim_reward_assignments/`:

- `_global_reward_conditioning_14_image_panel.json` for the cohort-wide image panel
- `<mouse_id>_reward_conditioning_assignment.json` for the per-mouse role mapping

These are persistent longitudinal protocol files. They are created atomically
under a filesystem lock and reused on future sessions. Do not delete or
manually edit them after training or data collection begins. `force_new=True`
is an explicit exceptional operation for deliberately replacing a mouse's
assignment; it does not replace the cohort-wide panel.

Session output is written under:

```text
/mnt/hd/<mouse_id>_<YYYYMMDDThhmmssZ>_vstim_natural_reward_conditioning/
```

The main files there are:

- `session_manifest.json`
- `<session>_event_log.csv`
- `<session>_planned_sequence.csv`
- `<session>_trial_summary.csv`
- `<session>_image_assignment.csv`
- `<session>_plan_summary.csv`
- `<session>_metadata.json`
- `<session>_session_qc.json`
- `raw_cache/` with the baked RPG raws

When calibrated volume fields are supplied in the hardware config, the runner
enforces `reward_volume_ul_per_train` and `maximum_session_reward_ul` before
hardware starts, includes manual test rewards in the allowance, and records
planned, delivered, and estimated reward volume in metadata/QC. Null values
explicitly disable the volume cap and volume estimate.

### Operator flow and live status

The reward-conditioning runner resolves CLI or interactive session inputs,
shows the final setup summary, and asks for preparation confirmation before it
creates the session. The operator may then run manual reward and suction tests.
If selected, the face camera is started and verified before the two-photon
gate.

The experimental sequence is:

1. `WAITING FOR 2P`: start two-photon acquisition, then press Enter.
2. `PRE`: live gray-background countdown.
3. `TASK`: live trial progress and exact planned task ETA.
4. `POST`: live gray-background countdown.
5. Stop the camera and freeze the displayed REC elapsed time.
6. Transfer and verify raw video, convert to MP4, and verify it with `ffprobe`.
7. Report the final session status and output paths.

Representative status lines are:

```text
REC 00:01:42 | WAITING FOR 2P | press Enter when acquisition is running | planned after Enter 00:52:18
REC 00:03:10 | PRE 01:50 remaining | protocol remaining 00:48:31 | finish ~14:58:22
REC 00:23:14 | TASK 221/500 (44.2%) | planned task remaining 00:24:08 | +POST 00:05:00 | finish ~15:32:11
REC 00:49:23 | POST 03:17 remaining | finish ~15:27:04
REC stopped 00:52:41 | camera stop confirmed
```

`WAITING FOR 2P` has no finish prediction. Planned ETA excludes the operator
gate and video transfer/conversion; TASK ETA uses the exact realized ITI
sequence and excludes the skipped final ITI. REC is a controller-side
operational timer, not a measurement of remote physical camera frames.
Photodiode/DAQ remains the ground truth for neural alignment.

### Control Pi telemetry

The runner sends optional, read-only, best-effort UDP telemetry to the Control
Pi at `192.168.1.150:5055` by default. Network I/O and JSON encoding run in a
separate process; the experiment hands off small pickled messages through a
bounded, nonblocking `AF_UNIX SOCK_DGRAM` socketpair, so monitor availability,
dropped packets, and local IPC pressure cannot delay stimulus, reward, suction,
GPIO processing, or session finalization. Disable it with:

```bash
python3 run_stringer_reward_conditioning.py --no-telemetry
```

Override the destination with `--telemetry-host` and `--telemetry-port`, or
with `RIG_MONITOR_HOST` and `RIG_MONITOR_PORT`. CLI values take precedence.
To characterize parent-side handoff overhead on Box151, run at least 1,000
calls against the live telemetry worker:

```bash
python3 benchmark_rig_telemetry_publish.py --host 192.168.1.150 --port 5055 --count 1000
```

The diagnostic reports median, p95, and maximum publish-call duration in
microseconds, plus dropped calls and parent errors. It is not part of normal
experiment runs.
Telemetry phases include `WAITING_FOR_2P`, `PRE`, `STIMULUS`, `ITI`, `POST`,
and `COMPLETE`; dashboard timestamps are operator telemetry, not scientific
synchronization. Photodiode/DAQ remains the physical visual-timing ground
truth.

`reward_volume_ul_per_train` is the current per-train volume estimate or
calibration. `task_water_delivered_ul_session` equals verified task rewards
times that volume, while `task_water_likely_consumed_ul_session` equals
verified rewards with a lick between actual valve-on and suction-on times,
times that volume. “Likely consumed” is a heuristic, not a direct volumetric
measurement. Manual test rewards are not included in task counters.

The metadata retains `session_completed` for compatibility. Its more detailed
`session_status` is also copied to `session_manifest.json` and can be:
`complete`, `interrupted`, `failed`, `protocol_complete_video_pending`,
`protocol_complete_camera_cleanup_failed`, `cleanup_failed`, or `incomplete`.

### Camera recovery and integrity

The reward-conditioning controller keeps its own camera state file and session
namespace. Its safe workflow is: remote camera recording → confirmed stop →
remote raw H.264 manifest (size + SHA-256) → resumable local copy without
deleting the remote source → local size/hash verification → temporary MP4
conversion → `ffprobe` validation → exact remote raw cleanup. Local `.h264`
files are retained after conversion for recovery.

If transfer or conversion fails, both local raw files and remote raw files are
kept. Run `remote_camera_control.py fetch` or `convert` again to recover; a
valid existing MP4 is reused only after `ffprobe` validation. Ctrl-C performs
cleanup and exits with status 130. An explicit camera-cleanup exception exits
nonzero even when TASK and POST completed.

Recommended test sequence:

```bash
python3 test_reward_conditioning_protocol.py
python3 test_reward_conditioning_runtime.py
python3 smoke_test_reward_gpio.py
python3 test_remote_camera_control.py
```

After that, run one simulated session with `--simulate-gpio` before touching
the valve or lick hardware. On the Pi, confirm the GPIO pins are wired as
BCM19 for reward, BCM25 for suction, and BCM26 for lick detection.

For conditioned-image trials, reward or omission occurs at 1.0 s after image
onset, gray begins at 1.5 s, and suction begins at 3.5 s after onset on both
rewarded and omitted conditioned-cue trials. Suction and reward are open-loop
and never depend on licking. Lick onset and offset events remain in the main
event log and are also exported to `<session>_lick_events.csv`.
