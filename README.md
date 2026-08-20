# AirPods → Birthday Surprise Detector

When a specific pair of AirPods Pro connects, a webcam window opens, finds
people in frame, and labels the enrolled person "Birthday girl". After 10
consecutive matching frames it plays a video.

Built for Apple Silicon: detection uses Apple's **Vision** framework (Neural
Engine, no model download, negligible battery cost). Only face *recognition*
needs a downloaded model, because macOS exposes no public face-embedding API.

## Layout

| Path | Role |
|---|---|
| [src/bt_watcher.py](src/bt_watcher.py) | Polls Bluetooth, launches the detector on connect |
| [src/detector.py](src/detector.py) | Webcam loop: detect → recognize → draw → trigger |
| [src/enroll.py](src/enroll.py) | Turns reference photos into stored embeddings |
| [src/vision_mac.py](src/vision_mac.py) | Vision framework wrappers (person, face, feature print) |
| [src/embedder.py](src/embedder.py) | Pluggable face-embedding backends |
| [config.json](config.json) | All tunables |
| [launchagent/watcher.plist.template](launchagent/watcher.plist.template) | Login LaunchAgent |

## Setup

Already done on this machine: `uv`, a Python 3.12 venv in `.venv/`, all
dependencies including the facenet model weights. To rebuild from scratch:

```bash
./install.sh --with-facenet
```

### 1. Grant camera permission — do this first

This is the one step that cannot be automated, and it is currently **not
granted**:

```bash
./run.sh detect --selftest
```

Run it from your own Terminal (or the VS Code terminal) and approve the
prompt. If no prompt appears, add your terminal app under **System Settings →
Privacy & Security → Camera** manually. You should see `camera OK, frame
1280x720`.

### 2. Add reference photos

Put **5–15 photos** of the birthday girl in `enroll_photos/`. Each must
contain exactly one clearly visible face — photos with several faces are
skipped rather than guessed at.

Vary them: different lighting, angles, expressions, hair. The spread across
these photos is what decides how forgiving recognition is; six near-identical
burst shots produce a brittle model, and `enroll` will warn you if it sees
that.

### 3. Add the video

Put the surprise video at `media/surprise.mp4` (or point `video.path` in
[config.json](config.json) elsewhere).

### 4. Enroll

```bash
./run.sh enroll
```

This prints the distance spread across your photos and a suggested
`match_threshold`. Set it in [config.json](config.json), or leave
`match_threshold` as `null` to use the backend default (0.9 for facenet).

### 5. Try it live

```bash
./run.sh detect
```

Each box shows its live distance `d=…`. Lower means more similar. Watch the
number for the birthday girl versus other people and set the threshold between
the two clusters. `q` or `Esc` quits.

### 6. Install the login watcher

```bash
./install.sh --launchagent
```

Starts at login, survives crashes, logs to `logs/watcher.{out,err}.log`.

```bash
launchctl print gui/$UID/com.shreya.birthday-surprise.watcher   # inspect
launchctl bootout gui/$UID/com.shreya.birthday-surprise.watcher # remove
```

## How the trigger works

Vision returns person boxes and face boxes in one pass. Each face is embedded
and compared to the enrolled set; a person box is labelled "Birthday girl"
when a matching face falls inside it. A face with no enclosing person box
still counts on its own — Vision often drops the body box when someone leans
close to the camera, which is exactly when you least want a miss.

Any frame with at least one match increments the streak; any frame without one
resets it to zero. At `consecutive_frames` (10) the match is "confirmed" and
the streak resets, but the video doesn't play yet -- the live camera feed
keeps running with boxes on screen for `pre_trigger_pause_seconds` (10), then
the video plays and `cooldown_seconds` (60) blocks re-triggering. Once
confirmed, playback is committed to; losing the match during the pause
doesn't cancel it.

## Recognition backends

| Backend | What it is | Accuracy |
|---|---|---|
| `facenet` | InceptionResnetV1 / VGGFace2, 512-d | Good — a real face-identity embedding. **Installed and default.** |
| `vision` | `VNGenerateImageFeaturePrintRequest`, 768-d | Weak for this purpose — see below |

The `vision` backend needs no extra dependencies and runs on the ANE, but it
is a *generic image descriptor*, not a face descriptor: it encodes hair,
lighting and background alongside the face, so it will confuse two people who
look broadly similar in similar light. It exists as a zero-install fallback.
Prefer `facenet`.

Embeddings from the two backends are not comparable. The detector refuses to
run if `config.json` selects a different backend than the one used to enroll,
rather than silently producing nonsense distances.

## Tuning

All in [config.json](config.json):

| Key | Default | Meaning |
|---|---|---|
| `bluetooth.device_address` | `38:c4:3a:64:1f:02` | MAC address of her AirPods (preferred match — see below). |
| `bluetooth.device_name_contains` | `Shreya's Airpods` | Fallback name substring, used only if `device_address` is unset. |
| `bluetooth.poll_seconds` | `3` | Poll interval |
| `bluetooth.stop_detector_on_disconnect` | `true` | Kill the detector when the AirPods drop |
| `detector.consecutive_frames` | `10` | Matching frames needed to confirm |
| `detector.pre_trigger_pause_seconds` | `10` | Live feed keeps running this long after confirming, before the video plays |
| `detector.match_threshold` | `null` | Distance cutoff; `null` = backend default |
| `detector.cooldown_seconds` | `60` | Silence after a trigger |
| `detector.mirror` | `true` | Flip the preview like a mirror |
| `detector.min_person_confidence` | `0.5` | Vision person-box cutoff |

**AirPods name keeps changing / two people's AirPods both match**: match by
`device_address` instead of name. AirPods' display name on macOS can drift
back to the generic "AirPods Pro - Find My" independent of any custom name
set on the phone, and a name substring can't tell two similar AirPods apart.
Find the address with:
```bash
system_profiler SPBluetoothDataType | grep -B2 "Address:"
```
Set it in `config.json` as `bluetooth.device_address` (colon or dash
separated, case-insensitive) — it takes priority over `device_name_contains`
whenever both are set.

**Too many false positives** (strangers labelled): lower `match_threshold`.
**She isn't recognized**: raise it, and add more varied enrollment photos.
At ~15–20 fps, 10 frames is under a second — raise `consecutive_frames` if it
feels twitchy.

## Verified

Confirmed working on this machine (macOS 26.1, arm64):

- Vision person + face detection, and the numpy→CGImage bridge
- Both embedding backends (512-d facenet, 768-d feature print, both normalized)
- Enrollment end-to-end, including the threshold report
- Streak logic: 6 matches → reset by 2 blanks → 12 matches fires exactly once
- Backend-mismatch and missing-embeddings guards
- Bluetooth detection via both IOBluetooth (~0.2s) and `system_profiler`,
  agreeing with each other against your real AirPods
- Watcher transitions: one spawn per connect, one stop per disconnect
- OpenCV GUI windows

Not yet verified, because they need a person and a granted camera:

- Live camera capture (permission is currently **denied** — step 1)
- Real-world recognition accuracy and the right threshold for her face

## Known gotchas

- **LaunchAgent camera permission.** A process started by `launchd` is
  attributed to the venv's `python` binary, not to your terminal, so it may
  need its own approval. Run step 1 from Terminal first, and check **Privacy &
  Security → Camera** after the first automatic launch.
- **HEIC photos.** iPhone HEICs may not decode. If `enroll` reports
  "unreadable", convert first:
  `sips -s format jpeg in.heic --out out.jpg`
- **AirPods connect on lid-open.** They reconnect whenever they come out of
  the case near the Mac, so the surprise can fire earlier than you intend.
  Pair a *different* set for the trigger, or keep them in the case until the
  moment.
