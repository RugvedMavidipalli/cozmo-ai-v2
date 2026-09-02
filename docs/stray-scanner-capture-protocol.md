# Stray Scanner -> cozmo-ai-v2 capture protocol

**Print this page. Follow every step in order.** The goal is one continuous, connected floor/area with the same device pose throughout. `[OFFICIAL]` means the app or its published data format says it; `[PROJECT]` is the capture rule for this pipeline.

## 1. Install and check the device

- `[OFFICIAL]` Install **Stray Scanner** (developer **Kenneth Blomqvist**, free) from the App Store. Grant Camera access. The App Store currently lists iOS 18.6 or later.
- `[OFFICIAL]` Use a LiDAR device only: **iPhone 12 Pro or 12 Pro Max and later Pro models**, or **iPad Pro 11-inch (2nd generation or later) / 12.9-inch (4th generation or later)**. Do not substitute a non-Pro iPhone or iPad. If the App Store refuses the iPad install, use an eligible iPhone.
- `[PROJECT]` Before leaving: check Settings > General > iPhone/iPad Storage and leave **at least 10 GB free**; charge to **80% or connect power**; close other apps; enable Do Not Disturb; set Auto-Lock to Never for the capture; silence calls/notifications.

## 2. Prepare the room

- `[PROJECT]` Turn on even room lighting. Avoid direct sun across walls, flicker, glare, and very dark corners. Open curtains/blinds so window jambs are visible; cover or flag mirrors and reflective glass. Do not use a mirror/glass surface as the only wall evidence.
- `[PROJECT]` Open every door fully and wedge it. Remove people and pets. Move small clutter out of the wall/floor view; **do not move furniture, doors, or curtains after recording starts**.
- `[PROJECT]` Tape a straight, non-reflective **measured reference marker** (preferably 2.00 m) where it can be seen face-on. Write its actual length above. Keep it fixed; show its full endpoints at both ends of the capture. This validates scale later; it does not silently calibrate the run.

## 3. Set the app and preflight the files

- Open **Record new session**. On the recording screen tap the fps control until it reads **60 fps**. `[PROJECT]` Record the full capture at 60 frames per second; the app also offers 30, 15, 5, and 1 fps. Keep the app in portrait.

## 4. Hold the device this way

- `[PROJECT]` Hold the device in **portrait**, screen toward you, rear camera/LiDAR toward the scene, top edge up, and **no roll**. Keep the lens at about **1.4 m above the floor** (within about 10 cm) for the whole run.
- `[PROJECT]` Keep the camera level for wall sweeps. At the named floor/ceiling pauses only, pitch about 15 degrees down/up and return to level. Never turn the device sideways or upside down.
- `[PROJECT]` Keep the target surface about **1-3 m** away where possible and never intentionally rely on a surface more than **3.5 m** away. Walk closer to large rooms. Keep camera and LiDAR windows uncovered.

## 5. Walk this exact route

`[PROJECT]` Use this route exactly:

1.  Tap Record.
2. Walk clockwise with the wall on your left. Advance no more than **0.5 m every 2 seconds**; pause **2 seconds at every corner**. Keep a visible wall band from roughly knee to head.
3. At each doorway: approach level; hold **2 seconds** with both jambs, header, and floor threshold visible; cross without changing height or portrait orientation; hold **2 seconds just inside**; then sweep that room clockwise.
4. In each room, capture every wall and corner. At the room centre hold **2 seconds level**, **2 seconds pitched down** to show a floor strip, and **2 seconds pitched up** to show the wall/ceiling edge. At every window hold **2 seconds face-on** showing both jambs plus sill and header.
5. Return through the same doorways and come back within **2 m of the start marker** after at least **4 seconds** have elapsed. Hold the marker motionless for **3 seconds**, then tap Stop. `[PROJECT]` This revisit gives the current pose-refinement code a chance to find loop closure.

## 6. Duration and split rules

- `[PROJECT]` Target **30-120 seconds** per connected floor/area. Hard operator cap: **5 minutes** for one continuous capture. This is a project safety rule, not an App Store limit; it limits accumulated drift, interruptions, and terminal frame loss.
- If the route will exceed 5 minutes, crosses floors, or enters a disconnected area, stop and start a new recording. **One pipeline run = one continuous coordinate frame; never join two folders.**

## 7. Avoid / restart

`[PROJECT]` **Stop, mark the folder BAD, fix the room, and restart from the marker** if any of these occurs: portrait/roll/orientation changes; a whip turn or motion blur; camera or LiDAR window covered; app interruption, call, lock, or recording button stops; AR view jumps or loses tracking; a room is missed; any required wall is beyond 3.5 m; a door/furniture/curtain moves; people or pets enter; or a mirror/glass surface is the only evidence. Capture solid adjacent surfaces and both jambs; flag unavoidable reflective spans for manual measurement. Do not resume a failed file.

## 8. Stop, export, and hand off

- `[OFFICIAL]` After Stop, open the recording detail and tap **Share**. Prefer **Save to Files** or AirDrop and keep the app-created ZIP unchanged. For an original folder, connect by cable: Finder > device > **Files** > **Stray Scanner**, then drag the dataset folder; on Windows use iTunes. Files app route: Browse > On My iPhone/iPad > Stray Scanner > Share/Save to Files.
- Name only the delivered top-level capture folder `site_floor_area_YYYYMMDD-HHMM`; retain the app hash in the handoff note. Do not rename children, rotate video, transcode video, resize PNGs, reorder files, or rearrange folders. `distortion/` is optional; everything else above is required.

## 9. Process on current main

From the repository root, after `uv sync`, run the current-main command (verified 2026-09-02):

```console
uv run python -m cozmo_ai_v2.pipeline run "/path/to/CAPTURE" \
  --out "out/CAPTURE" --no-damage
```

Expected outputs include `result.json`, `floorplan.svg`, `scene.glb`, `cloud.ply`, `mesh.ply`, `planes.json`, `fusion_manifest.json`, and scope/openings CSV files. **Recapture** for `terminal sidecar frame(s) unavailable`, `requested frame(s) rejected`, low IMU gravity consistency, `floor plane was not observed`/low confidence, `no ceiling plane found`, no rooms, or room-overlap warnings. `confidence intervals are uncalibrated` and `known reference ... not applied` are review/scale notes, not capture failures. The run defaults to confidence >=1 and depth <=3.5 m; invalid depth is discarded.

## 10. Final 20-second pass/fail

- `[ ] PASS` LiDAR device, 10 GB free, 80%/power, no interruptions
- `[ ] PASS` 60 fps full capture, portrait/no roll, lens about 1.4 m high, 1-3 m from surfaces
- `[ ] PASS` marker measured and seen at start and finish; all rooms/doors/corners/windows covered
- `[ ] PASS` floor strip and ceiling edge shown; no people/pets/moving furniture; revisit completed
- `[ ] PASS` folder/ZIP is original, required names/counts match, checksum recorded
- `[ ] PASS` pipeline output has no recapture warning

### Evidence and sources (checked 2026-09-02)

Official app facts: [App Store listing](https://apps.apple.com/us/app/stray-scanner/id1557051662) (name, developer, iOS compatibility, LiDAR description, fps/PNG/constant-rate history); [official data format](https://github.com/strayrobots/scanner/blob/main/docs/format.md) (file names, indexing, units); [official export guide](https://github.com/strayrobots/scanner/blob/main/docs/export.md); [official record screen source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Controllers/RecordSessionViewController.swift), [share/ZIP source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Helpers/ShareUtility.swift), and [portrait/device capability source](https://github.com/strayrobots/scanner/blob/main/StrayScanner/Info.plist). Apple's current LiDAR model list is [here](https://support.apple.com/en-sg/121825). Project contracts and behavior: [`ingest.py`](../src/cozmo_ai_v2/pipeline/ingest.py), [`frame_contract.py`](../src/cozmo_ai_v2/pipeline/frame_contract.py), [`poses.py`](../src/cozmo_ai_v2/pipeline/poses.py), [`cli.py`](../src/cozmo_ai_v2/pipeline/cli.py), and the tested fixture in [`tests/conftest.py`](../tests/conftest.py).
