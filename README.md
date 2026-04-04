# Camera Coverage and Planning Tool

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#python-version-support)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](#requirements)
[![Package Layout](https://img.shields.io/badge/layout-src-green.svg)](#project-layout)

<!-- COVER IMAGE PLACEHOLDER -->
<!-- Replace with your screenshot when ready -->
<!-- ![Camera Viewer Cover](docs/images/cover.png) -->

Interactive multi-camera planning tool for coverage and defect detectability analysis, with in-app COLMAP point cloud generation from video.

## Table of Contents

- [Features](#features)
- [Python Version Support](#python-version-support)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Managed Projects](#managed-projects)
- [COLMAP Workflow Setup (gitignored)](#colmap-workflow-setup-gitignored)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

## Features

- Multi-camera frustum and marker visualization
- Mesh and point-cloud object support
- Coverage computation:
  - selected camera coverage
  - overall multi-camera coverage
- Defect detectability overlays (mm-size based)
- Camera configuration presets (sensor format and resolution)
- Camera edit/delete + roll around optical axis
- Status chips for mode/coverage/detectability
- Managed project browser with search + sort
- New Project wizard with source choices:
  - custom cuboid
  - load mesh/point cloud from file
  - create point cloud from video using COLMAP
- Background COLMAP reconstruction with live logs
- Force rebuild option for existing scene output

## Python Version Support

- Supported: Python 3.10+
- Tested with Conda-based environments on Windows

## Requirements

- Windows 10/11
- Python 3.10+
- OpenGL-capable graphics driver (recommended for smoother Open3D rendering)

External tools for video-to-point-cloud flow:

- COLMAP binaries
- FFmpeg binaries

## Installation

### Option A: Requirements file

```bash
pip install -r requirements.txt
```

### Option B: Editable package install

```bash
pip install -e .
```

## Quick Start

Run with any of the following:

```bash
python tool.py
```

```bash
python -m camera_viewer.app
```

```bash
camera-viewer
```

Notes:

- `tool.py` is a compatibility launcher.
- `camera-viewer` is available after editable/package install.

## Project Layout

```text
camera_viewer_tool/
  src/camera_viewer/
    app.py
    __main__.py
  projects/
    <project-slug>/
      project.json
      metadata.json
  colmap_workflow/          (usually gitignored)
  tool.py
  pyproject.toml
  requirements.txt
  README.md
```

## Managed Projects

Projects are saved under:

```text
projects/<slug>/project.json
projects/<slug>/metadata.json
```

`metadata.json` includes fields used by the load browser:

- name
- created_at
- updated_at
- object_type
- object_source
- camera_count
- sample_points

## COLMAP Workflow Setup (gitignored)

If `colmap_workflow/` is excluded from git, recreate it on each machine as follows.

### 1) Create directory structure

```powershell
New-Item -ItemType Directory -Force -Path .\colmap_workflow\"01 COLMAP"
New-Item -ItemType Directory -Force -Path .\colmap_workflow\"02 VIDEOS"
New-Item -ItemType Directory -Force -Path .\colmap_workflow\"03 FFMPEG"
New-Item -ItemType Directory -Force -Path .\colmap_workflow\"04 SCENES"
New-Item -ItemType Directory -Force -Path .\colmap_workflow\"05 SCRIPTS"
```

### 2) Install COLMAP

Extract COLMAP release into:

```text
colmap_workflow/01 COLMAP/
```

Expected executable:

- `colmap_workflow/01 COLMAP/colmap.exe`
  or
- `colmap_workflow/01 COLMAP/bin/colmap.exe`

### 3) Install FFmpeg

Extract FFmpeg static build into:

```text
colmap_workflow/03 FFMPEG/
```

Expected executable:

- `colmap_workflow/03 FFMPEG/ffmpeg.exe`
  or
- `colmap_workflow/03 FFMPEG/bin/ffmpeg.exe`

### 4) Add reconstruction script

Place script at:

```text
colmap_workflow/05 SCRIPTS/batch_reconstruct.bat
```

### 5) Verify with one video (optional manual test)

```cmd
colmap_workflow\05 SCRIPTS\batch_reconstruct.bat "C:\path\to\video.mp4"
```

Output should appear in:

```text
colmap_workflow/04 SCENES/<video-name>/
```

and include `pointcloud.ply`.

## Troubleshooting

- `COLMAP script missing`
  - verify `colmap_workflow/05 SCRIPTS/batch_reconstruct.bat`
- `ffmpeg.exe not found` or `colmap.exe not found`
  - verify binary locations under `01 COLMAP` and `03 FFMPEG`
- Reconstruction finishes but no point cloud loads
  - check reconstruction log dialog
  - verify `pointcloud.ply` in scene output folder
- Progress appears slow
  - feature extraction/mapping can take significant time depending on video length and hardware

## Roadmap

- Split `app.py` into modular packages (`models`, `services`, `ui`, `io`)
- Add project operations (rename, duplicate, delete)
- Add reconstruction cancel control and richer progress estimation

## Attribution

This tool integrates with external binaries (COLMAP and FFmpeg). Follow their licenses for redistribution.
