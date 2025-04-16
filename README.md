# Paddle Game Heatmap Generator

This tool generates heatmaps showing player positions throughout paddle game videos. It processes the video through several steps to create an accurate visualization of player movement patterns.

## Features

- Lens distortion removal
- Court alignment and centering
- Player detection and tracking using YOLO
- Perspective transform to map players onto a standard court
- Heatmap generation showing player position frequency

## Requirements

```
numpy
opencv-python (cv2)
ultralytics
matplotlib
seaborn
tqdm
norfair
pip install opencv-python numpy matplotlib seaborn tqdm ultralytics norfair
```

## Usage

Basic usage:
```
python heatmap_generator.py video_path.mp4
```

### Options

- `--manual` or `-m`: Manually select court corners (useful for first-time setup, but not need for given examples)
- Add `False` as second argument to remove debugging temporary files:
```
python heatmap_generator.py video_path.mp4 False
```

## How It Works

1. The script first undistorts the video to remove lens distortion
2. It aligns and centers the court
3. If running with `--manual` or if no saved court points exist you'll be prompted to select the four corners of the court (top-left, top-right, bottom-right, bottom-left)
4. Player tracking is performed using YOLO
5. Player positions are projected onto a standardized court representation
6. A heatmap is generated showing the frequency of player positions

## Output

- `heatmap.png`: Visualization of player movement patterns
- Temporary files in `tmp/` directory (if debugging is enabled)
