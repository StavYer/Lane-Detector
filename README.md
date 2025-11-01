# Lane Detector

Lane Detector is a computer vision project that identifies lane boundaries and nearby vehicles in road footage. It uses OpenCV to extract a region of interest, perform edge detection, find lane lines with the Hough Transform, and track potential lane changes frame by frame. A complementary utility script helps you cut a reusable segment from a longer recording so it can be processed repeatedly during experimentation.

## Features
- **Configurable workflow** driven by environment variables loaded from a `.env` file.
- **Video segment extraction** (`VideoTools.extract_segment`) to create manageable clips for development and testing.
- **Lane detection pipeline** that masks the road surface, applies Canny edge detection, and fits left/right lanes with the Hough Transform.
- **Lane stability heuristics** that smooth noisy detections and approximate missing lane lines from previous frames.
- **Lane-change notifications** that monitor the detected lane center and display guidance text when the vehicle drifts.
- **Vehicle detection** via a Haar cascade that highlights cars and flags those overlapping the detected lane area.
- **Visual overlays** showing lane polygons, highlighted vehicles, and lane area statistics on every processed frame.

## Requirements
- Python 3.9+
- OpenCV (`opencv-python`)
- NumPy
- python-dotenv

You can install the dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install opencv-python numpy python-dotenv
```

## Environment configuration
Create a `.env` file in the project root with the paths relevant to your footage:

```dotenv
RAW_VIDEO_PATH=/absolute/path/to/raw_video.mp4
SEGMENT_PATH=/absolute/path/to/segment.mp4
RAW_NIGHT_VIDEO_PATH=/absolute/path/to/night_video.mp4
CARS_XML=/absolute/path/to/haarcascade_cars.xml
```

- `RAW_VIDEO_PATH` – original recording used by `VideoTools.extract_segment`.
- `SEGMENT_PATH` – output file written by `VideoTools.extract_segment` and default input for `LaneDetector.detect_lanes`.
- `RAW_NIGHT_VIDEO_PATH` – optional night-time recording if you want to process different footage.
- `CARS_XML` – Haar cascade classifier (for example, `haarcascade_car.xml`).

## Usage
1. **Extract a working clip** (optional but recommended):
   ```bash
   python VideoTools.py
   ```
   This command writes a 20-second clip starting at time 0 to `SEGMENT_PATH`.

2. **Run lane and vehicle detection** on a prepared clip:
   ```bash
   python LaneDetector.py
   ```
   The script reads frames from `SEGMENT_PATH` (or a path you supply to `detect_lanes`) and saves annotated output as `output.mp4`. A preview window opens during processing; press `q` to exit early.

## Customizing the pipeline
- Adjust the trapezoid in `roi_mask` to fine-tune the road area for different camera placements.
- Update the Canny thresholds or dilation kernel in `process_frame` to accommodate brighter or darker footage.
- Modify the lane-change sensitivity by tweaking the rolling averages and `left_diff` / `right_diff` thresholds in `detect_lanes`.

## Repository layout
```
LaneDetector.py   # Main lane detection pipeline and CLI entry point
VideoTools.py     # Helper to extract a shorter clip from longer videos
```

## Contributing
Issues and pull requests are welcome. Please open an issue describing the change you would like to make, along with sample footage or logs if applicable.
