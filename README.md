<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/gautamthampy/parking-space-detector">

![index](https://github.com/danielbob32/ParkingSpace/assets/120675110/c128eb3a-0221-49b5-b596-d167a89c4740)
  </a>

<h1 align="center">ParkingSpace Detection System</h1>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#learn-more">Learn More</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About

ParkingSpace is a research project for real-time parking space detection using YOLOv11 and RTSP cameras. It identifies available parking spaces in urban environments, including areas without pre-defined spots.

### Features
- Real-time detection with YOLOv11
- RTSP camera support
- Works in undefined parking areas
- Probability mapping for space availability

## Project Structure

- `src/parkingspace/` — Main source code (entry: `main.py`)
- `Demo/` — Demo videos and probability map
- `regions.json` — Region definitions for parking detection
- `requirements.txt` / `requirements-gpu.txt` — Dependencies
- `tests/` — Unit tests
- `scripts/` — Utility scripts

<!-- INSTALLATION -->

## Installation

**Requirements:**
- Python 3.8+
- 2.5GB free disk space
- CUDA-compatible GPU (optional)

**Setup:**
```bash
# Clone repository
git clone https://github.com/gautamthampy/parking-space-detector.git
cd ParkingSpace

# Create and activate virtual environment
python -m venv parkingspace-env
# Windows:
parkingspace-env\Scripts\activate
# Linux/Mac:
source parkingspace-env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run demo
python -c "from src.parkingspace import main; main()"
```

**GPU (CUDA 11.7):**
```bash
pip install -r requirements-gpu.txt
# Or
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

**Check CUDA:**
```bash
nvcc --version
```

## Testing

Run all tests with:
```bash
pytest tests/
```

<!-- USAGE -->

## Usage

On first run, models are downloaded automatically. Processing is performed at 15-second intervals by default. Output is shown in a window with detected parking spaces.

![Demo Output](https://github.com/danielbob32/ParkingSpace/assets/120675110/710c9c66-7d26-4056-b5d9-6ca8d663cb82)

![Processing Window](https://github.com/danielbob32/ParkingSpace/assets/120675110/fc9735be-4666-4770-a94f-cddddd735656)

### Performance
- Increase sampling interval for lower resource usage
  ![Sampling Interval](https://github.com/danielbob32/ParkingSpace/assets/120675110/7febc6be-1e54-4b6e-a028-9f947884602e)
- Reduce model resolution
  ![Model Resolution](https://github.com/danielbob32/ParkingSpace/assets/120675110/e2d4846d-19ef-479b-a6ac-2e8c3410b38b)
- Lower model accuracy for faster processing
  ![Model Accuracy](https://github.com/danielbob32/ParkingSpace/assets/120675110/f1b3e74e-b69f-4a95-b1be-e659e0611e39)

<!-- DATA -->

## Data

- [Raw Images](https://drive.google.com/drive/folders/19Vj134JEaQX1-8Ek_UGWvElroVuO2vXz?usp=sharing): Raw images from live videos
- [Augmented Images](https://drive.google.com/drive/folders/1gHNhspoRk9ewnf64yqftENCU64kqY5OP?usp=sharing): Augmented images for probability map
- [Segmented Images](https://drive.google.com/drive/folders/1Te31EDZKZ-XRcGjyaPD2qc1c4jH3Dpk9?usp=sharing): YOLOv11 segmented images
- [Binary Masks](https://drive.google.com/drive/folders/145pIsCr6CX0GDmDCKgXocOVQuMTDszI_?usp=sharing): Binary masks from segmentation
- [Probability Map](https://drive.google.com/file/d/1fwNCc_sKEZyjcrchR3vL8WX9ULql6Wt2/view?usp=sharing): Probability map from binary masks

## Acknowledgments

- [Ultralytics](https://docs.ultralytics.com/)
- [Roboflow](https://roboflow.com/)

<!-- MARKDOWN LINKS & IMAGES -->
