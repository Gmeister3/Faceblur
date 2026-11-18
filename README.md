# FaceBlur

FaceBlur is a Python command-line tool that finds and blurs faces in photos. It uses the pretrained YuNet detector from OpenCV Zoo and runs locally without uploading images. It saves a separate copy and refuses to overwrite an existing file.

## Setup on Windows

1. Install Python 3.12 and open PowerShell or Command Prompt in this folder.
2. Download the [YuNet ONNX model from OpenCV Zoo](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) using **Download raw file**. Save `face_detection_yunet_2023mar.onnx` next to `detect_faces.py`. If you already have that file, reuse it.
3. Create a virtual environment and install the dependencies.

```text
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```text
.\.venv\Scripts\python.exe detect_faces.py input.jpg output.jpg
.\.venv\Scripts\python.exe detect_faces.py input.jpg output-lower.jpg --threshold 0.60
```

Replace `input.jpg` with your photo's path. Use quotes around paths with spaces. Choose a new output filename each time.

## How it works

1. YuNet returns bounding boxes for faces above the confidence threshold.
2. FaceBlur expands each box slightly, clips it to the image boundaries, and applies a Gaussian blur inside it.
3. The script saves the processed image and reports the number of regions blurred.

The default threshold is `0.80`. Lowering it may catch additional faces and also introduce false detections. Small, partially covered, and side-facing faces can be missed. Inspect the output before sharing it. This project uses pretrained inference, not model training or identity recognition.
