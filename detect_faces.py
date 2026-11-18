"""Detect and blur faces in a photo using OpenCV and YuNet."""

import argparse
from pathlib import Path

import cv2


MODEL = Path(__file__).resolve().with_name("face_detection_yunet_2023mar.onnx")


def blur_faces(image, faces):
    """Blur padded detection boxes and return the number processed."""
    height, width = image.shape[:2]
    count = 0
    for face in faces:
        x, y, w, h = map(int, face[:4])
        pad_x, pad_y = int(w * 0.15), int(h * 0.15)
        left, top = max(0, x - pad_x), max(0, y - pad_y)
        right, bottom = min(width, x + w + pad_x), min(height, y + h + pad_y)
        if right <= left or bottom <= top:
            continue

        region = image[top:bottom, left:right]
        kernel = max(3, int(min(region.shape[:2]) * 0.5) | 1)
        image[top:bottom, left:right] = cv2.GaussianBlur(
            region, (kernel, kernel), 0
        )
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Photo to process")
    parser.add_argument("output", type=Path, help="Where to save the blurred copy")
    parser.add_argument(
        "--threshold", type=float, default=0.80, help="Detection score cutoff (default: 0.80)"
    )
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose a different output file so the original stays untouched.")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new filename.")
    if not MODEL.is_file():
        parser.exit(1, f"Model not found: {MODEL.name}. See README.md for setup.\n")

    try:
        image = cv2.imread(str(args.input))
        if image is None:
            parser.exit(1, f"Couldn't open {args.input}.\n")
        height, width = image.shape[:2]
        detector = cv2.FaceDetectorYN.create(str(MODEL), "", (width, height), args.threshold)
        _, faces = detector.detect(image)
        count = blur_faces(image, faces) if faces is not None else 0
        if not cv2.imwrite(str(args.output), image):
            parser.exit(1, f"Couldn't save {args.output}.\n")
    except cv2.error as error:
        parser.exit(1, f"OpenCV could not process this request:\n{error}\n")

    print(f"Blurred {count} face(s). Saved {args.output}.")


if __name__ == "__main__":
    main()
