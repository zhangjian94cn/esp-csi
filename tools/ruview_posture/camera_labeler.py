"""Local-only camera posture labels; no image or video frames are written."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time


def _angle(a, b, c) -> float:
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator == 0:
        return 180.0
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(first, second)) / denominator))
    return math.degrees(math.acos(cosine))


def classify(landmarks) -> tuple[str, float]:
    indexes = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
        "left_knee": 25,
        "right_knee": 26,
        "left_ankle": 27,
        "right_ankle": 28,
    }
    points = {name: landmarks[index] for name, index in indexes.items()}
    confidence = min(point.visibility for point in points.values())
    shoulder_x = (points["left_shoulder"].x + points["right_shoulder"].x) / 2
    shoulder_y = (points["left_shoulder"].y + points["right_shoulder"].y) / 2
    hip_x = (points["left_hip"].x + points["right_hip"].x) / 2
    hip_y = (points["left_hip"].y + points["right_hip"].y) / 2
    horizontal = abs(shoulder_x - hip_x)
    vertical = abs(shoulder_y - hip_y)
    if horizontal > vertical * 1.2:
        return "lying", confidence
    knee_angle = min(
        _angle(points["left_hip"], points["left_knee"], points["left_ankle"]),
        _angle(points["right_hip"], points["right_knee"], points["right_ankle"]),
    )
    return ("sitting" if knee_angle < 135.0 else "standing"), confidence


def run(args: argparse.Namespace) -> int:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        raise SystemExit(
            "camera labeler requires requirements-camera.txt"
        ) from error

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    period = 1.0 / args.fps
    try:
        with args.output.open("a") as output:
            while True:
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    break
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks:
                    label, confidence = classify(result.pose_landmarks.landmark)
                else:
                    label, confidence = "absent", 0.0
                output.write(
                    json.dumps(
                        {
                            "schema": "rvp-camera-label-v2",
                            "timestamp_ns": time.time_ns(),
                            "occupancy": (
                                "absent" if label == "absent" else "present"
                            ),
                            "posture": (
                                "unknown" if label == "absent" else label
                            ),
                            "confidence": confidence,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                output.flush()
                delay = period - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        pose.close()
        capture.release()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=float, default=5.0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
