"""
Shared ROI utilities for IntiVision.

This module contains the common hand ROI extraction logic
used by both realtime inference and dataset preprocessing.
"""

from collections.abc import Sequence


DEFAULT_HAND_PADDING_RATIO = 0.40


def get_hand_box(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    hand_landmarks: Sequence,
    padding_ratio: float = DEFAULT_HAND_PADDING_RATIO,
) -> tuple[int, int, int, int] | None:
    """
    Calculate a square hand bounding box from MediaPipe landmarks.

    The returned box may extend beyond the frame boundaries.
    Border handling will be performed by the ROI extraction function.
    """
    if padding_ratio < 0:
        raise ValueError("padding_ratio cannot be negative.")

    frame_height, frame_width = frame_shape[:2]

    if frame_height <= 0 or frame_width <= 0:
        raise ValueError("Frame dimensions must be greater than zero.")

    if not hand_landmarks:
        return None

    x_coordinates = [
        int(landmark.x * frame_width)
        for landmark in hand_landmarks
    ]

    y_coordinates = [
        int(landmark.y * frame_height)
        for landmark in hand_landmarks
    ]

    min_x = min(x_coordinates)
    max_x = max(x_coordinates)
    min_y = min(y_coordinates)
    max_y = max(y_coordinates)

    hand_width = max_x - min_x
    hand_height = max_y - min_y
    hand_size = max(hand_width, hand_height)

    if hand_size <= 0:
        return None

    padding = int(hand_size * padding_ratio)
    box_size = hand_size + (padding * 2)

    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2

    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size

    return x1, y1, x2, y2

import cv2
import numpy as np


def make_square_roi(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray | None:
    """
    Crop a square ROI from the frame.

    If the box extends outside the frame, missing areas are filled
    with black border padding.
    """
    x1, y1, x2, y2 = box

    frame_height, frame_width = frame.shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width <= 0 or box_height <= 0:
        return None

    padding_left = max(0, -x1)
    padding_top = max(0, -y1)
    padding_right = max(0, x2 - frame_width)
    padding_bottom = max(0, y2 - frame_height)

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(frame_width, x2)
    crop_y2 = min(frame_height, y2)

    roi = frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ]

    if roi.size == 0:
        return None

    square_roi = cv2.copyMakeBorder(
        roi,
        padding_top,
        padding_bottom,
        padding_left,
        padding_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    roi_height, roi_width = square_roi.shape[:2]

    if roi_height != roi_width:
        square_size = max(roi_height, roi_width)

        extra_vertical = square_size - roi_height
        extra_horizontal = square_size - roi_width

        top = extra_vertical // 2
        bottom = extra_vertical - top
        left = extra_horizontal // 2
        right = extra_horizontal - left

        square_roi = cv2.copyMakeBorder(
            square_roi,
            top,
            bottom,
            left,
            right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    return square_roi


def extract_hand_roi(
    frame: np.ndarray,
    hand_landmarks: Sequence,
    padding_ratio: float = DEFAULT_HAND_PADDING_RATIO,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """
    Calculate the hand box and extract its square ROI.

    Returns:
        A tuple containing the square ROI and bounding box.
        None if a valid ROI cannot be produced.
    """
    hand_box = get_hand_box(
        frame_shape=frame.shape,
        hand_landmarks=hand_landmarks,
        padding_ratio=padding_ratio,
    )

    if hand_box is None:
        return None

    square_roi = make_square_roi(
        frame=frame,
        box=hand_box,
    )

    if square_roi is None:
        return None

    return square_roi, hand_box