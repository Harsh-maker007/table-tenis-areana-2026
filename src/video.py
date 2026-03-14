import cv2


def open_video_source(video_path: str, camera_index: int) -> cv2.VideoCapture:
    if video_path:
        return cv2.VideoCapture(video_path)
    return cv2.VideoCapture(camera_index)
