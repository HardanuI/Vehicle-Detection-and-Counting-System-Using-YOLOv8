import os
import subprocess
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import imageio_ffmpeg

warnings.filterwarnings("ignore", category=FutureWarning, module="supervision")

MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.3
CAR_CLASS_ID = 2

model = YOLO(MODEL_PATH)


def _read_image(path):
    """Membaca gambar secara aman termasuk path non-ASCII di Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path)


def _write_image(path, img):
    """Menyimpan gambar secara aman termasuk path non-ASCII di Windows."""
    try:
        ext = os.path.splitext(path)[1]
        success, buf = cv2.imencode(ext, img)
        if success:
            buf.tofile(path)
            return True
    except Exception:
        pass
    return cv2.imwrite(path, img)


def _convert_to_h264(input_path, output_path):
    """Konversi video mp4v ke H.264 agar dapat diputar langsung di browser web."""
    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y",
            "-i", input_path,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if os.path.exists(input_path) and input_path != output_path:
                try:
                    os.remove(input_path)
                except OSError:
                    pass
            return True
    except Exception:
        pass

    # Fallback: jika konversi gagal, pindahkan file asli
    if os.path.exists(input_path) and input_path != output_path:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass
        os.replace(input_path, output_path)
    return False


def detect_image(image_path, output_path):
    """Deteksi mobil pada satu gambar. Return jumlah mobil."""
    frame = _read_image(image_path)
    if frame is None:
        return 0

    results = model(frame, classes=[CAR_CLASS_ID], conf=CONF_THRES)[0]
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    labels = [f"Mobil {c:.2f}" for c in detections.confidence]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    _write_image(output_path, frame)
    return len(detections)


def count_video(video_path, output_path):
    """Proses video, hitung mobil melewati garis. Return dict hasil."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"masuk": 0, "keluar": 0, "total": 0}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    line_y = h // 2

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    line_zone = sv.LineZone(start=sv.Point(0, line_y), end=sv.Point(w, line_y))

    temp_output = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, classes=[CAR_CLASS_ID], conf=CONF_THRES)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        line_zone.trigger(detections)

        # Labels dengan tracker id jika ada
        if detections.tracker_id is not None:
            labels = [f"#{tid} Mobil {c:.2f}" for tid, c in zip(detections.tracker_id, detections.confidence)]
        else:
            labels = [f"Mobil {c:.2f}" for c in detections.confidence]

        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        # Garis virtual penghitung
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        # Overlay statistik semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 115), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        in_cnt = line_zone.in_count
        out_cnt = line_zone.out_count
        tot_cnt = in_cnt + out_cnt

        cv2.putText(frame, f"Masuk : {in_cnt}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, f"Keluar: {out_cnt}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        cv2.putText(frame, f"Total : {tot_cnt}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    _convert_to_h264(temp_output, output_path)

    return {
        "masuk": line_zone.in_count,
        "keluar": line_zone.out_count,
        "total": line_zone.in_count + line_zone.out_count,
    }
