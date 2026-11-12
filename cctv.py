import threading
import time
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Hindari peringatan deprecation dari supervision
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.25

# Kelas kendaraan COCO
CLASS_NAMES = {
    2: "Mobil",
    3: "Motor",
    5: "Bus",
    7: "Truk",
}

# State global hitungan & status di memori (tanpa database)
cctv_state = {
    "masuk": 0,
    "keluar": 0,
    "total": 0,
    "is_connected": False,
    "error": None,
    "fps": 0.0,
    "orientation": "horizontal",  # 'horizontal' atau 'vertical'
    "line_pos": 50,               # persentase (10 - 90)
    "vehicle_mode": "all",        # 'all' (mobil, bus, truk, motor) atau 'car' (mobil, bus, truk)
}

# Lock untuk thread-safety
_state_lock = threading.Lock()
_latest_frame = None
_latest_frame_lock = threading.Lock()
_worker_instance = None
_worker_lock = threading.Lock()


def get_cctv_stats():
    """Mengembalikan statistik hitungan dan status CCTV saat ini."""
    with _state_lock:
        return {
            "masuk": cctv_state["masuk"],
            "keluar": cctv_state["keluar"],
            "total": cctv_state["total"],
            "is_connected": cctv_state["is_connected"],
            "error": cctv_state["error"],
            "fps": round(cctv_state["fps"], 1),
            "orientation": cctv_state["orientation"],
            "line_pos": cctv_state["line_pos"],
            "vehicle_mode": cctv_state["vehicle_mode"],
        }


def reset_cctv_counts():
    """Mereset hitungan mobil sementara di memori."""
    with _state_lock:
        cctv_state["masuk"] = 0
        cctv_state["keluar"] = 0
        cctv_state["total"] = 0
    if _worker_instance:
        _worker_instance.request_reset()


def update_cctv_config(orientation=None, line_pos=None, vehicle_mode=None):
    """Memperbarui arah/posisi garis deteksi secara realtime dari antarmuka web."""
    with _state_lock:
        if orientation in ["horizontal", "vertical"]:
            cctv_state["orientation"] = orientation
        if line_pos is not None:
            try:
                pos = int(line_pos)
                if 10 <= pos <= 90:
                    cctv_state["line_pos"] = pos
            except ValueError:
                pass
        if vehicle_mode in ["all", "car"]:
            cctv_state["vehicle_mode"] = vehicle_mode

    if _worker_instance:
        _worker_instance.update_line_geometry()


def create_error_frame(title, detail=""):
    """Buat frame status/error bergaya neo-brutalisme jika stream tidak dapat dibuka."""
    h, w = 480, 720
    frame = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)

    # Border tebal 4px
    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (10, 10, 10), 4)

    # Header Box Merah Neo-Brutalis
    cv2.rectangle(frame, (30, 35), (w - 30, 135), (46, 46, 255), -1)
    cv2.rectangle(frame, (30, 35), (w - 30, 135), (10, 10, 10), 4)

    cv2.putText(frame, "CCTV LIVE STREAM", (50, 75),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(frame, title, (50, 115),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)

    # Kotak Pesan Info / Solusi
    cv2.rectangle(frame, (30, 165), (w - 30, 440), (255, 255, 255), -1)
    cv2.rectangle(frame, (30, 165), (w - 30, 440), (10, 10, 10), 3)

    cv2.putText(frame, "KETERANGAN & STATUS:", (50, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)

    words = detail.split()
    lines = []
    curr = ""
    for wd in words:
        if len(curr) + len(wd) + 1 > 50:
            lines.append(curr)
            curr = wd
        else:
            curr = f"{curr} {wd}".strip()
    if curr:
        lines.append(curr)

    y_pos = 245
    for ln in lines[:5]:
        cv2.putText(frame, ln, (50, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        y_pos += 34

    cv2.putText(frame, "// Ganti CCTV_URL di app.py untuk menyambungkan stream .m3u8", (50, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 0), 1)

    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ret else b""


class CCTVProcessor(threading.Thread):
    """
    Worker tunggal (Singleton) yang membaca stream, menjalankan YOLOv8 + ByteTrack + LineZone,
    dan menyajikan frame terenkripsi JPEG ke seluruh request klien secara real-time tanpa lag.
    """

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.running = True
        self.reset_flag = False
        self.line_zone = None
        self.w = 640
        self.h = 360
        self.base_in = 0
        self.base_out = 0

    def request_reset(self):
        self.reset_flag = True

    def _build_line(self, w, h):
        with _state_lock:
            orientation = cctv_state["orientation"]
            line_pos = cctv_state["line_pos"]

        if orientation == "vertical":
            x = int(w * (line_pos / 100.0))
            x = max(10, min(w - 10, x))
            p_start = sv.Point(x, 0)
            p_end = sv.Point(x, h)
        else:
            y = int(h * (line_pos / 100.0))
            y = max(10, min(h - 10, y))
            p_start = sv.Point(0, y)
            p_end = sv.Point(w, y)

        # Menggunakan CENTER anchor agar kendaraan terhitung secara instan saat melewati garis
        # tanpa masalah deadlock straddling dari 4 sudut bounding box
        return sv.LineZone(
            start=p_start,
            end=p_end,
            triggering_anchors=[sv.Position.CENTER, sv.Position.BOTTOM_CENTER]
        )

    def update_line_geometry(self):
        if self.w and self.h:
            prev_in = self.line_zone.in_count if self.line_zone else 0
            prev_out = self.line_zone.out_count if self.line_zone else 0
            self.base_in += prev_in
            self.base_out += prev_out
            self.line_zone = self._build_line(self.w, self.h)

    def run(self):
        global _latest_frame

        if not self.url or not str(self.url).strip():
            with _state_lock:
                cctv_state["is_connected"] = False
                cctv_state["error"] = "URL stream (.m3u8) belum diatur di app.py (CCTV_URL masih kosong)."
            err_frame = create_error_frame(
                "URL STREAM BELUM DIISI",
                "Silakan isi variabel CCTV_URL di app.py dengan URL .m3u8 yang valid."
            )
            with _latest_frame_lock:
                _latest_frame = err_frame
            return

        try:
            cap = cv2.VideoCapture(self.url)
        except Exception as e:
            with _state_lock:
                cctv_state["is_connected"] = False
                cctv_state["error"] = f"Gagal membuka VideoCapture: {e}"
            err_frame = create_error_frame("GAGAL BUKA STREAM", str(e))
            with _latest_frame_lock:
                _latest_frame = err_frame
            return

        if not cap.isOpened():
            with _state_lock:
                cctv_state["is_connected"] = False
                cctv_state["error"] = f"Koneksi gagal ke URL: {self.url}. Pastikan URL aktif."
            err_frame = create_error_frame(
                "STREAM TIDAK DAPAT DIBUKA",
                f"Koneksi ke URL .m3u8 gagal. Periksa koneksi atau validitas link: {self.url}"
            )
            with _latest_frame_lock:
                _latest_frame = err_frame
            cap.release()
            return

        with _state_lock:
            cctv_state["is_connected"] = True
            cctv_state["error"] = None

        # Load YOLO model & Supervision annotators
        model = YOLO(MODEL_PATH)
        tracker = sv.ByteTrack(track_activation_threshold=0.25, minimum_matching_threshold=0.7)
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
        self.line_zone = self._build_line(self.w, self.h)

        fps_timer = time.time()
        fps_counter = 0
        consecutive_failures = 0
        max_failures = 30

        try:
            while self.running:
                if self.reset_flag:
                    if self.line_zone:
                        self.base_in = -self.line_zone.in_count
                        self.base_out = -self.line_zone.out_count
                    self.reset_flag = False

                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        with _state_lock:
                            cctv_state["is_connected"] = False
                            cctv_state["error"] = "Stream CCTV terputus atau tidak menerima frame baru."
                        err_frame = create_error_frame("STREAM TERPUTUS", "Tidak menerima frame dari URL stream.")
                        with _latest_frame_lock:
                            _latest_frame = err_frame
                        time.sleep(1.0)
                        # Coba buka kembali stream
                        cap.release()
                        cap = cv2.VideoCapture(self.url)
                        consecutive_failures = 0
                    time.sleep(0.04)
                    continue

                consecutive_failures = 0

                # Sesuaikan ukuran jika ada perubahan resolusi frame stream
                fh, fw = frame.shape[:2]
                if fw != self.w or fh != self.h:
                    self.w, self.h = fw, fh
                    self.update_line_geometry()

                # Tentukan kelas target berdasarkan vehicle_mode
                with _state_lock:
                    v_mode = cctv_state["vehicle_mode"]
                if v_mode == "car":
                    target_classes = [2, 5, 7]  # Mobil sedan/MPV, Bus, Truk
                else:
                    target_classes = [2, 3, 5, 7]  # Mobil, Motor, Bus, Truk

                # Deteksi objek dengan YOLOv8
                results = model(frame, classes=target_classes, conf=CONF_THRES, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(results)

                # Tracking dengan ByteTrack & Penghitungan garis dengan LineZone
                detections = tracker.update_with_detections(detections)
                self.line_zone.trigger(detections)

                # Hitungan kumulatif
                tot_in = max(0, self.base_in + self.line_zone.in_count)
                tot_out = max(0, self.base_out + self.line_zone.out_count)
                total = tot_in + tot_out

                # Update state global
                fps_counter += 1
                now = time.time()
                current_fps = cctv_state["fps"]
                if now - fps_timer >= 1.0:
                    current_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                with _state_lock:
                    cctv_state["masuk"] = tot_in
                    cctv_state["keluar"] = tot_out
                    cctv_state["total"] = total
                    cctv_state["is_connected"] = True
                    cctv_state["error"] = None
                    cctv_state["fps"] = current_fps

                # Buat label deteksi yang informatif
                labels = []
                if len(detections) > 0:
                    for i in range(len(detections)):
                        cid = int(detections.class_id[i]) if detections.class_id is not None else 2
                        c_name = CLASS_NAMES.get(cid, "Mobil")
                        conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                        tid = detections.tracker_id[i] if detections.tracker_id is not None else None
                        if tid is not None:
                            labels.append(f"#{tid} {c_name} {conf:.2f}")
                        else:
                            labels.append(f"{c_name} {conf:.2f}")

                    frame = box_annotator.annotate(scene=frame, detections=detections)
                    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

                # Gambar garis virtual penghitung
                with _state_lock:
                    orientation = cctv_state["orientation"]
                p_s = (int(self.line_zone.vector.start.x), int(self.line_zone.vector.start.y))
                p_e = (int(self.line_zone.vector.end.x), int(self.line_zone.vector.end.y))

                # Garis neon kuning 3px dengan titik penanda ujung
                cv2.line(frame, p_s, p_e, (0, 212, 255), 3)
                cv2.circle(frame, p_s, 6, (0, 0, 255), -1)
                cv2.circle(frame, p_e, 6, (0, 0, 255), -1)

                # Label teks di garis
                if orientation == "vertical":
                    mid_x = p_s[0]
                    cv2.putText(frame, "GARIS HITUNG", (max(10, mid_x - 60), 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)
                else:
                    mid_y = p_s[1]
                    cv2.putText(frame, "GARIS HITUNG", (30, max(25, mid_y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)

                # Overlay Neo-Brutalist Teks Besar di kiri atas
                overlay = frame.copy()
                cv2.rectangle(overlay, (15, 15), (295, 140), (10, 10, 10), -1)
                frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
                cv2.rectangle(frame, (15, 15), (295, 140), (255, 212, 0), 2)

                cv2.putText(frame, f"MASUK : {tot_in}", (25, 50),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 224, 106), 2)
                cv2.putText(frame, f"KELUAR: {tot_out}", (25, 85),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 212, 255), 2)
                cv2.putText(frame, f"TOTAL : {total}", (25, 124),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

                # Encode frame ke JPEG
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with _latest_frame_lock:
                        _latest_frame = buffer.tobytes()

        finally:
            cap.release()


def _ensure_worker(url):
    """Menjamin hanya satu worker singleton yang berjalan membaca stream."""
    global _worker_instance
    with _worker_lock:
        if _worker_instance is None or not _worker_instance.is_alive() or _worker_instance.url != url:
            if _worker_instance and _worker_instance.is_alive():
                _worker_instance.running = False
                _worker_instance.join(timeout=1.0)
            _worker_instance = CCTVProcessor(url)
            _worker_instance.start()


def generate_cctv_frames(url_m3u8):
    """
    Generator streaming MJPEG ke browser.
    Membaca frame terbaru dari worker tanpa memblokir request lain.
    """
    _ensure_worker(url_m3u8)

    while True:
        with _latest_frame_lock:
            frame_bytes = _latest_frame

        if frame_bytes:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.04)  # ~25 FPS streaming
