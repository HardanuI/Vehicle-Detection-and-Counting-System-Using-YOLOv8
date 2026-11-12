from collections import deque
import os
import threading
import time
import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Hindari peringatan deprecation dari supervision
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# KONFIGURASI AI & PERFORMA STREAMING
# ============================================================
# Gunakan model YOLO yang lebih besar jika tersedia di lokal,
# jika menggunakan yolov8n.pt perbesar resolusi input (imgsz=800) agar objek kecil terdeteksi
if os.path.exists("yolov8m.pt"):
    MODEL_PATH = "yolov8m.pt"
    YOLO_IMGSZ = 640
elif os.path.exists("yolov8s.pt"):
    MODEL_PATH = "yolov8s.pt"
    YOLO_IMGSZ = 640
else:
    MODEL_PATH = "yolov8n.pt"
    YOLO_IMGSZ = 800

CONF_THRES = 0.15     # Diturunkan ke 0.15 agar mobil jauh / tertutup sebagian tetap terdeteksi
IOU_THRES = 0.50      # IoU 0.5 agar kotak deteksi mobil yang tumpang tindih tetap dihitung
FRAME_SKIP = 2        # Jalankan YOLO hanya setiap 2 frame sekali
JPEG_QUALITY = 75     # Kualitas kompresi JPEG (70-80) untuk pengiriman ke browser

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
    "line_pos": 50,               # persentase (10 - 90), default 50 (tengah area jalan)
    "vehicle_mode": "all",        # 'all' (mobil, bus, truk, motor) atau 'car' (mobil, bus, truk)
    "reset_version": 0,
    "config_version": 0,
}

# Lock & Singleton instances
_state_lock = threading.Lock()
_stream_lock = threading.Lock()
_active_stream = None
_processor_lock = threading.Lock()
_active_processor = None
_model_lock = threading.Lock()
_yolo_model = None


def _get_yolo_model():
    """Singleton loader untuk model YOLOv8 agar tidak dimuat berulang kali."""
    global _yolo_model
    with _model_lock:
        if _yolo_model is None:
            _yolo_model = YOLO(MODEL_PATH)
        return _yolo_model


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
        cctv_state["reset_version"] += 1


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
        cctv_state["config_version"] += 1


def create_error_frame(title, detail=""):
    """Buat frame status/error bergaya neo-brutalisme jika stream tidak dapat dibuka."""
    h, w = 480, 720
    frame = np.full((h, w, 3), (245, 245, 245), dtype=np.uint8)

    cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (10, 10, 10), 4)
    cv2.rectangle(frame, (30, 35), (w - 30, 135), (46, 46, 255), -1)
    cv2.rectangle(frame, (30, 35), (w - 30, 135), (10, 10, 10), 4)

    cv2.putText(frame, "CCTV LIVE STREAM", (50, 75),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (10, 10, 10), 2)
    cv2.putText(frame, title, (50, 115),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)

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

    ret, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ret else b""


# ============================================================
# 1. PRODUCER THREAD: VideoStream (Khusus Baca Stream Cepat)
# ============================================================
class VideoStream(threading.Thread):
    """
    Thread Producer khusus HANYA untuk membaca frame dari cv2.VideoCapture(url_m3u8)
    di background. Menyimpan frame terbaru di variabel & ring-buffer terbatas agar
    buffer internal OpenCV/FFmpeg tidak pernah menumpuk dan bebas dari jeda segmen HLS.
    """

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cap = None
        self.grabbed = False
        self.frame = None
        self.frame_id = 0
        self.fps = 25.0
        self.running = False
        self.error = None
        self.lock = threading.Lock()
        # Ring buffer terbatas (maks 45 frame) untuk menyerap burst segmen .ts HLS
        # sekaligus membuang frame usang jika penuh agar tidak terjadi penumpukan latency
        self.queue = deque(maxlen=45)

    def start_stream(self):
        if not self.running:
            self.running = True
            self.start()
        return self

    def _open_capture(self):
        try:
            cap = cv2.VideoCapture(self.url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        except Exception as e:
            self.error = f"Gagal membuka VideoCapture: {e}"
            return None

    def run(self):
        if not self.url or not str(self.url).strip():
            with self.lock:
                self.grabbed = False
                self.error = "URL stream (.m3u8) belum diatur di app.py (CCTV_URL masih kosong)."
            with _state_lock:
                cctv_state["is_connected"] = False
                cctv_state["error"] = self.error
            return

        self.cap = self._open_capture()
        if self.cap is None or not self.cap.isOpened():
            err_msg = self.error or f"Koneksi gagal ke URL: {self.url}. Pastikan URL aktif."
            with self.lock:
                self.grabbed = False
                self.error = err_msg
            with _state_lock:
                cctv_state["is_connected"] = False
                cctv_state["error"] = err_msg
            if self.cap is not None:
                self.cap.release()
            return

        detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if detected_fps and 5.0 <= detected_fps <= 60.0:
            self.fps = float(detected_fps)
        else:
            self.fps = 25.0

        frame_interval = 1.0 / self.fps
        consecutive_failures = 0
        max_failures = 35

        try:
            while self.running:
                t_start = time.time()
                ret, raw_frame = self.cap.read()

                if not ret or raw_frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        err_msg = "Stream CCTV terputus atau sedang menghubungkan ulang..."
                        with self.lock:
                            self.grabbed = False
                            self.error = err_msg
                        with _state_lock:
                            cctv_state["is_connected"] = False
                            cctv_state["error"] = err_msg
                        time.sleep(0.5)
                        self.cap.release()
                        self.cap = self._open_capture()
                        consecutive_failures = 0
                    else:
                        time.sleep(0.01)
                    continue

                consecutive_failures = 0
                with self.lock:
                    self.grabbed = True
                    self.frame = raw_frame
                    self.frame_id += 1
                    self.queue.append((raw_frame, self.frame_id))
                    q_len = len(self.queue)
                    self.error = None

                with _state_lock:
                    cctv_state["is_connected"] = True
                    cctv_state["error"] = None

                # Jika antrean sudah memiliki cadangan >= 10 frame (dari segmen .ts HLS),
                # baca sesuai interval FPS stream. Jika < 10, kuras buffer OpenCV secepat mungkin!
                if q_len >= 10:
                    elapsed = time.time() - t_start
                    sleep_dur = (frame_interval * 0.92) - elapsed
                    if sleep_dur > 0.001:
                        time.sleep(sleep_dur)
                else:
                    time.sleep(0.001)
        finally:
            if self.cap is not None:
                self.cap.release()

    def read(self):
        """
        Mengambil frame berikutnya untuk diproses oleh Consumer.
        Jika antrean terlalu penuh (> 30 frame), buang frame lama agar selalu real-time.
        """
        with self.lock:
            if not self.grabbed or self.frame is None:
                return False, None, self.frame_id, self.error

            # Cegah penumpukan lag: jika antrean menumpuk > 30 frame, lompat ke 10 frame terakhir
            while len(self.queue) > 30:
                self.queue.popleft()

            if self.queue:
                f, fid = self.queue.popleft()
                return True, f.copy(), fid, None

            return True, self.frame.copy(), self.frame_id, None

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=1.0)


# ============================================================
# 4. LOGIKA LINEZONE ANTI DOUBLE-COUNT & ANTI MISSED-COUNT
# ============================================================
class RobustLineZone:
    """
    Menggunakan sv.LineZone(start=..., end=...) di tengah area jalan dengan parameter tambahan:
    1. triggering_anchors=[sv.Position.CENTER] & minimum_crossing_threshold=2
    2. Zona Hysteresis / Deadband (deadband_px=6.0) di atas garis: jika ada mobil yang berhenti
       tepat di atas garis dan bounding box-nya bergetar, tidak terjadi double-counting.
    3. Memori sisi awal jangka panjang (max_absent_frames=150) & penguncian tracker_id
       (_counted_ids): mencegah missed-count saat mobil tertutup sebagian di dekat garis,
       serta menjamin 1 tracker_id hanya dihitung tepat 1 kali.
    """

    def __init__(self, start: sv.Point, end: sv.Point, deadband_px: float = 6.0, max_absent_frames: int = 150):
        self.line_zone = sv.LineZone(
            start=start,
            end=end,
            triggering_anchors=[sv.Position.CENTER],
            minimum_crossing_threshold=2,
        )
        self.start = start
        self.end = end
        self.deadband_px = deadband_px
        self.max_absent_frames = max_absent_frames

        self.in_count = 0
        self.out_count = 0

        self._initial_side = {}
        self._last_seen_frame = {}
        self._counted_ids = set()
        self._step_counter = 0

    @property
    def vector(self):
        return self.line_zone.vector

    def _signed_distance_and_projection(self, px: float, py: float):
        x1, y1 = float(self.start.x), float(self.start.y)
        x2, y2 = float(self.end.x), float(self.end.y)
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-6:
            return 0.0, 0.0
        length = float(np.sqrt(length_sq))
        cross = (px - x1) * dy - (py - y1) * dx
        signed_dist = cross / length
        t = ((px - x1) * dx + (py - y1) * dy) / length_sq
        return signed_dist, t

    def trigger(self, detections: sv.Detections):
        self._step_counter += 1

        # Jalankan trigger sv.LineZone
        sv_in, sv_out = self.line_zone.trigger(detections)

        if len(detections) == 0 or detections.tracker_id is None:
            self._cleanup_stale()
            return

        xyxy = detections.xyxy
        centers_x = (xyxy[:, 0] + xyxy[:, 2]) * 0.5
        centers_y = (xyxy[:, 1] + xyxy[:, 3]) * 0.5

        for i, raw_tid in enumerate(detections.tracker_id):
            tid = int(raw_tid)
            if tid < 0:
                continue

            self._last_seen_frame[tid] = self._step_counter

            # Anti Double-Counting: jika ID mobil ini sudah terhitung, lewati
            if tid in self._counted_ids:
                continue

            px, py = float(centers_x[i]), float(centers_y[i])
            signed_dist, t = self._signed_distance_and_projection(px, py)

            # Pastikan mobil berada di area bentang garis jalan
            if t < -0.05 or t > 1.05:
                continue

            # Jika sv.LineZone mendeteksi crossing valid & di luar zona deadband
            if bool(sv_in[i]) and abs(signed_dist) >= self.deadband_px:
                self.in_count += 1
                self._counted_ids.add(tid)
                continue
            elif bool(sv_out[i]) and abs(signed_dist) >= self.deadband_px:
                self.out_count += 1
                self._counted_ids.add(tid)
                continue

            # Catat sisi awal ketika mobil pertama kali terdeteksi sebelum garis
            if tid not in self._initial_side:
                if abs(signed_dist) >= 1.5:
                    self._initial_side[tid] = -1 if signed_dist < 0 else 1
                continue

            # Jika mobil berhenti tepat di atas garis (dalam radius deadband_px), tahan dulu
            if abs(signed_dist) < self.deadband_px:
                continue

            current_side = -1 if signed_dist < 0 else 1
            start_side = self._initial_side[tid]

            # Jika mobil telah berpindah sisi melewati batas deadband_px, hitung & kunci ID-nya
            if current_side != start_side:
                if current_side < 0:
                    self.in_count += 1
                else:
                    self.out_count += 1
                self._counted_ids.add(tid)

        self._cleanup_stale()

    def _cleanup_stale(self):
        stale_tids = [
            tid for tid, last_f in self._last_seen_frame.items()
            if self._step_counter - last_f > self.max_absent_frames
        ]
        for tid in stale_tids:
            self._last_seen_frame.pop(tid, None)
            self._initial_side.pop(tid, None)
            if len(self._counted_ids) > 2000:
                self._counted_ids.discard(tid)


def _build_line_zone(w: int, h: int) -> RobustLineZone:
    """Membangun LineZone tepat di tengah area jalan sesuai orientasi & posisi."""
    with _state_lock:
        orientation = cctv_state["orientation"]
        line_pos = cctv_state["line_pos"]

    if orientation == "vertical":
        x = int(w * (line_pos / 100.0))
        x = max(10, min(w - 10, x))
        p_start = sv.Point(x, int(h * 0.05))
        p_end = sv.Point(x, int(h * 0.95))
    else:
        y = int(h * (line_pos / 100.0))
        y = max(10, min(h - 10, y))
        p_start = sv.Point(int(w * 0.02), y)
        p_end = sv.Point(int(w * 0.98), y)

    return RobustLineZone(start=p_start, end=p_end, deadband_px=6.0, max_absent_frames=150)


# ============================================================
# 2 & 3. CONSUMER THREAD: CCTVProcessor (YOLO + Skip Frame + JPEG)
# ============================================================
class CCTVProcessor(threading.Thread):
    """
    Thread kedua (Consumer) yang mengambil frame terbaru dari VideoStream,
    menjalankan YOLO setiap FRAME_SKIP (2) frame dengan sensitivitas tinggi
    (conf=0.15, iou=0.5, imgsz=800), menggambar ulang bounding box & tracker_id
    pada frame yang di-skip, menggambar LineZone & teks jumlah mobil di setiap frame,
    serta mengenkode JPEG (kualitas 75) untuk dikirim ke browser tanpa lag.
    """

    def __init__(self, stream: VideoStream):
        super().__init__(daemon=True)
        self.stream = stream
        self.running = True

        self.frame_lock = threading.Lock()
        self. frame_cond = threading.Condition(self.frame_lock)
        self.latest_jpeg = None
        self.latest_jpeg_id = 0

    def start_processor(self):
        self.start()
        return self

    def get_jpeg_frame(self, last_seen_id: int, timeout: float = 0.5):
        """Mengambil frame JPEG terbaru hasil olahan Consumer thread untuk dikirim ke browser."""
        with self.frame_cond:
            if self.latest_jpeg_id == last_seen_id:
                self.frame_cond.wait(timeout=timeout)
            return self.latest_jpeg, self.latest_jpeg_id

    def run(self):
        model = _get_yolo_model()
        # Konfigurasi ByteTrack sensitif agar mobil jauh/tertutup sebagian langsung mendapat tracker_id
        tracker = sv.ByteTrack(
            track_activation_threshold=CONF_THRES,
            lost_track_buffer=60,
            minimum_matching_threshold=0.85,
            frame_rate=15,
            minimum_consecutive_frames=1,
        )
        if hasattr(tracker, "det_thresh"):
            tracker.det_thresh = CONF_THRES

        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

        orig_w, orig_h = 0, 0
        line_zone = None

        with _state_lock:
            base_in = cctv_state["masuk"]
            base_out = cctv_state["keluar"]
            local_reset_ver = cctv_state["reset_version"]
            local_config_ver = cctv_state["config_version"]

        frame_count = 0
        last_raw_id = -1
        last_detections = sv.Detections.empty()
        last_labels = []

        fps_timer = time.time()
        fps_counter = 0

        while self.running and self.stream.running:
            t_cycle_start = time.time()
            grabbed, frame, raw_id, err_msg = self.stream.read()

            if not grabbed or frame is None:
                if err_msg:
                    err_bytes = create_error_frame("STREAM TIDAK TERSEDIA", err_msg)
                    with self.frame_cond:
                        self.latest_jpeg = err_bytes
                        self.latest_jpeg_id += 1
                        self.frame_cond.notify_all()
                    time.sleep(0.3)
                else:
                    time.sleep(0.015)
                continue

            if raw_id == last_raw_id:
                time.sleep(0.005)
                continue
            last_raw_id = raw_id
            frame_count += 1

            fh, fw = frame.shape[:2]
            if line_zone is None or fw != orig_w or fh != orig_h:
                if line_zone is not None:
                    base_in += line_zone.in_count
                    base_out += line_zone.out_count
                orig_w, orig_h = fw, fh
                line_zone = _build_line_zone(orig_w, orig_h)

            with _state_lock:
                curr_reset_ver = cctv_state["reset_version"]
                curr_config_ver = cctv_state["config_version"]
                v_mode = cctv_state["vehicle_mode"]
                orientation = cctv_state["orientation"]

            if curr_reset_ver != local_reset_ver:
                base_in = -line_zone.in_count
                base_out = -line_zone.out_count
                line_zone._counted_ids.clear()
                line_zone._initial_side.clear()
                local_reset_ver = curr_reset_ver

            if curr_config_ver != local_config_ver:
                base_in += line_zone.in_count
                base_out += line_zone.out_count
                line_zone = _build_line_zone(orig_w, orig_h)
                local_config_ver = curr_config_ver

            # ------------------------------------------------------------------
            # 2 & 3. FRAME SKIPPING + YOLO SENSITIVITAS TINGGI (conf=0.15, iou=0.5, imgsz=800)
            # ------------------------------------------------------------------
            if frame_count == 1 or frame_count % FRAME_SKIP == 0:
                target_classes = [2, 5, 7] if v_mode == "car" else [2, 3, 5, 7]

                results = model.predict(
                    source=frame,
                    classes=target_classes,
                    conf=CONF_THRES,
                    iou=IOU_THRES,
                    imgsz=YOLO_IMGSZ,
                    verbose=False,
                )[0]
                detections = sv.Detections.from_ultralytics(results)

                # Update ByteTrack & trigger LineZone
                detections = tracker.update_with_detections(detections)
                line_zone.trigger(detections)

                labels = []
                for i in range(len(detections)):
                    cid = int(detections.class_id[i]) if detections.class_id is not None else 2
                    c_name = CLASS_NAMES.get(cid, "Mobil")
                    conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                    tid = detections.tracker_id[i] if detections.tracker_id is not None else None
                    if tid is not None and int(tid) >= 0:
                        labels.append(f"#{int(tid)} {c_name} {conf:.2f}")
                    else:
                        labels.append(f"{c_name} {conf:.2f}")

                last_detections = detections
                last_labels = labels

            # Di antara frame yang di-skip, gambar ulang bounding box & tracker_id dari frame terakhir
            if len(last_detections) > 0:
                frame = box_annotator.annotate(scene=frame, detections=last_detections)
                frame = label_annotator.annotate(scene=frame, detections=last_detections, labels=last_labels)

            # Hitung jumlah mobil masuk/keluar/total
            tot_in = max(0, base_in + line_zone.in_count)
            tot_out = max(0, base_out + line_zone.out_count)
            total = tot_in + tot_out

            # Gambar garis (LineZone) di setiap frame
            p_s = (int(line_zone.vector.start.x), int(line_zone.vector.start.y))
            p_e = (int(line_zone.vector.end.x), int(line_zone.vector.end.y))
            cv2.line(frame, p_s, p_e, (0, 212, 255), 3)
            cv2.circle(frame, p_s, 6, (0, 0, 255), -1)
            cv2.circle(frame, p_e, 6, (0, 0, 255), -1)

            if orientation == "vertical":
                cv2.putText(frame, "GARIS HITUNG", (max(10, p_s[0] - 60), 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)
            else:
                cv2.putText(frame, "GARIS HITUNG", (30, max(25, p_s[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2)

            # Gambar teks jumlah mobil di setiap frame agar responsif
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

            # Hitung FPS aktual
            fps_counter += 1
            now = time.time()
            with _state_lock:
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

            # Kompresi output ke browser menggunakan kualitas JPEG 75 (70-80)
            ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ret:
                with self.frame_cond:
                    self.latest_jpeg = buffer.tobytes()
                    self.latest_jpeg_id += 1
                    self.frame_cond.notify_all()

            # Pada frame yang di-skip (sangat cepat ~2ms), jaga pacing halus agar ritme ~30 FPS stabil
            if frame_count % FRAME_SKIP != 0:
                target_dt = 1.0 / max(20.0, min(30.0, self.stream.fps))
                elapsed = time.time() - t_cycle_start
                if elapsed < target_dt * 0.6:
                    time.sleep((target_dt * 0.6) - elapsed)


def _ensure_pipeline(url):
    """Menjalankan pipeline Producer (VideoStream) & Consumer (CCTVProcessor) sebagai singleton."""
    global _active_stream, _active_processor
    with _stream_lock:
        stream_restarted = False
        if (
            _active_stream is None
            or not _active_stream.running
            or not _active_stream.is_alive()
            or _active_stream.url != url
        ):
            if _active_stream is not None:
                _active_stream.stop()
            _active_stream = VideoStream(url).start_stream()
            stream_restarted = True

    with _processor_lock:
        if (
            stream_restarted
            or _active_processor is None
            or not _active_processor.running
            or not _active_processor.is_alive()
        ):
            if _active_processor is not None:
                _active_processor.running = False
            _active_processor = CCTVProcessor(_active_stream).start_processor()

    return _active_processor


def generate_cctv_frames(url_m3u8):
    """
    Generator Flask non-blocking yang mengirim stream frame JPEG hasil olahan
    pipeline Producer-Consumer (VideoStream + CCTVProcessor) ke browser.
    """
    processor = _ensure_pipeline(url_m3u8)
    last_sent_id = 0

    while True:
        jpeg_bytes, current_id = processor.get_jpeg_frame(last_sent_id, timeout=0.5)
        if jpeg_bytes and current_id != last_sent_id:
            last_sent_id = current_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        else:
            time.sleep(0.01)
