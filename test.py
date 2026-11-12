import cv2
import supervision as sv
from ultralytics import YOLO

# ============================================================
# KONFIGURASI
# ============================================================
VIDEO_PATH   = "test.mp4"        # ganti dengan path video Anda
MODEL_PATH   = "yolov8n.pt"       # model YOLO pretrained
CONF_THRES   = 0.3                # ambang keyakinan deteksi
CAR_CLASS_ID = 2                  # class id 'car' di COCO

# ============================================================
# INISIALISASI MODEL, TRACKER, DAN GARIS
# ============================================================
model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()

cap = cv2.VideoCapture(VIDEO_PATH)

# Ambil ukuran frame untuk menentukan posisi garis secara otomatis
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Garis horizontal di tengah frame (bisa disesuaikan)
LINE_Y = frame_h // 2
line_zone = sv.LineZone(
    start=sv.Point(0, LINE_Y),
    end=sv.Point(frame_w, LINE_Y)
)

print(f"Resolusi video: {frame_w}x{frame_h}, garis di y={LINE_Y}")

# ============================================================
# PROSES VIDEO FRAME PER FRAME
# ============================================================
frame = None  # inisialisasi agar aman jika video gagal dibuka

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Deteksi ---
    results = model(frame, classes=[CAR_CLASS_ID], conf=CONF_THRES)[0]
    detections = sv.Detections.from_ultralytics(results)

    # --- Tracking ---
    detections = tracker.update_with_detections(detections)

    # --- Hitung melewati garis ---
    line_zone.trigger(detections)

    # --- Gambar kotak deteksi ---
    frame = box_annotator.annotate(frame, detections)

    # --- Gambar garis virtual ---
    cv2.line(frame, (0, LINE_Y), (frame_w, LINE_Y), (0, 255, 255), 2)

    # === TAMBAHAN #1: panel jumlah ===
    in_count = line_zone.in_count
    out_count = line_zone.out_count
    total = in_count + out_count

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 130), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, f"Masuk  : {in_count}", (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Keluar : {out_count}", (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    cv2.putText(frame, f"Total  : {total}", (25, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    # === AKHIR TAMBAHAN #1 ===

    # --- Tampilkan ---
    cv2.imshow("Deteksi & Hitung Mobil", frame)
    if cv2.waitKey(1) == 27:  # tekan ESC untuk keluar
        break

# ============================================================
# SETELAH VIDEO SELESAI
# ============================================================

# === TAMBAHAN #3: simpan frame akhir ===
if frame is not None:
    cv2.imwrite("hasil_penghitungan_akhir.jpg", frame)
    print("Frame akhir disimpan sebagai 'hasil_penghitungan_akhir.jpg'")
# === AKHIR TAMBAHAN #3 ===

# === TAMBAHAN #2: ringkasan di terminal ===
print("\n" + "=" * 40)
print("     HASIL PENGHITUNGAN AKHIR")
print("=" * 40)
print(f"  Mobil masuk  : {line_zone.in_count}")
print(f"  Mobil keluar : {line_zone.out_count}")
print(f"  Total mobil  : {line_zone.in_count + line_zone.out_count}")
print("=" * 40 + "\n")
# === AKHIR TAMBAHAN #2 ===

cap.release()
cv2.destroyAllWindows()