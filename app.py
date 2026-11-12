import os
import uuid
from flask import Flask, render_template, request, url_for, Response, stream_with_context, jsonify
from werkzeug.utils import secure_filename

from detector import detect_image, count_video
from analysis import analisis_gambar, konvolusi_kernel
from cctv import generate_cctv_frames, get_cctv_stats, reset_cctv_counts, update_cctv_config

# ============================================================
# KONFIGURASI CCTV STREAM
# ============================================================
# Masukkan URL stream HLS (.m3u8) Anda di sini.
# Contoh: CCTV_URL = "https://example.com/live/cctv.m3u8"
CCTV_URL = "https://cctvjss.jogjakota.go.id/kotabaru/ANPR-Jl-FM-Noto-Raminten.stream/chunklist_w852002735.m3u8"

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def bersihkan_uploads():
    """Hapus semua file lama di folder static/uploads/ agar penyimpanan tidak menumpuk."""
    if not os.path.exists(UPLOAD_FOLDER):
        return
    for nama_file in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, nama_file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"[CLEANUP ERROR] Gagal menghapus {file_path}: {e}")


def _nama_unik(filename):
    """Buat nama file unik agar tidak saling menimpa dan selalu memiliki ekstensi yang valid."""
    ext = os.path.splitext(filename)[1].lower()
    base = os.path.splitext(secure_filename(filename))[0]
    if not base:
        base = "upload"
    return f"{uuid.uuid4().hex}_{base}{ext}"


def _url(nama_file):
    """URL statis untuk file di folder uploads."""
    return url_for("static", filename=f"uploads/{nama_file}")


@app.route("/", methods=["GET", "POST"])
@app.route("/gambar", methods=["GET", "POST"])
def gambar():
    hasil = None
    error = None
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename:
            bersihkan_uploads()
            nama = _nama_unik(f.filename)
            in_path = os.path.join(UPLOAD_FOLDER, nama)
            out_path = os.path.join(UPLOAD_FOLDER, f"hasil_{nama}")
            f.save(in_path)
            try:
                jumlah = detect_image(in_path, out_path)
                if os.path.exists(out_path):
                    hasil = {
                        "jumlah": jumlah,
                        "img_url": _url(os.path.basename(out_path)),
                    }
                else:
                    error = "Gagal memproses gambar. Format file tidak didukung atau gambar rusak."
            except Exception as e:
                error = f"Terjadi kesalahan saat deteksi: {e}"
        else:
            error = "Silakan pilih file gambar terlebih dahulu."
    return render_template("gambar.html", hasil=hasil, error=error)


@app.route("/video", methods=["GET", "POST"])
def video():
    hasil = None
    error = None
    if request.method == "POST":
        f = request.files.get("file")
        if f and f.filename:
            bersihkan_uploads()
            nama = _nama_unik(f.filename)
            in_path = os.path.join(UPLOAD_FOLDER, nama)
            out_name = f"hasil_{nama.rsplit('.', 1)[0]}.mp4"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            f.save(in_path)
            try:
                res = count_video(in_path, out_path)
                if res and os.path.exists(out_path):
                    hasil = res
                    hasil["video_url"] = _url(out_name)
                else:
                    error = "Gagal memproses video. File video tidak dapat dibaca."
            except Exception as e:
                error = f"Terjadi kesalahan saat memproses video: {e}"
        else:
            error = "Silakan pilih file video terlebih dahulu."
    return render_template("video.html", hasil=hasil, error=error)


@app.route("/cctv")
def cctv():
    stats = get_cctv_stats()
    return render_template("cctv.html", cctv_url=CCTV_URL, stats=stats)


@app.route("/cctv_feed")
def cctv_feed():
    response = Response(
        stream_with_context(generate_cctv_frames(CCTV_URL)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/cctv_data")
def cctv_data():
    """Mengembalikan data hitungan terbaru (JSON) untuk update realtime di halaman."""
    return jsonify(get_cctv_stats())


@app.route("/cctv_reset", methods=["POST"])
def cctv_reset():
    """Mereset hitungan sementara."""
    reset_cctv_counts()
    return jsonify({"status": "ok", "stats": get_cctv_stats()})


@app.route("/cctv_config", methods=["GET", "POST"])
def cctv_config():
    """Mengatur posisi atau orientasi garis deteksi secara dinamis."""
    orientation = request.values.get("orientation")
    line_pos = request.values.get("line_pos")
    vehicle_mode = request.values.get("vehicle_mode")
    update_cctv_config(orientation=orientation, line_pos=line_pos, vehicle_mode=vehicle_mode)
    return jsonify(get_cctv_stats())


@app.route("/analisis", methods=["GET", "POST"])
def analisis():
    hasil = None
    error = None
    kernel_values = [0, -1, 0, -1, 5, -1, 0, -1, 0]  # default sharpen
    if request.method == "POST":
        # Ambil 9 nilai kernel dari form
        try:
            kernel_values = [float(request.form.get(f"k{i}", 0)) for i in range(9)]
        except ValueError:
            kernel_values = [0, -1, 0, -1, 5, -1, 0, -1, 0]

        f = request.files.get("file")
        if f and f.filename:
            bersihkan_uploads()
            nama = _nama_unik(f.filename)
            in_path = os.path.join(UPLOAD_FOLDER, nama)
            f.save(in_path)

            try:
                hasil_analisis = analisis_gambar(in_path)
                hasil_konvolusi = konvolusi_kernel(in_path, kernel_values)

                if hasil_analisis and hasil_konvolusi:
                    hasil = {
                        "asli": _url(nama),
                        "grayscale": hasil_analisis["grayscale"],
                        "histogram": hasil_analisis["histogram"],
                        "blur": hasil_analisis["blur"],
                        "sharpen": hasil_analisis["sharpen"],
                        "sobel": hasil_analisis["sobel"],
                        "canny": hasil_analisis["canny"],
                        "konv_manual": hasil_konvolusi["manual"],
                        "konv_lib": hasil_konvolusi["library"],
                    }
                else:
                    error = "Gagal menganalisis gambar. Format tidak didukung atau file rusak."
            except Exception as e:
                error = f"Terjadi kesalahan saat analisis gambar: {e}"
        else:
            error = "Silakan pilih file gambar terlebih dahulu."
    return render_template("analisis.html", hasil=hasil, kernel=kernel_values, error=error)


if __name__ == "__main__":
    from waitress import serve
    print("Server berjalan di http://127.0.0.1:5000 (host=0.0.0.0, port=5000, threads=8)")
    serve(app, host="0.0.0.0", port=5000, threads=8)
