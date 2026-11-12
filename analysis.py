import os
import tempfile
import cv2
import numpy as np

# Arahkan cache matplotlib ke folder temporary agar tidak ada peringatan permission
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")  # backend tanpa GUI, aman untuk Flask
import matplotlib.pyplot as plt


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


def _simpan_histogram(gray_or_rgb, output_path):
    """Simpan grafik histogram RGB ke file."""
    img = _read_image(gray_or_rgb)
    if img is None:
        return
    warna = ("b", "g", "r")  # OpenCV BGR
    plt.figure(figsize=(7, 3.2), dpi=110)
    for i, col in enumerate(warna):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col, linewidth=1.4)
    plt.xlim([0, 256])
    plt.xlabel("Intensitas")
    plt.ylabel("Jumlah piksel")
    plt.title("Histogram RGB")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def konvolusi_manual(gray, kernel):
    """Konvolusi manual matematika 3x3 (vektorisasi neighborhood untuk performa optimal)."""
    h, w = gray.shape
    k = kernel.astype(np.float32)
    gray_f = gray.astype(np.float32)
    hasil = np.zeros((h, w), dtype=np.float32)

    # Akumulasi 3x3 kernel secara manual tanpa modul eksternal filter2D
    for ki in range(3):
        for kj in range(3):
            hasil[1:h - 1, 1:w - 1] += k[ki, kj] * gray_f[ki:h - 2 + ki, kj:w - 2 + kj]

    hasil = np.clip(hasil, 0, 255).astype(np.uint8)
    return hasil


def analisis_gambar(input_path, upload_dir):
    """Jalankan semua analisis, simpan hasil ke upload_dir. Return dict path."""
    img = _read_image(input_path)
    if img is None:
        return None

    base = os.path.splitext(os.path.basename(input_path))[0]
    hasil = {}

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_path = os.path.join(upload_dir, f"{base}_gray.jpg")
    _write_image(gray_path, gray)
    hasil["grayscale"] = os.path.basename(gray_path)

    # Histogram RGB
    hist_path = os.path.join(upload_dir, f"{base}_hist.png")
    _simpan_histogram(input_path, hist_path)
    hasil["histogram"] = os.path.basename(hist_path)

    # Filter blur & sharpen (contoh tetap)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    blur_path = os.path.join(upload_dir, f"{base}_blur.jpg")
    _write_image(blur_path, blur)
    hasil["blur"] = os.path.basename(blur_path)

    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(gray, -1, kernel_sharpen)
    sharpen_path = os.path.join(upload_dir, f"{base}_sharpen.jpg")
    _write_image(sharpen_path, sharpen)
    hasil["sharpen"] = os.path.basename(sharpen_path)

    # Sobel
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    sobel_path = os.path.join(upload_dir, f"{base}_sobel.jpg")
    _write_image(sobel_path, sobel)
    hasil["sobel"] = os.path.basename(sobel_path)

    # Canny
    canny = cv2.Canny(gray, 100, 200)
    canny_path = os.path.join(upload_dir, f"{base}_canny.jpg")
    _write_image(canny_path, canny)
    hasil["canny"] = os.path.basename(canny_path)

    return hasil


def konvolusi_kernel(input_path, upload_dir, kernel_values):
    """Hitung konvolusi manual & library dengan kernel dari user."""
    img = _read_image(input_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = os.path.splitext(os.path.basename(input_path))[0]
    kernel = np.array(kernel_values, dtype=np.float32).reshape(3, 3)

    # Manual
    manual = konvolusi_manual(gray, kernel)
    manual_path = os.path.join(upload_dir, f"{base}_konv_manual.jpg")
    _write_image(manual_path, manual)

    # Library
    library = cv2.filter2D(gray, -1, kernel)
    library_path = os.path.join(upload_dir, f"{base}_konv_lib.jpg")
    _write_image(library_path, library)

    return {
        "manual": os.path.basename(manual_path),
        "library": os.path.basename(library_path),
    }
