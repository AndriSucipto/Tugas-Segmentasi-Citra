import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

# --------- Variabel global ----------
original_img = None
current_result = None


# --------- Tampilkan gambar ----------
def show_image_on_label(img, label, is_bgr=True, max_size=(420, 420)):
    if img is None:
        return

    if is_bgr:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img

    pil_img = Image.fromarray(img_rgb)
    pil_img.thumbnail(max_size, Image.LANCZOS)

    photo = ImageTk.PhotoImage(pil_img)
    label.image = photo
    label.config(image=photo)


# --------- Region Growing ----------
def region_growing(gray, seed, thresh=20):
    h, w = gray.shape
    mask = np.zeros_like(gray, np.uint8)
    visited = np.zeros_like(gray, bool)

    sy, sx = seed
    seed_val = int(gray[sy, sx])

    stack = [(sy, sx)]

    while stack:
        y, x = stack.pop()

        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x]:
            continue

        visited[y, x] = True

        if abs(int(gray[y, x]) - seed_val) <= thresh:
            mask[y, x] = 255

            for ny in range(y - 1, y + 2):
                for nx in range(x - 1, x + 2):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny, nx))

    return mask


# --------- Split & Merge ----------
def split_region(gray, x, y, w, h, min_size, var_thresh, regions):
    roi = gray[y:y+h, x:x+w]
    var = np.var(roi)

    if w <= min_size or h <= min_size or var <= var_thresh:
        regions.append((x, y, w, h, int(np.mean(roi))))
        return

    w2, h2 = w // 2, h // 2

    split_region(gray, x,       y,       w2,     h2,     min_size, var_thresh, regions)
    split_region(gray, x+w2,    y,       w-w2,   h2,     min_size, var_thresh, regions)
    split_region(gray, x,       y+h2,    w2,     h-h2,   min_size, var_thresh, regions)
    split_region(gray, x+w2,    y+h2,    w-w2,   h-h2,   min_size, var_thresh, regions)


def split_and_merge(gray, min_size=16, var_thresh=500):
    h, w = gray.shape
    regions = []
    split_region(gray, 0, 0, w, h, min_size, var_thresh, regions)

    result = np.zeros_like(gray, np.uint8)

    for x, y, ww, hh, val in regions:
        result[y:y+hh, x:x+ww] = val

    return result


# --------- Load gambar ----------
def pilih_gambar():
    global original_img, current_result

    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )

    if not path:
        return

    original_img = cv2.imread(path)
    current_result = None

    show_image_on_label(original_img, lbl_original, is_bgr=True)
    lbl_info.config(text=f"File: {path}")
    lbl_result_text.config(text="Hasil: -")
    lbl_result.config(image='')


# --------- Proses segmentasi ----------
def proses_metode():
    global original_img, current_result

    if original_img is None:
        messagebox.showwarning("Peringatan", "Silakan pilih gambar terlebih dahulu.")
        return

    metode = combo_metode.get()

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # === 1. CANNY ===
    if metode == "Discontinuity - Canny Edge":
        edges = cv2.Canny(gray, 100, 200)
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        current_result = edges_color
        lbl_result_text.config(text="Discontinuity - Canny")

    # === 2. Threshold ===
    elif metode == "Similarity - Threshold (Otsu)":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_color = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        current_result = th_color
        lbl_result_text.config(text="Similarity - Threshold (Otsu)")

    # === 3. Region Growing Improved ===
    elif metode == "Similarity - Region Growing":

        h, w = gray.shape
        seed = (h // 2, w // 2)

        mask = region_growing(gray, seed, thresh=25)

        hasil = original_img.copy()
        hasil[mask == 0] = 0

        current_result = hasil
        lbl_result_text.config(text="Similarity - Region Growing (Warna)")

    # === 4. Split & Merge ===
    elif metode == "Similarity - Split & Merge":
        sm = split_and_merge(gray)
        sm_color = cv2.cvtColor(sm, cv2.COLOR_GRAY2BGR)
        current_result = sm_color
        lbl_result_text.config(text="Similarity - Split & Merge")

    # === 5. K-Means Optimized ===
    elif metode == "Similarity - Clustering (K-Means)":

        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        # Resize agar cepat
        h, w, _ = img_rgb.shape
        max_size = 300

        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_rgb = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Siapkan data
        pixel_values = img_rgb.reshape((-1, 3)).astype(np.float32)

        # Sampling jika besar
        if len(pixel_values) > 150000:
            pixel_values = pixel_values[::5]

        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.2)

        # Jalankan cepat
        _, labels, centers = cv2.kmeans(pixel_values, K, None,
                                        criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        segmented = centers[labels.flatten()]
        segmented = segmented.reshape((-1, 3))

        # Reconstruct ke ukuran gambar
        segmented_img = segmented.reshape(img_rgb.shape)

        current_result = segmented_img
        lbl_result_text.config(text="Similarity - Clustering (Fast)")

    show_image_on_label(current_result, lbl_result, is_bgr=False)


# --------- Reset ----------
def reset_gui():
    global original_img, current_result

    original_img = None
    current_result = None

    lbl_original.config(image='')
    lbl_result.config(image='')
    lbl_info.config(text="Belum ada gambar dipilih")
    lbl_result_text.config(text="Hasil: -")

    combo_metode.set("Discontinuity - Canny Edge")


# --------- GUI ----------
root = tk.Tk()
root.title("Aplikasi Segmentasi Citra")
root.geometry("1000x620")
root.configure(bg="#1e1e2f")

style = ttk.Style()
style.theme_use("clam")

style.configure("TButton", padding=6, font=("Segoe UI", 10, "bold"))
style.configure("TLabel", background="#1e1e2f", foreground="white", font=("Segoe UI", 10))

# Title
ttk.Label(root, text="Metode Segmentasi Citra", font=("Segoe UI", 14, "bold"), background="#1e1e2f", foreground="white").pack(pady=(10, 0))
ttk.Label(root, text="Discontinuity & Similarity", background="#1e1e2f", foreground="white").pack(pady=(0, 10))

# Control panel
frame_top = tk.Frame(root, bg="#1e1e2f")
frame_top.pack(fill=tk.X, padx=20, pady=10)

ttk.Button(frame_top, text="Pilih Gambar", command=pilih_gambar).pack(side=tk.LEFT)

combo_metode = ttk.Combobox(
    frame_top, state="readonly", width=35,
    values=[
        "Discontinuity - Canny Edge",
        "Similarity - Threshold (Otsu)",
        "Similarity - Region Growing",
        "Similarity - Split & Merge",
        "Similarity - Clustering (K-Means)"
    ]
)
combo_metode.pack(side=tk.LEFT, padx=5)
combo_metode.set("Discontinuity - Canny Edge")

ttk.Button(frame_top, text="Proses", command=proses_metode).pack(side=tk.LEFT, padx=5)
ttk.Button(frame_top, text="Reset", command=reset_gui).pack(side=tk.LEFT, padx=5)

lbl_info = ttk.Label(frame_top, text="Belum ada gambar dipilih")
lbl_info.pack(side=tk.LEFT, padx=10)

# Tampilan gambar
frame_mid = tk.Frame(root, bg="#1e1e2f")
frame_mid.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

# Original
frame_left = tk.Frame(frame_mid, bg="#1e1e2f")
frame_left.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

ttk.Label(frame_left, text="Citra Asli").pack(pady=(0, 5))
lbl_original = tk.Label(frame_left, bg="#2b2b3c")
lbl_original.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

# Result
frame_right = tk.Frame(frame_mid, bg="#1e1e2f")
frame_right.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)

lbl_result_text = ttk.Label(frame_right, text="Hasil: -")
lbl_result_text.pack(pady=(0, 5))

lbl_result = tk.Label(frame_right, bg="#2b2b3c")
lbl_result.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

root.mainloop()
