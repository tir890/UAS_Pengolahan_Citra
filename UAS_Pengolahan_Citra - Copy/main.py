# main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import processing_utils as utils

class PCDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Pengolahan Citra Digital - UAS")
        self.root.geometry("1100x650")
        self.root.configure(bg="#f0f0f0")

        # Variabel Penyimpan Citra
        self.path_gambar = None
        self.img_asli = None
        self.img_gray = None
        self.img_biner = None
        self.img_aktif = None # Menampung gambar grayscale aktif untuk filter/tepi

        self.buat_widget()

    def buat_widget(self):
        # --- PANEL ATAS: Tombol Utama ---
        frame_atas = tk.Frame(self.root, bg="#333333", padx=10, pady=10)
        frame_atas.pack(fill=tk.X)

        btn_upload = tk.Button(frame_atas, text="📁 Upload Gambar Daun", command=self.upload_gambar, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_upload.pack(side=tk.LEFT, padx=10)

        lbl_judul = tk.Label(frame_atas, text="Mini Project PCD - Tema: Daun Tanaman", bg="#333333", fg="white", font=("Arial", 12, "bold"))
        lbl_judul.pack(side=tk.RIGHT, padx=10)


        # --- PANEL KIRI: Menu Kontrol/Proses ---
        frame_kiri = tk.LabelFrame(self.root, text=" Pilihan Proses (Minimal 5 Proses) ", font=("Arial", 10, "bold"), bg="#f9f9f9", padx=10, pady=10)
        frame_kiri.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # 1. Konversi Citra [cite: 36]
        tk.Label(frame_kiri, text="1. Konversi Citra:", bg="#f9f9f9", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(5,0))
        tk.Button(frame_kiri, text="RGB ke Grayscale", width=22, command=self.proses_grayscale).pack(pady=2)
        tk.Button(frame_kiri, text="RGB ke Biner", width=22, command=self.proses_biner).pack(pady=2)

        # 2. Perbaikan Kualitas [cite: 37, 38]
        tk.Label(frame_kiri, text="2. Perbaikan Kualitas:", bg="#f9f9f9", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        tk.Button(frame_kiri, text="Histogram Equalization", width=22, command=lambda: self.proses_perbaikan('histogram')).pack(pady=2)
        tk.Button(frame_kiri, text="Contrast Stretching", width=22, command=lambda: self.proses_perbaikan('contrast')).pack(pady=2)

        # 3. Filtering [cite: 39]
        tk.Label(frame_kiri, text="3. Filtering (Penapisan):", bg="#f9f9f9", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        self.cb_filter = ttk.Combobox(frame_kiri, values=["mean", "median", "gaussian"], state="readonly", width=20)
        self.cb_filter.set("gaussian")
        self.cb_filter.pack(pady=2)
        tk.Button(frame_kiri, text="Terapkan Filter", width=22, command=self.proses_filter, bg="#2196F3", fg="white").pack(pady=2)

        # 4. Deteksi Tepi [cite: 40]
        tk.Label(frame_kiri, text="4. Deteksi Tepi:", bg="#f9f9f9", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        self.cb_tepi = ttk.Combobox(frame_kiri, values=["canny", "sobel"], state="readonly", width=20)
        self.cb_tepi.set("canny")
        self.cb_tepi.pack(pady=2)
        tk.Button(frame_kiri, text="Terapkan Deteksi Tepi", width=22, command=self.proses_tepi, bg="#9C27B0", fg="white").pack(pady=2)

        # 5. Segmentasi Citra [cite: 41]
        tk.Label(frame_kiri, text="5. Segmentasi Citra:", bg="#f9f9f9", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10,0))
        tk.Button(frame_kiri, text="Otsu Thresholding", width=22, command=self.proses_segmentasi, bg="#FF9800", fg="white").pack(pady=2)


        # --- PANEL KANAN: Tempat Menampilkan Gambar ---
        frame_kanan = tk.Frame(self.root, bg="#ffffff", bd=1, relief=tk.SUNKEN)
        frame_kanan.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tempat Gambar Asli
        self.frame_img_asli = tk.Frame(frame_kanan, bg="#ffffff")
        self.frame_img_asli.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(self.frame_img_asli, text="Gambar Asli (Sebelum)", bg="#ffffff", font=("Arial", 10, "bold")).pack(pady=5)
        self.canvas_asli = tk.Label(self.frame_img_asli, text="Belum ada gambar", bg="#e0e0e0")
        self.canvas_asli.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tempat Gambar Hasil Proses
        self.frame_img_hasil = tk.Frame(frame_kanan, bg="#ffffff")
        self.frame_img_hasil.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        tk.Label(self.frame_img_hasil, text="Gambar Hasil Proses (Sesudah)", bg="#ffffff", font=("Arial", 10, "bold")).pack(pady=5)
        self.canvas_hasil = tk.Label(self.frame_img_hasil, text="Belum ada proses", bg="#e0e0e0")
        self.canvas_hasil.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # --- FUNGSI LOGIKA LOGISTIK ---
    def upload_gambar(self):
        try:
            # Ditambahkan filter ekstensi huruf besar (*.JPG, *.JPEG, *.PNG) agar kebal dari case-sensitive dataset
            self.path_gambar = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.JPEG *.PNG")]
            )
            if not self.path_gambar:
                return
                
            # Panggil utils untuk inisialisasi gambar awal
            self.img_asli, self.img_gray, self.img_biner = utils.konversi_citra(self.path_gambar)
            
            if self.img_asli is None:
                messagebox.showerror("Error", "Gagal membaca file gambar. Pastikan file tidak rusak.")
                return
                
            self.img_aktif = self.img_gray.copy() # Set default aktif ke gray
            
            # Tampilkan gambar asli ke GUI
            self.tampilkan_gambar(self.img_asli, asli=True)
            messagebox.showinfo("Sukses", "Gambar berhasil dimuat!")
            
        except Exception as e:
            messagebox.showerror("Error Upload", f"Terjadi kesalahan saat memuat gambar:\n{str(e)}")

    def tampilkan_gambar(self, cv_img, asli=True):
        try:
            if cv_img is None:
                return
                
            # Mengubah OpenCV (BGR) ke PIL format (RGB) agar bisa dibaca Tkinter
            if len(cv_img.shape) == 3:
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                
            img_pil = Image.fromarray(cv_img)
            # Resize otomatis agar pas di layar GUI tanpa merusak aspek rasio
            img_pil.thumbnail((400, 400))
            img_tk = ImageTk.PhotoImage(image=img_pil)

            if asli:
                self.canvas_asli.configure(image=img_tk, text="")
                self.canvas_asli.image = img_tk
            else:
                self.canvas_hasil.configure(image=img_tk, text="")
                self.canvas_hasil.image = img_tk
                
            # Memaksa Tkinter untuk langsung menyegarkan (refresh) tampilan GUI
            self.root.update_idletasks()
            
        except Exception as e:
            messagebox.showerror("Error Tampilan", f"Gagal merender gambar di layar:\n{str(e)}")

    # --- FUNGSI EKSEKUSI PROSES PCD ---
    def proses_grayscale(self):
        if self.img_gray is None: return
        self.tampilkan_gambar(self.img_gray, asli=False)

    def proses_biner(self):
        if self.img_biner is None: return
        self.tampilkan_gambar(self.img_biner, asli=False)

    def proses_perbaikan(self, metode):
        if self.img_gray is None: return
        hasil = utils.perbaikan_kualitas(self.img_gray, metode=metode)
        self.img_aktif = hasil.copy() # Perbarui gambar aktif untuk filter lanjutan
        self.tampilkan_gambar(hasil, asli=False)

    def proses_filter(self):
        if self.img_aktif is None: return
        tipe_filter = self.cb_filter.get()
        hasil = utils.penapisan_filter(self.img_aktif, tipe=tipe_filter)
        self.img_aktif = hasil.copy()
        self.tampilkan_gambar(hasil, asli=False)

    def proses_tepi(self):
        if self.img_aktif is None: return
        metode_tepi = self.cb_tepi.get()
        hasil = utils.deteksi_tepi(self.img_aktif, metode=metode_tepi)
        self.tampilkan_gambar(hasil, asli=False)

    def proses_segmentasi(self):
        if self.img_gray is None: return
        hasil = utils.segmentasi_citra(self.img_gray)
        self.tampilkan_gambar(hasil, asli=False)

if __name__ == "__main__":
    root = tk.Tk()
    app = PCDApp(root)
    root.mainloop()