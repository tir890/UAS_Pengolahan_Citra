# batch_process.py
import os
import cv2
import numpy as np
import processing_utils as utils

def simpan_gambar_aman(img, path_tujuan):
    """Fungsi pembantu untuk menyimpan gambar dengan aman walaupun path Windows ada spasi"""
    try:
        ekstensi = os.path.splitext(path_tujuan)[1]
        sukses, array_buf = cv2.imencode(ekstensi, img)
        if sukses:
            array_buf.tofile(path_tujuan)
            return True
    except Exception as e:
        print(f"Gagal menyimpan {path_tujuan}: {e}")
    return False

def jalankan_automasi_batch():
    # 1. Tentukan path folder sesuai struktur project
    folder_input = os.path.join("dataset", "original")
    folder_output = os.path.join("dataset", "output")
    
    # 2. Buat folder output jika belum ada
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
        print(f"[INFO] Folder '{folder_output}' berhasil dibuat.")

    # 3. Ambil semua file gambar di folder original
    list_file = [f for f in os.listdir(folder_input) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(list_file) == 0:
        print("[PERINGATAN] Tidak ditemukan gambar (.jpg/.png) di folder 'dataset/original/'.")
        print("Silakan masukkan minimal 20 gambar daun ke folder tersebut terlebih dahulu!")
        return

    print(f"[START] Memulai pemrosesan otomatis untuk {len(list_file)} gambar...\n")

    # 4. Looping proses untuk setiap gambar
    for indeks, nama_file in enumerate(list_file, start=1):
        path_file_input = os.path.join(folder_input, nama_file)
        print(f"--> Memproses [{indeks}/{len(list_file)}]: {nama_file}")
        
        # --- PROSES 1: Akuisisi & Konversi ---
        img_rgb, img_gray, img_biner = utils.konversi_citra(path_file_input)
        
        if img_rgb is None:
            print(f"    [ERROR] Gagal membaca gambar {nama_file}. Dilewati.")
            continue
            
        # --- PROSES 2: Perbaikan Kualitas ---
        img_enhanced = utils.perbaikan_kualitas(img_gray, metode='histogram')
        
        # --- PROSES 3: Filtering (Penapisan) ---
        img_filtered = utils.penapisan_filter(img_enhanced, tipe='gaussian')
        
        # --- PROSES 4: Deteksi Tepi ---
        img_edge = utils.deteksi_tepi(img_filtered, metode='canny')
        
        # --- PROSES 5: Segmentasi Citra ---
        img_segmented = utils.segmentasi_citra(img_gray)
        
        # --- TAHAP PENYIMPANAN HASIL ---
        # Kita beri nama file yang terstruktur agar mudah disusun di Word/Laporan
        nama_dasar = f"daun_{indeks}"
        
        simpan_gambar_aman(img_rgb, os.path.join(folder_output, f"{nama_dasar}_1_original.jpg"))
        simpan_gambar_aman(img_gray, os.path.join(folder_output, f"{nama_dasar}_2_grayscale.jpg"))
        simpan_gambar_aman(img_biner, os.path.join(folder_output, f"{nama_dasar}_3_biner.jpg"))
        simpan_gambar_aman(img_enhanced, os.path.join(folder_output, f"{nama_dasar}_4_perbaikan.jpg"))
        simpan_gambar_aman(img_filtered, os.path.join(folder_output, f"{nama_dasar}_5_filter.jpg"))
        simpan_gambar_aman(img_edge, os.path.join(folder_output, f"{nama_dasar}_6_tepi.jpg"))
        simpan_gambar_aman(img_segmented, os.path.join(folder_output, f"{nama_dasar}_7_segmentasi.jpg"))

    print("\n[SELESAI] Semua gambar berhasil diproses!")
    print(f"Silakan cek hasilnya di folder: {os.path.abspath(folder_output)}")

if __name__ == "__main__":
    jalankan_automasi_batch()