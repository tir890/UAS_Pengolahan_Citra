# processing_utils.py
import cv2
import numpy as np

def konversi_citra(img_path):
    import numpy as np
    try:
        # Membaca gambar menggunakan numpy agar aman dari spasi nama folder Windows
        img_array = np.fromfile(img_path, np.uint8)
        img_rgb = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        img_rgb = cv2.imread(img_path)

    if img_rgb is None:
        return None, None, None
        
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_rgb, img_gray, img_binary

def perbaikan_kualitas(img_gray, metode='histogram'):
    """
    2. Perbaikan Kualitas Citra: Histogram Equalization atau Contrast Stretching
    """
    if metode == 'histogram':
        # Histogram Equalization
        return cv2.equalizeHist(img_gray)
    elif metode == 'contrast':
        # Contrast Stretching sederhana
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        return cv2.LUT(img_gray, table)
    else:
        return img_gray

def penapisan_filter(img_gray, tipe='gaussian'):
    """
    3. Filtering: Mean Filter, Median Filter, dan Gaussian Filter
    """
    if tipe == 'mean':
        # Mean Filter (Blur dengan kernel 5x5)
        return cv2.blur(img_gray, (5, 5))
    elif tipe == 'median':
        # Median Filter (Bagus untuk menghilangkan noise)
        return cv2.medianBlur(img_gray, 5)
    elif tipe == 'gaussian':
        # Gaussian Filter
        return cv2.GaussianBlur(img_gray, (5, 5), 0)
    else:
        return img_gray

def deteksi_tepi(img_filtered, metode='canny'):
    """
    4. Deteksi Tepi: Sobel atau Canny
    """
    if metode == 'canny':
        # Deteksi Tepi Canny
        return cv2.Canny(img_filtered, 50, 150)
    elif metode == 'sobel':
        # Sobel kombinasi X dan Y
        sobelx = cv2.Sobel(img_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return np.uint8(sobel_combined)
    else:
        return img_filtered

def segmentasi_citra(img_gray):
    """
    5. Segmentasi Citra: Thresholding / Otsu Segmentation
    Memisahkan objek daun dari latar belakang abu-abunya
    """
    # Menggunakan inverse thresholding karena background daun ini agak terang/abu-abu
    # dan kita ingin daunnya tersegmen dengan jelas
    _, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Opsional: membersihkan lubang kecil menggunakan operasi morfologi penutupan (Closing)
    kernel = np.ones((5,5), np.uint8)
    segmented = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    
    return segmented