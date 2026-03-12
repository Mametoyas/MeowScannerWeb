from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif

# 1. ลงทะเบียน HEIC Opener
pillow_heif.register_heif_opener()

def get_decimal_from_dms(dms, ref):
    """แปลง (องศา, ลิปดา, ฟิลิปดา) เป็นทศนิยม"""
    try:
        degrees = dms[0]
        minutes = dms[1]
        seconds = dms[2]
        
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        
        # ถ้าอยู่ซีกโลกใต้ (S) หรือ ตะวันตก (W) ต้องติดลบ
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception:
        return None

def extract_gps(image_path):
    try:
        image = Image.open(image_path)
        
        gps_info = {}
        
        # ---------------------------------------------------------
        # วิธีที่ 1: ดึงจาก get_ifd (เหมาะสำหรับ HEIC และไฟล์ใหม่ๆ)
        # ---------------------------------------------------------
        exif = image.getexif()
        gps_ifd = exif.get_ifd(34853) # 34853 คือ ID ของ GPS Info
        
        if gps_ifd:
            # ถ้าเจอ ให้ใช้ข้อมูลชุดนี้
            for key, value in gps_ifd.items():
                tag_name = GPSTAGS.get(key, key)
                gps_info[tag_name] = value

        # ---------------------------------------------------------
        # วิธีที่ 2: ดึงจาก _getexif (Fallback สำหรับ JPG เก่า หรือ PNG)
        # ---------------------------------------------------------
        if not gps_info: # ถ้าวิธีแรกไม่เจอ ให้ลองวิธีนี้
            # _getexif เป็นฟังก์ชันเก่าแต่ work ดีกับ JPG
            # Note: _getexif ไม่มีใน HEIF object บางทีจึงต้อง try/except หรือเช็ค attribute
            if hasattr(image, '_getexif'):
                exif_data = image._getexif()
                if exif_data:
                    for tag, value in exif_data.items():
                        tag_name = TAGS.get(tag, tag)
                        if tag_name == 'GPSInfo':
                            # แปลง key ตัวเลขใน GPSInfo ให้เป็นชื่อ
                            for t in value:
                                sub_tag = GPSTAGS.get(t, t)
                                gps_info[sub_tag] = value[t]

        # ---------------------------------------------------------
        # ตรวจสอบและคำนวณพิกัด
        # ---------------------------------------------------------
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info.get('GPSLatitudeRef', 'N'))
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info.get('GPSLongitudeRef', 'E'))
            
            if lat is not None and lon is not None:
                return {"lat": lat, "lon": lon}
        
        # ถ้าหาทุกวิธีแล้วไม่เจอ
        return None

    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None
    
# --- ตอนใช้งาน ---
# location = extract_gps("photo.jpg")
# print(location) 
# ผลลัพธ์: {'lat': 13.736717, 'lon': 100.523186}

# location = extract_gps(r"C:\Users\ASUS\Downloads\IMG_2844.HEIC")
location = extract_gps(r"C:\Users\ASUS\Downloads\IMG_20190213_201402.jpg")
# location = extract_gps(r"C:\Users\ASUS\Downloads\sddsadsda.jpg")
# location = extract_gps(r"C:\Users\ASUS\Downloads\IMG_2844.HEIC")

print(location) 