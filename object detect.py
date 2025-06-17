import customtkinter as ctk
from tkinter import filedialog, messagebox
import cv2
import torch
import time
from ultralytics import YOLO

# تنظیمات GUI
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# بررسی GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ پردازش روی: {device}")

# بارگذاری مدل YOLO با نسخه دقیق‌تر
try:
    model = YOLO("yolov8l.pt").to(device)  # اگر کند بود از "yolov8m.pt" استفاده کنید
    print("✅ مدل با موفقیت بارگذاری شد!")
except Exception as e:
    print(f"⛔ خطا در بارگذاری مدل: {e}")
    messagebox.showerror("خطا", "مدل YOLO یافت نشد! بررسی کنید که فایل موجود باشد.")

# تابع انتخاب فایل
def select_file(file_types):
    root = ctk.CTk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=file_types)
    root.destroy()
    return file_path

# پردازش تصویر
def process_image():
    image_path = select_file([("Images", "*.jpg;*.png;*.jpeg")])
    if not image_path:
        messagebox.showerror("خطا", "هیچ تصویری انتخاب نشد!")
        return

    print(f"📂 تصویر انتخاب شده: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("خطا", f"تصویر پیدا نشد: {image_path}")
        return

    cv2.destroyAllWindows()

    start_time = time.time()
    results = model(image, conf=0.3, iou=0.5)  
    end_time = time.time()

    detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]
    detected_text = ", ".join(set(detected_objects)) if detected_objects else "هیچ شی‌ای شناسایی نشد!"

    for result in results:
        image = result.plot()

    cv2.imshow("تشخیص اشیا در تصویر", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    messagebox.showinfo("نتایج پردازش", f"✅ تعداد اشیای تشخیص داده‌شده: {len(results[0].boxes)}\n✅ زمان پردازش: {round(end_time - start_time, 2)} ثانیه\n✅ اشیای شناسایی‌شده: {detected_text}")

# پردازش ویدیو
def process_video():
    video_path = select_file([("Videos", "*.mp4;*.avi;*.mov")])
    if not video_path:
        messagebox.showerror("خطا", "هیچ ویدئویی انتخاب نشد!")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("خطا", f"ویدیو پیدا نشد: {video_path}")
        return

    detected_objects = set()
    total_frames = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, iou=0.5)
        detected_objects.update([model.names[int(box.cls)] for box in results[0].boxes])
        frame = results[0].plot()
        total_frames += 1

        cv2.imshow("تشخیص اشیا در ویدیو", frame)
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()

    detected_text = ", ".join(detected_objects) if detected_objects else "هیچ شی‌ای شناسایی نشد!"
    messagebox.showinfo("نتایج پردازش", f"✅ تعداد فریم‌های پردازش شده: {total_frames}\n✅ زمان پردازش: {round(end_time - start_time, 2)} ثانیه\n✅ اشیای شناسایی‌شده در ویدیو: {detected_text}")

# پردازش وب‌کم
def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("خطا", "وب‌کم در دسترس نیست!")
        return

    detected_objects = set()
    total_frames = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.3, iou=0.5)
        detected_objects.update([model.names[int(box.cls)] for box in results[0].boxes])
        frame = results[0].plot()
        total_frames += 1

        cv2.imshow("تشخیص اشیا در وب‌کم", frame)
        if cv2.waitKey(1) & 0xFF in [ord("q"), 27]:
            break

    end_time = time.time()
    cap.release()
    cv2.destroyAllWindows()

    detected_text = ", ".join(detected_objects) if detected_objects else "هیچ شی‌ای شناسایی نشد!"
    messagebox.showinfo("نتایج پردازش", f"✅ تعداد فریم‌های پردازش شده: {total_frames}\n✅ زمان پردازش: {round(end_time - start_time, 2)} ثانیه\n✅ اشیای شناسایی‌شده در وب‌کم: {detected_text}")

# طراحی GUI
class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("YOLO Object Detection")
        self.geometry("400x400")

        label = ctk.CTkLabel(self, text="چه کاری می‌خواهید انجام دهید؟", font=("Arial", 18))
        label.pack(pady=20)

        btn_image = ctk.CTkButton(self, text="  عکس پردازش ", command=process_image)
        btn_image.pack(pady=10)

        btn_video = ctk.CTkButton(self, text=" ویدیو پردازش", command=process_video)
        btn_video.pack(pady=10)

        btn_webcam = ctk.CTkButton(self, text=" وب‌کم پردازش", command=process_webcam)
        btn_webcam.pack(pady=10)

if __name__ == "__main__":
    app = App()
    app.mainloop()
