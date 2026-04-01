import cv2
import os

def extract_frames_optimized(video_path, output_folder, interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    target_size = (336, 336)
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps # Độ dài video tính bằng giây
    
    frame_id = 0
    # Tính toán thời điểm cần lấy ảnh (giây thứ 0, 1, 2...)
    for sec in range(0, int(duration), interval):
        # Tính toán số thứ tự frame cần nhảy tới
        target_frame = int(sec * fps)
        
        # KIỂM TRA: Nếu frame mục tiêu vượt quá tổng số frame thì dừng
        if target_frame >= total_frames:
            break
            
        # LỆNH QUAN TRỌNG: Nhảy thẳng đến frame mục tiêu
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        success, image = vidcap.read()
        if success:
            image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            frame_name = f"frame_{frame_id}.jpg"
            
            save_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(save_path, image_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            frame_id += 1
        else:
            break

    vidcap.release()
    print(f"Đã trích xuất xong {frame_id} frames từ {os.path.basename(video_path)}")