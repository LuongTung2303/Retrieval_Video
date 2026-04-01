import cv2
import os

def extract_frames(video_path, output_folder, interval=1):
    """
    video_path: Đường dẫn file mp4
    output_folder: Nơi lưu ảnh
    interval: Cứ mỗi 'interval' giây thì lấy 1 ảnh
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS) # Số khung hình trên giây
    hop = int(fps * interval) # Số khung hình cần bỏ qua
    
    count = 0
    frame_id = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % hop == 0:
            # Lưu frame với tên: videoID_frameID.jpg
            frame_name = f"frame_{frame_id}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_name), image)
            frame_id += 1
        count += 1
    vidcap.release()

# Thử nghiệm với 1 video
# extract_frames("path_to_video.mp4", f"{PROJECT_PATH}/data/processed/test_video")