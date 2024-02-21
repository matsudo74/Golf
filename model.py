import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def process_video(video_path):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return  # 動画が開けない場合は処理を終了
    while True:
        success, frame = cap.read()

        # フレームの読み込みに失敗した場合、ループを抜ける
        if not success:
            print("Failed to read frame or end of video reached.")
            break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    # 動画キャプチャを解放する
    cap.release()

def display_frames(frames):
    for i, frame in enumerate(frames):
        # フレームをウィンドウで表示
        cv2.imshow(f'Frame {i}', frame)
        
        # 'q'キーが押されるまで待機
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    # すべてのウィンドウを閉じる
    cv2.destroyAllWindows()


# 動画ファイルのパス
video_path = 'path_to_your_video.mp4'

# フレームの抽出
process_video(video_path)
