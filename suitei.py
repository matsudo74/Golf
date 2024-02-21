import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def estimate_pose_for_video(video_path):
    cap = cv2.VideoCapture(video_path)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # BGRからRGBへ変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        # 姿勢のランドマークを描画
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
            break
    
    cap.release()
    cv2.destroyAllWindows()

def extract_frames(video_path):
    # 動画ファイルを読み込む
    cap = cv2.VideoCapture(video_path)
    
    # フレームが正しく読み込まれたか確認するフラグ
    success, frame = cap.read()
    
    frames = []
    while success:
        # フレームをリストに追加
        frames.append(frame)
        
        # 次のフレームを読み込む
        success, frame = cap.read()
    
    # 動画キャプチャを解放する
    cap.release()
    
    return frames

def display_frames(frames):
    for i, frame in enumerate(frames):
        # フレームをウィンドウで表示
        cv2.imshow(f'Frame {i}', frame)
        
        # 'q'キーが押されるまで待機
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    # すべてのウィンドウを閉じる
    cv2.destroyAllWindows()

# 動画パスを指定
video_path1 = 'path_to_your_first_video.mp4'
video_path2 = 'path_to_your_second_video.mp4'

# 二つの動画に対して姿勢推定を実行
estimate_pose_for_video(video_path1)
estimate_pose_for_video(video_path2)
