import cv2
import numpy as np

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

# 動画ファイルのパス
video_path = 'path_to_your_video.mp4'

# フレームの抽出
frames = extract_frames(video_path)

# フレームの表示
display_frames(frames)
