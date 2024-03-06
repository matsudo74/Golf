import cv2
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
#static_image_mode=Trueのとき画像、動画の方が精度は高い
#min_detection_confidenceは信頼度であり、0から1まで指定できる
#model_complexityは複雑さであり、0,1,2が指定できる。リアルタイムなら0にしておく。
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)

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

#フレームになっているか確かめる
def display_frames(frames):
    for i, frame in enumerate(frames):
        # フレームをウィンドウで表示
        cv2.imshow(f"Frame {i}", frame)

        # 'q'キーが押されるまで待機
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break

    # すべてのウィンドウを閉じる
    cv2.destroyAllWindows()

#フレームごとに姿勢推定する関数
def estimate_pose_on_frames(frames):
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        for frame in frames:
            # BGR画像をRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # MediaPipeによる姿勢推定処理
            results = pose.process(frame_rgb)
            ######ここに動きが少ないフレームを削除するコードを入れる
            keypoints = []
            # 姿勢推定の結果を元のフレームに描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for landmark in results.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
            
            pose_vector = np.array(keypoints).flatten()
            
            cv2.imshow('Pose Estimation', frame)
            if cv2.waitKey(0) == 27:  # ESCで終了
                break
        
        cv2.destroyAllWindows()
    return pose_vector

def calculate_cosine_similarity(pose_vector1, pose_vector2):
    # コサイン類似度の計算
    similarity = cosine_similarity([pose_vector1], [pose_vector2])
    return similarity[0][0]

# 動画ファイルのパス
video_path_1 = "path_to_your_first_video.mp4"
video_path_2 = "path_to_your_second_video.mp4"

# フレームの抽出
frames_1 = extract_frames(video_path_1)
frames_2 = extract_frames(video_path_2)

# フレームの表示
#display_frames(frames_1)
#display_frames(frames_2)

#各フレームごとに姿勢推定されたものが出てくるコード
vector_1=estimate_pose_on_frames(frames_1)
vector_2=estimate_pose_on_frames(frames_2)

similarity = calculate_cosine_similarity(vector_1, vector_2)
print(f"コサイン類似度: {similarity}")