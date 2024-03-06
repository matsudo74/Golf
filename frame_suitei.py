import cv2
import mediapipe as mp

# MediaPipeの姿勢推定モデルを初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_video(video_path):
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # BGR画像をRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # MediaPipeによる処理
        results = pose.process(frame_rgb)
        
        # 姿勢推定の結果を元のフレームに描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 結果を表示
        cv2.imshow(f'Pose Estimation - {video_path}', frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESCキーで終了
            break

    cap.release()
    cv2.destroyAllWindows()

# 二つの異なる動画ファイルのパス
video_path1 = 'path_to_your_first_video.mp4'
video_path2 = 'path_to_your_second_video.mp4'

# それぞれの動画で姿勢推定を行う
process_video(video_path1)
process_video(video_path2)
