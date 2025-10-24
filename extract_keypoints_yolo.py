from prompt_toolkit.key_binding.key_bindings import key_binding
from ultralytics import YOLO
import os
import json
import torch
import sys

# load YOLO 11 pose model
model = YOLO("yolo11n-pose.pt")

# path to folder wanting to get poses
folder_path = "Images/body segmentation and gesturere cognition cleaned/hand_fand/"

pose_data = []

# number of videos
num_videos = 0

imgs = sorted(os.listdir(folder_path))
for img_id, img in enumerate(imgs):
    frame_id = 0

    # asked gpt for how to make a progress bar
    bar_length = 30
    filled_length = int(bar_length * num_videos // 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\rProgress: |{bar}| {num_videos/len(imgs) * 100:.2f}%')
    sys.stdout.flush()

    num_videos += 1
    print(num_videos)

    img_path = os.path.join(folder_path, img)

    results = model.predict(
        source=img_path,
        save=False,
        show=False,
        conf=0.5,
        verbose=False
    )

    for result in results:
        if result.keypoints is None:
            continue

        # convert tensor to numpy
        kp = result.keypoints.data.cpu().numpy()

        for person_id, person_keypoints in enumerate(kp):
            frame_entry = {
                "frame_id": frame_id,
                "person_id": person_id,
                "img_id": img_id,
                "keypoints": person_keypoints.tolist()
            }
            pose_data.append(frame_entry)

    frame_id += 1

# save the data to json for later use
with open("pose_data.json", "w") as f:
    json.dump(pose_data, f, indent=2)

print("pose data saved to pose_data.json")