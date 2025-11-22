import torch
from PIL import Image
import cv2
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Faster model (patch32 recommended for speed)
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16").to(device)

weapon_labels = [
    "knife", "pistol", "gun", "revolver", "rifle",
    "assault rifle", "AK47", "grenade", "weapon", "blade",
    "machete", "bazooka", "sniper rifle", "sword"
]


# -----------------------------
# IMAGE DETECTION
# -----------------------------
def detect_image(image_path, show=True, save_output=True):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=weapon_labels, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, threshold=0.30, target_sizes=target_sizes
    )[0]

    detections = []
    img_cv = cv2.imread(image_path)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        score = float(score)
        if score < 0.30:
            continue

        label_name = weapon_labels[label]
        x1, y1, x2, y2 = map(int, box.tolist())

        detections.append({
            "label": label_name,
            "confidence": score,
            "box": [x1, y1, x2, y2]
        })

        # Draw bounding box
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, f"{label_name} ({score:.2f})",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Show result
    if show:
        cv2.imshow("Detections", img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Save output
    if save_output:
        os.makedirs("output_images", exist_ok=True)
        out_path = f"output_images/detected_{os.path.basename(image_path)}"
        cv2.imwrite(out_path, img_cv)

    return detections


# -----------------------------
# VIDEO DETECTION (FAST)
# -----------------------------
def detect_video(video_path, skip_frames=8, save_frames=True):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    
    os.makedirs("output_video_frames", exist_ok=True)
    detections = []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("Video FPS:", fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # SKIP FRAMES FOR SPEED
        if frame_id % skip_frames != 0:
            frame_id += 1
            continue

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(text=weapon_labels, images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_img.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.30, target_sizes=target_sizes
        )[0]

        found_weapon = False

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            score = float(score)
            if score < 0.30:
                continue

            found_weapon = True
            label_name = weapon_labels[label]
            x1, y1, x2, y2 = map(int, box.tolist())

            detections.append({
                "frame": frame_id,
                "label": label_name,
                "confidence": score,
                "box": [x1, y1, x2, y2]
            })

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} ({score:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Save frame ONLY if weapon detected
        if found_weapon and save_frames:
            out_path = f"output_video_frames/frame_{frame_id}.jpg"
            cv2.imwrite(out_path, frame)

        frame_id += 1

    cap.release()
    return detections





# =============================
# VIDEO DECISION LOGIC
# =============================
def is_weapon_in_video(detections, min_conf=0.30, min_frames_required=3):
    """
    Returns 1 only if weapons appear in at least `min_frames_required` frames.
    This avoids false positives.
    """
    valid_count = 0

    for det in detections:
        if det["confidence"] >= min_conf:
            valid_count += 1

    if valid_count >= min_frames_required:
        return 1
    else:
        return 0


# =============================
# IMAGE DECISION LOGIC
# =============================
def is_weapon_in_image(detections):
    """
    If image has ANY detection → return 1
    If empty → return 0
    """
    return 1 if len(detections) > 0 else 0


# =============================
# MAIN WRAPPER FUNCTION
# =============================
def is_weapon_detected(path):
    """
    Automatically detects if file is image or video,
    runs the correct detection method,
    applies logic,
    returns 1 or 0.
    """

    ext = os.path.splitext(path)[1].lower()

    image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]

    # --- IMAGE ---
    if ext in image_exts:
        print(f"[INFO] Processing image: {path}")
        detections = detect_image(path, show=False, save_output=False)
        result = is_weapon_in_image(detections)

        print("Image result:", "WEAPON DETECTED" if result == 1 else "SAFE (no weapon)")
        return result

    # --- VIDEO ---
    elif ext in video_exts:
        print(f"[INFO] Processing video: {path}")
        detections = detect_video(path, skip_frames=8, save_frames=False)
        result = is_weapon_in_video(detections)

        print("Video result:", "WEAPON DETECTED" if result == 1 else "SAFE (no weapon)")
        return result

    # --- UNKNOWN FILE ---
    else:
        print("[ERROR] Unsupported file format:", ext)
        return 0


# -----------------------------
# TESTING
# -----------------------------

# if __name__ == "__main__":
#     result = is_weapon_detected("test/images/fakeKnife.jpg")
#     print("Final Output:", result)

#     result = is_weapon_detected("test/videos/test2..mp4")
#     print("Final Output:", result)
