import cv2
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cosine
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor

# ---------------------------------------------------------
# Helper Functions & Anatomical Bone Map
# ---------------------------------------------------------
def compute_iou(box1, box2):
    xx1, yy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    xx2, yy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_w, inter_h = max(0, xx2 - xx1), max(0, yy2 - yy1)
    union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - (inter_w * inter_h)
    return (inter_w * inter_h) / union_area if union_area > 0 else 0.0

# Clean body bones only — EXCLUDES chaotic face polygons (0 to 4)
BODY_LIMBS = [
    (5, 7), (7, 9),         # Left Arm
    (6, 8), (8, 10),        # Right Arm
    (5, 6),                 # Shoulders line
    (11, 12),               # Hips line
    (5, 11), (6, 12),       # Left & Right Torso
    (11, 13), (13, 15),     # Left Leg
    (12, 14), (14, 16)      # Right Leg
]

def draw_pill_badge(img, text, center_x, center_y, bg_color, text_color=(255, 255, 255)):
    """Draws a smooth HUD pill badge with centered text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    pad_x, pad_y = 10, 6
    x1 = int(center_x - text_w // 2 - pad_x)
    y1 = int(center_y - text_h // 2 - pad_y)
    x2 = int(center_x + text_w // 2 + pad_x)
    y2 = int(center_y + text_h // 2 + pad_y + baseline)

    # Rounded rectangle badge
    radius = (y2 - y1) // 2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), bg_color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, bg_color, -1, cv2.LINE_AA)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, bg_color, -1, cv2.LINE_AA)
    
    # Text
    tx = int(center_x - text_w // 2)
    ty = int(center_y + text_h // 2)
    cv2.putText(img, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
MIN_BBOX_WIDTH = 30
MIN_BBOX_HEIGHT = 60
SIMILARITY_THRESHOLD = 0.65
IOU_OVERLAP_THRESHOLD = 0.15
SELF_HEAL_FRAMES = 20  
REQUIRED_AFFIRMATIONS = 3

# ---------------------------------------------------------
# 1. Initialize Models
# ---------------------------------------------------------
print("Loading YOLOv8-Pose and OSNet...")
yolo_model = YOLO('yolov8n-pose.pt') 
reid_model = FeatureExtractor(model_name='osnet_x0_25', device='cpu')

# ---------------------------------------------------------
# 2. Tracking State
# ---------------------------------------------------------
yolo_id_counts = defaultdict(int)
yolo_to_seq_id = {}
known_visitor_embeddings = {}
last_reid_frame = {}  
pending_embeddings = defaultdict(list)
next_sequential_id = 1
frame_count = 0

# ---------------------------------------------------------
# 3. Video Pipeline Setup
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_skeleton.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Processing video... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    results = yolo_model.track(frame, classes=[0], conf=0.4, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
        raw_boxes = results[0].boxes.xyxy.cpu().numpy()
        raw_track_ids = results[0].boxes.id.cpu().numpy()
        raw_keypoints = results[0].keypoints.data.cpu().numpy() 

        valid_detections = []
        for box, track_id, kpts in zip(raw_boxes, raw_track_ids, raw_keypoints):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if (x2 - x1) >= MIN_BBOX_WIDTH and (y2 - y1) >= MIN_BBOX_HEIGHT:
                valid_detections.append({'box': (x1, y1, x2, y2), 'track_id': track_id, 'kpts': kpts})

        for i, det in enumerate(valid_detections):
            track_id, (x1, y1, x2, y2), kpts = det['track_id'], det['box'], det['kpts']
            yolo_id_counts[track_id] += 1
            is_occluded = any(i != j and compute_iou(det['box'], other['box']) > IOU_OVERLAP_THRESHOLD for j, other in enumerate(valid_detections))
            is_assigned = track_id in yolo_to_seq_id

            # -----------------------------------------------------
            # Re-ID & Affirmation Logic (Unchanged)
            # -----------------------------------------------------
            if not is_assigned:
                if not is_occluded:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        embedding = reid_model([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)])[0].cpu().numpy()
                        pending_embeddings[track_id].append(embedding)

                        if len(pending_embeddings[track_id]) >= REQUIRED_AFFIRMATIONS:
                            avg_emb = np.mean(pending_embeddings[track_id], axis=0)
                            avg_emb = avg_emb / np.linalg.norm(avg_emb)

                            best_match_id, highest_sim = None, -1.0
                            for seq_id, known_emb in known_visitor_embeddings.items():
                                sim = 1.0 - cosine(avg_emb, known_emb)
                                if sim > highest_sim:
                                    highest_sim, best_match_id = sim, seq_id

                            if highest_sim > SIMILARITY_THRESHOLD and best_match_id is not None:
                                yolo_to_seq_id[track_id] = best_match_id
                                known_visitor_embeddings[best_match_id] = 0.9 * known_visitor_embeddings[best_match_id] + 0.1 * avg_emb
                            else:
                                yolo_to_seq_id[track_id] = next_sequential_id
                                known_visitor_embeddings[next_sequential_id] = avg_emb
                                next_sequential_id += 1

                            last_reid_frame[track_id] = frame_count
                            del pending_embeddings[track_id]
            else:
                if (frame_count - last_reid_frame.get(track_id, 0)) >= SELF_HEAL_FRAMES:
                    if not is_occluded:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            embedding = reid_model([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)])[0].cpu().numpy()
                            embedding = embedding / np.linalg.norm(embedding)

                            best_match_id, highest_sim = None, -1.0
                            for seq_id, known_emb in known_visitor_embeddings.items():
                                sim = 1.0 - cosine(embedding, known_emb)
                                if sim > highest_sim:
                                    highest_sim, best_match_id = sim, seq_id

                            if highest_sim > SIMILARITY_THRESHOLD and best_match_id is not None:
                                yolo_to_seq_id[track_id] = best_match_id
                                known_visitor_embeddings[best_match_id] = 0.9 * known_visitor_embeddings[best_match_id] + 0.1 * embedding
                            
                            last_reid_frame[track_id] = frame_count

            # -----------------------------------------------------
            # Clean Aesthetic Skeleton Rendering
            # -----------------------------------------------------
            if not is_assigned:
                color = (0, 215, 255)       # Amber/Yellow during scanning
                label_text = f"ANALYZING {len(pending_embeddings[track_id])}/{REQUIRED_AFFIRMATIONS}"
            else:
                color = (0, 140, 255) if is_occluded else (75, 220, 100) # Orange if occluded, Soft Emerald Green if stable
                label_text = f"VISITOR #{yolo_to_seq_id[track_id]}"

            # 1. Draw Body Limbs
            for edge in BODY_LIMBS:
                pt1, pt2 = kpts[edge[0]], kpts[edge[1]]
                if pt1[2] > 0.4 and pt2[2] > 0.4:
                    p1 = (int(pt1[0]), int(pt1[1]))
                    p2 = (int(pt2[0]), int(pt2[1]))
                    # Main bone line with Anti-Aliasing
                    cv2.line(frame, p1, p2, color, 3, cv2.LINE_AA)

            # 2. Draw Anatomical Joints (Only key body joints: 5 to 16)
            for idx in range(5, 17):
                pt = kpts[idx]
                if pt[2] > 0.4:
                    center = (int(pt[0]), int(pt[1]))
                    # Outer colored ring + white inner core for modern look
                    cv2.circle(frame, center, 4, color, -1, cv2.LINE_AA)
                    cv2.circle(frame, center, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # 3. Clean Stylized Head (No face wireframe spiderwebs)
            nose = kpts[0]
            l_shoulder, r_shoulder = kpts[5], kpts[6]
            
            # Determine neck position (midpoint between shoulders)
            neck_pt = None
            if l_shoulder[2] > 0.4 and r_shoulder[2] > 0.4:
                neck_pt = (int((l_shoulder[0] + r_shoulder[0]) / 2), int((l_shoulder[1] + r_shoulder[1]) / 2))

            # Render Head circle
            head_center = None
            if nose[2] > 0.4:
                head_center = (int(nose[0]), int(nose[1]))
            elif neck_pt is not None:
                head_center = (neck_pt[0], neck_pt[1] - 25)

            if head_center is not None:
                # Estimate head radius from shoulder distance if possible
                if l_shoulder[2] > 0.4 and r_shoulder[2] > 0.4:
                    shoulder_dist = np.linalg.norm(l_shoulder[:2] - r_shoulder[:2])
                    radius = int(np.clip(shoulder_dist * 0.28, 8, 30))
                else:
                    radius = 12

                # Connect neck to head
                if neck_pt is not None:
                    cv2.line(frame, neck_pt, head_center, color, 2, cv2.LINE_AA)

                # Head outline circle with central dot
                cv2.circle(frame, head_center, radius, color, 2, cv2.LINE_AA)
                cv2.circle(frame, head_center, 2, (255, 255, 255), -1, cv2.LINE_AA)

                # 4. Floating Badge Label above Head
                badge_x = head_center[0]
                badge_y = head_center[1] - radius - 18
            else:
                badge_x = (x1 + x2) // 2
                badge_y = max(20, y1 - 15)

            draw_pill_badge(frame, label_text, badge_x, badge_y, bg_color=color, text_color=(20, 20, 20))

    # Clean Top-Left Dashboard HUD
    unique_count = len(known_visitor_embeddings)
    draw_pill_badge(frame, f"TOTAL UNIQUE: {unique_count}", 110, 40, bg_color=(40, 40, 40), text_color=(0, 255, 200))

    out.write(frame)
    cv2.imshow('Live Skeleton Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()