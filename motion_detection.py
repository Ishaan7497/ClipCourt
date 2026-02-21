import cv2
import os

# =============================
# VIDEO SETUP
# =============================
VIDEO_PATH = "tt.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps == 0:
    fps = 30

delay = int(1000 / fps)

# =============================
# PARAMETERS (TUNABLE)
# =============================
SHOT_HEIGHT_RATIO = 0.38    # allow higher tolerance while debugging
MIN_VERTICAL_MOVE = 6         # ignore tiny jitter
MIN_PEAK_GAP = int(0.5 * fps) # minimum frames between peaks

CLIP_BEFORE_SECONDS = 1
CLIP_AFTER_SECONDS  = 2

BEFORE_FRAMES = int(CLIP_BEFORE_SECONDS * fps)
AFTER_FRAMES  = int(CLIP_AFTER_SECONDS * fps)

# =============================
# STATE VARIABLES
# =============================
prev_gray = None
prev_y = None
prev_direction = None
last_peak_frame = -999
frame_idx = 0
peak_frames = []

# =============================
# ANALYSIS LOOP (LIVE VIEW)
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # draw shot height reference
    cv2.line(
        frame,
        (0, int(height * SHOT_HEIGHT_RATIO)),
        (width, int(height * SHOT_HEIGHT_RATIO)),
        (255, 255, 0),
        2
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        cv2.imshow("Video", frame)
        cv2.waitKey(delay)
        continue

    # -------------------------
    # MOTION DETECTION
    # -------------------------
    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.erode(motion_mask, None, iterations=2)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    # -------------------------
    # CONTOURS
    # -------------------------
    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best_ball = None
    best_area = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < 50 or area > 800:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        x, y, radius = int(x), int(y), int(radius)

        if radius < 3 or radius > 15:
            continue

        if area > best_area:
            best_area = area
            best_ball = (x, y, radius)

    # -------------------------
    # BALL TRACKING + PEAK
    # -------------------------
    if best_ball:
        x, y, radius = best_ball
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

        if prev_y is not None:
            # robust direction logic
            if prev_y - y > MIN_VERTICAL_MOVE:
                direction = "UP"
            elif y - prev_y > MIN_VERTICAL_MOVE:
                direction = "DOWN"
            else:
                direction = prev_direction

            # DEBUG PRINT (IMPORTANT)
            print(
                f"frame {frame_idx} | y={y} | prev_y={prev_y} | "
                f"dir={direction} | prev_dir={prev_direction}"
            )

            # PEAK DETECTION
            if (
                prev_direction == "UP"
                and direction == "DOWN"
                and y < SHOT_HEIGHT_RATIO * height
                and frame_idx - last_peak_frame > MIN_PEAK_GAP
            ):
                peak_frames.append(frame_idx)
                last_peak_frame = frame_idx
                print("🔥 PEAK CONFIRMED at frame", frame_idx)

                cv2.putText(
                    frame,
                    "PEAK",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2
                )

            prev_direction = direction

        prev_y = y

    # -------------------------
    # DISPLAY
    # -------------------------
    cv2.imshow("Video", frame)
    cv2.imshow("Motion Mask", motion_mask)

    prev_gray = gray

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Detected shot peaks:", peak_frames)

# =============================
# CLIP EXTRACTION
# =============================
os.makedirs("clips", exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

for i, peak in enumerate(peak_frames):
    start = max(0, peak - BEFORE_FRAMES)
    end   = min(total_frames - 1, peak + AFTER_FRAMES)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    out = cv2.VideoWriter(
        f"clips/shot_{i}.mp4",
        fourcc,
        fps,
        (width, height)
    )

    for _ in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()

cap.release()

print("Saved", len(peak_frames), "clips")

