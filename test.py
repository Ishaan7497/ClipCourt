import cv2
import os

# -----------------------------
# OPEN THE VIDEO
# -----------------------------
cap = cv2.VideoCapture("game.mp4")


# -----------------------------
# VIDEO PROPERTIES
# -----------------------------
fps = int(cap.get(cv2.CAP_PROP_FPS))    # frames per second
delay = int(1000 / fps)         
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# -----------------------------
# ADJUSTABLE CLIP SETTINGS
# -----------------------------
CLIP_BEFORE_SECONDS = 10
CLIP_AFTER_SECONDS = 3

BEFORE_FRAMES = CLIP_BEFORE_SECONDS * fps
AFTER_FRAMES  = CLIP_AFTER_SECONDS * fps

# -----------------------------
# FAKE BALL PHYSICS (TEMPORARY)
# -----------------------------
x, y = 200, 400
dx = 3
dy = -20
gravity = 1

# -----------------------------
# VARIABLES
# -----------------------------
prev_direction = None
frame_idx = 0
peak_frames = []

# -----------------------------
# MAIN VIDEO LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # ---- MOVE BALL ----
    x += dx
    y += dy
    dy += gravity

    # ---- DETECT DIRECTION ----
    if dy < 0:
        direction = "UP"
        ball_color = (0, 255, 0)
    else:
        direction = "DOWN"
        ball_color = (0, 0, 255)

    # ---- PEAK DETECTION ----
    peak_detected = False
    if prev_direction == "UP" and direction == "DOWN":
        peak_detected = True
        peak_frames.append(frame_idx)

    # ---- DRAW BALL ----
    cv2.circle(frame, (x, y), 25, ball_color, -1)

    # ---- PEAK DETECTION ----
    if peak_detected:
        cv2.putText(frame, "PEAK",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2)

    cv2.imshow("Basketball Video", frame)

    # ---- NORMAL PLAYBACK SPEED ----
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

    prev_direction = direction

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# CLIP EXTRACTION
# -----------------------------
os.makedirs("clips", exist_ok=True)

cap = cv2.VideoCapture("game.mp4")
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

    for x in range(start, end):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()

cap.release()

print("Shot peaks detected at frames:", peak_frames)
