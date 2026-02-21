import cv2
import os

# -----------------------------
# OPEN VIDEO
# -----------------------------
cap = cv2.VideoCapture("tt.mp4")

# -----------------------------
# VIDEO INFO
# -----------------------------
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
SHOT_HEIGHT_RATIO = 0.4

if fps == 0:
    fps = 30
delay = int(1000 / fps)

# -----------------------------
# CLIP SETTINGS
# -----------------------------
CLIP_BEFORE_SECONDS = 2
CLIP_AFTER_SECONDS = 2

BEFORE_FRAMES = CLIP_BEFORE_SECONDS * fps
AFTER_FRAMES  = CLIP_AFTER_SECONDS * fps

# -----------------------------
# BALL COLOR RANGE (BROWN/ORANGE)
# -----------------------------
LOWER_BALL = (10, 90, 80)
UPPER_BALL = (25, 220, 220)

# -----------------------------
# STATE VARIABLES
# -----------------------------
frame_idx = 0
prev_y = None
prev_direction = None
peak_frames = []


# -----------------------------
# MAIN ANALYSIS LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.line(
    frame,
    (0, int(height * SHOT_HEIGHT_RATIO)),
    (width, int(height * SHOT_HEIGHT_RATIO)),
    (255, 255, 0),
    2
)

    frame_idx += 1

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #converts from bgr scale 2 hsv scale = hue saturation brightness(value)

    # Keep only basketball color
    mask = cv2.inRange(hsv, LOWER_BALL, UPPER_BALL)
    

    # Find shapes - ball = white region ,rest black, outline of the ball is the countour
    contours, _ = cv2.findContours( # we use _ as findcounters returns 2 values, second value is useless(hierarchy values)
        mask,
        cv2.RETR_EXTERNAL,#only return outer lines,
        cv2.CHAIN_APPROX_SIMPLE#instead of saving every pixel on the edge it only saves the important corner points which saves memory
    )

    if contours: # if counters is an empty list there is no mask =white region=ball and it wont continue
        largest = max(contours, key=cv2.contourArea) #compares other counter areas and picks the largest to ignore the noise
        area = cv2.contourArea(largest) # area of the basketball(largest contour assumption)

        if area <800:
            (x, y), radius = cv2.minEnclosingCircle(largest) #returns centre of circle and radius of smallest circle = ball
            x, y, radius = int(x), int(y), int(radius)

            # Draw detected ball
            cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

            # Detect direction
            if prev_y is not None:
                if y < prev_y:
                    direction = "UP"
                else:
                    direction = "DOWN"

                # Detect peak
                if prev_direction == "UP" and direction == "DOWN" and y < SHOT_HEIGHT_RATIO * height:
                    peak_frames.append(frame_idx)
                    print("PEAK at frame", frame_idx, "y =", y)
                    cv2.putText(
                        frame, "PEAK", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2
                    )

                prev_direction = direction

            prev_y = y
    cv2.imshow("Mask", mask)
    cv2.imshow("Analysis", frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# CLIPPING PHASE
# -----------------------------
os.makedirs("clips", exist_ok=True)

cap = cv2.VideoCapture("game.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

for (i, peak) in enumerate(peak_frames):
    start = max(0, peak - BEFORE_FRAMES) # peak - before frames cannot be negative
    end   = min(total_frames - 1, peak + AFTER_FRAMES) #last frame cannot be over total frames - 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start) # jumps and reads directly from desired frame

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
print("Detected shot peaks at frames:", peak_frames)




