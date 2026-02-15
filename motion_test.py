import cv2

cap = cv2.VideoCapture("tt.mp4")

prev_gray = None
prev_y = None            # previous vertical position of the ball
prev_direction = None    # previous motion direction (UP / DOWN)
direction = None
frame_idx = 0            # frame counter
peak_frames = []         # list of detected shot peaks


while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        continue

    diff = cv2.absdiff(prev_gray, gray)
    _, motion_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY) # binary mask

    motion_mask = cv2.erode(motion_mask, None, iterations=2)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    contours, _ = cv2.findContours(
        motion_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        area = cv2.contourArea(c)

        # Ignore tiny noise and big player blobs
        if area < 50 or area > 800:
            continue

        (x, y), radius = cv2.minEnclosingCircle(c)
        x, y, radius = int(x), int(y), int(radius)

        if radius < 3 or radius > 15:
            continue

        # the ball
        cv2.circle(frame, (x, y), radius, (0, 255, 0), 2)

            # ----- BALL MOTION DIRECTION -----
        if prev_y is not None:
            if y < prev_y:
                direction = "UP"
            else:
                direction = "DOWN"

        # Detect peak (UP to DOWN)
            if prev_direction == "UP" and direction == "DOWN":
                peak_frames.append(frame_idx)
                print("PEAK at frame", frame_idx)

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


    cv2.imshow("Motion Mask", motion_mask)
    cv2.imshow("Frame", frame)

    prev_gray = gray

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
