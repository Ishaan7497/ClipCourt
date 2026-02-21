import cv2

cap = cv2.VideoCapture("game.mp4")

x, y = 200, 400        # start lower
dx, dy = 3, -2         # moving up
prev_y = y             # store previous y

trail = []             # store past positions
MAX_POINTS = 25        # trail length

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Move ball
    x += dx
    y += dy

    # Detect upward motion + choose color
    if y < prev_y:
        direction = "UP"
        ball_color = (0, 255, 0)   # green
    else:
        direction = "DOWN"
        ball_color = (0, 0, 255)   # red

    #Save position to trail
    trail.append((x, y))
    if len(trail) > MAX_POINTS:
        trail.pop(0)

    #Draw the trail
    for i in range(1, len(trail)):
        cv2.line(
            frame,
            trail[i - 1],
            trail[i],
            (255, 0, 0),   # blue trail
            2
        )

    # Draw the ball
    cv2.circle(frame, (x, y), 50, ball_color, -1)

    # Draw direction text
    cv2.putText(
        frame,
        direction,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        ball_color,
        2
    )

    prev_y = y  # update previous y

    cv2.imshow("Basketball Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
