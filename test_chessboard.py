import cv2

cap = cv2.VideoCapture(1)
rows = 9
cols = 6
# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

while True:
    ret, frame = cap.read()
    if ret:
        grayL = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retL, cornersL = cv2.findChessboardCorners(grayL, (rows, cols), None)
        if retL:
            # Refine the corner position
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

            # Draw the corners on the image
            cv2.drawChessboardCorners(frame, (rows, cols), cornersL, retL)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break
cv2.destroyAllWindows()