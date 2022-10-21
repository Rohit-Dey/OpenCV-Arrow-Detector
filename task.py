import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)   # uses default webcam

point = [[701, 200]]


# Function to find coordinates of the point & store in a list
def coordinate(event, x, y, flags, parameters):
    if event == cv2.EVENT_LBUTTONDOWN:
        point[0][0] = x
        point[0][1] = y


while True:

    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Color Detection

    # Masking
    low1 = np.array([0, 100, 20])
    high1 = np.array([5, 255, 255])
    red_range = cv2.inRange(hsv_frame, low1, high1)
    red_mask = cv2.bitwise_and(frame, frame, mask=red_range)
    # To make finer mask
    kernel = np.ones((15, 15), np.uint8)
    red_mask = cv2.erode(red_mask, kernel)

    # Contour Detection
    frame_contour = frame.copy()
    frame_canny = cv2.Canny(frame, 400, 200)

    contours, hierarchy = cv2.findContours(frame_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 7 and area > 400:
            cv2.drawContours(frame_contour, [approx], 0, (0, 0, 0), 5)
            # put text telling it's an arrow
            cv2.putText(frame_contour, "Red Arrow", (720, 180), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0))

    # Drawing an axis & getting the slope
    cv2.line(frame_contour, (700, 0), (700, 200), (0, 0, 0), 5)
    m1 = math.asin(1)  # gives angle in radians
    angle1 = math.degrees(m1)

    # Finding Angle of Arrow
    cv2.setMouseCallback("Frame", coordinate)
    x1 = point[0][0]
    y1 = point[0][1]

    gradient = (y1-200)/(x1-700)
    m2 = math.atan(gradient)
    angle2 = math.degrees(m2)

    # Inclination wrt Axis

    if x1 > 700:
        angle = angle2 + angle1
    else:
        angle = 180+(angle1 + angle2)

    cv2.putText(frame_contour, "Angle->" + str(angle) + " degrees", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0))

    # Displaying Output Windows
    cv2.imshow("Frame", frame_contour)  # borders the arrow
    cv2.imshow("Mask", red_mask)        # color detection output

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):  # pressing q exits
        break
