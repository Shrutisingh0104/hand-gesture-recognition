import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, frame = cam.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "Unknown"

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            thumb_tip = landmarks[4][0]
            thumb_base = landmarks[2][0]

            index_tip = landmarks[8][1]
            index_base = landmarks[6][1]

            middle_tip = landmarks[12][1]
            middle_base = landmarks[10][1]

            ring_tip = landmarks[16][1]
            ring_base = landmarks[14][1]

            pinky_tip = landmarks[20][1]
            pinky_base = landmarks[18][1]

            fingers = []

            fingers.append(
                1 if abs(thumb_tip - thumb_base) > 30 else 0
            )

            fingers.append(
                1 if index_tip < index_base else 0
            )

            fingers.append(
                1 if middle_tip < middle_base else 0
            )

            fingers.append(
                1 if ring_tip < ring_base else 0
            )

            fingers.append(
                1 if pinky_tip < pinky_base else 0
            )

            if fingers == [0,0,0,0,0]:
                gesture = "Fist"

            elif sum(fingers) >= 4:
                gesture = "Palm"

            elif fingers[0] == 1 and sum(fingers[1:]) == 0:
                gesture = "Thumb"

            elif fingers == [0,1,0,0,0]:
                gesture = "Index"

            elif fingers == [1,1,0,0,0]:
                gesture = "L"

            elif fingers[1] == 1 and fingers[2] == 1:
                gesture = "OK"

            else:
                gesture = "Unknown"

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()