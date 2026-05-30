import cv2
import os

gesture_name = input("Enter gesture name: ")

path = f"dataset/{gesture_name}"

os.makedirs(path, exist_ok=True)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("Camera not opening!")
    exit()

count = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Collecting Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_name = f"{path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        count += 1
        print("Saved:", count)

    elif key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()