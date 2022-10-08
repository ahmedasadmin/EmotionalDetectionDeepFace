import cv2


from deepface import DeepFace
# import tensorflow as tf
# img1 = cv2.imread('happyFace.jpg')
# img2 = cv2.imread('img1.jpg')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("camera is not opened")

while True:
    ret, img  = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = DeepFace.analyze(img, actions=['emotion'])
    # print(result)
    region = result['region']
    dominant_emotion = result['dominant_emotion']

    print(dominant_emotion)
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    cv2.putText(img, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    #print(region['x'])



    cv2.rectangle(img,(x, y), (x+w, y+h),(255, 255, 0), 2)
    #cv2.imwrite("ahmed.jpg", img)
    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == 113:
        break
cap.release()
cv2.destroyAllWindows()

    #print(result)
