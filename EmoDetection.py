import cv2
import matplotlib.pyplot as plt

from deepface import DeepFace

# while True:

img = cv2.imread('happyFace.jpg')
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	#cv2.imshow('image', img)
	# key = cv2.waitKey(1)
	# if key == 113:
	# 	break


# cv2.destroyAllWindows()

img = cv2.imread('happyFace.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


prediction = DeepFace.analyze(img)

print(prediction)
plt.show()
