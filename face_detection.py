import cv2
from cv2.data import haarcascades
from keras_vggface.vggface import VGGFace

image_path = "face_test.jpg"

face_cascade = cv2.CascadeClassifier(haarcascades + 'haarcascade_frontalface_default.xml')

image = cv2.imread(image_path)


def detect_faces(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        color = (0, 255, 255)  # rgb
        stroke = 5
        cv2.rectangle(img, (x, y), (x + w, y + h), color, stroke)


def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_faces(image)
show_image(image)

model = VGGFace(model='senet50')  # vgg16 is default, could avoid the param altogether
