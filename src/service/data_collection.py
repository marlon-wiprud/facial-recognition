import cv2
from service.utils import predict_faces, get_class_list, load_local_model, get_face_cascade, SAVE_PATH


def detect_faces(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_cascade = get_face_cascade()

    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        color = (0, 255, 255)  # rgb
        stroke = 5
        cv2.rectangle(img, (x, y), (x + w, y + h), color, stroke)


def show_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cleanup_webcam(stream):
    stream.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def run_webcam():
    stream = cv2.VideoCapture(0)
    class_list = get_class_list()

    model = load_local_model(SAVE_PATH)

    face_cascade = get_face_cascade()

    while(True):
        (_, frame) = stream.read()
        # detect_faces(frame)
        prediction = predict_faces(model, face_cascade, class_list, frame)

        print("PREDICTION: ", prediction)
        # show the frame
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):    # Press q to break out
            break

    cleanup_webcam()


# detect_faces(image)
# show_image(image)

# vgg16 is default, could avoid the param altogether
# model = VGGFace(model='senet50')
