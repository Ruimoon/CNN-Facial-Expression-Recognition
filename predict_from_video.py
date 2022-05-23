import imutils
import cv2
import time
import dlib

from parameters import NETWORK, DATASET, VIDEO_PREDICTOR
from predict import load_model, predict


class EmotionRecognizer:
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (0, 255, 0)

    def __init__(self):

        # 初始化视频-打开笔记本内置摄像头
        self.video_stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # 加载分类器文件-获得人脸框位置的检测器
        self.face_detector = cv2.CascadeClassifier(VIDEO_PREDICTOR.face_detection_classifier)

        self.shape_predictor = None
        if NETWORK.use_landmarks or NETWORK.use_hog_and_landmarks:
            # 得人脸关键点检测器
            self.shape_predictor = dlib.shape_predictor(DATASET.shape_predictor_path)

        self.model = load_model()

    def predict_emotion(self, image):
        image.resize([NETWORK.input_size, NETWORK.input_size], refcheck=False)
        emotion, confidence = predict(image, self.model, self.shape_predictor)
        return emotion, confidence

    def recognize_emotions(self):
        failedFramesCount = 0
        detected_faces = []
        time_last_sent = 0
        while True:
            grabbed, frame = self.video_stream.read()

            if grabbed:
                # 检测阶段
                frame = imutils.resize(frame, width=600)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 进行多尺度检测
                # 1.3 是 图像的缩放因子
                # 5   是 每一个级联矩形应该保留的邻近个数，可以理解为一个人周边有几个人脸
                faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    # 边界框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), self.BOX_COLOR, 2)

                    # 尝试识别情绪
                    face = gray[y:y + h, x:x + w].copy()

                    label, confidence = self.predict_emotion(face)

                    # 框上显示标签
                    if VIDEO_PREDICTOR.show_confidence:
                        text = "{0} ({1:.1f}%)".format(label, confidence * 100)
                    else:
                        text = label
                    if label is not None:
                        cv2.putText(frame, text, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.TEXT_COLOR, 2)

                cv2.imshow("FER", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            else:
                failedFramesCount += 1
                print("failed_Frames_Count")
                if failedFramesCount > 10:
                    print("抓不到帧")
                    break

        self.video_stream.release()
        cv2.destroyAllWindows()


r = EmotionRecognizer()
r.recognize_emotions()
