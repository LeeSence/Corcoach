import numpy as np
import cv2
import dlib
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector

predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')

hand_dect = HandDetector(detectionCon=0.7, maxHands=2)

count = 0

cap_count = 0

# 얼굴의 각 구역의 포인트들을 구분해 놓기

JAWLINE_POINTS = list(range(0, 17))

RIGHT_EYEBROW_POINTS = list(range(17, 22))

LEFT_EYEBROW_POINTS = list(range(22, 27))

NOSE_POINTS = list(range(27, 36))

RIGHT_EYE_POINTS = list(range(36, 42))

LEFT_EYE_POINTS = list(range(42, 48))

MOUTH_OUTLINE_POINTS = list(range(48, 61))

MOUTH_INNER_POINTS = list(range(61, 68))

def time_check(start, end):
    result = start - end
    return result    

def detect(gray,frame):
    global check
    
    # 일단, 등록한 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
 
    # 얼굴에서 랜드마크를 찾자
    for (x, y, w, h) in faces:
        # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # 랜드마크 포인트들 지정
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        # 원하는 포인트들을 넣는다, 지금은 전부
        # [0:68] = 모든 랜드마크 접근
        # [48:68] = 입술 랜드마크 접근
        global landmarks_display
        # outer mouth
        landmarks_display = landmarks[48:60]
        
        print(landmarks_display.shape)
        print(landmarks_display[0])
        
        # outer mouth
        nor_left = str(landmarks_display[0])
        nor_top = str(landmarks_display[3])
        nor_right = str(landmarks_display[6])
        nor_bottom = str(landmarks_display[9])        
        
        
        LEFT_X = int(nor_left[2:5])
        LEFT_Y = int(nor_top[6:9])
        
        RIGHT_X = int(nor_right[2:5])
        RIGHT_Y = int(nor_bottom[6:9])
        
        
        
        cv2.rectangle(frame, (LEFT_X, LEFT_Y), (RIGHT_X, RIGHT_Y), (255, 0, 0), 2)
        
        print('nor_left = ', nor_left)
        print('nor_top = ', nor_top)
        print('nor_right = ', nor_right)
        print('nor_bottom = ', nor_bottom)
        
        
        
        for idx, point in enumerate(landmarks_display):
            pos = (point[0, 0], point[0, 1])
            cv2.circle(frame, pos, 2, color=(0, 255, 255), thickness=-1)
            
        detector = FaceDetector()
        
        global bbox
        
        img, bbox = detector.findFaces(frame, draw=False)
        
        # print(bbox)
        
        # print(bbox[0].values())
        
        list_bbox = list(bbox[0].values())
        
        global x1
        global y1
        global x2 
        global y2
        

        x1 = LEFT_X
        y1 = LEFT_Y
        x2 = RIGHT_X
        y2 = RIGHT_Y
        
        print(f'x1 = {x1} y1 = {y1} x2 = {x2} y2 = {y2}')
        
        if bbox:
            # print(bbox[0]['score'])
            if bbox[0]['score'][0] >= 0.3:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0]['bbox']
                cv2.rectangle(img, (bbox_x1, bbox_y1), ((bbox_x1+bbox_x2), (bbox_y1+bbox_y2)), (0, 0, 255), 2)
                center = bbox[0]['center']
                cv2.circle(img, center, 2, (255, 0, 0))
                # pass
        
    detect_datas = hand_dect.findHands(frame, draw=False)
    
    for detectData in detect_datas:
        global count
        d_imgList = detectData['lmList']
        d_type = detectData['type']
        
        if d_type == 'Left':
            # 8번과 12번에 원
            cv2.circle(frame, (d_imgList[4][0], d_imgList[4][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[8][0], d_imgList[8][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[12][0], d_imgList[12][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[16][0], d_imgList[16][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[20][0], d_imgList[20][1]), 10, (255, 0, 0), cv2.FILLED)
            
            for i in range(4, 21, 4):
                if (x1 < d_imgList[i][0] and d_imgList[i][1] > y1) and ((x1+x2) > d_imgList[i][1] and d_imgList[i][1] < (y1+y2)):
                    count = count + 1
                    if bbox:
                            # print(bbox[0]['score'])
                        if bbox[0]['score'][0] >= 0.7:
                            pass
                        
        if d_type == 'Right':
            cv2.circle(frame, (d_imgList[4][0], d_imgList[4][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[8][0], d_imgList[8][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[12][0], d_imgList[12][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[16][0], d_imgList[16][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[20][0], d_imgList[20][1]), 10, (255, 0, 0), cv2.FILLED)
            
            
            for i in range(4, 21, 4):
                if (x1 < d_imgList[i][0] and d_imgList[i][1] > y1) and (x2 > d_imgList[i][1] and d_imgList[i][1] < y2):
                    count = count+10
                    # 3초이상이면 캡쳐 아니면 패스
                    print(count)
                    if bbox:
                            # print(bbox[0]['score'])
                        if bbox[0]['score'][0] >= 0.7:
                            pass
                else:
                    count = count - 1
                    if count < 0:
                        count = 0

    return frame, count

# 웹캠에서 이미지 가져오기
video_capture = cv2.VideoCapture(0)

while True:
    # 웹캠 이미지를 프레임으로 자름
    _, frame = video_capture.read()
    # 그리고 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame = cv2.flip(frame, 1)
    
    # 만들어준 얼굴 입 찾기
    canvas, count_check = detect(gray, frame)
    
    if count_check > 450:
        cap_count = cap_count + 1
        img_cap = cv2.imwrite(f'./media/cap{cap_count}.png', frame)
        count = 0
    # 찾은 이미지 보여주기
    cv2.imshow("haha", canvas)
 
    # q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 끝

video_capture.release()
cv2.destroyAllWindows()