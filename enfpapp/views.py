from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import auth
from django.contrib.auth import authenticate
from .forms import User_update_form
from django.http import StreamingHttpResponse
import uuid
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import os
import numpy as np
from playsound import playsound
from django.db.models import Q

# opencv 라이브러리
import cv2
#dlib 라이브러리
import dlib
# 핸드트래킹, 페이스트래킹 모듈 호출
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
# 미디어파이프 라이브러리 호출
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import datetime
from datetime import timedelta
import winsound as sd

from django.contrib.auth.hashers import check_password
from . models import coach_turtle, coach_shoulder, coach_nail



# 페이징 처리
from django.core.paginator import Paginator

######################################################################
# 전역변수
######################################################################
cap = cv2.VideoCapture(0)
if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
    cap.release()	# 영상 파일(카메라) 사용을 종료

######################################################################
# nail 전역변수(손톱)
######################################################################
count = 0
x1,x2,y1,y2 = 0,0,0,0


# dlib 학습파일 불러오기
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + './haarcascade_frontalface_default.xml')

hand_dect = HandDetector(detectionCon=0.7, maxHands=2)

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

def alert_change(request):
    # request.POST['']
    # 받아온 값이 None이면 False 저장 "on" 이면 True 저장
    # 경고음 울리는 부분에서 
    log_id = request.user.username
    alert_check = request.user.first_name
    
    if request.POST['onoff'] == "on":
        pass
    elif request.POST['onoff'] == None:
        pass
    pass

def detect(gray,frame):
    global check
    
    # 일단, 등록한 dlib 학습파일 Cascade classifier 를 이용 얼굴을 찾음
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
 
    # 얼굴에서 랜드마크를 찾음
    for (x, y, w, h) in faces:
        # 오픈 CV 이미지를 dlib용 사각형으로 변환하고
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # 랜드마크 포인트들 지정
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dlib_rect).parts()])
        global landmarks_display
        # [48:68] = 입술 랜드마크 접근
        landmarks_display = landmarks[48:60]
        
        #print(landmarks_display.shape)
        #print(landmarks_display[0])
        
        # 좌표값을 스트링 처리하고
        nor_left = str(landmarks_display[0])
        nor_top = str(landmarks_display[3])
        nor_right = str(landmarks_display[6])
        nor_bottom = str(landmarks_display[9])        
        
        # 정수값으로 변경하여 슬라이싱해 좌표값을 얻어옴
        LEFT_X = int(nor_left[2:5])
        LEFT_Y = int(nor_top[6:9])
        RIGHT_X = int(nor_right[2:5])
        RIGHT_Y = int(nor_bottom[6:9])
        
        # print(LEFT_X)
        # print(LEFT_Y)
        # print(RIGHT_X)
        # print(RIGHT_Y)
        
        # 좌표값을 이용해 사각형 범위 생성
        cv2.rectangle(frame, (LEFT_X, LEFT_Y), (RIGHT_X, RIGHT_Y), (255, 0, 0), 2)
        
        print('nor_left = ', nor_left)
        print('nor_top = ', nor_top)
        print('nor_right = ', nor_right)
        print('nor_bottom = ', nor_bottom)
        
        # print(landmarks_display[0][1]) # 데이터 구조 [[x1, y1], [x1, y2], [], [], [], [], [], []]
        
        #print(landmarks.shape)
        #print('*' * 100)
        #print(landmarks_display)
        #print('*' * 100)
        
        # print(landmarks_display[6][0])
        # 눈만 = landmarks_display = landmarks[RIGHT_EYE_POINTS, LEFT_EYE_POINTS]

        # dets = detector(frame, 1)
        
        # dlib 포인트 출력
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
        
        #x1 = list_bbox[1][0]
        #y1 = list_bbox[1][1]
        #x2 = list_bbox[1][2]
        #y2 = list_bbox[1][3]

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
    
    # opencv HandTrackingModule을 활용해 hand landmark접근
    detect_datas = hand_dect.findHands(frame, draw=False)
    
    for detectData in detect_datas:
        global count
        d_imgList = detectData['lmList']
        d_type = detectData['type']
        
        # 왼손 끝 landmark 좌표값을 추적
        if d_type == 'Left':
            # 8번과 12번에 원
            cv2.circle(frame, (d_imgList[4][0], d_imgList[4][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[8][0], d_imgList[8][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[12][0], d_imgList[12][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[16][0], d_imgList[16][1]), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (d_imgList[20][0], d_imgList[20][1]), 10, (255, 0, 0), cv2.FILLED)
            
            # 왼손 끝 landmark 좌표값이 일정시간 범위를 침범했는지 확인하는 로직 오른손도 같은코드
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
    # frame 정보와 count값 리턴
    return frame, count

######################################################################
# shoulder, turtle 사용함수
######################################################################
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    # 두 점 사이의 각도 구하는 함수
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    # abs = 절대값 구하는 함수
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

# Create your views here.


def video_view(request):
    return StreamingHttpResponse(
        stream(),
        content_type='multipart/x-mixed-replace;boundary=frame'
    )
    
def func_nail(request):
    return StreamingHttpResponse(
        stream_nail(request),
        content_type='multipart/x-mixed-replace;boundary=frame'
    )

def func_shoulder(request):
    return StreamingHttpResponse(
        stream_shoulder(request),
        content_type='multipart/x-mixed-replace;boundary=frame'
    )

def updateCheck(request):
    return render(request, 'updateCheck.html')

def passwordCheck(request):
    if request.method == "POST":
        user = request.user
        password = request.POST["password"]
        print(user)
        print(password)
        if check_password(password, user.password):
            auth.login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            return redirect('update')
        else:
            messages.error(request, 'fail')
            return render(request, 'updateCheck.html')
    else:
        return render(request, 'updateCheck.html')

# template web 페이지에서 호출되면
def func_turtle(request):
    return StreamingHttpResponse(
        # 기능화면 호출
        stream_turtle(request),
        content_type='multipart/x-mixed-replace;boundary=frame'
    )

def func_leg_cross(request):
    return StreamingHttpResponse(
        stream_leg_cross(request),
        content_type='multipart/x-mixed-replace;boundary=frame'
    )

def stream():
    global cap
    
    if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()	# 영상 파일(카메라) 사용을 종료
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        
        # 카메라가 없다면
        if not ret:
            print('error: 카메라가 존재하지 않습니다.')
            break

        # frame 정보를 받았다면 서버에 전송
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield(b'--frame\r\n'
              b'Content-type: image/jpeg\r\n\r\n'
              + image_bytes + b'\r\n')
#              + open('./demo.jpg', 'rb').read() + b'\r\n')

# 자세교정 기능
def stream_turtle(request):
    global cap
    
    if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()	# 영상 파일(카메라) 사용을 종료

    # 캠 호출
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0 
    stage = None

    # pose 최소 인식률, 랜드마크 추적 신뢰도 조정 및 mediapipe landmark 호출
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            # 캠화면을 읽어옮
            ret, frame = cap.read()

            # 캠화면 좌우반
            frame = cv2.flip(frame, 1)
            
            # 카메라가 없다면
            if not ret:
                print('error: 카메라가 존재하지 않습니다.')
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        

            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # landmarks 추출
            try:
                # landmark 정보저장
                landmarks = results.pose_landmarks.landmark
                
                ##############################################################################################################################
                # landmark 정보값을 변경만 해주면 새로운기능으로 변경이 가능함
                
                # landmark 좌표값 얻어오기
                L_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                R_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                L_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                R_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                
                # 랜드마크 간 각도를 계산하는 함수 이를 통해 실시간으로 랜드마크간의 각도값을 리턴받아옴
                angle = calculate_angle(L_mouth, L_shoulder, R_shoulder)
                angle2 = calculate_angle(R_mouth, L_shoulder, R_shoulder)
                
                ##############################################################################################################################
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(L_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(R_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # 만약 사용자의 행동에따른 랜드마크간 각도값이 이렇다면
                if (angle > 40) and (angle2 > 40):
                    # 평상시
                    stage = 'normal'
                    
                if (angle < 35 and stage == 'normal') and (angle2 < 35 and stage == 'normal'):
                    # 안좋은 행동을 할 시
                    stage = 'turtle neck'
                    counter += 1
                    # DB에 저장
                    # 만약 DB의 마지막 값이 있다면
                    try:
                        # coach_turtle의 idx마지막 값을 가져온다
                        info = coach_turtle.objects.latest('idx')
                    # DB의 값이 없다면 None 처리
                    except coach_turtle.DoesNotExist:
                        info = None
                        
                    # DB 값이 없었을때 첫 사진 캡쳐 및 DB저장
                    if info == None:
                        cap_count = 1
                        name = request.user.username
                        path = f'./media/{name}_cap{cap_count}_turtle.png'
                        cv2.imwrite(path, frame)
                        # 데이터저장
                        save_coach = coach_turtle()
                        save_coach.id = request.user.username
                        save_coach.path = '.' + path
                        save_coach.dates = datetime.datetime.now()
                        save_coach.save()
                        playsound("./media/result.mp3")
                    # DB값이 존재할 경우 캡쳐 및 DB저장
                    else:
                        value = coach_turtle.objects.latest('idx')
                        cap_count = int(value.idx)+1
                        name = request.user.username
                        path = f'./media/{name}_cap{cap_count}_turtle.png'
                        cv2.imwrite(path, frame)
            
                        # 데이터저장
                        save_coach = coach_turtle()
                        save_coach.id = request.user.username
                        save_coach.path = '.' + path
                        save_coach.dates = datetime.datetime.now()
                        save_coach.save()
                        playsound("./media/result.mp3")
                        print(counter)
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            # Stage data
            if stage == 'normal':
                pass
            else : 
                cv2.putText(image, 'STAGE', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )   

            # frame 정보를 받았다면 서버에 전송
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield(b'--frame\r\n'
                b'Content-type: image/jpeg\r\n\r\n'
                + image_bytes + b'\r\n')
        
def stream_shoulder(request):
    global cap
    
    #PoseLandmark 클래스 ( enum . IntEnum ):
    #    """33가지 포즈의 랜드마크입니다."""
    #    코  =  0
    #    LEFT_EYE_INNER  =  1
    #    왼쪽 _눈  =  2
    #    LEFT_EYE_OUTER  =  3
    #    RIGHT_EYE_INNER  =  4
    #    오른쪽 눈  =  5
    #    RIGHT_EYE_OUTER  =  6
    #    LEFT_EAR  =  7
    #    RIGHT_EAR  =  8
    #    MOUTH_LEFT  =  9
    #    MOUTH_RIGHT  =  10
    #    왼쪽 어깨  =  11
    #    오른쪽 어깨  =  12
    #    왼쪽 팔꿈치  =  13
    #    RIGHT_ELBOW  =  14
    #    LEFT_WRIST  =  15
    #    RIGHT_WRIST  =  16
    #    왼쪽_ 핑키  =  17
    #    RIGHT_PINKY  =  18
    #    왼쪽_ 인덱스  =  19
    #    RIGHT_INDEX  =  20
    #    LEFT_THUMB  =  21
    #    RIGHT_THUMB  =  22
    #    왼쪽_ 엉덩이  =  23
    #    RIGHT_HIP  =  24
    #    왼쪽_ 무릎  =  25
    #    RIGHT_KNEE  =  26
    #    왼쪽_ 발목  =  27
    #    오른쪽_ 발목  =  28
    #    LEFT_HEEL  =  29
    #    RIGHT_HEEL  =  30
    #    LEFT_FOOT_INDEX  =  31
    #    RIGHT_FOOT_INDEX  =  32
    
    if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()	# 영상 파일(카메라) 사용을 종료

    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            
            frame = cv2.flip(frame, 1)
            
            # 카메라가 없다면
            if not ret:
                print('error: 카메라가 존재하지 않습니다.')
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                # 일부수정필요
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                left_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y]
                right_mouth = [landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].x,landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y]
                # check(left_knee, right_knee)
                
                # Calculate angle
                
                angle = calculate_angle(left_shoulder, right_shoulder, right_mouth)
                angle2 = calculate_angle(right_shoulder, left_shoulder, left_mouth)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(right_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(left_shoulder, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                # Curl counter logic
                if (angle > 48) and (angle2 > 48):
                    stage = 'normal'
                    
                if (angle < 43 and stage == 'normal') or (angle2 < 43 and stage == 'normal'):
                    stage = 'shoulder down'
                    counter += 1
                    try:
                        # coach_shoulder의 idx마지막 값을 가져오는 쿼리셋
                        info = coach_shoulder.objects.latest('idx')
                    except coach_shoulder.DoesNotExist:
                        info = None
                    if info == None:
                        cap_count = 1
                        name = request.user.username
                        path = f'./media/{name}_cap{cap_count}_shoulder.png'
                        cv2.imwrite(path, frame)
                        # 데이터저장
                        save_coach = coach_shoulder()
                        save_coach.id = request.user.username
                        save_coach.path = '.' + path
                        save_coach.dates = datetime.datetime.now()
                        save_coach.save()
                        playsound("./media/result.mp3")
                        count = 0
                    else:
                        value = coach_shoulder.objects.latest('idx')
                        cap_count = int(value.idx)+1
                        name = request.user.username
                        path = f'./media/{name}_cap{cap_count}_shoulder.png'
                        cv2.imwrite(path, frame)
            
                        # 데이터저장
                        save_coach = coach_shoulder()
                        save_coach.id = request.user.username
                        save_coach.path = '.' + path
                        save_coach.dates = datetime.datetime.now()
                        save_coach.save()
                        playsound("./media/result.mp3")
                    print(counter)
                    
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            if stage == 'normal':
                pass
            else : 
                cv2.putText(image, 'STAGE', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
 
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )   
            
            # frame 정보를 받았다면 서버에 전송
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield(b'--frame\r\n'
                b'Content-type: image/jpeg\r\n\r\n'
                + image_bytes + b'\r\n')
    #              + open('./demo.jpg', 'rb').read() + b'\r\n')

def stream_leg_cross(request):
    global cap
    
    #PoseLandmark 클래스 ( enum . IntEnum ):
    #    """33가지 포즈의 랜드마크입니다."""
    #    코  =  0
    #    LEFT_EYE_INNER  =  1
    #    왼쪽 _눈  =  2
    #    LEFT_EYE_OUTER  =  3
    #    RIGHT_EYE_INNER  =  4
    #    오른쪽 눈  =  5
    #    RIGHT_EYE_OUTER  =  6
    #    LEFT_EAR  =  7
    #    RIGHT_EAR  =  8
    #    MOUTH_LEFT  =  9
    #    MOUTH_RIGHT  =  10
    #    왼쪽 어깨  =  11
    #    오른쪽 어깨  =  12
    #    왼쪽 팔꿈치  =  13
    #    RIGHT_ELBOW  =  14
    #    LEFT_WRIST  =  15
    #    RIGHT_WRIST  =  16
    #    왼쪽_ 핑키  =  17
    #    RIGHT_PINKY  =  18
    #    왼쪽_ 인덱스  =  19
    #    RIGHT_INDEX  =  20
    #    LEFT_THUMB  =  21
    #    RIGHT_THUMB  =  22
    #    왼쪽_ 엉덩이  =  23
    #    RIGHT_HIP  =  24
    #    왼쪽_ 무릎  =  25
    #    RIGHT_KNEE  =  26
    #    왼쪽_ 발목  =  27
    #    오른쪽_ 발목  =  28
    #    LEFT_HEEL  =  29
    #    RIGHT_HEEL  =  30
    #    LEFT_FOOT_INDEX  =  31
    #    RIGHT_FOOT_INDEX  =  32
    
    if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()	# 영상 파일(카메라) 사용을 종료

    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            
            frame = cv2.flip(frame, 1)
            
            # 카메라가 없다면
            if not ret:
                print('error: 카메라가 존재하지 않습니다.')
                break
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                
                
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                
                # check(left_knee, right_knee)
                
                # Calculate angle
                
                angle = calculate_angle(left_hip, right_hip, right_knee)
                
                angle2 = calculate_angle(right_hip, left_hip, left_knee)
                
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                cv2.putText(image, str(angle2), 
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                
                
                # Curl counter logic
                if (angle > 65) and (angle2 > 65):
                    stage = 'normal'
                    
                if (angle < 60 and stage == 'normal') or (angle2 < 60 and stage == 'normal'):
                    stage = 'cross'
                    counter += 1
                    try:
                        # coach_shoulder의 idx마지막 값을 가져오는 쿼리셋
                        info = coach_shoulder.objects.latest('idx')
                    except coach_shoulder.DoesNotExist:
                        info = None
                    if info == None:
                        cap_count = 1
                        name = request.user.username
                        path = f'./media/{name}_cap{cap_count}_shoulder.png'
                        cv2.imwrite(path, frame)
                        # 데이터저장
                        save_coach = coach_shoulder()
                        save_coach.id = request.user.username
                        save_coach.path = '.' + path
                        save_coach.dates = datetime.datetime.now()
                        save_coach.save()
                        count = 0
                    else:
                        value = coach_shoulder.objects.latest('idx')
                        cap_count = int(value.idx)+1
                        name = request.user.username
                        path = f'./media/{name}_cap{cap_count}_shoulder.png'
                        cv2.imwrite(path, frame)
            
                        # 데이터저장
                        save_coach = coach_shoulder()
                        save_coach.id = request.user.username
                        save_coach.path = '.' + path
                        save_coach.dates = datetime.datetime.now()
                        save_coach.save()
                    print(counter)
                    
            except:
                pass
            
            # Render curl counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
            
            # Stage data
            if stage == 'normal':
                pass
            else : 
                cv2.putText(image, 'STAGE', (65,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                cv2.putText(image, stage, 
                            (60,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
 
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )   
            
            # frame 정보를 받았다면 서버에 전송
            image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
            yield(b'--frame\r\n'
                b'Content-type: image/jpeg\r\n\r\n'
                + image_bytes + b'\r\n')
    #              + open('./demo.jpg', 'rb').read() + b'\r\n')


# 버릇 교정 기능
def stream_nail(request):
    global cap, count, cap_count

    if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
        cap.release()	# 영상 파일(카메라) 사용을 종료

    # 웹캠 호출
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        # 카메라가 없다면
        if not ret:
            print('error: 카메라가 존재하지 않습니다.')
            break

        # 그리고 이미지를 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame = cv2.flip(frame, 1)
        # 만들어준 얼굴과 입부분범위, 손을 찾고, mediapipe hand landmark값을 추적하여 그 좌표값이 몇초간 범위를 침범하였는지 확인해주는 return 함수  
        canvas, count_check = detect(gray, frame)
        
        # 만약 반환받은 카운트값이 450을 넘었다면
        if count_check > 450:
            # DB내부에 값이 있을경우
            try:
                # coach_nail의 idx마지막 값을 가져오는 쿼리셋
                info = coach_nail.objects.latest('idx')
            # DB내부에 값이 없을 경우
            except coach_nail.DoesNotExist:
                info = None
            # DB 값이 없었을때 첫 사진 캡쳐 및 DB저장
            if info == None:
                cap_count = 1
                name = request.user.username
                path = f'./media/{name}_cap{cap_count}_nail.png'
                cv2.imwrite(path, frame)
                # 데이터저장
                save_coach = coach_nail()
                save_coach.id = request.user.username
                save_coach.path = '.' + path
                save_coach.dates = datetime.datetime.now()
                save_coach.save()
                playsound("./media/result.mp3")
                count = 0
            # DB 값이 있었을때 첫 사진 캡쳐 및 DB저장
            else:
                value = coach_nail.objects.latest('idx')
                cap_count = int(value.idx)+1
                name = request.user.username
                path = f'./media/{name}_cap{cap_count}_nail.png'
                cv2.imwrite(path, frame)
            
                # 데이터저장
                save_coach = coach_nail()
                save_coach.id = request.user.username
                save_coach.path = '.' + path
                save_coach.dates = datetime.datetime.now()
                save_coach.save()
                playsound("./media/result.mp3")
                count = 0
                

        # frame 정보를 받았다면 서버에 전송
        image_bytes = cv2.imencode('.jpg', canvas)[1].tobytes()
        yield(b'--frame\r\n'
              b'Content-type: image/jpeg\r\n\r\n'
              + image_bytes + b'\r\n')
#              + open('./demo.jpg', 'rb').read() + b'\r\n')

def index(request):
    #return HttpResponse("enfp파이팅!")
    return render(request, 'index.html')

def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        print("user_id = ", username)
        print("password =", password)
        
        # Django 회원멤버DB에서  사용자가 있는지 확인
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('index')
        else:
            return render(request, 'login.html', {'error' : 'user_id or user_pw is incorrect.'})
    else:
        return render(request, 'login.html')
    
def logout(request):
    auth.logout(request)
    return redirect('index')

def about(request):
    return render(request, 'about.html')

def join(request):
    return render(request, 'join.html')

def change_password(request):
    if request.method == "POST":
        user = request.user
        origin_password = request.POST["origin_password"]
        if check_password(origin_password, user.password):
            new_password = request.POST["new_password"]
            confirm_password = request.POST["confirm_password"]
            if new_password == confirm_password:
                user.set_password(new_password)
                user.save()
                auth.login(request, user, backend='django.contrib.auth.backends.ModelBackend')
                return redirect('index')
            else:
                messages.error(request, 'Password not same')
        else:
            messages.error(request, 'Password not correct')
        return render(request, 'updatePassword.html')
    else:
        return render(request, 'updatePassword.html')
    
def ar_graph(request):
    # id값이 일치하는 데이터셋 호출
    # 데이터 없을때 예외처리
    chk_list = []
    try:
        date_nail = coach_nail.objects.filter(id=request.user.username).latest('dates')
        chk_list.append(date_nail)
    except coach_nail.DoesNotExist:
        date_nail = None

    try:
        date_turtle = coach_turtle.objects.filter(id=request.user.username).latest('dates')
        chk_list.append(date_turtle)
    except coach_turtle.DoesNotExist:
        date_turtle = None

    try:
        date_shoulder = coach_shoulder.objects.filter(id=request.user.username).latest('dates')
        chk_list.append(date_shoulder)
    except coach_shoulder.DoesNotExist:
        date_shoulder = None
            
    
    
    ############################################################################
    
    if len(chk_list) == 0:
        ####################################
        # 마지막날
        today_cal = datetime.datetime.now()
        ####################################
                
        # result_data = day_cal(today_slice, before_one_day_slice, before_two_day_slice, before_three_day_slice, before_four_day_slice)
        
        print('='*100)
        print('현재 시각 : ', str(today_cal)[0:10])
        print('1일전 : ', str(today_cal - timedelta(days=1))[0:10])
        print('2일전 : ', str(today_cal - timedelta(days=2))[0:10])
        print('='*100)
        print('데이트타임 타입: ', type(today_cal))
        print('='*100)
        
        data = {
            "today1" : str(today_cal)[0:10],
            "today2" : str(today_cal - timedelta(days=1))[0:10],
            "today3" : str(today_cal - timedelta(days=2))[0:10],
            "today4" : str(today_cal - timedelta(days=3))[0:10],
            "today5" : str(today_cal - timedelta(days=4))[0:10],
            "today_turtle_count" : 0,
            "today_shoulder_count" : 0,
            "today_nail_count" : 0,
            "before_one_turtle_count" : 0,
            "before_one_shoulder_count" : 0,
            "before_one_nail_count" : 0,
            "before_two_turtle_count" : 0,
            "before_two_shoulder_count" : 0,
            "before_two_nail_count" : 0,
            "before_three_turtle_count" : 0,
            "before_three_shoulder_count" : 0,
            "before_three_nail_count" : 0,
            "before_four_turtle_count" : 0,
            "before_four_shoulder_count" : 0,
            "before_four_nail_count" : 0,
        }
        
        return render(request, 'ar_graph.html', context=data)
    
    else:
        
        if len(chk_list) == 1:
            print("오늘의날짜 : ", str(chk_list[0].dates)[0:10])
            today_cal = str(chk_list[0].dates)[0:10]
            final_before_one_day = str(chk_list[0].dates - timedelta(days=1))[0:10]
            final_before_two_day = str(chk_list[0].dates - timedelta(days=2))[0:10]
            final_before_three_day = str(chk_list[0].dates - timedelta(days=3))[0:10]
            final_before_four_day = str(chk_list[0].dates - timedelta(days=4))[0:10]
                    
            try:
                q = Q()
                q.add(Q(dates__startswith=today_cal), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                today_nail_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                today_nail_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_one_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_one_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_one_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_two_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_two_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_two_count = 0
            try:    
                q = Q()
                q.add(Q(dates__startswith=final_before_three_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_three_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_three_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_four_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_four_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_four_count = 0
        
            try:
                q = Q()
                q.add(Q(dates__startswith=today_cal), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                today_shoulder_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                    today_shoulder_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_one_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_one_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_one_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_two_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_two_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_two_count = 0
            try:    
                q = Q()
                q.add(Q(dates__startswith=final_before_three_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_three_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_three_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_four_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_four_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_four_count = 0
        
            try:
                q = Q()
                q.add(Q(dates__startswith=today_cal), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                today_turtle_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                today_turtle_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_one_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_one_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_one_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_two_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_two_count = len(coach_shoulder.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_two_count = 0
            try: 
                q = Q()
                q.add(Q(dates__startswith=final_before_three_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_three_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_three_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_four_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_four_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_four_count = 0
                    
            data = {
                "today1" : today_cal,
                "today2" : final_before_one_day,
                "today3" : final_before_two_day,
                "today4" : final_before_three_day,
                "today5" : final_before_four_day,
                "today_turtle_count" : today_turtle_count,
                "today_shoulder_count" : today_shoulder_count,
                "today_nail_count" : today_nail_count,
                "before_one_turtle_count" : before_turtle_one_count,
                "before_one_shoulder_count" : before_shoulder_one_count,
                "before_one_nail_count" : before_nail_one_count,
                "before_two_turtle_count" : before_turtle_two_count,
                "before_two_shoulder_count" : before_shoulder_two_count,
                "before_two_nail_count" : before_nail_two_count,
                "before_three_turtle_count" : before_turtle_three_count,
                "before_three_shoulder_count" : before_shoulder_three_count,
                "before_three_nail_count" : before_nail_three_count,
                "before_four_turtle_count" : before_turtle_four_count,
                "before_four_shoulder_count" : before_shoulder_four_count,
                "before_four_nail_count" : before_nail_four_count,
            }
        else:
            max = int(str(chk_list[0].dates)[0:10].replace('-', '999')) # 00009990099900
            print(max)
            for i, max_date in enumerate(chk_list, start=1):
                if max < int(str(max_date.dates)[0:10].replace('-', '999')):
                    max = int(str(max_date.dates)[0:10].replace('-', '999'))
            
            today = str(max).replace('999', '-')
            print(today)
            today_cal = datetime.datetime.strptime(today,'%Y-%m-%d')
            print(today_cal)
            final_before_one_day = str(today_cal - timedelta(days=1))[0:10]
            final_before_two_day = str(today_cal - timedelta(days=2))[0:10]
            final_before_three_day = str(today_cal - timedelta(days=3))[0:10]
            final_before_four_day = str(today_cal - timedelta(days=4))[0:10]
                    
            try:
                q = Q()
                q.add(Q(dates__startswith=str(today_cal)[0:10]), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                today_nail_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                today_nail_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_one_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_one_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_one_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_two_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_two_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_two_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_three_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_three_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_three_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_four_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_nail_four_count = len(coach_nail.objects.filter(q))
            except coach_nail.DoesNotExist:
                before_nail_four_count = 0
        
            try:
                q = Q()
                q.add(Q(dates__startswith=str(today_cal)[0:10]), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                today_shoulder_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                    today_shoulder_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_one_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_one_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_one_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_two_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_two_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_two_count = 0
            try:    
                q = Q()
                q.add(Q(dates__startswith=final_before_three_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_three_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_three_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_four_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_shoulder_four_count = len(coach_shoulder.objects.filter(q))
            except coach_shoulder.DoesNotExist:
                before_shoulder_four_count = 0
        
            try:
                q = Q()
                q.add(Q(dates__startswith=str(today_cal)[0:10]), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                today_turtle_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                today_turtle_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_one_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_one_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_one_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_two_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_two_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_two_count = 0
            try:    
                q = Q()
                q.add(Q(dates__startswith=final_before_three_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_three_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_three_count = 0
            try:
                q = Q()
                q.add(Q(dates__startswith=final_before_four_day), q.AND)
                q.add(Q(path__contains=str(request.user.username)), q.AND)
                before_turtle_four_count = len(coach_turtle.objects.filter(q))
            except coach_turtle.DoesNotExist:
                before_turtle_four_count = 0
                    
            data = {
                "today1" : str(today_cal)[0:10],
                "today2" : final_before_one_day,
                "today3" : final_before_two_day,
                "today4" : final_before_three_day,
                "today5" : final_before_four_day,
                "today_turtle_count" : today_turtle_count,
                "today_shoulder_count" : today_shoulder_count,
                "today_nail_count" : today_nail_count,
                "before_one_turtle_count" : before_turtle_one_count,
                "before_one_shoulder_count" : before_shoulder_one_count,
                "before_one_nail_count" : before_nail_one_count,
                "before_two_turtle_count" : before_turtle_two_count,
                "before_two_shoulder_count" : before_shoulder_two_count,
                "before_two_nail_count" : before_nail_two_count,
                "before_three_turtle_count" : before_turtle_three_count,
                "before_three_shoulder_count" : before_shoulder_three_count,
                "before_three_nail_count" : before_nail_three_count,
                "before_four_turtle_count" : before_turtle_four_count,
                "before_four_shoulder_count" : before_shoulder_four_count,
                "before_four_nail_count" : before_nail_four_count,
            }     
                
        return render(request, 'ar_graph.html', context=data)
    
def forgot(request):
    
    return render(request, 'forgot.html')

def save_pic(request):
    # 데이터베이스의 총 개수를 전달
    pic_nail_list_cnt = coach_nail.objects.all().count()
    # 데이터베이스 불러오기(전체)
    pic_nail_list = coach_nail.objects.all().order_by('-id')    
    
    pic_nail_list = coach_nail.objects.filter(id=request.user.username)
    
    # pic_nail_list = coach_nail.objects.filter(id=request.user.username)
    # pic_shoulder_list = coach_shoulder.objects.filter(id=request.user.username)
    # pic_turtle_list = coach_turtle.objects.filter(id=request.user.username)
    
    pages = Paginator(pic_nail_list, 9)
    page = request.GET.get('page', '1')
    db_data = pages.get_page(page)
    
    # print(db_data.path)
    
    # 가공
    datas = {
        "pic_nail_list_cnt" : pic_nail_list_cnt,
        "pic_nail_list" : db_data, # pic_list_list, # path
    }
    
    return render(request, 'save_pic.html', datas)

def nail_pic(request):
    # 데이터베이스의 총 개수를 전달
    pic_nail_list_cnt = coach_nail.objects.all().count()
    # 데이터베이스 불러오기(전체)
    pic_nail_list = coach_nail.objects.all().order_by('-id')    
    
    pic_nail_list = coach_nail.objects.filter(id=request.user.username)
    
    # pic_nail_list = coach_nail.objects.filter(id=request.user.username)
    # pic_shoulder_list = coach_shoulder.objects.filter(id=request.user.username)
    # pic_turtle_list = coach_turtle.objects.filter(id=request.user.username)
    
    pages = Paginator(pic_nail_list, 9)
    page = request.GET.get('page', '1')
    db_data = pages.get_page(page)
    
    # print(db_data.path)
    
    # 가공
    datas = {
        "pic_nail_list_cnt" : pic_nail_list_cnt,
        "pic_nail_list" : db_data, # pic_list_list, # path
    }
    
    return render(request, 'nail_pic.html', datas)

def shoulder_pic(request):
    # 데이터베이스의 총 개수를 전달
    pic_shoulder_list_cnt = coach_shoulder.objects.all().count()
    # 데이터베이스 불러오기(전체)
    pic_shoulder_list = coach_shoulder.objects.all().order_by('-id')    
    
    pic_shoulder_list = coach_shoulder.objects.filter(id=request.user.username)
    
    # pic_nail_list = coach_nail.objects.filter(id=request.user.username)
    # pic_shoulder_list = coach_shoulder.objects.filter(id=request.user.username)
    # pic_turtle_list = coach_turtle.objects.filter(id=request.user.username)
    
    pages = Paginator(pic_shoulder_list, 9)
    page = request.GET.get('page', '1')
    db_data = pages.get_page(page)
    
    # print(db_data.path)
    
    # 가공
    datas = {
        "pic_shoulder_list_cnt" : pic_shoulder_list_cnt,
        "pic_shoulder_list" : db_data, # pic_list_list, # path
    }
    
    return render(request, 'shoulder_pic.html', datas)

def turtle_pic(request):
    # 데이터베이스의 총 개수를 전달
    pic_turtle_list_cnt = coach_turtle.objects.all().count()
    # 데이터베이스 불러오기(전체)
    pic_turtle_list = coach_turtle.objects.all().order_by('-id')    
    
    pic_turtle_list = coach_turtle.objects.filter(id=request.user.username)
    
    # pic_nail_list = coach_nail.objects.filter(id=request.user.username)
    # pic_shoulder_list = coach_shoulder.objects.filter(id=request.user.username)
    # pic_turtle_list = coach_turtle.objects.filter(id=request.user.username)
    
    pages = Paginator(pic_turtle_list, 9)
    page = request.GET.get('page', '1')
    db_data = pages.get_page(page)
    
    # print(db_data.path)
    
    # 가공
    datas = {
        "pic_turtle_list_cnt" : pic_turtle_list_cnt,
        "pic_turtle_list" : db_data, # pic_list_list, # path
    }
    
    return render(request, 'turtle_pic.html', datas)



def update(request):
    if request.method == "POST":
        form = User_update_form(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            return redirect('index')
    else:
        form = User_update_form(instance=request.user)
    return render(request, 'update.html', {'form': form})

def start(request):
    return render(request, 'start.html')

def turtle(request):
    return render(request, 'turtle.html')
def nail(request):
    return render(request, 'nail.html')
def leg(request):
    return render(request, 'leg.html')
def shoulder(request):
    return render(request, 'shoulder.html')

def signup(request):
    if request.method == 'POST':
        if request.POST['password1'] == request.POST['password2']:
            user = User.objects.create_user(
                                            username=request.POST['username'],
                                            password=request.POST['password1'],
                                            last_name=request.POST['last_name'],
                                            email=request.POST['email'],
                                            #first_name = False,
                                            )
            
            return redirect('/')
        return render(request, 'join.html')
    return render(request, 'join.html')

def ext_name(filename):
    name, ext = os.path.splitext(filename)
    
    return ext

def up_image(request):
    print("up_image 경로입니다.")
    uuid_img = uuid.uuid4().hex
    print("uuid = ", uuid_img)
    
    # 첨부파일이 등록되었는지를 확인
    if request.method == "POST":
        try:
            image = request.FILES['image']
        except:
            messages.info(request, '업로드된 파일이 존재하지 않습니다.')
            return redirect('index')

    # 실제 처리
    # (1) 이미지 서버에 저장
    image_name = image.name
    print("업로드 된 이미지 이름 = ", image_name)
    image_ext = ext_name(image_name)
    print("업로드 확장자 = ", image_ext)
    # 최종 파일명
    up_image = uuid_img + image_ext
    # 이미지 저장
    fs = FileSystemStorage(location='media/upimg', base_url='media/upimg')
    save_file = fs.save(up_image, image)
    
    # 파일명, 파일이미지 index에 보여주는 처리
    upload_image_path = '/' + fs.url(save_file)
    print("실제 이미지경로 = ", upload_image_path)
    
    datas = {
    "upload_image_path" : upload_image_path,
    "up_image" : up_image,
    }
    
    return render(request, 'start.html', datas)