import cv2

cap = cv2.VideoCapture(0)

# Frame Per Second (FPS) 정보 얻기
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

while True: # 무한 루프
    ret, frame = cap.read() # 두 개의 값을 반환하므로 두 변수 지정

    if not ret: # 새로운 프레임을 못받아 왔을 때 braek
        break
        
    # 정지화면에서 윤곽선을 추출
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
    sobel2 = cv2.Sobel(gray, cv2.CV_8U, 0, 1, 3)
    sobel3 = cv2.Sobel(gray, cv2.CV_8U, 2, 2, 3)

    # Frame의 size (height, width) 얻기
    height, width, channels = frame.shape
    print(f"Frame size: {width} x {height}, Channels: {channels}")
    
    cv2.imshow('frame', frame)
    cv2.imshow("sobel", sobel)
    cv2.imshow("sobel2", sobel2)
    cv2.imshow("sobel3", sobel3)
    # 10ms 기다리고 다음 프레임으로 전환, Esc누르면 while 강제 종료
    if cv2.waitKey(1) == 27:
        break

cap.release() # 사용한 자원 해제
cv2.destroyAllWindows()
