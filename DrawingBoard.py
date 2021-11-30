import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



global xc, yc, ix, iy, operation, toolSet
operation = "draw"
toolClicked = False
ix, iy = -1, -1


xc = 640
yc = 480

def drawing(event, x, y, flags, param):
    global ix, iy, operation, operationIcon, toolClicked
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
    if(flags == 1):
        if x>0 and x<50 and y>0 and y<50 :
            if toolClicked:
                return
            toolClicked = True
            if operation == "erase":
                operation = "draw"
                operationIcon = eraser
                return
            operation="erase"
            operationIcon = pencile
            return
        toolClicked = False
        if operation == "erase":
            if ix == -1 and iy == -1:
                cv2.line(board, (x, y), (x, y), (255, 255, 255), 20)
                # cv2.circle(board,(100,100), 100, (0,255,0), -1)  
                # cv2.rectangle(board,(x,y),(x+30,y+30),(0,255,255),15)  
                ix, iy = x, y
                return
            else:
                cv2.line(board, (ix, iy), (x, y), (255, 255, 255), 20)
                # cv2.rectangle(board,(15,25),(200,150),(0,255,255),15)  
                ix, iy = x, y
                return
        if ix == -1 and iy == -1:
            cv2.line(board, (x, y), (x, y), (0, 0, 0), 2)
            ix, iy = x, y

        else:
            cv2.line(board, (ix, iy), (x, y), (0, 0, 0), 2)
            ix, iy = x, y


board = np.ones((480, 640, 3), np.uint8)
board.fill(255)

eraser = cv2.imread("./eraser.png")

eraser = cv2.resize(eraser, (48,48));
eraser = cv2.copyMakeBorder(eraser, 1,  1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)

pencile = cv2.imread("./pencile.png")

pencile = cv2.resize(pencile, (48,48));
pencile = cv2.copyMakeBorder(pencile, 1,  1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)


operationIcon = eraser
# cv2.imshow("Eraser", eraser)

cv2.namedWindow("Media Board")
# cv2.namedWindow("Media Board")
cv2.setMouseCallback("Media Board", drawing)



cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1) as hands:
  while cap.isOpened():
    board[0:50,0:50] = operationIcon

    success, image = cap.read()
    MediaBoard = np.ones((480, 640, 3), np.uint8)
    MediaBoard.fill(255)

    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.flip(image, 1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


        x, y = int(results.multi_hand_landmarks[0].landmark[8].x*xc), int(results.multi_hand_landmarks[0].landmark[8].y*yc)
        x2, y2 = int(results.multi_hand_landmarks[0].landmark[12].x*xc), int(results.multi_hand_landmarks[0].landmark[12].y*yc)


        if((x2 <= x+20 and x2 >= x-20) or (y2 <= y+20 and y2 >= y-20)):
            drawing("", x, y, 1, "")
        else:
            drawing("", x, y, 0, "")


        # mask[x-50:x, x-50:x] = operationIcon
        if operation == "erase":
            cv2.line(MediaBoard, (x, y), (x, y), (255, 0, 0), 20)
        else:
            cv2.line(MediaBoard, (x, y), (x, y), (0, 0, 0), 5)


        cv2.line(image, (x, y), (x, y), (0, 0, 0), 10)
        cv2.line(image, (x2, y2), (x2, y2), (0, 0, 0), 10)
        ix, iy = x, y


    cv2.imshow('Hand Landmarks', image)
    # mask[200:250, 200:250] = operationIcon
    MediaBoard = cv2.bitwise_and(board, MediaBoard)
    # mask = cv2.bitwise_and(image, mask)
    # cv2.imshow('Board', board)
    cv2.imshow('Media Board', MediaBoard)
    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()
cv2.destroyAllWindows()