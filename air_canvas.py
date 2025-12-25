import cv2
import numpy as np
import mediapipe as mp
import math
import time

# --- INITIALIZATION ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.5
)
cap = cv2.VideoCapture(0)

# Canvas Setup
canvas = None 
undo_stack = [] 
prev_x, prev_y = 0, 0   
curr_x, curr_y = 0, 0   
smoothing_factor = 0.5 

draw_color = (255, 0, 0) 
brush_thickness = 5 # CONSTANT SIZE

# CALIBRATION
THUMB_THRESHOLD = 50 

def save_canvas_state(c):
    if len(undo_stack) > 10: undo_stack.pop(0)
    undo_stack.append(c.copy())

print("Air Canvas: Constant Brush Size.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)
        save_canvas_state(canvas)

    # --- UI LAYOUT ---
    cv2.rectangle(frame, (30, 1), (120, 65), (122, 122, 122), -1); cv2.putText(frame, "CLEAR", (45, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (140, 1), (220, 65), (255, 0, 0), -1)   
    cv2.rectangle(frame, (240, 1), (320, 65), (0, 255, 0), -1)   
    cv2.rectangle(frame, (340, 1), (420, 65), (0, 0, 255), -1)   
    cv2.rectangle(frame, (440, 1), (520, 65), (0, 255, 255), -1) 
    
    # Active Color Indicator
    cv2.rectangle(frame, (540, 1), (630, 65), draw_color, -1)
    cv2.putText(frame, "ACTIVE", (555, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # --- TRACKING ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            
            # --- CUSTOM VISUALIZATION ---
            lm_list = []
            for id, lm in enumerate(hand_lms.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Draw Lines (Thumb & Index Only)
            # Thumb (Green)
            cv2.line(frame, lm_list[0], lm_list[1], (0, 255, 0), 2)
            cv2.line(frame, lm_list[1], lm_list[2], (0, 255, 0), 2)
            cv2.line(frame, lm_list[2], lm_list[3], (0, 255, 0), 2)
            cv2.line(frame, lm_list[3], lm_list[4], (0, 255, 0), 2)
            
            # Index (Pink)
            cv2.line(frame, lm_list[0], lm_list[5], (255, 0, 255), 2)
            cv2.line(frame, lm_list[5], lm_list[6], (255, 0, 255), 2)
            cv2.line(frame, lm_list[6], lm_list[7], (255, 0, 255), 2)
            cv2.line(frame, lm_list[7], lm_list[8], (255, 0, 255), 2)

            # Draw Tips
            cv2.circle(frame, lm_list[4], 8, (0, 255, 0), -1)   # Thumb
            cv2.circle(frame, lm_list[8], 8, (255, 0, 255), -1) # Index

            # --- LOGIC ---
            raw_x, raw_y = lm_list[8] # Index Tip
            xt, yt = lm_list[4]       # Thumb Tip
            x_mcp, y_mcp = lm_list[5] # Index Base
            
            # Smoothing
            if prev_x == 0 and prev_y == 0: curr_x, curr_y = raw_x, raw_y
            else:
                curr_x = int(prev_x + (raw_x - prev_x) * smoothing_factor)
                curr_y = int(prev_y + (raw_y - prev_y) * smoothing_factor)

            # Thumb Logic (Draw vs Hover)
            thumb_dist = math.hypot(xt - x_mcp, yt - y_mcp)
            thumb_open = thumb_dist > THUMB_THRESHOLD 

            # Debug distance
            cv2.putText(frame, f'Dist: {int(thumb_dist)}', (curr_x+30, curr_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # --- STATE MACHINE ---
            if curr_y < 65: # Selection Mode
                if prev_x != 0: save_canvas_state(canvas)
                prev_x, prev_y = curr_x, curr_y 
                
                if 30 < curr_x < 120: save_canvas_state(canvas); canvas = np.zeros_like(frame)
                elif 140 < curr_x < 220: draw_color = (255, 0, 0)
                elif 240 < curr_x < 320: draw_color = (0, 255, 0)
                elif 340 < curr_x < 420: draw_color = (0, 0, 255)
                elif 440 < curr_x < 520: draw_color = (0, 255, 255)
                
                cv2.circle(frame, (curr_x, curr_y), 15, (255, 255, 255), 2)

            elif thumb_open: # HOVER
                if prev_x != 0 and prev_y != 0: pass
                prev_x, prev_y = curr_x, curr_y 
                
                # Show brush preview (hollow circle)
                cv2.circle(frame, (curr_x, curr_y), brush_thickness, (100, 100, 100), 1)

            else: # DRAW
                if prev_x == 0 and prev_y == 0:
                    save_canvas_state(canvas)
                    prev_x, prev_y = curr_x, curr_y
                
                cv2.circle(frame, (curr_x, curr_y), brush_thickness, draw_color, -1)
                cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), draw_color, brush_thickness)
                prev_x, prev_y = curr_x, curr_y
    else:
        prev_x, prev_y = 0, 0

    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, img_inv)
    frame = cv2.bitwise_or(frame, canvas)
    cv2.line(frame, (0, 65), (w, 65), (70, 70, 70), 2)
    
    cv2.imshow('ECE Air Canvas - Constant', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('s'): cv2.imwrite(f"Art_{int(time.time())}.jpg", frame)
    elif key == ord('u'): 
        if len(undo_stack) > 0: canvas = undo_stack.pop()

cap.release()
cv2.destroyAllWindows()