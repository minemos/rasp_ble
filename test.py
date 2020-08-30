import cv2  
import numpy as np
import matplotlib.pyplot as plt #이미지 표시
import pytesseract  #글씨 읽기
from picamera import PiCamera #PICAM 연결
import time

def test(is_test=True):
    plt.style.use('dark_background') 

    if is_test == False:
        camera = PiCamera()

        camera.start_preview()
        time.sleep(5)
        camera.capture('test.png')
        camera.stop_preview()

    longest_text = 0
    img_ori = cv2.imread('test.png')  #이미지 로드

    height, width, channel = img_ori.shape  #너비 높이 채널 저장

    plt.figure(figsize=(12, 10))

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY) # BGR -> GRAY 변경

    plt.figure(figsize=(12, 10))

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    plt.figure(figsize=(12, 10))

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) #가우시안 블러 노이즈 감소

    img_thresh = cv2.adaptiveThreshold(    # 임계값 127기준 0,255로 
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )

    plt.figure(figsize=(12, 10))

    contours, _ = cv2.findContours(  #이미지에서 윤곽선 찾기
        img_thresh, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))  #윤곽선 그리기  -1은 전체컨트어를 그린다

    plt.figure(figsize=(12, 10))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []  

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  #컨투어의 사각형 범위를 찾는다
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)  #rectangle 함수로 사각형을 그린다
        
        # insert to dict 
        contours_dict.append({  
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    plt.figure(figsize=(12, 10))

    MIN_AREA = 80  # Bounding Rect 범위
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 최소너비와 높이
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 가로 세로 비율 범위

    possible_contours = []  # Bounding Rect 범위 안 저장

    cnt = 0
    for d in contours_dict:  
        area = d['w'] * d['h'] #넓이 계산
        ratio = d['w'] / d['h'] #비율 계산
        
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO: # 조건 
            d['idx'] = cnt  #인덱스로 저장 (나중에 편하기위해)
            cnt += 1
            possible_contours.append(d) # 조건 포함 저장
            
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
    #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))

    MAX_DIAG_MULTIPLYER = 5 # 5 컨투어 대각선 길이의 5배
    MAX_ANGLE_DIFF = 12.0 # 12.0 #첫번째 컨투어 두번째 컨투어 중심을 이었을때 각도의 최대값
    MAX_AREA_DIFF = 0.5 # 0.5 # 면적의 차이
    MAX_WIDTH_DIFF = 0.8 # 너비 차이
    MAX_HEIGHT_DIFF = 0.2 # 높이 차이
    MIN_N_MATCHED = 3 # 3 # 만족하는 그룹의 개수 최소값

    def find_chars(contour_list): # 재귀함수로 반복찾기
        matched_result_idx = []  #최종적으로 남는 인덱스 값 저장
        
        for d1 in contour_list: #컨투어 2개비교하여 조건 
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']: # 같은 조건이면 continue
                    continue

                dx = abs(d1['cx'] - d2['cx']) 
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))  # 중심부 사이의 거리를 구한다
                if dx == 0:
                    angle_diff = 90  # 각도가 0일때는 오류가나기때문에 90으로 설정해서 아니라고 판단
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))  #artan이용 각도구하기 
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) #면적의 비율
                width_diff = abs(d1['w'] - d2['w']) / d1['w'] #너비의 비율
                height_diff = abs(d1['h'] - d2['h']) / d1['h'] #높이의 비율

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx']) #위 조건의 맞는 인덱스만 넣음 

            # append this contour
            matched_contours_idx.append(d1['idx']) 

            if len(matched_contours_idx) < MIN_N_MATCHED: # 3개이하일 경우 조건에 취급안함
                continue

            matched_result_idx.append(matched_contours_idx) # 위의조건 다 충족하면 최종후보군취급

            unmatched_contour_idx = [] 
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx']) # 조건 포함안되는 그룹 저장

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            
            # recursive
            recursive_contour_list = find_chars(unmatched_contour)
            
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx
        
    result_idx = find_chars(possible_contours)  # 조건 포함안되는 그룹 재귀함수 이용

    matched_result = []
    for idx_list in result_idx: # 다시 조건에 넣는과정
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:  # 조건에 만족하는 후보군 결과
        for d in r:
    #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

    plt.figure(figsize=(12, 10))

    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx']) # x방향으로 순차적으로 정렬

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2  #센터 x좌표 
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2  #센터 y좌표
        
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING 
        
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING) # 후보군 사이의 높이
        
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])  # 후보군 중심사이의 거리
        )
        
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) #arsin으로 각 구하기
        
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0) # 회전변환 행렬
        
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height)) # 이미지 변형 
        
        img_cropped = cv2.getRectSubPix(  # 회전된 이미지에서 숫자부분만 표시
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
        
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
        
        plt.subplot(len(matched_result), 1, i+1)

        longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)  
        
        # find contours again (same as above)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)  
        
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            area = w * h
            ratio = w / h

            if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h
                    
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        # chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0 -c tessedit_char_whitelist=0123456789')
        
        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c
        
        print(result_chars)
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

        plt.subplot(len(plate_imgs), 1, i+1)

    return ','.join(plate_chars)


if __name__ == "__main__":
    print(test())