# utils_road_damage_analyzer.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import os
from PIL import ImageFont, ImageDraw, Image

# 폰트 파일을 로드하여 한글 텍스트를 이미지에 그릴 수 있도록 설정합니다.
FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "fonts", "malgun.ttf")
try:
    # 다양한 용도의 한글 폰트를 로드합니다.
    FONT_TITLE = ImageFont.truetype(FONT_PATH, 30)
    FONT_REPORT = ImageFont.truetype(FONT_PATH, 24)
    FONT_LABEL = ImageFont.truetype(FONT_PATH, 20)
    # Matplotlib에서도 한글 폰트를 사용하도록 설정합니다.
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호가 깨지는 것을 방지합니다.
except Exception as e:
    # 폰트 로드 실패 시 경고 메시지를 출력하고 기본 폰트를 사용합니다.
    print(f"경고: '{FONT_PATH}' 폰트 설정 중 오류 발생. 기본 폰트를 사용합니다. - {e}")
    FONT_TITLE, FONT_REPORT, FONT_LABEL = [ImageFont.load_default()] * 3

def draw_korean_text(image, text, position, font, color):
    """OpenCV 이미지에 한글 텍스트를 그리는 함수입니다. Pillow 라이브러리를 사용해 한글 폰트를 지원합니다."""
    # OpenCV 이미지를 PIL(Pillow) 이미지로 변환합니다.
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    # 지정된 위치, 폰트, 색상으로 텍스트를 그립니다.
    draw.text(position, text, font=font, fill=color)
    # 다시 OpenCV 이미지 형식으로 변환하여 반환합니다.
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def add_panel_title(panel, title, font):
    """이미지 패널 상단에 반투명한 배경과 함께 제목을 추가하는 함수입니다."""
    panel_with_title = panel.copy()
    overlay = panel_with_title.copy()
    # 제목 배경으로 사용할 검은색 사각형 오버레이를 그립니다.
    cv2.rectangle(overlay, (0, 0), (len(title) * 30 + 20, 45), (0, 0, 0), -1)
    alpha = 0.6
    # 원본 이미지와 오버레이를 혼합하여 반투명 효과를 만듭니다.
    panel_with_title = cv2.addWeighted(overlay, alpha, panel_with_title, 1 - alpha, 0)
    # 한글 제목을 그립니다.
    panel_with_title = draw_korean_text(panel_with_title, title, (10, 5), font, (255, 255, 255))
    return panel_with_title

def generate_depth_and_points(image_roi):
    """
    이미지의 ROI(관심 영역)를 분석하여 가상의 깊이 맵과 3D 포인트 클라우드를 생성합니다.
    이미지의 밝기를 깊이로 가정하는 단순화된 방식을 사용합니다.
    """
    # 입력 값이 Numpy 배열인지 확인하고, 아니면 TypeError를 발생시킵니다.
    if not isinstance(image_roi, np.ndarray):
        raise TypeError("입력값은 Numpy 배열(이미지)이어야 합니다.")
    # 입력된 이미지 영역이 비어있는지 확인하고, 비어있으면 ValueError를 발생시킵니다.
    if image_roi.size == 0:
        raise ValueError("입력된 이미지 영역이 비어있습니다.")
    
    # 이미지를 회색조로 변환하여 각 픽셀의 밝기를 '깊이'로 사용합니다.
    gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    # 깊이 맵을 시각화하기 위해 컬러맵을 적용합니다.
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    h, w = image_roi.shape[:2]
    # 2차원 그리드(X, Y)를 생성하고 회색조 이미지를 Z값으로 사용하여 3D 포인트 클라우드를 만듭니다.
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)
    # X, Y, Z 채널을 합쳐 (높이, 너비, 3) 형태의 3D 포인트 클라우드 배열을 만듭니다.
    points_3d = np.dstack((X, Y, Z))
    
    return depth_map, points_3d.astype(np.float32)

def analyze_severity_from_points(points_3d):
    """
    3D 포인트 클라우드 데이터를 분석하여 도로 파손의 심각도를 판별하는 함수입니다.
    평균 깊이와 표면의 밝기(깊이) 변화(표준편차)를 기준으로 판단합니다.
    """
    # 입력 값이 Numpy 배열인지 확인하고, 아니면 TypeError를 발생시킵니다.
    if not isinstance(points_3d, np.ndarray):
        raise TypeError("입력값은 Numpy 배열(포인트 클라우드)이어야 합니다.")
    # 입력된 포인트 클라우드가 비어있는지 확인하고, 비어있으면 ValueError를 발생시킵니다.
    if points_3d.size == 0:
        raise ValueError("입력된 포인트 클라우드가 비어있습니다.")
        
    # 깊이(Z) 값만 추출합니다.
    z_values = points_3d[:, :, 2]
    # Z값의 평균과 표준편차를 계산합니다.
    average_depth = np.mean(z_values)
    std_dev_depth = np.std(z_values)

    # 정의된 임계값을 기준으로 심각도 등급과 색상을 결정합니다.
    if average_depth < 80 or std_dev_depth > 45:
        level = "높음"  # 깊이가 매우 낮거나 표면이 매우 거칠면 심각도가 높음
        color = (0, 0, 255) # 빨간색
    elif average_depth < 150 or std_dev_depth > 30:
        level = "중간"
        color = (0, 255, 255) # 노란색
    else:
        level = "낮음"
        color = (0, 255, 0) # 초록색
        
    return level, color, average_depth, std_dev_depth

def create_3d_plot_image(points_3d, title="3D Point Cloud"):
    """
    3D 포인트 클라우드를 Matplotlib를 사용하여 시각화하고, 이를 이미지로 변환합니다.
    """
    fig = None  # fig 변수를 미리 선언
    try:
        # 포인트 클라우드를 샘플링하여 렌더링 성능을 개선합니다.
        points_sampled = points_3d[::5, ::5, :].reshape(-1, 3)
        
        # Matplotlib 3D 플롯을 생성합니다.
        fig = plt.figure(figsize=(6.4, 4.8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 샘플링된 포인트의 X, Y, Z 값을 추출합니다.
        x, y, z = points_sampled[:, 0], points_sampled[:, 1], points_sampled[:, 2]
        # 산점도(scatter)로 3D 포인트를 그립니다.
        ax.scatter(x, y, z, c=z, cmap='jet', s=1)
        
        # Y축을 반전시켜 이미지의 원점(좌측 상단)에 맞춥니다.
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("너비 (Width)", fontsize=8)
        ax.set_ylabel("높이 (Height)", fontsize=8)
        ax.set_zlabel("깊이 (Depth)", fontsize=8)

        # 플롯을 메모리 버퍼에 PNG 이미지로 저장합니다.
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=fig.dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        
        # 메모리에서 읽은 이미지 데이터를 OpenCV 이미지로 디코딩합니다.
        plot_image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Matplotlib 플롯 디코딩에 실패했을 경우를 처리합니다.
        if plot_image is None:
            print("경고: Matplotlib 플롯 디코딩에 실패했습니다. 빈 이미지를 반환합니다.")
            return np.zeros((480, 640, 3), dtype=np.uint8)

        return plot_image

    except Exception as e:
        # 3D 플롯 생성 중 오류 발생 시 예외를 처리하고 빈 이미지를 반환합니다.
        print(f"오류: 3D 플롯 생성 중 예외 발생 - {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    finally:
        # 함수 실행이 끝나면 Matplotlib figure를 닫아 메모리 누수를 방지합니다.
        if fig is not None:
            plt.close(fig)
