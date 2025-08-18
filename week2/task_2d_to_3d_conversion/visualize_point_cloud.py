import numpy as np
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_with_matplotlib():
    """
    NumPy 파일로 저장된 3D 포인트 클라우드 데이터를 불러와 Matplotlib으로 시각화합니다.
    """
    point_cloud_path = os.path.join('outputs', 'point_cloud.npy')

    if not os.path.exists(point_cloud_path):
        # 포인트 클라우드 파일이 없을 경우 경고 메시지를 출력하고 종료합니다.
        print(f"경고: '{point_cloud_path}' 파일이 없습니다. 먼저 depth_processing.py를 실행하세요.")
        return

    points_3d = np.load(point_cloud_path) # 저장된 .npy 파일을 불러옵니다.
    
    print("데이터 샘플링을 시작합니다...")
    # 전체 데이터가 너무 많아 렌더링이 느릴 수 있으므로 1/4만 샘플링합니다.
    points_sampled = points_3d[::2, ::2, :].reshape(-1, 3)
    
    print("Matplotlib 3D 뷰어를 생성합니다...")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d') # 3D 그래프를 그릴 서브플롯을 생성합니다.

    # 샘플링된 데이터에서 X, Y, Z 좌표를 각각 추출합니다.
    x = points_sampled[:, 0]
    y = points_sampled[:, 1]
    z = points_sampled[:, 2]
    
    # 3D 산점도(scatter plot)를 그립니다. 점 색깔은 Z(깊이) 값에 따라 다르게 표시됩니다.
    scatter = ax.scatter(x, y, z, c=z, cmap='jet', s=1)

    # 그래프 제목과 축 레이블을 설정합니다.
    ax.set_title('3D Point Cloud from Image Brightness', fontsize=16, pad=20)
    ax.set_xlabel('X-axis (Width)')
    ax.set_ylabel('Y-axis (Height)')
    ax.set_zlabel('Z-axis (Depth/Brightness)')
    
    # Y축을 반전시켜 이미지의 원점(0,0)을 좌측 상단에 위치시킵니다.
    ax.invert_yaxis()
    
    # 깊이 값에 따른 색상을 보여주는 컬러바를 추가합니다.
    fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, label='Depth Value (Brightness)')

    print("그래프를 화면에 표시합니다. 창을 닫으면 프로그램이 종료됩니다.")
    plt.show() # 그래프 창을 화면에 표시합니다.

if __name__ == "__main__":
    visualize_with_matplotlib() # 시각화 함수 실행
