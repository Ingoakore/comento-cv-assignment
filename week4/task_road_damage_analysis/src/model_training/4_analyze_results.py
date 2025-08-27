# analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    
    # 분석할 실험 모델들의 이름과 해당 훈련 결과 파일(results.csv)의 경로를 정의합니다.
    # 각 CSV 파일은 에포크별 훈련 및 검증 성능 지표를 담고 있습니다.
    experiments = {
        "Baseline (YOLOv8s, 30e)": "../../runs/baseline/road_damage_detector_baseline/results.csv",
        "Over-Tuned (100e, All Aug)": "../../runs/tuned/road_damage_detector_over_tuned/results.csv",
        "Refined-Tuned (50e, Select Aug)": "../../runs/tuned/road_damage_detector_refined_tuned/results.csv"
    }

    # Matplotlib를 사용하여 3개의 서브플롯(그래프)을 담을 figure와 axes 객체를 생성합니다.
    # figsize는 전체 그림의 크기를 지정합니다.
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))
    # 전체 그림의 제목을 설정합니다.
    fig.suptitle('Model Performance Comparison by Tuning Strategy', fontsize=16)
    
    print("===== 3개 모델 성능 비교 분석을 시작합니다. =====")

    # 정의된 각 실험에 대해 반복문을 실행합니다.
    for name, path in experiments.items():
        try:
            # 지정된 경로에 파일이 있는지 확인하고, 없으면 백업 경로를 시도합니다.
            # 이는 Ultralytics가 같은 이름으로 훈련 시 폴더명 뒤에 숫자를 추가하는 경우를 대비합니다.
            if not os.path.exists(path):
                path_alt = path.replace("/results.csv", "2/results.csv")
                if os.path.exists(path_alt):
                    path = path_alt
                else:
                    print(f"경고: '{path}' 또는 '{path_alt}' 파일을 찾을 수 없습니다. 해당 실험을 건너뜁니다.")
                    continue

            # pandas를 사용하여 CSV 파일을 데이터프레임으로 읽어옵니다.
            df = pd.read_csv(path)
            # 열 이름의 공백을 제거하여 접근을 용이하게 합니다.
            df.columns = df.columns.str.strip()

            # 그래프에 사용할 데이터(에포크, mAP, 정밀도, 재현율)를 추출합니다.
            epochs = df['epoch']
            map_scores = df['metrics/mAP50-95(B)']
            precision_scores = df['metrics/precision(B)']
            recall_scores = df['metrics/recall(B)']

            # 각 성능 지표를 해당하는 서브플롯에 그립니다.
            # label은 범례에 표시될 실험 이름, marker와 linestyle은 그래프 스타일을 지정합니다.
            axs[0].plot(epochs, map_scores, label=name, marker='o', linestyle='--')
            axs[1].plot(epochs, precision_scores, label=name, marker='o', linestyle='--')
            axs[2].plot(epochs, recall_scores, label=name, marker='o', linestyle='--')
            
            print(f"- '{name}' 데이터 로드 및 그래프 추가 완료.")
        except Exception as e:
            # 파일 읽기 또는 처리 중 오류가 발생하면 경고 메시지를 출력합니다.
            print(f"경고: '{name}' 처리 중 에러 발생 - {e}")

    # ---
    ## 그래프 설정 및 저장

    # 첫 번째 서브플롯(mAP)의 제목과 y축 레이블을 설정합니다.
    axs[0].set_title("mAP50-95 Score (Overall Performance)")
    axs[0].set_ylabel("mAP")
    # 두 번째 서브플롯(정밀도)의 제목과 y축 레이블을 설정합니다.
    axs[1].set_title("Precision (How accurate are the predictions?)")
    axs[1].set_ylabel("Precision")
    # 세 번째 서브플롯(재현율)의 제목과 y축 레이블을 설정합니다.
    axs[2].set_title("Recall (How many objects were found?)")
    axs[2].set_ylabel("Recall")
    
    # 모든 서브플롯에 대해 공통 설정을 적용합니다.
    for ax in axs:
        ax.set_xlabel("Epochs")  # x축 레이블을 에포크로 설정
        ax.legend()              # 범례(label)를 표시
        ax.grid(True)            # 그리드(격자)를 표시
        ax.set_ylim(bottom=0)    # y축의 최솟값을 0으로 설정

    # 서브플롯 간의 간격을 자동으로 조정하여 겹치지 않게 합니다.
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # 완성된 그래프를 이미지 파일로 저장합니다.
    output_path = "../../outputs/performance_comparison_graph.png"
    plt.savefig(output_path)
    print(f"\n성공: 비교 분석 그래프가 '{output_path}'에 저장되었습니다.")
    
    # 그래프를 화면에 표시합니다.
    plt.show()
