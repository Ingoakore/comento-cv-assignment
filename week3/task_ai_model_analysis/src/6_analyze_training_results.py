# analyze_training_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    
    experiments = {
        "Baseline (YOLOv8n, 10e)": "../runs/training/yolov8n/results.csv",
        "Augmented (YOLOv8n, 10e)": "../runs/tuning/yolov8n_augmented/results.csv",
        "Larger Model (YOLOv8s, 10e)": "../runs/tuning/yolov8s/results.csv",
        "Longer Training (YOLOv8n, 50e)": "../runs/tuning/yolov8n_50_epoch/results.csv"
    }

    plt.figure(figsize=(12, 8))
    
    print("===== 모델 성능 비교 분석을 시작합니다. =====")

    styles = {
        "Baseline (YOLOv8n, 10e)": {"color": "blue", "linestyle": "-", "linewidth": 4, "alpha": 0.7},
        "Augmented (YOLOv8n, 10e)": {"color": "orange", "linestyle": "--", "linewidth": 2, "alpha": 1.0},
        "Larger Model (YOLOv8s, 10e)": {"color": "green", "linestyle": ":", "linewidth": 2, "alpha": 1.0},
        "Longer Training (YOLOv8n, 50e)": {"color": "red", "linestyle": "-.", "linewidth": 2, "alpha": 1.0}
    }

    for name, path in experiments.items():
        try:
            results_df = pd.read_csv(path)
            results_df.columns = results_df.columns.str.strip()
            epochs = results_df['epoch']
            map_scores = results_df['metrics/mAP50-95(B)']
            

            style = styles.get(name, {})
            plt.plot(epochs, map_scores, label=name, marker='o', **style)
            
            print(f"- '{name}' 데이터 로드 및 그래프 추가 완료.")
        except Exception as e:
            print(f"경고: '{name}' 처리 중 에러 발생 - {e}")

    plt.title("Analysis of Training Results by Experiment", fontsize=16)
    plt.xlabel("Epochs")
    plt.ylabel("mAP50-95 Score")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    print("\n분석 그래프를 화면에 표시합니다. 창을 닫으면 프로그램이 종료됩니다.")
    plt.show()
