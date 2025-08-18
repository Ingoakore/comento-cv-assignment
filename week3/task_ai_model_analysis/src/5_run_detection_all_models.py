#run_detection_all_models.py

from ultralytics import YOLO
import cv2
import os

if __name__ == '__main__':
    
    # --- 1. 기본 설정 ---
    # 테스트할 이미지 경로를 고정합니다.
    TEST_IMAGE_PATH = "../input_data/test_image.jpg" # 'input_data' 폴더의 테스트 이미지
    OUTPUT_DIR = '../outputs' # 결과물을 저장할 별도의 폴더

    # 테스트할 모델들의 경로와, 결과 파일에 사용할 이름을 정의합니다.
    models_to_test = {
        "baseline": "../runs/training/yolov8n/weights/best.pt",
        "augmented": "../runs/tuning/yolov8n_augmented/weights/best.pt",
        "yolov8s": "../runs/tuning/yolov8s/weights/best.pt",
        "50_epochs": "../runs/tuning/yolov8n_50_epoch/weights/best.pt"
    }

    # --- 2. 모든 모델에 대해 순서대로 객체 탐지 수행 ---
    print("===== 모든 훈련된 모델에 대한 객체 탐지를 시작합니다. =====")

    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"오류: 테스트 이미지 '{TEST_IMAGE_PATH}'를 찾을 수 없습니다.")
    else:
        # 결과물 저장 폴더가 없으면 생성합니다.
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # 정의된 모델 목록을 하나씩 순회하며 테스트를 진행합니다.
        for model_name, model_path in models_to_test.items():
            print(f"\n--- 모델 '{model_name}' 테스트 중 ---")

            if not os.path.exists(model_path):
                print(f"경고: 모델 파일 '{model_path}'를 찾을 수 없어 건너뜁니다.")
                continue # 다음 모델로 넘어감

            # 1. 모델 로드
            model = YOLO(model_path)
            
            # 2. 객체 탐지 수행
            results = model(TEST_IMAGE_PATH)
            
            # 3. 탐지 결과를 이미지에 그리기
            annotated_image = results[0].plot()

            # 4. 결과 저장
            output_filename = f"detection_result_{model_name}.jpg"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, annotated_image)
            print(f"성공: 결과가 '{output_path}'에 저장되었습니다.")

            cv2.imshow(f"Result: {model_name}", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("\n===== 모든 모델에 대한 객체 탐지가 완료되었습니다. =====")
