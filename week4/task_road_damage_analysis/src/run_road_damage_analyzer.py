# run_road_damage_analyzer.py

import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
from PIL import Image, ImageTk
import threading
from ultralytics import YOLO
import cv2
import numpy as np
import os
from utils_road_damage_analyzer import (
    FONT_TITLE, FONT_REPORT, FONT_LABEL,
    draw_korean_text, add_panel_title,
    generate_depth_and_points,
    analyze_severity_from_points,
    create_3d_plot_image
)

# AI 도로 파손 분석기 애플리케이션의 메인 GUI 클래스입니다.
class RoadDamageAnalyzerApp:
    def __init__(self, root):
        # Tkinter 윈도우(root)를 초기화하고 제목과 크기를 설정합니다.
        self.root = root
        self.root.title("AI 도로 파손 분석기")
        self.root.geometry("1400x850")
        
        # 사용할 YOLO 모델들의 정보(경로, 저장 이름, 인스턴스)를 딕셔너리로 관리합니다.
        self.models = {
            "기준 모델 (Baseline)": {"path": "../runs/baseline/road_damage_detector_baseline/weights/best.pt", "savename": "baseline", "instance": None},
            "최종 튜닝 모델 (Tuned)": {"path": "../runs/tuned/road_damage_detector_refined_tuned/weights/best.pt", "savename": "tuned", "instance": None}
        }
        # 분석할 이미지의 경로와 대시보드 이미지를 저장할 변수를 초기화합니다.
        self.image_path = ""
        self.dashboard_image = None
        self.displayed_image = None

        # GUI 레이아웃을 설정합니다.
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 컨트롤 버튼 및 선택 위젯을 포함하는 상단 프레임을 생성합니다.
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.grid(row=0, column=0, sticky="ew")
        frame_row1 = ttk.Frame(control_frame)
        frame_row1.pack(fill=tk.X, pady=2)
        ttk.Label(frame_row1, text="분석 모델 선택:").pack(side=tk.LEFT, padx=(0, 5))
        self.model_selector = ttk.Combobox(frame_row1, values=list(self.models.keys()), width=30, state="readonly")
        self.model_selector.pack(side=tk.LEFT, padx=5)
        self.model_selector.current(0)
        ttk.Frame(frame_row1).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.analyze_button = ttk.Button(frame_row1, text="분석 실행", command=self.start_analysis, state="disabled")
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(frame_row1, text="결과 저장", command=self.save_result, state="disabled")
        self.save_button.pack(side=tk.LEFT, padx=5)
        frame_row2 = ttk.Frame(control_frame)
        frame_row2.pack(fill=tk.X, pady=2)
        self.select_button = ttk.Button(frame_row2, text="이미지 파일 선택", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=(0, 5))
        self.path_label = ttk.Label(frame_row2, text="선택된 파일 없음")
        self.path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 분석 결과 이미지를 표시할 캔버스 프레임을 생성합니다.
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=1, column=0, sticky="nsew")
        self.canvas = tk.Canvas(main_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_resize)
        
        # 진행 상황 바와 로그 텍스트 영역을 포함하는 하단 프레임을 생성합니다.
        log_frame = ttk.Frame(self.root, padding="10")
        log_frame.grid(row=2, column=0, sticky="ew")
        self.progress_bar = ttk.Progressbar(log_frame, orient="horizontal", length=100, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=5, wrap=tk.WORD)
        self.log_text.pack(fill=tk.X, expand=True)
        
    # 이미지 파일 선택 다이얼로그를 띄우고, 선택된 파일 경로를 업데이트합니다.
    def select_image(self):
        path = filedialog.askopenfilename(title="분석할 이미지 선택", filetypes=(("이미지 파일", "*.jpg *.jpeg *.png"),))
        if path:
            self.image_path = path
            self.path_label.config(text=path)
            self.analyze_button.config(state="normal")
            self.save_button.config(state="disabled")
            self.log_message("이미지가 선택되었습니다. '분석 실행' 버튼을 눌러주세요.")

    # 분석 시작을 위한 스레드를 시작합니다. GUI가 멈추지 않도록 백그라운드에서 실행됩니다.
    def start_analysis(self):
        if not self.image_path:
            self.log_message("오류: 먼저 이미지 파일을 선택해주세요.")
            return
        
        self.analyze_button.config(state="disabled")
        self.save_button.config(state="disabled")
        self.progress_bar["value"] = 0
        self.log_message("분석을 시작합니다. 잠시만 기다려주세요...")
        
        selected_model_name = self.model_selector.get()
        
        analysis_thread = threading.Thread(
            target=self.run_analysis_thread, 
            args=(selected_model_name,)
        )
        analysis_thread.start()

    # 분석 작업을 수행하는 스레드의 메인 함수입니다.
    def run_analysis_thread(self, selected_model_name):
        try:
            # 선택된 모델이 로드되지 않았다면 로드합니다.
            model_dict = self.models[selected_model_name]
            if model_dict.get("instance") is None:
                self.log_message(f"'{selected_model_name}' 모델을 처음 로드하는 중입니다...")
                model_path = model_dict["path"]
                if not os.path.exists(model_path):
                    self.log_message(f"오류: 모델 파일 '{model_path}'를 찾을 수 없습니다.")
                    return
                model_dict["instance"] = YOLO(model_path)
                self.log_message("모델 로드 완료.")
            
            # 이미지 분석을 실행하고 진행 상황을 업데이트합니다.
            model = model_dict["instance"]
            self.root.after(0, self.update_progress, 20)

            self.log_message(f"'{os.path.basename(self.image_path)}' 이미지 분석 중...")
            image = cv2.imread(self.image_path)
            results = model(image)
            self.root.after(0, self.update_progress, 50)
            self.log_message("AI 객체 탐지 완료.")

            analysis_result_data = {
                "original_image": image,
                "detection_results": results[0],
                "selected_model_name": selected_model_name
            }
            
            # GUI 업데이트 함수를 메인 스레드에서 호출합니다.
            self.root.after(0, self.update_gui_with_results, analysis_result_data)

        except Exception as e:
            self.log_message(f"분석 스레드에서 치명적 오류 발생: {e}")
        finally:
            # 분석이 완료되거나 오류가 발생하면 '분석 실행' 버튼을 다시 활성화합니다.
            self.root.after(0, self.analyze_button.config, {"state": "normal"})

    # 진행 상황 바의 값을 업데이트합니다.
    def update_progress(self, value):
        self.progress_bar["value"] = value

    # 분석 결과를 바탕으로 GUI를 업데이트하는 함수입니다.
    def update_gui_with_results(self, result_data):
        try:
            image = result_data["original_image"]
            results = result_data["detection_results"]
            selected_model_name = result_data["selected_model_name"]

            H, W = 480, 640
            
            # 1. 원본 이미지 패널 생성
            panel1 = cv2.resize(image, (W, H))
            panel1 = add_panel_title(panel1, "1. 원본 이미지", FONT_TITLE)
            
            # 2. 분석 결과 패널 초기화 및 파손 영역 시각화
            panel2 = image.copy()
            panel3 = np.zeros((H, W, 3), dtype=np.uint8)
            panel3 = draw_korean_text(panel3, "탐지된 파손 없음", (150, H//2), FONT_TITLE, (255, 255, 255))
            analysis_texts = ["4. 상세 3D 분석 리포트", f"    (모델: {selected_model_name})"]
            main_object_info = None
            largest_area = 0

            if len(results.boxes) > 0:
                for i, box in enumerate(results.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if y2 > y1 and x2 > x1:
                        # 파손 영역(ROI)에 대해 3D 분석을 수행합니다.
                        roi = image[y1:y2, x1:x2]
                        depth_map, points_3d = generate_depth_and_points(roi)
                        level, color, avg_d, std_d = analyze_severity_from_points(points_3d)
                        
                        # 탐지된 파손에 라벨과 경계 상자를 그립니다.
                        label = f"{results.names[int(box.cls[0])]} (심각도: {level})"
                        panel2 = draw_korean_text(panel2, label, (x1, y1 - 30), FONT_LABEL, color)
                        cv2.rectangle(panel2, (x1, y1), (x2, y2), color, 2)
                        
                        # 가장 큰 파손 영역을 찾아 상세 분석을 위한 데이터를 저장합니다.
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            main_object_info = {"points_3d": points_3d, "name": results.names[int(box.cls[0])], "level": level, "avg_d": avg_d, "std_d": std_d}
                
                # 가장 큰 파손에 대한 3D 시각화 및 리포트를 생성합니다.
                if main_object_info:
                    self.log_message("가장 큰 파손 영역에 대한 3D 분석을 시작합니다...")
                    panel3 = create_3d_plot_image(main_object_info["points_3d"], title=f"3D View of '{main_object_info['name']}'")
                    self.root.after(0, self.update_progress, 70)
                    analysis_texts.append(f"- 객체 종류: {main_object_info['name']}")
                    analysis_texts.append(f"- 심각도 등급: {main_object_info['level']}")
                    analysis_texts.append("--------------------")
                    analysis_texts.append(f"- 평균 깊이(밝기): {main_object_info['avg_d']:.2f}")
                    analysis_texts.append(f"- 표면 거칠기: {main_object_info['std_d']:.2f}")
                    self.log_message("3D 분석 완료.")

            # 각 패널의 크기를 조정하고 제목을 추가합니다.
            panel2 = cv2.resize(panel2, (W, H))
            panel2 = add_panel_title(panel2, "2. AI 분석 결과", FONT_TITLE)
            panel3 = cv2.resize(panel3, (W, H))
            panel3 = add_panel_title(panel3, "3. 주요 탐지 영역 3D 뷰", FONT_TITLE)
            panel4 = np.full((H, W, 3), 220, dtype=np.uint8)
            for i, text in enumerate(analysis_texts):
                panel4 = draw_korean_text(panel4, text, (20, 20 + i*35), FONT_REPORT, (0, 0, 0))

            # 모든 패널을 하나의 대시보드 이미지로 합칩니다.
            top_row = cv2.hconcat([panel1, panel2])
            bottom_row = cv2.hconcat([panel3, panel4])
            self.dashboard_image = cv2.vconcat([top_row, bottom_row])
            
            # 최종 대시보드를 GUI에 표시하고 진행 상황을 업데이트합니다.
            self.update_progress(90)
            self.display_image(self.dashboard_image)
            self.log_message("분석이 완료되었습니다! 대시보드에서 결과를 확인하세요.")
            self.save_button.config(state="normal")
            
            self.root.after(200, self.update_progress, 100)

        except Exception as e:
            self.log_message(f"GUI 업데이트 중 치명적 오류 발생: {e}")

    # 생성된 대시보드 이미지를 파일로 저장합니다.
    def save_result(self):
        if self.dashboard_image is None:
            self.log_message("오류: 저장할 분석 결과가 없습니다.")
            return
        
        # 저장할 디렉토리를 생성합니다.
        output_dir = "../outputs/final_dashboard_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 파일 이름을 생성하여 이미지를 저장합니다.
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        selected_model_name = self.model_selector.get()
        model_savename = self.models[selected_model_name]['savename']
        output_filename = f"{base_name}_{model_savename}_dashboard.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, self.dashboard_image)
        self.log_message(f"성공: 통합 대시보드가 '{output_path}'에 저장되었습니다.")

    # 로그 메시지를 스레드 안전하게(thread-safe) 로그 텍스트 영역에 추가합니다.
    def log_message(self, msg):
        self.root.after(0, self._log_message_thread_safe, msg)

    # 실제 로그 메시지를 Tkinter GUI에 삽입하는 내부 함수입니다.
    def _log_message_thread_safe(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    # 이미지를 GUI 캔버스에 표시하기 위해 준비합니다.
    def display_image(self, img):
        self.dashboard_image = img
        self.root.after(0, self._display_image_thread_safe)

    # 윈도우 크기 변경 시 이미지 크기를 조정하여 다시 그립니다.
    def on_resize(self, event):
        self.root.after(0, self._display_image_thread_safe)
        
    # 이미지를 캔버스 크기에 맞게 조정하고 표시하는 내부 함수입니다.
    def _display_image_thread_safe(self):
        if self.dashboard_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width > 1 and canvas_height > 1:
            img_pil = Image.fromarray(cv2.cvtColor(self.dashboard_image, cv2.COLOR_BGR2RGB))
            img_pil.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            new_image = ImageTk.PhotoImage(image=img_pil)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width / 2, 
                canvas_height / 2, 
                anchor=tk.CENTER, 
                image=new_image
            )
            self.displayed_image = new_image

# 애플리케이션의 시작점입니다.
if __name__ == "__main__":
    root = tk.Tk()
    app = RoadDamageAnalyzerApp(root)
    root.mainloop()
