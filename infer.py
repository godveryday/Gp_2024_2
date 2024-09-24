import os
import tensorflow as tf
import time
import numpy as np

# GPU 설정 및 스레드 설정
USE_GPU = True  # GPU 사용 여부 설정
THREADS = None  # 스레드 수 설정 (None이면 자동으로 설정됨)

MODEL_PATH = "resnet34_cifar10_bs64_epochs10.keras"  # 저장된 모델 경로

def configure_gpu_settings():
    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # GPU 사용 비활성화
    if THREADS:
        tf.config.threading.set_inter_op_parallelism_threads(THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(THREADS)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus and USE_GPU:
        try:
            # 모든 GPU를 최대한 활용하는 설정
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # 필요한 만큼만 메모리 할당
            print(f"{len(gpus)} Physical GPUs are available and configured.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found or GPU usage disabled. Running on CPU.")

def load_model_and_predict():
    # GPU 및 스레드 설정
    configure_gpu_settings()

    # Custom layer인 ResidualBlock을 import
    from train import ResidualBlock  # 'train.py' 파일에서 ResidualBlock 가져오기
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'ResidualBlock': ResidualBlock})
    model.summary()

    # CIFAR-10 데이터셋 불러오기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 모델 로딩 후 초기 추론(예열)
    if USE_GPU:
        model.predict(x_test[:1])  # CUDA 라이브러리를 로드하기 위해 작은 예측 수행

    # 추론 시간 측정
    infer_time = []
    for i in range(10):
        print('Predicting start!')
        start = time.time()
        result = model.predict(x_test)
        end = time.time()
        elapsed_time = end - start
        print(f'Prediction took {elapsed_time} seconds.')
        infer_time.append(elapsed_time)

    # 추론 시간 통계 출력
    infer_time = np.array(infer_time)
    avg_time = np.mean(infer_time)
    max_time = np.max(infer_time)
    min_time = np.min(infer_time)
    
    # 파일 이름 생성 (모델 이름, GPU 사용 여부, 스레드 수 포함)
    gpu_status = "GPU" if USE_GPU else "CPU"
    thread_info = f"{THREADS}threads" if THREADS else "default_threads"
    results_filename = f"inference_results_{os.path.basename(MODEL_PATH).split('.')[0]}_{gpu_status}_{thread_info}.txt"

    # 예측 결과 저장
    with open(results_filename, "w") as f:
        f.write(f"Model used: {MODEL_PATH}\n")
        f.write(f"GPU used: {'Yes' if USE_GPU else 'No'}\n")
        if not USE_GPU:
            f.write(f"CPU Threads used: {THREADS if THREADS else 'Default (automatic)'}\n")
        f.write(f"Average inference time: {avg_time} seconds\n")
        f.write(f"Max inference time: {max_time} seconds\n")
        f.write(f"Min inference time: {min_time} seconds\n")
    
    print(f"Inference results saved to {results_filename}")

if __name__ == '__main__':
    load_model_and_predict()
