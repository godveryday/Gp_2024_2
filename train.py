# 이게 진짜 resnet34 original



import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.mixed_precision import set_global_policy  # 혼합 정밀도 설정을 위해

# 혼합 정밀도 정책 설정
set_global_policy('mixed_float16')
EPOCHS = 30
BATCH_SIZE = 64

# ResNet-34의 Residual Block 정의
class ResidualBlock(Model):
    def __init__(self, filters, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = ReLU()
        self.conv2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.bn2 = BatchNormalization()

        if stride != 1:
            self.shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)
            self.bn_shortcut = BatchNormalization()
        else:
            self.shortcut = lambda x: x  # Identity shortcut

    def call(self, x):
        shortcut = self.shortcut(x)
        if hasattr(self, 'bn_shortcut'):
            shortcut = self.bn_shortcut(shortcut)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = Add()([x, shortcut])
        x = self.relu(x)
        return x

# ResNet-34 모델 생성 함수
def create_resnet34(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    # Original ResNet-34의 첫 Conv 레이어
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # 3x3 MaxPooling 레이어 추가
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # 이후 Residual Blocks
    x = ResidualBlock(64)(x)
    x = ResidualBlock(64)(x)
    x = ResidualBlock(64)(x)

    x = ResidualBlock(128, stride=2)(x)
    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)
    x = ResidualBlock(128)(x)

    x = ResidualBlock(256, stride=2)(x)
    x = ResidualBlock(256)(x)
    x = ResidualBlock(256)(x)
    x = ResidualBlock(256)(x)
    x = ResidualBlock(256)(x)
    x = ResidualBlock(256)(x)

    x = ResidualBlock(512, stride=2)(x)
    x = ResidualBlock(512)(x)
    x = ResidualBlock(512)(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax', dtype = 'float32')(x)

    model = Model(inputs, outputs, name="ResNet34")
    return model


# CIFAR-10 데이터셋으로 학습하기
def train_resnet34_on_cifar10():
    # 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 데이터 전처리
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 모델 생성
    model = create_resnet34(input_shape=(32, 32, 3), num_classes=10)

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

    # 모델 저장 (batch size와 epoch를 포함한 파일명)
    model_filename = f"resnet34_cifar10_bs{BATCH_SIZE}_epochs{EPOCHS}.keras"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")

# 모델 학습 실행
if __name__ == '__main__':
    train_resnet34_on_cifar10()
