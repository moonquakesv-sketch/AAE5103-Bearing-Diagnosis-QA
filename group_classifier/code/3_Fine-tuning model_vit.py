# finetune_vit_head_only.py
import os
from pathlib import Path
import numpy as np
import cv2
import tensorflow_addons as tfa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from vit_keras import vit

# =========== 配置 =============
# 更换为自己的训练集和测试集文件路径
DATA_ROOT = Path(__file__).resolve().parents[1] / "Data"
TRAIN_DIR = str(DATA_ROOT / "train_1s")
TEST_DIR = str(DATA_ROOT / "test_1s")
CLASSES   = ["B", "IR", "N", "OR"]

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS_HEAD   = 50         # 只训练分类头，可以适当增加 epoch 数
LR_HEAD       = 1e-2
WEIGHT_DECAY  = 0.01
SEED          = 42
CKPT_PATH     = "best_vit_cwru.weights.h5"
NORMALIZE_INPUT = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ============================== 数据集 ==============================
def _load_npy(path):
    path = path.decode("utf-8") if isinstance(path, bytes) else path
    arr = np.load(path).astype(np.float32)
    if NORMALIZE_INPUT:
        arr = arr * 2.0 - 1.0
    return np.transpose(arr, (1, 2, 0))

def _parse(path, label):
    x = tf.numpy_function(_load_npy, [path], tf.float32)
    x.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    y = tf.cast(label, tf.int32)
    return x, y

def make_dataset(root, classes, training=False):
    paths, labels = [], []
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        for f in os.listdir(cls_dir):
            if f.endswith(".npy"):
                paths.append(os.path.join(cls_dir, f))
                labels.append(i)
    print(f"[{root}] {len(paths)} samples")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED)
    ds = ds.map(_parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, np.array(labels)

train_ds, train_labels = make_dataset(TRAIN_DIR, CLASSES, training=True)
test_ds,  test_labels  = make_dataset(TEST_DIR,  CLASSES, training=False)

# ================== 构建 ViT 模型（仅训练分类head，vit骨干用来提取特征） ===================
backbone = vit.vit_b16(
    image_size=IMG_SIZE[0],
    activation='linear',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
)
backbone.trainable = False   # 冻结整个 backbone

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="pixel_values")
x = backbone(inputs)
# x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="head")(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

# ============================== 回调 ==============================
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        CKPT_PATH, monitor="val_accuracy", mode="max",
        save_best_only=True, save_weights_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max",
        patience=10, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5,
        min_lr=1e-7, verbose=1),
]

# ==================== 训练分类头 ====================
model.compile(
    optimizer=tfa.optimizers.AdamW(
        learning_rate=LR_HEAD, weight_decay=WEIGHT_DECAY),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print("\n===== 只训练分类头 =====")
model.fit(train_ds, validation_data=test_ds,
          epochs=EPOCHS_HEAD, callbacks=callbacks)

# ================ 最终评估 ===================
print("\n===== 加载最佳权重并评估 =====")
model.load_weights(CKPT_PATH)
probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1)

acc = (preds == test_labels).mean()
print(f"\n最终测试准确率：{acc:.4f}\n")
print("分类报告：")
print(classification_report(test_labels, preds,
                            target_names=CLASSES, digits=4))
print("混淆矩阵：")
print(confusion_matrix(test_labels, preds))