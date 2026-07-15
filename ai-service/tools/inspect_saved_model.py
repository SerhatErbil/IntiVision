import tensorflow as tf

from config import MODEL_PATH


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    model.summary()

    print()
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Model path: {MODEL_PATH}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")


if __name__ == "__main__":
    main()