import click
import mlflow
import mlflow.keras
import tensorflow as tf

@click.command()
@click.option('--epochs', type=int)
@click.option('--batch-size', type=int)
def main(epochs, batch_size):

    # 何かしらtensorflowで学習
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    accuracy = model.evaluate(x_test, y_test)

    # paramとmetricsの保存
    mlflow.log_metrics({'accuracy': accuracy[1]})
    # modelの保存
    mlflow.keras.log_model(model, 'mnist_digit')

if __name__ == "__main__":
    main()
