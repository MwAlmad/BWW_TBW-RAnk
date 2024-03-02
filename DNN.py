from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Model
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, classification_report
from keras.regularizers import l2
from contextlib import redirect_stdout


# Set random seed for reproducibility
np.random.seed(123)
#tf.random.set_seed(123)

# Clear TensorFlow session
#tf.keras.backend.clear_session()

# Constants
DATA_FILE = 'Dataset.xlsx'
TEST_SIZE = 0.2
VAL_SIZE = 0.5
EPOCHS = 100
BATCH_SIZE = 50


def preprocess_data():
    df = pd.read_excel(DATA_FILE)

    # Drop unnecessary columns
    drop_columns = [col for col in df.columns if any(keyword in col for keyword in ["Name", "Score"])]
    df = df.drop(columns=drop_columns)

    # Separate features and scale them
    X = df.drop(columns=[col for col in df.columns if "Ranking" in col])
    X_scaled = StandardScaler().fit_transform(X)

    # One-hot encode the target columns
    target_columns = [col for col in df.columns if "Ranking" in col]
    encoders = {col: OneHotEncoder() for col in target_columns}
    y_encoded = {col: encoders[col].fit_transform(df[col].values.reshape(-1, 1)).toarray() for col in target_columns}

    return X_scaled, df, target_columns, y_encoded, encoders


def partition_data(X_scaled, df, target_columns, y_encoded):
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, df[target_columns], test_size=TEST_SIZE,
                                                        random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=42)

    encode = lambda y, col: y_encoded[col][y.index]
    y_train_encoded, y_val_encoded, y_test_encoded = {col: encode(y_train, col) for col in target_columns}, \
                                                     {col: encode(y_val, col) for col in target_columns}, \
                                                     {col: encode(y_test, col) for col in target_columns}
    return X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded


def build_model(input_shape, target_columns, y_train_encoded, units=72, dropout_rate=0.3, activation='relu'):
    """
    Build a multi-output neural network model with L2 regularization using the specified lambda for weight decay.

    Args:
    - input_shape (tuple): The shape of the input features.
    - target_columns (list): A list of names for each target output column.
    - y_train_encoded (dict): Encoded training data for each target column.
    - units (int): Number of units in the dense layer.
    - dropout_rate (float): Dropout rate for regularization.
    - activation (str): Activation function for the dense layers.
    - l2_lambda (float): Lambda value for L2 regularization.

    Returns:
    - A compiled TensorFlow Keras model with multiple outputs.
    """
    input_layer = Input(shape=(input_shape,))
    x = Dense(units, activation=activation)(input_layer)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # Creating a dense layer for each target column with L2 regularization
    outputs = [Dense(y_train_encoded[col].shape[1], activation='softmax', name=f'y{idx + 1}')(x) for idx, col in enumerate(target_columns)]

    model = Model(inputs=input_layer, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train_encoded, X_val, y_val_encoded):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.001, verbose=1),
        TensorBoard(log_dir=f"logs/fit/{datetime.now().strftime('%Y%m%d-%H%M%S')}", histogram_freq=1)
    ]
    # Logging training information
    start_time = datetime.now()
    history = model.fit(
        X_train, {f'y{idx}': y_train_encoded[col] for idx, col in enumerate(target_columns, 1)},
        validation_data=(X_val, {f'y{idx}': y_val_encoded[col] for idx, col in enumerate(target_columns, 1)}),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks
    )

    end_time = datetime.now()
    duration = (end_time - start_time).seconds / 60.0
    with open('training_info-5.txt', 'a') as f:
        f.write(f"Date: {start_time.strftime('%Y-%m-%d')}\n"
                f"Training Started: {start_time.strftime('%H:%M:%S')}\n"
                f"Training Finished: {end_time.strftime('%H:%M:%S')}\n"
                f"Training Duration: {duration:.2f} minutes\n"
                "-------------------------------------------------\n")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model.save(f'trained_model_{timestamp}.h5')

    return history


def evaluate_model(model, X_test, y_test_encoded, target_columns):
    test_metrics = model.evaluate(X_test, {f'y{idx}': y_test_encoded[col] for idx, col in enumerate(target_columns, 1)})

    metrics_dict = {}
    predictions = model.predict(X_test)

    for idx, col in enumerate(target_columns):
        y_true = np.argmax(y_test_encoded[col], axis=1)
        y_pred = np.argmax(predictions[idx], axis=1)

        print(f"Classification Report for {col}:")
        report = classification_report(y_true, y_pred, output_dict=True)
        print(report)

        metrics_dict[col] = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1-score': f1_score(y_true, y_pred, average='macro')
        }

    return test_metrics, metrics_dict


def cross_validate_model(X, y_encoded, target_columns, create_model_fn, n_splits=5, batch_size=32, epochs=100, validation_split=0.2):
    """
    Perform K-Fold cross-validation for each target column separately and calculate the cross-validation scores,
    mean accuracy, and standard deviation for each target.

    Args:
    - X (np.array): Input features.
    - y_encoded (dict): Dictionary of encoded targets.
    - target_columns (list): List of target column names.
    - create_model_fn (function): Function to create the model.
    - n_splits (int): Number of splits for K-Fold cross-validation.
    - batch_size (int): Batch size for training.
    - epochs (int): Number of epochs for training.
    - validation_split (float): Fraction of the training data to be used as validation data.

    Returns:
    - dict: Aggregated cross-validation metrics for each target column.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = {}

    for col in target_columns:
        print(f"Processing Target Column: {col}")
        accuracy_scores = []  # List to store accuracy for each fold

        for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
            print(f"  Processing Fold {fold_idx + 1}/{n_splits} for {col}")

            X_train, X_test = X[train_index], X[test_index]
            y_train_fold = y_encoded[col][train_index]
            y_test_fold = y_encoded[col][test_index]

            model = create_model_fn(input_shape=X_train.shape[1], target_columns=[col], y_train_encoded={col: y_train_fold})

            # Train the model
            model.fit(X_train, y_train_fold, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)

            # Evaluate the model performance on the test fold
            scores = model.evaluate(X_test, y_test_fold, verbose=0)
            accuracy_scores.append(scores[1])  # Assuming scores[1] is accuracy

        # Calculate average and standard deviation of accuracies across all folds for the current target
        avg_accuracy = np.mean(accuracy_scores)
        std_dev_accuracy = np.std(accuracy_scores)

        print(f"  Average Accuracy for {col}: {avg_accuracy}")
        print(f"  Standard Deviation of Accuracy for {col}: {std_dev_accuracy}")

        # Store results for the current target column
        all_results[col] = {
            'cross_validation_scores': accuracy_scores,
            'mean_accuracy': avg_accuracy,
            'std_dev_accuracy': std_dev_accuracy
        }

    return all_results

#plotting avreage precision, recall, F1-score
def plot_average_metrics(metrics_dict):
    avg_accuracy = np.mean([metrics['accuracy'] for metrics in metrics_dict.values()])
    avg_precision = np.mean([metrics['precision'] for metrics in metrics_dict.values()])
    avg_recall = np.mean([metrics['recall'] for metrics in metrics_dict.values()])
    avg_f1 = np.mean([metrics['f1-score'] for metrics in metrics_dict.values()])

    labels = ['accuracy','Precision', 'Recall', 'F1-score']
    scores = [avg_accuracy,avg_precision, avg_recall, avg_f1]

    # Calculate positions for bars and shadows
    bar_positions = np.arange(len(labels))

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 2))

    # Create the actual bars
    bars = ax.barh(labels, scores, color=['#B85450FF', '#6C8EBFFF', '#82B366FF', '#D6B656FF'], edgecolor='none', height=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.3))

    # Set font size for tick labels
    ax.tick_params(axis='x', labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

    # Annotate each bar
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width(),  # x-coordinate (the end of the bar)
                 bar.get_y() + bar.get_height(),  # y-coordinate (top of the bar)
                 f"{score:.2f}",  # Text
                 va='bottom',  # Vertical alignment
                 ha='right',  # Horizontal alignment
                 fontdict={'size': 6})  # Font size

    # Label the y-axis with the categories
    ax.set_yticks(bar_positions)
    ax.set_yticklabels(labels)

    # Hide spines (axis lines) on the top and left
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)

    # Set linewidth for visible spines
    ax.spines['right'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)

    # Remove x-ticks and y-ticks
    ax.tick_params(axis='x', which='both', bottom=True, top=False)  # keep bottom x-tick

    # Layout adjustment
    plt.tight_layout()

    # Show the plot
    fig = plt.gcf()
    fig.show()

    with PdfPages('overall_average_precision_recall_F1-score_9.pdf') as pdf:
        pdf.savefig(fig)

def plot_metrics(metrics_dict, target_columns):
    with PdfPages('precision_recall_F1-score_9.pdf') as bar_chart_pdf:
        for idx, col in enumerate(target_columns):

            labels = ['Accuracy','Precision', 'Recall', 'F1-score'] #metrics
            scores = [metrics_dict[col][label.lower()] for label in labels] #values
            bar_positions = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(4, 2))

            bars = ax.barh(labels, scores, label=f'{col}', color=['#B85450FF', '#6C8EBFFF', '#82B366FF', '#D6B656FF' ] , edgecolor='none', height=0.4)

            ax.set_title(f'Matrices for {col}', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.3))

            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)

            for bar, value in zip(bars, scores):
                plt.text(bar.get_width(),  # x-coordinate (the end of the bar)
                         bar.get_y() + bar.get_height(),  # y-coordinate (top of the bar)
                         f"{value:.2f}",  # Text
                         va='bottom',  # Vertical alignment
                         ha='right',  # Horizontal alignment
                         fontdict={'size': 6})  # Font size

            ax.set_yticks(np.arange(len(bar_positions)))
            ax.set_yticklabels(labels)

            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)

            ax.spines['right'].set_linewidth(0.3)
            ax.spines['bottom'].set_linewidth(0.3)

            ax.tick_params(axis='x', which='both', bottom=True, top=False)

            plt.tight_layout()
            # Show the plot
            fig = plt.gcf()
            fig.show()
            # Save the Bar chart figure to the Bar Chart PDF file
            bar_chart_pdf.savefig(fig)


# Plotting average loss and accuracy
def plot_results_overall_average(history, target_columns):
    # Importing seaborn for improved aesthetics
    import seaborn as sns
    # Apply the default Seaborn theme
    sns.set_theme()
    plt.figure(figsize=(14, 5))

    # Initialize lists to collect all metrics
    all_train_loss = []
    all_val_loss = []
    all_train_accuracy = []
    all_val_accuracy = []

    for idx, col in enumerate(target_columns, 1):
        # Accessing the 'history' attribute of the History object
        train_loss = np.array(history.history[f'y{idx}_loss'])
        val_loss = np.array(history.history[f'val_y{idx}_loss'])
        train_accuracy = np.array(history.history[f'y{idx}_accuracy'])
        val_accuracy = np.array(history.history[f'val_y{idx}_accuracy'])

        # Collecting metrics for all outputs
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        all_train_accuracy.append(train_accuracy)
        all_val_accuracy.append(val_accuracy)

    # Calculate the average metrics (element-wise division)
    avg_train_loss = np.mean(all_train_loss, axis=0)
    avg_val_loss = np.mean(all_val_loss, axis=0)
    avg_train_accuracy = np.mean(all_train_accuracy, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracy, axis=0)

    epochs = range(1, len(avg_train_loss) + 1)

    # Plotting average loss with improved aesthetics
    plt.subplot(1, 2, 1)
    plt.plot(epochs, avg_train_loss, 'b', label='Training Loss', linewidth=2, marker='o')
    plt.plot(epochs, avg_val_loss, 'r', label='Validation Loss', linewidth=2, marker='s')
    plt.title('Training and Validation Loss', fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=8)

    # Plotting average accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, avg_train_accuracy, 'b', label='Training Accuracy', linewidth=2, marker='o')
    plt.plot(epochs, avg_val_accuracy, 'r', label='Validation Accuracy', linewidth=2, marker='s')
    plt.title('Training and Validation Accuracy', fontsize=12)
    plt.xlabel('Epochs', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig = plt.gcf()
    fig.show()
    with PdfPages('overall_average_training_validation_9.pdf') as pdf:
        pdf.savefig(fig)

def plot_results(history, target_columns):
    num_outputs = len(target_columns)
    plt.figure(figsize=(14, 5 * num_outputs))

    for idx, col in enumerate(target_columns, 1):
        train_loss = history.history[f'y{idx}_loss']
        val_loss = history.history[f'val_y{idx}_loss']
        train_accuracy = history.history[f'y{idx}_accuracy']
        val_accuracy = history.history[f'val_y{idx}_accuracy']

        epochs = range(1, len(train_loss) + 1)

        # Plotting loss for current output
        plt.subplot(num_outputs, 2, 2 * idx - 1)
        plt.plot(epochs, train_loss, 'b', label='Training Loss')
        plt.plot(epochs, val_loss, 'r', label='Validation Loss')
        plt.title(f'Training and Validation Loss for y{idx} ({col})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plotting accuracy for current output
        plt.subplot(num_outputs, 2, 2 * idx)
        plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
        plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
        #plt.axhline(y=test_accuracies[int(idx - 1)], color='g', linestyle='--',label='Test Accuracy')  # Adjusted the indexing here
        plt.title(f'Training and Validation Accuracy for y{idx} ({col})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.tight_layout()
    #plt.savefig('training_validation_plot.pdf')
    fig = plt.gcf()
    fig.show()
    with PdfPages('training_validation_plot-9.pdf') as pdf:
        pdf.savefig(fig)

X_scaled, df, target_columns, y_encoded, encoders  =preprocess_data()
X_train, X_val, X_test, y_train_encoded, y_val_encoded, y_test_encoded=partition_data(X_scaled, df, target_columns, y_encoded)

cross_validate_model(X_train, y_train_encoded, target_columns, create_model_fn=build_model, n_splits=5)

model = build_model(X_train.shape[1], target_columns, y_train_encoded)
# Specify the filename where you want to save the model summary
filename = 'model_summary.txt'

# Redirect the output of model.summary() to the file
with open(filename, 'w') as f:
    with redirect_stdout(f):
        model.summary()

history = train_model(model, X_train, y_train_encoded, X_val, y_val_encoded)

test_metrics, metrics_dict = evaluate_model(model, X_test, y_test_encoded, target_columns)

plot_average_metrics(metrics_dict)

plot_metrics(metrics_dict, target_columns)

plot_results_overall_average(history, target_columns)

plot_results(history, target_columns)
