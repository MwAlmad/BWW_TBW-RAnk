import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from datetime import datetime


DATA_FILE = 'Dataset.xlsx'
df = pd.read_excel(DATA_FILE)

# Drop unnecessary columns
drop_columns = [col for col in df.columns if any(keyword in col for keyword in ["Name", "Score"])]
df = df.drop(columns=drop_columns)

def perform_cross_validation(classifier, X, y):
    cross_val_scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
    if np.isnan(cross_val_scores).any():
        raise ValueError("Cross-validation scores contain nan values.")
    return cross_val_scores

def train_and_evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    start_time = datetime.now()
    # Create PDF files for Bar chart
    with PdfPages('bar_chart_figures_RF.pdf') as bar_chart_pdf:
        accuracies = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        labels = y_test.columns

        for i, classifier in enumerate(classifiers):
            classifier.fit(X_train, y_train.iloc[:, i])
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test.iloc[:, i], y_pred)
            precision = precision_score(y_test.iloc[:, i], y_pred, average='macro')
            recall = recall_score(y_test.iloc[:, i], y_pred, average='macro')
            f1 = f1_score(y_test.iloc[:, i], y_pred, average='macro')
            accuracies.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            print(f"Evaluation for label: {labels[i]}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Bar chart
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [accuracy, precision, recall, f1]

            # Calculate positions for bars and shadows
            bar_positions = np.arange(len(metrics))

            fig, ax = plt.subplots(figsize=(4, 2))
            bars = ax.barh(metrics, values, color=['#B85450FF', '#6C8EBFFF', '#82B366FF', '#D6B656FF'], edgecolor='none', height=0.4)
            ax.set_title(f'Matrices for {labels[i]}', fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.3))
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)

            for bar, value in zip(bars, values):
                plt.text(bar.get_width(),  # x-coordinate (the end of the bar)
                         bar.get_y() + bar.get_height(),  # y-coordinate (top of the bar)
                         f"{value:.2f}",  # Text
                         va='bottom',  # Vertical alignment
                         ha='right',  # Horizontal alignment
                         fontdict={'size': 6})  # Font size

            ax.set_yticks(np.arange(len(bar_positions)))
            ax.set_yticklabels(metrics)

            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)

            ax.spines['right'].set_linewidth(0.3)
            ax.spines['bottom'].set_linewidth(0.3)

            ax.tick_params(axis='x', which='both', bottom=True, top=False)

            plt.tight_layout()

            # Save the Bar chart figure to the Bar Chart PDF file
            bar_chart_pdf.savefig(fig)
            plt.close(fig)

        # Create a PDF file for the average metrics
        with PdfPages('average_metrics_chart_RF.pdf') as pdf_file:
            overall_accuracies = []
            overall_precision_scores = []
            overall_recall_scores = []
            overall_f1_scores = []

            labels = y_test.columns

            for i, classifier in enumerate(classifiers):
                classifier.fit(X_train, y_train.iloc[:, i])
                y_pred = classifier.predict(X_test)

                accuracy = accuracy_score(y_test.iloc[:, i], y_pred)
                precision = precision_score(y_test.iloc[:, i], y_pred, average='macro')
                recall = recall_score(y_test.iloc[:, i], y_pred, average='macro')
                f1 = f1_score(y_test.iloc[:, i], y_pred, average='macro')

                overall_accuracies.append(accuracy)
                overall_precision_scores.append(precision)
                overall_recall_scores.append(recall)
                overall_f1_scores.append(f1)

            # Calculate average metrics
            avg_accuracy = np.mean(overall_accuracies)
            avg_precision = np.mean(overall_precision_scores)
            avg_recall = np.mean(overall_recall_scores)
            avg_f1 = np.mean(overall_f1_scores)

            # Bar chart for average metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [avg_accuracy, avg_precision, avg_recall, avg_f1]

            fig, ax = plt.subplots(figsize=(4, 2))
            bars = ax.barh(metrics, values, color=['#B85450FF', '#6C8EBFFF', '#82B366FF', '#D6B656FF'],
                           edgecolor='none', height=0.4)
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.1, 0.3))
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)

            for bar, value in zip(bars, values):
                plt.text(bar.get_width(),  # x-coordinate (the end of the bar)
                         bar.get_y() + bar.get_height(),  # y-coordinate (top of the bar)
                         f"{value:.2f}",  # Text
                         va='bottom',  # Vertical alignment
                         ha='right',  # Horizontal alignment
                         fontdict={'size': 6})  # Font size

            ax.set_yticks(np.arange(len(bar_positions)))
            ax.set_yticklabels(metrics)

            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)

            ax.spines['right'].set_linewidth(0.3)
            ax.spines['bottom'].set_linewidth(0.3)

            ax.tick_params(axis='x', which='both', bottom=True, top=False)

            plt.tight_layout()

            # Save the Bar chart figure to the PDF file
            pdf_file.savefig(fig)
            plt.close(fig)

    with PdfPages('confusion_matrix_figures_RF.pdf') as confusion_matrix_pdf:
        for i, classifier in enumerate(classifiers):
            classifier.fit(X_train, y_train.iloc[:, i])
            y_pred = classifier.predict(X_test)

            # Confusion Matrix
            cm = confusion_matrix(y_test.iloc[:, i], y_pred)
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f'Confusion Matrix for {labels[i]}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()

            # Save the Confusion Matrix figure to the Confusion Matrix PDF file
            confusion_matrix_pdf.savefig(fig)
            plt.close(fig)

    end_time = datetime.now()
    duration = (end_time - start_time).seconds / 60.0
    with open('training_info-RF.txt', 'a') as f:
        f.write(f"Date: {start_time.strftime('%Y-%m-%d')}\n"
                f"Training Started: {start_time.strftime('%H:%M:%S')}\n"
                f"Training Finished: {end_time.strftime('%H:%M:%S')}\n"
                f"Training Duration: {duration:.2f} minutes\n"
                "-------------------------------------------------\n")

    return accuracies


def main():
    # Assuming df is your dataset with input features and labels for classification
    features = ['Support TOTP', 'Support Facial Recognition', 'Multiple Cryptocurrencies', 'Wallet Age', 'Non-Custodial',
                'Custodial', 'Rating', 'Security Level']

    labels = ['Ranking', 'Ranking by Support TOTP',
              'Ranking by Support Facial Recognition', 'Ranking by Multiple Cryptocurrencies',
              'Ranking by Wallet Age', 'Ranking by Non-Custodial', 'Ranking by Custodial',
              'Ranking by Rating', 'Ranking by Security Level']  # Replace with your actual label columns
    X = df[features]
    y = df[labels]


    # Create RandomForestClassifier for each label with model initialization
    classifiers = [
        RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            max_depth=20,  # limit on the maximum depth of each tree
            min_samples_split=2,  # Minimum number of samples required to split an internal node
            min_samples_leaf=1,  # Minimum number of samples required to form a leaf node
            random_state=42
        ) for _ in labels
    ]

    # Save the trained model
    joblib.dump(classifiers, 'random_forest_model1.pkl')

    # Perform cross-validation
    for i, label in enumerate(labels):
        cross_val_scores = perform_cross_validation(classifiers[i], X, y[label])
        print(f"Cross-Validation Scores for {label}:", cross_val_scores)
        print(f"Mean Accuracy for {label}: {cross_val_scores.mean():.4f}")
        print(f"Standard Deviation for {label}: {cross_val_scores.std():.4f}")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the classifiers on the test set
    accuracies = train_and_evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test)

    for label, accuracy in zip(labels, accuracies):
        print(f"Accuracy for {label}: {accuracy:.4f}")

if __name__ == "__main__":
    main()