# --- Import Necessary Python Modules ----
import numpy as np
import pandas as pd
from keras.layers import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from termcolor import colored
import tensorflow as tf
import shap





def Metric_Evaluation(y_true, pred):
    cm = confusion_matrix(y_true, pred)
    TP = cm[0, 0]  # True Positive
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TN = cm[1, 1]  # True Negative
    Acc = (TP + TN) / (TP + TN + FP + FN)
    Pre = TP / (TP + FP)
    Rec = TP / (TP + FN)
    F1score = 2 * (Pre * Rec) / (Pre + Rec)
    roc_score = roc_auc_score(y_true, pred)
    return [Acc, Pre, Rec, F1score]



class METHODS_:
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.ytrain_ = to_categorical(ytrain)
        self.num_classes = self.ytrain_.shape[1]

    def Logistic_Regression(self):
        print(colored("Logistic Regression", color='blue', on_color='on_grey'))
        clf = LogisticRegression(max_iter=1000)
        clf.fit(self.xtrain, self.ytrain)
        pred = clf.predict(self.xtest)
        metrics = Metric_Evaluation(self.ytest, pred)
        return metrics

    def Decision_Tree(self):
        print(colored("Decision Tree", color='blue', on_color='on_grey'))
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(self.xtrain, self.ytrain)
        pred = clf.predict(self.xtest)
        metrics = Metric_Evaluation(self.ytest, pred)
        return metrics

    def Random_Forest(self):
        print(colored("Random Forest", color='blue', on_color='on_grey'))
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(self.xtrain, self.ytrain)
        pred = clf.predict(self.xtest)
        metrics = Metric_Evaluation(self.ytest, pred)
        return metrics


    def MLP_(self, explainability = True):
        print(colored("MLP Classifier", color='blue', on_color='on_grey'))
        # Create an MLPClassifier model
        mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                            max_iter=1000, random_state=42)
        # Train the model on the training data
        mlp.fit(self.xtrain, self.ytrain)
        # Make predictions on the test data
        y_pred = mlp.predict(self.xtest)

        if explainability:
            background = shap.utils.sample(self.xtrain, 100, random_state=42)
            explainer = shap.KernelExplainer(mlp.predict_proba, background)
            test_data = self.xtest[:50]
            test_array = test_data.to_numpy() if hasattr(test_data, 'to_numpy') else test_data
            feature_names = self.xtest.columns.tolist() if hasattr(self.xtest, "columns") else [f"feature_{i}" for i in
                                                                                                range(test_array.shape[
                                                                                                          1])]
            shap_values = explainer.shap_values(test_data)
            explanations = []
            num_classes = shap_values.shape[2]

            for class_idx in range(num_classes):
                explanation = shap.Explanation(
                    values=shap_values[:, :, class_idx],
                    base_values=explainer.expected_value[class_idx],
                    data=test_array,
                    feature_names=feature_names
                )
                explanations.append(explanation)

            #  Beeswarm plots for each class
            for i, explanation in enumerate(explanations):
                shap.plots.beeswarm(explanation, show=False)
                plt.title(f"Beeswarm Plot - Class {i}")
                plt.savefig(f"shap_beeswarm_class_{i}.png", bbox_inches='tight')
                plt.close()

            #  Waterfall plot for first instance (per class)
            for i, explanation in enumerate(explanations):
                shap.plots.waterfall(explanation[0], show=False)
                plt.title(f"Waterfall Plot - Class {i}, Sample 0")
                plt.savefig(f"shap_waterfall_class_{i}.png", bbox_inches='tight')
                plt.close()

        return Metric_Evaluation(self.ytest, y_pred)

    def Neural_Network(self):
        print(colored("Neural Network", color='blue', on_color='on_grey'))

        input_layer = Input(shape=(self.xtrain.shape[1],))
        x = Dense(32, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(self.xtrain, self.ytrain_, epochs=25, batch_size=256)

        pred = model.predict(self.xtest)
        pred_classes = tf.argmax(pred, axis=1).numpy()
        metrics = Metric_Evaluation(self.ytest, pred_classes)
        return metrics

    def LSTM_model(self):
        print(colored("LSTM model...", color='blue', on_color='on_grey'))

        xtrain = self.xtrain
        xtest = self.xtest

        xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1)
        xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1)

        input_layer = Input(shape=(xtrain.shape[1], 1))
        x = LSTM(16, return_sequences=False)(input_layer)
        x = Dense(16, activation='relu')(x)
        x = Dense(8, activation='relu')(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(xtrain, self.ytrain_, epochs=25, batch_size=256)

        pred = model.predict(xtest)
        pred_classes = tf.argmax(pred, axis=1).numpy()
        metrics = Metric_Evaluation(self.ytest, pred_classes)
        return metrics


def plot_metrics(val, M):
    # val should be a 1D array-like input of metric values
    metrics = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'MLP', 'Neural Network', 'LSTM']

    filename_png = f'Visualizations/comp_{M}.png'
    x = np.arange(len(metrics))  # x-axis positions
    bar_width = 0.6
    colors = ['#e6ac00', '#666699', '#999966', '#00ffcc', '#0099ff', '#ac3973']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, val, width=bar_width, color=colors, edgecolor='black', linewidth=3,  alpha=1)


    for bar in bars:
        yval = bar.get_height()

        min_value = max(val)
        color = 'green' if yval == min_value else 'red'
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.4f}',
                 ha='center', va='bottom', fontsize=15, fontweight='bold', color=color)

    plt.xlabel('Models', fontsize=14, fontweight='bold')
    plt.ylabel(M, fontsize=14, fontweight='bold')
    plt.xticks(x, metrics, rotation=30, ha='right', fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11, fontweight='bold')
    plt.ylim(0, max(val) + 0.06)
    plt.tight_layout()
    plt.savefig(filename_png, dpi=800)
    plt.show()

def Plot_graphs():
    acc_ = np.load('ACC.npy')
    pre_ = np.load('PRE.npy')
    rec_ = np.load('REC.npy')
    f1score_ = np.load('F1score.npy')



    # --- saved to csv ---
    metrics = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'MLP', 'Neural Network', 'LSTM']
    # Create DataFrame
    df = pd.DataFrame({
        'Model': metrics,
        'MAE': acc_,
        'MSE': pre_,
        'RMSE': rec_,
        'R2': f1score_
    })

    # Save to CSV
    df.to_csv('model_performance_metrics.csv', index=False)

    plot_metrics(acc_, 'Accuracy')
    plot_metrics(pre_, 'Precision')
    plot_metrics(rec_, 'Recall')
    plot_metrics(f1score_, 'F1score')
