# ---- Import Necessary python Modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Models_ import *


def Preprocessing():
    df = pd.read_csv("Morticd10_part1.csv")

    # === Define ICD-10 tobacco-related cause codes ===
    tobacco_icd_codes = [
        'C00', 'C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10', 'C11', 'C12', 'C13', 'C14',
        'C15', 'C16', 'C33', 'C34', 'C53', 'C64', 'C65', 'C66', 'C67', 'C68',
        'I20', 'I21', 'I22', 'I23', 'I24', 'I25',
        'J40', 'J41', 'J42', 'J43', 'J44', 'J45', 'J46', 'J47'
    ]

    df['Tobacco_Related'] = df['Cause'].isin(tobacco_icd_codes).astype(int)
    death_cols = [col for col in df.columns if col.startswith("Deaths")]
    df['Total_Deaths'] = df[death_cols].sum(axis=1)
    df_cleaned = df[['Country', 'Year', 'Sex', 'Frmat', 'Total_Deaths', 'Tobacco_Related']].copy()
    df_cleaned = df_cleaned[df_cleaned['Sex'] != 9]

    df_cleaned['Country_Code'] = le.fit_transform(df_cleaned['Country'])
    df_cleaned['Sex_Code'] = le.fit_transform(df_cleaned['Sex'])
    df_cleaned['Deaths_Per_Year_Country'] = df_cleaned.groupby(['Country', 'Year'])['Total_Deaths'].transform('mean')
    df_cleaned['Normalized_Deaths'] = df_cleaned['Total_Deaths'] / (df_cleaned['Deaths_Per_Year_Country'] + 1e-5)

    # Normalize year relative to the dataset (e.g., from 0 to 1)
    df_cleaned['Year_Norm'] = (df_cleaned['Year'] - df_cleaned['Year'].min()) / (
                df_cleaned['Year'].max() - df_cleaned['Year'].min())

    df_cleaned = pd.get_dummies(df_cleaned, columns=['Frmat', 'Sex'], prefix=['Frmat', 'Sex'])

    country_stats = df_cleaned.groupby('Country_Code')['Total_Deaths'].agg(['mean', 'std']).reset_index()
    country_stats.columns = ['Country_Code', 'Country_Mean_Deaths', 'Country_Std_Deaths']
    df_cleaned = df_cleaned.merge(country_stats, on='Country_Code', how='left')

    df_cleaned['Deaths_by_Sex'] = df_cleaned['Total_Deaths'] * df_cleaned['Sex_Code']
    df_cleaned['Deaths_by_Year'] = df_cleaned['Total_Deaths'] * df_cleaned['Year_Norm']

    feature_cols = [col for col in df_cleaned.columns if
                    col not in ['Country', 'Tobacco_Related', 'Deaths_Per_Year_Country']]
    features = df_cleaned[feature_cols]
    labels = df_cleaned['Tobacco_Related']

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
    np.save('features.npy', features)
    np.save('labels.npy', labels)


def cross_validate_method(method_class, method_name, X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    all_metrics = []
    for train_index, test_index in kf.split(X):
        xtrain, xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]

        model_instance = method_class(xtrain, xtest, ytrain, ytest)
        method = getattr(model_instance, method_name)
        metrics = method()
        all_metrics.append(metrics)

    metric_ = np.mean(all_metrics, axis=0).tolist()
    return metric_



def Analysis():
    feat = np.load('features.npy', allow_pickle=True)
    lab = np.load('labels.npy')
    feat = feat.astype('float32')
    scaler = StandardScaler()
    feat = scaler.fit_transform(feat)
    oversample = SMOTE()
    feat, lab = oversample.fit_resample(feat, lab)

    C1 = cross_validate_method(METHODS_, 'Logistic_Regression', feat, lab, folds=5)
    C2 = cross_validate_method(METHODS_, 'Decision_Tree', feat, lab, folds=5)
    C3 = cross_validate_method(METHODS_, 'Random_Forest', feat, lab, folds=5)
    C4 = cross_validate_method(METHODS_, 'MLP_', feat, lab, folds=5)
    C5 = cross_validate_method(METHODS_, 'Neural_Network', feat, lab, folds=5)
    C6 = cross_validate_method(METHODS_, 'LSTM_model', feat, lab, folds=5)

    comp = [C1, C2, C3, C4, C5, C6]
    perf_names = ["ACC", "PRE", "REC", "F1score"]
    # file name creation
    file_names = [f'{name}.npy' for name in perf_names]
    for j in range(0, len(perf_names)):
        new = []
        for i in range(len(comp)):
            new.append(comp[i][j])
        np.save(file_names[j], np.array(new))

if __name__ == "__main__":
    Preprocessing()
    Analysis()
    Plot_graphs()
