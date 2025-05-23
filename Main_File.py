# ---- Import Necessary python Modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


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
    np.save('features_enhanced.npy', features)
    np.save('labels.npy', labels)


