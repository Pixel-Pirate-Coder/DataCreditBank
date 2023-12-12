import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

url = "https://raw.githubusercontent.com/Pixel-Pirate-Coder/DataCreditScore/main/final_bd.csv"
df = pd.read_csv(url, sep=";", index_col=0)
df_num = df.drop(["GENDER", "SOCSTATUS_WORK_FL",
                  "SOCSTATUS_PENS_FL", "AGREEMENT_RK"], axis=1)

df['GENDER'] = df['GENDER'].replace({0: 'Мужчина', 1: 'Женщина'})
df['TARGET'] = df['TARGET'].replace({0: 'Отклик не получен',
                                     1: 'Отклик получен'})
df['SOCSTATUS_WORK_FL'] = df['SOCSTATUS_WORK_FL'].replace({0: 'Не работает',
                                                           1: 'Работает'})
df['SOCSTATUS_PENS_FL'] = df['SOCSTATUS_PENS_FL'].replace({0: 'Не пенсионер',
                                                           1: 'Пенсионер'})

df_no_targ_id = df.drop(["TARGET", "AGREEMENT_RK"], axis=1)
df_no_id = df.drop("AGREEMENT_RK", axis=1)

rus = {'GENDER': 'ПОЛ', 'AGE': 'ВОЗРАСТ', 'CHILD_TOTAL': 'КОЛ-ВО ДЕТЕЙ',
                        'DEPENDANTS': 'КОЛ-ВО ИЖДИВЕНЦЕВ', 'PERSONAL_INCOME': 'ПЕРСОНАЛЬНЫЙ ДОХОД',
                        'LOAN_NUM_TOTAL': 'КОЛ-ВО КРЕДИТОВ', 'LOAN_NUM_CLOSED': 'КОЛ-ВО ЗАКРЫТЫХ КРЕДИТОВ',
                        'SOCSTATUS_WORK_FL': 'СОЦИАЛЬНЫЙ СТАТУС ОТНОСИТЕЛЬНО РАБОТЫ',
                        'SOCSTATUS_PENS_FL': 'СОЦИАЛЬНЫЙ СТАТУС ОТНОСИТЕЛЬНО ПЕНСИИ',
                        'AGREEMENT_RK': 'ID объекта', 'TARGET': 'ОТКЛИК НА МАРКЕТИНГОВУЮ КАМПАНИЮ'}

def on_rus(feature):
    return f'{feature} - {rus[feature]}'

def count_target(target_col):

    st.subheader('Распределение целевой переменной')
    labels = df[target_col].value_counts().index
    fig, ax = plt.subplots()
    ax.pie(df[target_col].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90,
           wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    plt.title(f"Отклик клиента на маркетинговую кампанию банка {target_col}")
    st.pyplot(fig)
    st.write('- Большинство клиентов не заинтересовалось предложением банка, '
             'это может сигнализировать о том, что кампания была не сильно эффективна в привлечении клиентов. '
             'Предполагается изменить маркетинговую стратегию, чтобы повысить отклик.')

# Построение диаграммы распределения
def count_features(df):

    st.subheader('Распределение признаков')
    feature = st.selectbox("Выберите признак:", df.columns, format_func=on_rus, key='1')

    if feature == 'GENDER' or feature == 'SOCSTATUS_WORK_FL' or feature == 'SOCSTATUS_PENS_FL':

        labels = df[feature].value_counts().index
        fig, ax = plt.subplots()
        ax.pie(df[feature].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor': 'black', 'linewidth': 1})
        plt.title(f"Распределение признака {feature}")
        st.pyplot(fig)

    else:

        fig = plt.figure(figsize=(10, 5))
        sns.histplot(df[feature], kde=False, label=feature, color='blue',
                     edgecolor='black', linestyle='-', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"Распределение признака {feature}")
        plt.xlabel("Значение признака")
        plt.ylabel("Частота")
        plt.legend()
        st.pyplot(fig)

    st.write('''
    - В датасете присутствуют два вещественных непрерывных признака: PERSONAL_INCOME и AGE;
    - Остальные признаки - категориальные, из них бинарные признаки: GENDER, SOCSTATUS_WORK_FL, SOCSTATUS_PENS_FL.
            ''')
def mattrix(df):

    st.subheader('Матрица корреляции признаков')
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', vmin=-1, vmax=1, center=0, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    st.write('''
        - Наиболее скоррелированные (положительно) пары признаков: LOAN_NUM_TOTAL - LOAN_NUM_CLOSED 
        и CHILD_TOTAL и DEPENDANTS;
        - Наименее скоррелированные пары признаков: LOAN_NUM_CLOSED - CHILD_TOTAL и LOAN_NUM_CLOSED - AGE;
        - Целевая переменная TARGET слабо коррелируют (слабая связь) с признаками.
                ''')

def info(df):

    st.subheader('Числовые характеристики признаков')
    feature = st.selectbox("Выберите признак:", df.columns, format_func=on_rus, key='2')
    st.write(df[feature].describe())

def diagram_feature(df):

    st.subheader('Попарные распределения признаков')
    feature_1 = st.selectbox("Выберите первый признак:", df.columns, format_func=on_rus, key='3')
    feature_2 = st.selectbox("Выберите второй признак:", df.columns, format_func=on_rus, key='4')

    fig = plt.figure(figsize=(10,5))
    sns.scatterplot(x=df[feature_1], y=df[feature_2], data=df, color='blue')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f"Диаграмма рассеяния для пары {feature_1} - {feature_2}")
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    st.pyplot(fig)

    st.write('''
            - Некоторые признаки имеет отрицательную линейную зависимость, например CHILD_TOTAL - PERSONAL_INCOME 
            и CHILD_TOTAL И LOAN_NUM_TOTAL;
            - Положительную линейную зависимость, например, имеет пара CHILD_TOTAL - DEPENDANTS ;
            - Какие-то не имеют четкой зависимости: AGE - LOAN_NUM_TOTAL.
                    ''')

def diagram_with_target(df):

    st.subheader('Распределение целевой переменной в зависимости от признаков')
    feature = st.selectbox("Выберите признак:", df.columns, format_func=on_rus, key='5')

    if feature == 'PERSONAL_INCOME' or feature == 'AGE':

        fig = plt.figure(figsize=(10, 5))
        sns.histplot(data=df, x=feature, hue='TARGET', bins=30, palette='viridis')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"Распределение целевой переменной TARGET относительно {feature}")
        plt.xlabel(feature)
        plt.ylabel("Частота")
        st.pyplot(fig)

    else:
        fig = plt.figure(figsize=(10, 5))
        sns.countplot(x=feature, hue='TARGET', data=df, palette='viridis')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title(f"Распределение целевой переменной TARGET относительно {feature}")
        plt.xlabel(feature)
        plt.ylabel('Частота')
        st.pyplot(fig)

    st.write('''
            - Можно сказать, что c уменьшением дискретных значений для категориальных признаков 
            наблюдается увеличение вероятности отклика или отсутствия отклика;
            - Чем меньше PERSONAL_INCOME тем выше вероятность реакции или отсутствия реакции;
            - Для AGE высокие показатели по отклику/отсутствию отклика приходятся с 22 лет до 40 лет.
                    ''')

if __name__ == "__main__":

    st.title('Разведочный анализ данных клиентов банка')
    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader('Исследуем признаки и их взаимосвязь с целевой переменной, '
                 'числовые характеристики признаков, корреляцию признаков и т.д.')
    st.write('Исходные данные - база данных с информацией о клиентах банка и их персональных данных, '
             'таких как пол, количество детей и т.д.')
    st.info(''' Таблица с данными состоит из:
    - AGREEMENT_RK — уникальный идентификатор объекта в выборке;
    - TARGET — целевая переменная: отклик на маркетинговую кампанию (1 — отклик был зарегистрирован, 0 — отклика не было);
    - AGE — возраст клиента;
    - SOCSTATUS_WORK_FL — социальный статус клиента относительно работы (1 — работает, 0 — не работает);
    - SOCSTATUS_PENS_FL — социальный статус клиента относительно пенсии (1 — пенсионер, 0 — не пенсионер);
    - GENDER — пол клиента (1 — мужчина, 0 — женщина);
    - CHILD_TOTAL — количество детей клиента;
    - DEPENDANTS — количество иждивенцев клиента;
    - PERSONAL_INCOME — личный доход клиента (в рублях);
    - LOAN_NUM_TOTAL — количество ссуд клиента;
    - LOAN_NUM_CLOSED — количество погашенных ссуд клиента.
                    ''')

    st.markdown("<br><br>", unsafe_allow_html=True)
    count_target("TARGET")

    st.markdown("<br><br>", unsafe_allow_html=True)
    count_features(df_no_targ_id)

    st.markdown("<br><br>", unsafe_allow_html=True)
    info(df_no_id)

    st.markdown("<br><br>", unsafe_allow_html=True)
    diagram_feature(df_no_targ_id)

    st.markdown("<br><br>", unsafe_allow_html=True)
    diagram_with_target(df_no_id)

    st.markdown("<br><br>", unsafe_allow_html=True)
    mattrix(df_num)
