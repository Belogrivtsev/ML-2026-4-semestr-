import streamlit as st
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from catboost import CatBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="РГР Машинное обучение", page_icon="🎮", layout="wide")


# Загрузка моделей

@st.cache_resource
def load_models():
    loaded = {}
    base_path = os.path.dirname(__file__) if '__file__' in locals() else "."
    
    model_files = {
        "ML1: Логистическая регрессия": ("ML1_LogisticRegression.pkl", "pickle"),
        "ML2: Gradient Boosting": ("ML2_GradientBoosting.pkl", "pickle"),
        "ML3: CatBoost Classifier": ("ML3_CatBoost.cbm", "catboost"),
        "ML4: Random Forest": ("ML4_RandomForest.pkl", "pickle"),
        "ML5: Stacking Classifier": ("ML5_Stacking.pkl", "pickle"),
        "ML6: Нейронная сеть (MLP)": ("ML6_NeuralNetwork.pkl", "pickle")
    }
    
    for name, (file_name, method) in model_files.items():
        full_path = os.path.join(base_path, file_name)
        if os.path.exists(full_path):
            try:
                if method == "pickle":
                    with open(full_path, "rb") as f:
                        loaded[name] = pickle.load(f)
                elif method == "catboost":
                    model = CatBoostClassifier()
                    model.load_model(full_path)
                    loaded[name] = model
            except Exception:
                loaded[name] = None
        else:
            loaded[name] = None
    return loaded

models = load_models()

@st.cache_data
def load_data():
    base_path = os.path.dirname(__file__) if '__file__' in locals() else "."
    p = os.path.join(base_path, 'cs_clean.csv')
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

df = load_data()

st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Перейти на страницу:",
    ["О разработчике",
     "О датасете",
     "Визуализация зависимостей",
     "Прогнозирование"]
)

# СТРАНИЦА 1: О разработчике (полное оформление)

if page == "О разработчике":
    st.title("Прогноз закладки бомбы в CS:GO")
    col1, col2 = st.columns([1, 3])
    with col1:
        photo_path = os.path.join(os.path.dirname(__file__), 'photo.jpg') if '__file__' in locals() else 'photo.jpg'
        if os.path.exists(photo_path):
            img = Image.open(photo_path)
            st.image(img, caption="Фотография разработчика", width=200, use_column_width=False)
        else:
            st.info("Файл photo.jpg не найден")
    with col2:
        st.markdown("### Информация о разработчике")
        st.info("**ФИО:** Белогривцев Андрей Дмитриевич \n\n **Группа:** МО-241 \n\n **Дисциплина:** Машинное обучение и большие данные")
        st.markdown("---")
        st.markdown("### Тема расчетно-графической работы")
        st.success("**Разработка Web-приложения (дашборда) для инференса моделей машинного обучения и анализа данных**")
        st.markdown("### Цель работы")
        st.write("Создание интерактивного интерфейса для демонстрации работы 6 моделей машинного обучения на реальном игровом датасете CS:GO")

# СТРАНИЦА 2: О датасете 

elif page == "О датасете":
    st.title("Описание набора данных")
    st.markdown("### Предметная область")
    st.write("Датасет содержит статистику раундов игры Counter-Strike: Global Offensive. Задача классификации: предсказать, будет ли заложена бомба (`bomb_planted = 1`) на основе игровых признаков в момент окончания раунда.")
    
    st.markdown("### Признаки (13 шт.)")
    feature_names_display = [
        "time_left", "t_score", "ct_score", "map", "ct_health", "t_health",
        "ct_armor", "t_armor", "ct_money", "t_money", "ct_defuse_kits",
        "ct_players_alive", "t_players_alive"
    ]
    descriptions = [
        "Оставшееся время (сек)", "Победы T", "Победы CT", "Карта", "ХП CT (абс.)",
        "ХП T (абс.)", "Броня CT (абс.)", "Броня T (абс.)", "Деньги CT ($)",
        "Деньги T ($)", "Дифьюзы CT (шт)", "Живые CT (чел)", "Живые T (чел)"
    ]
    desc_df = pd.DataFrame({
        "Признак": feature_names_display,
        "Описание": descriptions
    })
    st.dataframe(desc_df, use_container_width=True)
    
    st.markdown("### Предобработка данных")
    st.markdown("""
    - **Разделение выборки:** 80% обучающая / 20% тестовая (stratify=y для сохранения баланса классов)
    - **Масштабирование:** StandardScaler (приведение признаков к нулевому среднему и единичной дисперсии)
    - **Целевая переменная:** `bomb_planted` (бинарная: 0/1) удалена из матрицы признаков X перед обучением моделей
    - **Баланс классов:** ~72% (бомба не заложена) / ~28% (бомба заложена)
    """)
    
    st.markdown("### Разведочный анализ данных (EDA)")
    with st.expander("1. Заполнение пропущенных данных"):
        st.markdown("""
        Пропущенные значения были заполнены с использованием известных данных:
        - Пропущенные значения `t_score` были заполнены с помощью времени и карты, на которой играют
        - Пропущенные значения `t_health` были заполнены при помощи известных данных о здоровье команды, а также времени
        - Упущенные значения в `ct_defuse_kits` были заполнены по известным данным и времени
        - Неизвестные значения `t_players_alive` были заполнены, исходя из известных данных об игроках
        - Значения `map` были заполнены благодаря известным значениям столбца и времени
        """)
    with st.expander("2. Изменение типов данных"):
        st.markdown("""
        Были выполнены следующие преобразования типов данных:
        - Типы данных столбцов `ct_health`, `t_health`, `ct_armor`, `t_armor`, `ct_money`, `t_money`, `ct_defuse_kits`, `ct_players_alive`, `t_players_alive` были преобразованы из чисел с плавающей запятой в целочисленный тип (int)
        - Тип данных столбца `bomb_planted` был преобразован из bool в int
        - Тип данных столбца `map` был преобразован из str в int (каждому уникальному значению присвоен свой индекс с помощью функции replace)
        """)
    with st.expander("3. Редактирование данных"):
        st.markdown("""
        В процессе предобработки были выполнены следующие действия:
        - Удалены столбцы `ct_helmets` и `t_helmets` из-за их ненадобности (для анализа достаточно знать столбцы armor, так как шлем не оказывает существенного влияния на статистику брони)
        - Проведена проверка на наличие дубликатов в датафрейме, после чего они были удалены
        """)
    with st.expander("4. Проверка наличия выбросов"):
        st.markdown("""
        Для выявления выбросов был применен метод `describe()` к исходному датафрейму, что позволило обнаружить первые признаки аномальных значений. Построенные столбчатые диаграммы интересующих столбцов подтвердили наличие проблемы: выбросы действительно присутствуют в столбцах `t_score`, `ct_score`, `t_health`, `t_players_alive`.
        """)
    with st.expander("5. Удаление выбросов"):
        st.markdown("""
        Так как количество выбросов было невелико, а сами значения являлись невозможными с точки зрения игровой механики, было принято решение об их удалении путем применения условий к соответствующим столбцам.
        """)
    
    if df is not None:
        st.markdown("### Статистические характеристики данных")
        st.dataframe(df.describe().T, use_container_width=True)

# СТРАНИЦА 3: Визуализация зависимостей (доработанная)

elif page == "Визуализация зависимостей":
    st.title("Визуализация зависимостей в данных")
    if df is None:
        st.error("Датасет cs_clean.csv не найден.")
    else:
        # Создаём 4 вкладки
        tab1, tab2, tab3, tab4 = st.tabs(["Распределение", "Boxplot", "Корреляция", "PCA"])
        
        # Распределение целевой переменной
        with tab1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(x='bomb_planted', data=df, ax=ax, palette=['#4C72B0', '#55A868'])
            ax.set_title("Распределение целевого признака", fontsize=12)
            ax.set_xlabel("bomb_planted", fontsize=10)
            ax.set_ylabel("Количество (count)", fontsize=10)
            ax.set_xticklabels(['0 (не заложена)', '1 (заложена)'])
            # Добавляем значения над столбцами
            for p in ax.patches:
                ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=9)
            st.pyplot(fig)
        
        # Boxplot (time_left и ct_health)
        with tab2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # График 1: time_left vs bomb_planted
            sns.boxplot(x='bomb_planted', y='time_left', data=df, ax=axes[0], palette='Set2')
            axes[0].set_title("Время раунда vs установка бомбы", fontsize=10)
            axes[0].set_xlabel("bomb_planted", fontsize=9)
            axes[0].set_ylabel("Оставшееся время (сек)", fontsize=9)
            axes[0].set_xticklabels(['0', '1'])
            # График 2: ct_health vs bomb_planted
            sns.boxplot(x='bomb_planted', y='ct_health', data=df, ax=axes[1], palette='Set2')
            axes[1].set_title("Здоровье CT vs установка бомбы", fontsize=10)
            axes[1].set_xlabel("bomb_planted", fontsize=9)
            axes[1].set_ylabel("Суммарное здоровье CT (HP)", fontsize=9)
            axes[1].set_xticklabels(['0', '1'])
            plt.tight_layout()
            st.pyplot(fig)
        
        # Полная корреляционная матрица
        with tab3:
            # Берём только числовые столбцы
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(14, 10))  # увеличен размер для читаемости
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                        square=True, linewidths=0.5, ax=ax,
                        annot_kws={"size": 8})
            ax.set_title("Матрица корреляций (все признаки)", fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
        
        # PCA проекция
        with tab4:
            st.write("### Проекция данных на 2 главные компоненты (PCA)")
            # Подготовка данных: исключаем целевую переменную и нечисловые
            feature_cols = [c for c in df.columns if c != 'bomb_planted' and df[c].dtype in [np.int64, np.float64]]
            X = df[feature_cols].copy()
            y = df['bomb_planted']
            
            # Масштабирование (StandardScaler)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Создание DataFrame для plotly или matplotlib
            pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            pca_df['target'] = y.values
            
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], 
                                 c=pca_df['target'], cmap='coolwarm', 
                                 alpha=0.6, edgecolors='k', linewidth=0.5)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
            ax.set_title('PCA проекция: bomb_planted = 0 (синий) / 1 (красный)')
            legend = ax.legend(*scatter.legend_elements(), title="bomb_planted")
            ax.add_artist(legend)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption(f"Объяснённая дисперсия: PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%, PC2 = {pca.explained_variance_ratio_[1]*100:.1f}%")

# СТРАНИЦА 4: Прогнозирование (с поддержкой CSV)

elif page == "Прогнозирование":
    st.title("Получение прогноза модели машинного обучения")
    st.markdown("---")
    
    selected_model_name = st.selectbox("Выберите модель для инференса:", list(models.keys()))
    model = models[selected_model_name]
    
    if model is None:
        st.error(f"Файл для модели '{selected_model_name}' не найден.")
    else:
        st.success(f"'{selected_model_name}' готов к работе.")
        
        # Переключатель способа ввода
        input_mode = st.radio("Способ ввода данных:", ["Ручной ввод параметров", "Загрузить CSV-файл"])
        
        # РЕЖИМ 1: Ручной ввод
        
        if input_mode == "Ручной ввод параметров":
            st.markdown("### Конфигуратор параметров игрового раунда")
            col1, col2, col3 = st.columns(3)
            with col1:
                time_left = st.slider("Оставшееся время раунда (сек)", 0, 175, 90)
                map_val = st.selectbox("Карта (Идентификатор ID)", [0, 1, 2, 3, 4, 5, 6, 7])
                ct_score = st.number_input("Текущий счет команды CT", min_value=0, max_value=30, value=5)
                t_score = st.number_input("Текущий счет команды T", min_value=0, max_value=30, value=4)
            with col2:
                ct_health = st.slider("Суммарное здоровье CT (HP)", 0, 500, 400)
                t_health = st.slider("Суммарное здоровье T (HP)", 0, 500, 420)
                ct_armor = st.slider("Суммарные очки брони CT", 0, 500, 300)
                t_armor = st.slider("Суммарные очки брони T", 0, 500, 350)
                ct_players_alive = st.slider("Количество живых игроков CT", 0, 5, 4)
                t_players_alive = st.slider("Количество живых игроков T", 0, 5, 4)
            with col3:
                ct_money = st.number_input("Экономические ресурсы CT ($)", min_value=0, max_value=100000, value=12000)
                t_money = st.number_input("Экономические ресурсы T ($)", min_value=0, max_value=100000, value=15500)
                ct_helmets = st.number_input("Количество шлемов у CT", min_value=0, max_value=5, value=3)
                t_helmets = st.number_input("Количество шлемов у T", min_value=0, max_value=5, value=4)
                ct_defuse_kits = st.number_input("Наборы сапера (Defuse Kits) у CT", min_value=0, max_value=5, value=2)
            
            ct_econom_power = ct_money // 1000 if ct_money > 0 else 0
            t_econom_power = t_money // 1000 if t_money > 0 else 0
            
            input_data = pd.DataFrame([{
                'time_left': time_left, 'ct_score': ct_score, 't_score': t_score, 'map': map_val,
                'ct_health': ct_health, 't_health': t_health, 'ct_armor': ct_armor, 't_armor': t_armor,
                'ct_money': ct_money, 't_money': t_money, 'ct_helmets': ct_helmets, 't_helmets': t_helmets,
                'ct_defuse_kits': ct_defuse_kits, 'ct_players_alive': ct_players_alive, 't_players_alive': t_players_alive,
                'ct_econom_power': ct_econom_power, 't_econom_power': t_econom_power
            }])
            
            if df is not None:
                columns_order = [c for c in df.columns if c != 'bomb_planted']
                input_data = input_data[columns_order]
            
            if st.button("Выполнить прогноз", type="primary"):
                pred = model.predict(input_data)[0]
                st.subheader("Результат классификации:")
                if pred == 1:
                    st.error("Прогноз: **бомба будет заложена**")
                else:
                    st.success("Прогноз: **бомба НЕ будет заложена**")
        
        # РЕЖИМ 2: Загрузка CSV-файла

        else:
            st.markdown("### Загрузка данных из CSV")
            st.info("Файл должен содержать **все признаки** (те же колонки, что и в обучающем датасете, **без** колонки `bomb_planted`).")
            
            uploaded_file = st.file_uploader("Выберите файл в формате .csv", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df_input = pd.read_csv(uploaded_file)
                    st.write("**Первые 5 строк загруженного файла:**")
                    st.dataframe(df_input.head())
                    
                    if df is None:
                        st.error("Не загружен эталонный датасет 'cs_clean.csv'. Невозможно проверить колонки.")
                    else:
                        expected_cols = [c for c in df.columns if c != 'bomb_planted']
                        missing_cols = [c for c in expected_cols if c not in df_input.columns]
                        extra_cols = [c for c in df_input.columns if c not in expected_cols and c != 'bomb_planted']
                        
                        if missing_cols:
                            st.error(f"В загруженном файле отсутствуют необходимые колонки: {missing_cols}")
                        else:
                            if extra_cols:
                                st.warning(f"Найдены лишние колонки (будут проигнорированы): {extra_cols}")
                            
                            # Приводим порядок колонок и делаем предсказание
                            X_pred = df_input[expected_cols].copy()
                            
                            if X_pred.isnull().sum().sum() > 0:
                                st.warning("Обнаружены пропущенные значения. Они будут заполнены нулями (это может повлиять на качество).")
                                X_pred = X_pred.fillna(0)
                            
                            # Получение предсказаний и вероятностей
                            predictions = model.predict(X_pred)
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(X_pred)[:, 1]
                            else:
                                probabilities = np.full(len(predictions), np.nan)
                            
                            # Формирование результата
                            result_df = X_pred.copy()
                            result_df["prediction"] = predictions
                            result_df["probability_%"] = (probabilities * 100).round(1)
                            result_df["interpretation"] = result_df["prediction"].apply(
                                lambda x: "Бомба будет заложена" if x == 1 else "Бомба НЕ будет заложена"
                            )
                            
                            st.success("Прогноз выполнен успешно!")
                            st.write("### Результаты предсказаний")
                            st.dataframe(result_df, use_container_width=True)
                            
                            csv_result = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Скачать результаты (CSV)",
                                data=csv_result,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {str(e)}")