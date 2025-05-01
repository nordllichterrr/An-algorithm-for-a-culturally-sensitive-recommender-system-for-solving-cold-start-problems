
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка модели и данных
@st.cache_resource
def load_model():
    df = pd.read_csv('/content/cleaned_qos_data.csv')  # или загрузка из другого источника
    recommender = GeoRecommender(n_epochs=100, verbose=False)
    recommender.fit(df)
    return recommender
    return joblib.load('/content/geo_recommender_model.pkl')

model = load_model()

# Интерфейс Streamlit
st.title("Гибридный подход матричной факторизации с интеграцией географических координат и культурных характеристик пользователей.")
st.markdown("""
Анализ географических данных и предсказание рейтингов на основе местоположения
""")

# Основные вкладки
tab1, tab2, tab3 = st.tabs(["Предсказание", "Визуализация", "О модели"])

with tab1:
    st.header("Предсказание рейтинга")
    
    # Создаем форму для ввода данных
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input("User ID", "user123")
            latitude = st.number_input("Широта", value=55.751244)
            
        with col2:
            item_id = st.text_input("Item ID", "item456")
            longitude = st.number_input("Долгота", value=37.618423)
            
        submitted = st.form_submit_button("Предсказать рейтинг")
        
    if submitted:
        # Создаем временный DataFrame для предсказания
        input_data = pd.DataFrame({
            '[User ID]': [user_id],
            '[IP No.]': [item_id],
            '[Latitude]': [latitude],
            '[Longitude]': [longitude]
        })
        
        # Обрабатываем данные и делаем предсказание
        processed_data = model._preprocess_data(input_data)
        prediction = model._predict_one(
            processed_data.iloc[0]['user_id'],
            processed_data.iloc[0]['item_id'],
            processed_data.iloc[0]['geo_cluster']
        )
        
        st.success(f"Предсказанный рейтинг: **{prediction:.2f}**")
        
        # Показываем дополнительные метрики
        with st.expander("Детали предсказания"):
            st.write(f"Географический кластер: {processed_data.iloc[0]['geo_cluster']}")
            if model.cultural_features:
                st.write("Культурные факторы:")
                cultural_factors = processed_data.filter(regex='cultural_').iloc[0]
                st.write(cultural_factors)

with tab2:
    st.header("Визуализация данных")
    
    # Показываем графики из модели
    if hasattr(model, 'train_history'):
        st.subheader("График обучения")
        fig, ax = plt.subplots(figsize=(10, 5))
        epochs = [h['epoch'] for h in model.train_history]
        ax.plot(epochs, [h['rmse'] for h in model.train_history], label='RMSE')
        ax.plot(epochs, [h['mae'] for h in model.train_history], label='MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Значение')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    
    # Показываем распределение рейтингов
    if model.df_processed is not None:
        st.subheader("Распределение рейтингов")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(model.df_processed['rating'], bins=20, kde=True, ax=ax)
        ax.set_title('Распределение рейтингов в данных')
        st.pyplot(fig)

with tab3:
    st.header("Информация о модели")
    st.markdown("""
    ### Параметры модели:
    - Географический вес: {model.geo_weight}
    - Культурный вес: {model.cultural_weight}
    - Количество факторов: {model.n_factors}
    - Количество эпох обучения: {model.n_epochs}
    """)
    
    st.subheader("Метрики кластеризации")
    if model.cluster_centers:
        st.write(f"Всего кластеров: {len(model.cluster_centers)}")
        st.write(f"Размеры кластеров:")
        cluster_sizes = {k: v['size'] for k, v in model.cluster_centers.items()}
        st.bar_chart(cluster_sizes)

# Загрузка своих данных
st.sidebar.header("Загрузите свои данные")
uploaded_file = st.sidebar.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.sidebar.success("Данные успешно загружены!")
        
        if st.sidebar.button("Обучить модель на новых данных"):
            with st.spinner('Обучение модели...'):
                new_model = GeoRecommender(n_epochs=50, verbose=False)
                new_model.fit(user_data)
                model = new_model
                st.success("Модель успешно переобучена!")
                
                # Сохранение модели (опционально)
                buffer = BytesIO()
                joblib.dump(model, buffer)
                st.sidebar.download_button(
                    label="Скачать модель",
                    data=buffer,
                    file_name="/content/geo_recommender_model.pkl",
                    mime="application/octet-stream"
                )
    except Exception as e:
        st.sidebar.error(f"Ошибка при загрузке файла: {e}")
