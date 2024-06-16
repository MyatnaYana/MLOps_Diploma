# Выпускная квалификационная работа бакалавра 
# Стельмах Яна Сергеевна ИУ5-84Б
# Предсказание: что будет чувствовать покупатель до приобретения продукта

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

**Формулировка проблемы**:  Для исторических данных клиента нам поручено предсказать оценку отзыва для следующего заказа или покупки. Мы будем использовать [набор данных общественной электронной коммерции в Бразилии от Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Этот набор данных содержит информацию о 100,000 заказах с 2016 по 2018 год, сделанных на нескольких торговых площадках в Бразилии. Его характеристики позволяют просматривать расходы с различных точек зрения: от статуса заказа, цены, оплаты, производительности доставки до местоположения клиента, атрибутов продукта и, наконец, отзывов, написанных клиентами. Цель здесь - предсказать оценку удовлетворенности клиента для данного заказа на основе характеристик, таких как статус заказа, цена, оплата и т.д. Для достижения этого в реальном мире мы будем использовать [ZenML](https://zenml.io/) для построения готовой к производству конвейера для прогнозирования оценки удовлетворенности клиента для следующего заказа или покупки.

Цель этого репозитория - продемонстрировать, как [ZenML](https://github.com/zenml-io/zenml) расширяет возможности вашего бизнеса по построению и развертыванию конвейеров машинного обучения множеством способов:

- Предлагая вам каркас и шаблон для основы вашей собственной работы.
- Интегрируясь с инструментами, такими как [MLflow](https://mlflow.org/) для развертывания, отслеживания и многого другого
- Позволяя вам легко строить и развертывать конвейеры машинного обучения

## 🐍 Требования Python

Теперь давайте перейдем к необходимым Python-пакетам. В выбранной вами среде Python выполните:

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
pip install -r requirements.txt
```

Начиная с ZenML 0.20.0, ZenML поставляется с встроенной панелью инструментов на основе React. Эта панель инструментов позволяет наблюдать за вашими стеками, компонентами стека и графами конвейеров (DAG) в интерфейсе панели инструментов. Для доступа к этому вам необходимо [запустить локальный сервер и панель инструментов ZenML](https://docs.zenml.io/user-guide/starter-guide#explore-the-dashboard), но сначала вы должны установить необходимые зависимости для сервера ZenML:


```bash
pip3 install zenml["server"]
zenml up
```

Если вы запускаете скрипт run_deployment.py, вам также необходимо установить некоторые интеграции с помощью ZenML:


```bash
zenml integration install mlflow -y
```

Проект может быть выполнен только с использованием стека ZenML, который содержит эксперимент-трекер и модуль развертывания моделей MLflow в качестве компонентов. Конфигурирование нового стека с двумя этими компонентами выглядит следующим образом:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
zenml stack describe
```
Чтобы остановить процесс:
```bash
zenml stack list 
zenml model-deployer models delete <model_uuid>
```
Чтобы запустить MLFlow: 
```bash
mlflow ui --backend-store-uri "file:/Users/Iana/Library/Application Support/zenml/local_stores/0c496041-e535-44a0-8d4c-b340cde8590e/mlruns"
```
## 📙 Источники

https://blog.zenml.io/customer_satisfaction/
https://streamlit.io/
https://github.com/zenml-io/zenml/tree/main
https://www.mlflow.org

## Решение

Для построения реального рабочего процесса для прогнозирования оценки удовлетворенности клиента для следующего заказа или покупки (что поможет принимать более обоснованные решения), недостаточно просто обучить модель один раз.

Вместо этого мы строим сквозной конвейер для непрерывного прогнозирования и развертывания машинной обучения модели, наряду с приложением для работы с данными, которое использует последнюю развернутую модель для использования бизнесом.

Этот конвейер можно развернуть в облаке, масштабировать в соответствии с нашими потребностями и обеспечить отслеживание параметров и данных, проходящих через каждый запущенный конвейер. Он включает в себя входные сырые данные, признаки, результаты, машинная обучения модель и ее параметры, а также выходные данные прогнозирования. ZenML помогает нам построить такой конвейер простым, но мощным способом.

В этом проекте мы уделяем особое внимание [интеграции с MLflow] в ZenML. В частности, мы используем трекинг MLflow для отслеживания наших метрик и параметров, а развертывание MLflow для развертывания нашей модели. Мы также используем [Streamlit](https://streamlit.io/) для демонстрации того, как эта модель будет использоваться в реальной среде.


### Конвейер обучения модели (Train Pipeline)

Наш стандартный конвейер обучения состоит из нескольких шагов:

1. ingest_df: Этот шаг будет загружать данные и создавать DataFrame.
2. clean_df: Этот шаг будет очищать данные и удалять ненужные столбцы.
3. train_model: Этот шаг будет обучать модель и сохранять ее, используя [автоматическую фиксацию MLflow](https://www.mlflow.org/docs/latest/tracking.html).
4. evaluation: Этот шаг будет оценивать модель и сохранять метрики - используя автоматическую фиксацию MLflow - в хранилище артефактов.

Таким образом, весь процесс обучения модели разбит на четкие, последовательные шаги, каждый из которых выполняет определенную задачу. Использование MLflow для автоматической фиксации параметров и метрик позволяет отслеживать и анализировать процесс обучения модели.

### Конвейер развертывания (Deployment Pipeline)

Помимо конвейера обучения, у нас есть еще один конвейер - deployment_pipeline.py, который расширяет конвейер обучения и реализует непрерывный рабочий процесс развертывания. Он загружает и обрабатывает входные данные, обучает модель, а затем (пере)разворачивает сервер прогнозирования, который обслуживает модель, если она соответствует нашим критериям оценки.

Критерий, который мы выбрали, - это пороговое значение [r2_score] обучения. Первые четыре шага конвейера такие же, как и выше, но мы добавили следующие дополнительные шаги:

1. deployment_trigger: Этот шаг проверяет, соответствует ли недавно обученная модель установленным критериям для развертывания.
2. model_deployer: Этот шаг разворачивает модель как сервис с использованием MLflow (если критерии развертывания выполнены).

В конвейере развертывания используется интеграция отслеживания ZenML с MLflow для ведения журнала значений гиперпараметров, самой обученной модели и метрик оценки модели - как артефактов отслеживания экспериментов MLflow - в локальный бэкенд MLflow. Этот конвейер также запускает локальный сервер развертывания MLflow для обслуживания последней модели MLflow, если ее точность выше заданного порогового значения.

Сервер развертывания MLflow работает локально как демон-процесс, который будет продолжать работать в фоновом режиме после завершения примера. Когда запускается новый конвейер, который производит модель, прошедшую проверку порога точности, конвейер автоматически обновляет текущий запущенный сервер развертывания MLflow, чтобы он обслуживал новую модель вместо старой.

Наконец, мы развертываем приложение Streamlit, которое асинхронно потребляет последний сервис модели из логики конвейера. Это можно легко сделать с помощью ZenML внутри кода Streamlit.

```python
service = prediction_service_loader(
   pipeline_name="continuous_deployment_pipeline",
   pipeline_step_name="mlflow_model_deployer_step",
   running=False,
)
...
service.predict(...)  # Прогнозирование на входящих данных из приложения
```

Хотя этот проект ZenML обучает и развертывает модель локально, другие интеграции ZenML, такие как развертыватель [Seldon](https://github.com/zenml-io/zenml/tree/main/examples/seldon_deployment), также могут использоваться аналогичным образом для развертывания модели в более производственной среде (например, на кластере Kubernetes). Мы используем MLflow здесь из-за удобства его локального развертывания.

![training_and_deployment_pipeline]

## Листинг

Мы можем запустить два конвейера:

- Training pipeline:

```bash
python run_pipeline.py
```

- The continuous deployment pipeline:

```bash
python run_deployment.py
```

## 🕹 Demo Streamlit App

There is a live demo of this project using [Streamlit](https://streamlit.io/) which you can find [here](https://share.streamlit.io/ayush714/customer-satisfaction/main). 
Программа берет несколько входных характеристик продукта и предсказывает уровень удовлетворенности клиентов, используя последние обученные модели.
Чтобы запустить streamlit приложение:

```bash
streamlit run streamlit_app.py
```

## FAQ

1. When running the continuous deployment pipeline, I get an error stating: `No Step found for the name mlflow_deployer`.

   Solution: It happens because your artifact store is overridden after running the continuous deployment pipeline. So, you need to delete the artifact store and rerun the pipeline. You can get the location of the artifact store by running the following command:

   ```bash
   zenml artifact-store describe
   ```

   and then you can delete the artifact store with the following command:

   **Note**: This is a dangerous / destructive command! Please enter your path carefully, otherwise it may delete other folders from your computer.

   ```bash
   rm -rf PATH
   ```

2. When running the continuous deployment pipeline, I get the following error: `No Environment component with name mlflow is currently registered.`

   Solution: You forgot to install the MLflow integration in your ZenML environment. So, you need to install the MLflow integration by running the following command:

   ```bash
   zenml integration install mlflow -y
   ```

