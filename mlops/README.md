# Финальный проект

В качестве финального проекта вам нужно будет собрать end-to-end MLOps пайплайн. Вы должны будете применить все знания, полученные из лекций и практических заданий, затем собрать их в единую автоматическую систему для деплоя и мониторинга моделей.

## Задача

В качестве ML-задачи мы будем использовать уже знакомую задачу по предсказанию частоты паролей. Поскольку в данном курсе мы изучаем не ML, а MLOps, вам не нужно фокусироваться на тюнинге модели или исследовании данных (они будут синтетические).

Ваша задача — написать http обертку для этой модели, которая будет автоматически обновляться MLOps пайплайном при поступлении новых данных. Новые данные будут генерироваться автоматически проверочным скриптом. При появлении новых данных ваш сервис получит уведомление, после чего должен запуститься ваш пайплайн, который обучит модель, положит ее в mlflow и обновит ваш сервис. Вы свободны в выборе инструментов и архитектуры для построения пайплайна и можете использовать любую комбинацию изученных решений. В последнем степе можно найти описание одного из возможных вариантов решения.

## Решение и способ проверки

В качестве решения вам нужно предоставить два url, один — к эндпоинту с предсказаниями (predictions_url), другой — для того, чтобы уведомлять о появлении новых данных (trigger_url). Они могут быть в одном и том же сервисе или в разных, это остается на ваше усмотрение.

Проверочный скрипт будет генерировать данные и отправлять ссылку на них в trigger_url. После этого должен автоматически запуститься ваш пайплайн, состоящий из следующих шагов:

1. Скачать данные по url
2. Проверить качество данных
3. Обучить новую версию модели (используя только новые данные)
4. Загрузить версию модели в MLflow, зарегистрировать ее под именем {ваш юзернейм}-mlops-project-model и повесить на новую версию alias prod
5. Обновить модель в вашем сервисе на новую версию

Далее проверочный скрипт убедится, что модель обновилась, сравнив предсказания с ожидаемыми.

Такой цикл будет запущен несколько раз в течение проверки с разными наборами данных, в том числе невалидными — в этом случае модель обучать не нужно.

## Форматы запросов

Запросы в predictions_url будут иметь такой формат:

```json
{"Password": ["pass1", "pass2", ...]}
```

В ответ ожидается такой формат:

```json
{"Times": [0.1, 0.2, ...]}
```

В trigger_url будут прилетать запросы такого формата:

```json
{"data_url": "https://....."}
```
