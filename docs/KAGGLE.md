# Обучение на Kaggle

Этот маршрут запускает тот же ResNet18 baseline из `kosti.training` на GPU Kaggle. Сейчас запуск заблокирован данными: проверенного `folds.json` и проверки текста, впечённого в пиксели, нет. Локальный `kaggle.json` остаётся вне репозитория и не входит в загружаемый пакет.

## Что подготовить локально

1. Заполнить шаблон ручной разметки по [PREPARATION.md](PREPARATION.md) и создать `artifacts/folds.json`. Каждое изображение должно иметь проверенную область, сторону и происхождение применимых меток. Каждый обучающий fold должен содержать обе области и оба значения каждого из семи выходов; скрипт это проверяет.
2. Проверить **каждое уникальное изображение** на текст или другие сведения пациента, впечённые в пиксели. Создать `artifacts/pixel-review.csv` с заголовком `pixel_hash,reviewed`, одной строкой на каждый `pixel_hash` из `folds.json` и значением `1` только после проверки. Метаданные DICOM сами по себе не доказывают, что пиксели чистые.
3. Получить исходный ImageNet `state_dict` ResNet18 с оригинальной головой на 1000 классов. Существующий `model_classifaer.pth` имеет другую голову и не подходит. Зафиксировать происхождение весов отдельно.
4. Установить локальный пакет и зависимости. Для публикации поставить официальный CLI Kaggle. Legacy Credential должен находиться в отдельном файле с правами `0600`.

```bash
python -m pip install -r requirements-core.lock
python -m pip install --no-deps .
python -m pip install kaggle

python scripts/kaggle_prepare.py \
  --folds artifacts/folds.json \
  --data-root ../data/Исследования \
  --pixel-review artifacts/pixel-review.csv \
  --initial-weights artifacts/imagenet-resnet18.pth \
  --output artifacts/kaggle-input-v1 \
  --dataset-id USERNAME/private-dxa-training-v1

python scripts/kaggle_run.py \
  --package artifacts/kaggle-input-v1 \
  --kernel-dir artifacts/kaggle-kernel-v1 \
  --kernel-id USERNAME/dxa-quality-baseline-v1
```

`kaggle_prepare.py` заново декодирует снимки, сверяет UID и хеш пикселей с проверенным манифестом, создаёт DICOM только из явного набора пиксельных и числовых полей, генерирует новые UID и пути, сохраняет групповые фолды и повторно сверяет пиксели. Исходные имена, идентификаторы и остальные теги не попадают в пакет. Неизвестные LUT-последовательности блокируют экспорт. Вкладка с пикселями может содержать текст, поэтому отдельная визуальная проверка обязательна. Результат и исходная таблица остаются в игнорируемом `artifacts/`.

## Публикация и запуск

После просмотра содержимого пакета и проверки прав на использование данных:

```bash
python scripts/kaggle_run.py \
  --package artifacts/kaggle-input-v1 \
  --kernel-dir artifacts/kaggle-kernel-v1 \
  --kernel-id USERNAME/dxa-quality-baseline-v1 \
  --credential ../kaggle.json --action upload-dataset

kaggle datasets status USERNAME/private-dxa-training-v1

python scripts/kaggle_run.py \
  --package artifacts/kaggle-input-v1 \
  --kernel-dir artifacts/kaggle-kernel-v1 \
  --kernel-id USERNAME/dxa-quality-baseline-v1 \
  --credential ../kaggle.json --action push-kernel
```

Создание dataset через CLI по умолчанию приватное; скрипт никогда не передаёт флаг публичной публикации. Kernel также создаётся приватным, с GPU и доступом в Интернет только для установки `pydicom==3.0.1`. Убедиться в приватности на странице Kaggle после загрузки. Секрет копируется только во временный каталог с правами `0700`, вывод CLI подавляется, после вызова каталог удаляется. Веса модели, `oof.csv`, `metrics.json` и журналы обучения остаются результатами приватного kernel; загрузить их для локальной проверки. Сбой или незавершённый kernel не означает готовую модель.

Команды Kaggle `datasets create`, `datasets status` и `kernels push`, а также поля метаданных сверены с [официальной документацией Kaggle datasets](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) и [kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md). API и установленный CLI следует проверить перед фактическим запуском, поскольку в текущей папке CLI пока отсутствует.
