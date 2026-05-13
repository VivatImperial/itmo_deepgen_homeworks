# Часть 2. Автоэнкодер для определения аномалий

## Что сделано

Во второй части реализован reconstruction-based пайплайн для поиска пролива:

- автоэнкодер обучается только на нормальных изображениях из `dataset/train`;
- для каждого изображения считается ошибка реконструкции `MSE`;
- порог выбирается по изображениям из `dataset/proliv`;
- на тесте изображение относится к классу `proliv`, если его anomaly score выше порога.

Архитектура `MNAD` не использовалась, так как в условии она относится к другому курсу.

## Структура

- `src/data.py` - загрузка данных и preprocessing;
- `src/model.py` - автоэнкодер с pretrained encoder и сверточным decoder;
- `src/pipeline.py` - обучение, выбор порога, расчет метрик и сохранение артефактов;
- `data/` - исходный датасет;
- `artifacts_*` - сохраненные результаты экспериментов.

## Данные

Использован оригинальный датасет из задания:

- `train` - `10000` изображений без пролива;
- `proliv` - `154` изображения с проливом;
- `test` - `3794` изображения с разметкой.

Изображения приводятся к фиксированному размеру с сохранением пропорций и padding.

## Основное решение

В качестве рабочего варианта оставлен `mobilevitv2_150.cvnets_in1k` с pretrained весами из Hugging Face через `timm`.

Пайплайн поддерживает:

- обучение с `early stopping`;
- сохранение лучшего checkpoint;
- расчет `TPR`, `TNR` и `accuracy`;
- сохранение `metrics.json`, `summary.md`, кривых обучения и примеров реконструкции;
- автоматическое использование `mps` на Apple Silicon.

Итоговая конфигурация основного run:

- encoder: `mobilevitv2_150.cvnets_in1k`;
- decoder: `plain`;
- image size: `96`;
- batch size: `64`;
- epochs: `6`;
- bottleneck: `32`;
- weight decay: `5e-4`;
- loss: `MSE`;
- score mode: `pixel_quantile`;
- score param: `0.8`;
- threshold mode: `balanced`.

## Результат

Для основного full-image run получены такие метрики на тесте:

- `TPR = 0.9535`
- `TNR = 0.9179`
- `accuracy = 0.9191`

Требование из условия `> 0.95` одновременно по `TPR` и `TNR` в рамках проверенных lightweight autoencoder-конфигураций достигнуть не удалось. Среди проверенных вариантов эта конфигурация показала лучший баланс.

## Сравнение бэкбонов

Базовое сравнение проводилось в одном протоколе с reconstruction-based thresholding:

- `mobilenetv3_small_100`: `TPR=0.8837`, `TNR=0.8035`, `accuracy=0.8063`
- `efficientnet_b2.ra_in1k`: `TPR=0.8062`, `TNR=0.7607`, `accuracy=0.7623`
- `efficientnetv2_rw_t.ra2_in1k`: `TPR=0.8217`, `TNR=0.8030`, `accuracy=0.8036`
- `mobilenetv3_large_150d` (trainable): `TPR=0.8140`, `TNR=0.8082`, `accuracy=0.8084`
- `mobilevitv2_150.cvnets_in1k` (основной вариант): `TPR=0.9535`, `TNR=0.9179`, `accuracy=0.9191`

## Локальные ROI/patch эксперименты

Дополнительно был проверен локальный трек, где предполагалось усилить чувствительность к пространственно локальному дефекту:

- `central_roi`, `train_roi=0.75`: `TPR=0.7829`, `TNR=0.9367`, `accuracy=0.9315`
- `random_patch`, `patch=64`, tiled inference `stride=16`, `agg=max`: `TPR=0.7907`, `TNR=0.9045`, `accuracy=0.9006`

Оба локальных варианта не улучшили минимум между `TPR` и `TNR`, поэтому основным оставлен full-image run.

## Запуск

```bash
uv sync
uv run python -m hw_1.task_2.src.pipeline \
  --encoder-name mobilevitv2_150.cvnets_in1k \
  --encoder-out-index 4 \
  --decoder-mode plain \
  --bottleneck-channels 32 \
  --train-encoder \
  --image-size 96 \
  --epochs 6 \
  --batch-size 64 \
  --num-workers 0 \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 1e-4 \
  --score-roi 1.0 \
  --score-mode pixel_quantile \
  --score-param 0.8 \
  --output-dir hw_1/task_2/artifacts_mobilevitv2_150_96_b32_wd5e4
```

Во время запуска выводятся логи по основным этапам: загрузка данных, выбор устройства, обучение, подбор порога, метрики и сохранение артефактов. Для подробного вывода можно добавить `--verbose`, для тихого режима без progress bar - `--quiet`.

## Артефакты

В репозитории сохранены легкие артефакты ключевых запусков:

- `artifacts_mobilevitv2_150_96_b32_wd5e4` - основной full-image run;
- `artifacts_mobilevitv2_150_96_b32_wd5e4_roi075` - локальный `central_roi` эксперимент;
- `artifacts_mobilevitv2_150_96_b32_wd5e4_randompatch64_t64s16_max` - локальный `random_patch` эксперимент;
- `artifacts_effb2`, `artifacts_effv2t`, `artifacts_mobilenetv3_large_150d_trainable` - сравнение с другими backbone;
- `artifacts_local_roi_patch_comparison.md` - краткая сводка по локальному треку.

Тяжелые веса модели в репозиторий не добавляются.
