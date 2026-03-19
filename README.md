# OmniSub2026 - lipRead Case

Лицензионно-чистый baseline для visual-only lip reading.

## Структура проекта

- `scripts/` - основной код обучения, инференса и постобработки
- `notebooks/` - запускные Jupyter-ноутбуки (включая all-in-one)
- `runs/` - чекпоинты и метрики обучения
- `artifacts/` - эталонные CSV-артефакты
- `assets/` - шаблоны и служебные данные
- `experiments/` - исследовательский ориентир по pretrained VSR + LM + MediaPipe
- `submissions/` - рабочие сабмиты и сводки прогонов

## Ключевые файлы

- `runs/run_char_warmstart_overfit/best.pt` - лучший checkpoint
- `OmniSub2026_lipRead_Case_AllInOne.ipynb` - all-in-one в корне
- `notebooks/00_all_in_one.ipynb` - all-in-one в папке ноутбуков
- `scripts/quick_ctc_smoke.py` - from-scratch CTC тренировка/оценка
- `scripts/infer_beam_submission.py` - beam decoding
- `scripts/apply_wordnorm_from_beam.py` - wordnorm postprocess
- `scripts/kaggle_infer_weights_only.py` - inference-only сценарий для Kaggle

## Формат входных данных

Ожидается `data-root` со структурой:

- `train/`
- `test/`
- `sample_submission.csv`

Поддерживаются два формата `path`:

- `test/<video_id>/<clip_id>.mp4`
- `<clip_id>.mp4` (файл находится в `test/`)

## Быстрый старт (рекомендуется)

Откройте:

- `notebooks/00_all_in_one.ipynb`
- `OmniSub2026_lipRead_Case_AllInOne.ipynb`

В ноутбуке есть весь pipeline:

1. обучение (опционально)
2. beam inference
3. wordnorm
4. проверка итогового submission

## CLI запуск (опционально)

```bash
python scripts/infer_beam_submission.py \
  --ckpt runs/run_char_warmstart_overfit/best.pt \
  --data-root /path/to/data_root \
  --sample-submission /path/to/data_root/sample_submission.csv \
  --output-csv submission_beam.csv \
  --beam-size 30 --batch-size 6 --num-workers 0
```

```bash
python scripts/apply_wordnorm_from_beam.py \
  --input-csv submission_beam.csv \
  --output-csv submission_beam_wordnorm.csv
```

## Требования к submission CSV

- ровно 2 колонки: `path,transcription`
- порядок строк совпадает с `sample_submission.csv`
- `transcription` в lowercase с нормализованными пробелами

## Примечание по исследовательскому блоку

`experiments/pretrained_vsr_lm_mediapipe/` добавлен как ориентир качества предобученных VSR-подходов.  
Финальный submission в этом репозитории формируется from-scratch pipeline из `scripts/` и `runs/`.
