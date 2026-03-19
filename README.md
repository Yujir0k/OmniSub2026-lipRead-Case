# OmniSub2026 - lipRead Case

Лицензионно-чистый baseline для visual-only lip reading.

## Что внутри

- `runs/run_char_warmstart_overfit/best.pt` - лучший checkpoint
- `quick_ctc_smoke.py` - from-scratch CTC тренировка/оценка
- `infer_beam_submission.py` - beam decoding для submission
- `apply_wordnorm_from_beam.py` - postprocess (word normalization)
- `kaggle_infer_weights_only.py` - inference-only сценарий для Kaggle
- `OmniSub2026_lipRead_Case_AllInOne.ipynb` - единый ноутбук запуска

## Формат данных

Ожидается `data-root` со структурой:

- `train/`
- `test/`
- `sample_submission.csv`

Поддерживаются оба формата `path`:

- `test/<video_id>/<clip_id>.mp4`
- `<clip_id>.mp4` (файл находится в `test/`)

## Быстрый старт (ноутбук)

Откройте и запустите:

- `OmniSub2026_lipRead_Case_AllInOne.ipynb`

В ноутбуке есть отдельные секции:

1. from-scratch обучение (`TinyLipCTC`)
2. beam inference
3. word normalization
4. проверка итогового CSV

## CLI запуск (опционально)

```bash
python infer_beam_submission.py \
  --ckpt runs/run_char_warmstart_overfit/best.pt \
  --data-root /path/to/data_root \
  --sample-submission /path/to/data_root/sample_submission.csv \
  --output-csv submission_beam.csv \
  --beam-size 30 --batch-size 6 --num-workers 0
```

```bash
python apply_wordnorm_from_beam.py \
  --input-csv submission_beam.csv \
  --output-csv submission_beam_wordnorm.csv
```

## Требования к итоговому CSV

- ровно 2 колонки: `path,transcription`
- порядок строк строго как в `sample_submission.csv`
- `transcription`: lowercase, нормализованные пробелы

## Примечание по исследовательским материалам

Папка `Pretrained VSR + LM + MediaPipe (ориентир верхней планки)` добавлена как ориентир анализа сильных предобученных подходов.  
Финальный submission в этом проекте строится from-scratch pipeline из текущего репозитория.
