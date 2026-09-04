# Обвязка замеров диаризации

Исследовательские скрипты для карты
[Карта: диаризация спикеров в транскрипте](https://git.dementev.space/ddmitry/local-transcriber/issues/8) (#8).
Не часть пакета: они опираются на `sherpa-onnx`, которого нет в зависимостях
проекта, и живут в `.scratch/`, а не в `src/`.

Код этого каталога временно закреплен в Git: его используют несколько задач на
нескольких компьютерах, пока идет исследование диаризации. Общее правило
`/.scratch/` относится к новым локальным материалам; уже отслеживаемые файлы
этого стенда продолжают версионироваться.

Результаты первого прогона описаны в
[разведочном замере](../../docs/benchmarks/2026-08-12-diarization-feasibility.md).

## Модели

Скачиваются один раз в `models/`, в git не попадают (см. `.gitignore` рядом).

```bash
mkdir -p models && cd models
curl -sSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xjf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
curl -sSL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx
```

Сегментация — 6,9 МБ, эмбеддинги — 26,5 МБ. Опечатка `recongition` в URL
относится к самому релизу sherpa-onnx, это не ошибка набора.

## Скрипты

| Скрипт | Что делает | Тикеты |
|---|---|---|
| `bench_asr.py` | ASR тем же путём, что CLI: время, RTF, память | #13 |
| `bench_diar.py` | один прогон диаризации, сохраняет разметку в `segments-<порог>.tsv` | #12, #13 |
| `bench_sweep.py` | свип порога кластеризации и явного числа говорящих | #10 |
| `bench_conflict.py` | доля ASR-сегментов, внутри которых меняется говорящий | #11 |
| `common.py` | пути, конфигурация диаризатора, замер памяти | — |

## Запуск

Из корня репозитория. `PYTHONIOENCODING=utf-8` нужен, иначе вывод падает на
консоли cp1251.

```bash
export PYTHONIOENCODING=utf-8

uv run python .scratch/diarization/bench_asr.py "<путь к записи>"
uv run --with sherpa-onnx python .scratch/diarization/bench_diar.py "<путь>" 8 0.89
uv run --with sherpa-onnx python .scratch/diarization/bench_sweep.py "<путь>"
uv run --with sherpa-onnx python .scratch/diarization/bench_conflict.py "<путь>"
```

## Что стоит знать до запуска

- **Порог кластеризации откалиброван.** По умолчанию стоит 0,89 — единственное
  проверенное значение, которое без знания числа участников дало правильные
  3 / 2 / 2 кластера на трёх калибровочных фрагментах. Решение и ограничения
  описаны в
  [отчёте о калибровке](../../docs/benchmarks/2026-08-14-diarization-calibration.md).
- **Свип дорогой.** Каждая конфигурация — полный прогон сегментации и
  эмбеддингов, около 2,5 минут на 26-минутную запись, и время от настроек
  кластеризации практически не зависит. Свип вести на коротком фрагменте.
- **Чистота сегментов меряется относительно диаризации.** Если её границы
  систематически смещены, метрика измеряет не то, что кажется. Проверка границ
  на слух — тикет #12, и он намеренно идёт до калибровки.
- **Замер памяти чинился.** В разведке `psapi.GetProcessMemoryInfo` молча
  возвращал ноль; `common.peak_rss_mb()` теперь зовёт `K32GetProcessMemoryInfo`
  из kernel32 и проверяет код возврата. На Linux и macOS используется
  `resource.getrusage()` с поправкой на разные единицы измерения.
