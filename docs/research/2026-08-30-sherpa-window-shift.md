# Поддерживаемый шаг окна в sherpa-onnx

**Дата:** 2026-08-30

**Статус:** исследовательская записка. Production-зависимости и код не менялись.

## Вопрос и ответ

Первый опубликованный релиз, в котором `window_shift_ratio` доступен из
готового Python wheel для offline speaker diarization: **sherpa-onnx 1.13.6**
от 18 августа 2026 года. В нем параметр входит в публичный Python-конструктор
`OfflineSpeakerSegmentationPyannoteModelConfig` и доступен как изменяемое поле.
Это подтверждают [binding тега
v1.13.6](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/python/csrc/offline-speaker-diarization.cc#L50-L59),
[пример Python API](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/python-api-examples/offline-speaker-diarization.py#L62-L80)
и [состав релиза](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.6).
Готовые wheels появились на PyPI в тот же день
([метаданные 1.13.6](https://pypi.org/pypi/sherpa-onnx/1.13.6/json)). На дату
исследования 1.13.6 также остается [текущей версией пакета на
PyPI](https://pypi.org/pypi/sherpa-onnx/json).

С 1.13.5 можно перейти **обычным обновлением до 1.13.6**. Частный binding,
backport и новый запрос upstream не нужны: нужный follow-up уже принят и
опубликован upstream. Модели и рабочее значение `0,1` при этом не меняются.

## Хронология

| Дата и версия | Что было доступно |
|---|---|
| До 23.07.2026 | Шаг был зашит в C++ как 10% окна. Это исходная проблема, описанная upstream в [PR #3769](https://github.com/k2-fsa/sherpa-onnx/pull/3769). |
| 23.07.2026, commit [`797e6e9`](https://github.com/k2-fsa/sherpa-onnx/commit/797e6e9) | `window_shift_ratio` добавлен в C++-ядро и CLI. Допустимый диапазон: `(0, 1]`, значение по умолчанию: `0,1`. C API и языковые binding намеренно оставлены для follow-up ([PR #3769](https://github.com/k2-fsa/sherpa-onnx/pull/3769)). |
| 11.08.2026, v1.13.5 | Первый релиз C++-ядра и CLI с параметром ([release notes](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.5)). Python binding этой версии принимает только `model` и не экспортирует поле ([исходник v1.13.5](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/python/csrc/offline-speaker-diarization.cc#L50-L57)). |
| 13.08.2026, commit [`3e40933`](https://github.com/k2-fsa/sherpa-onnx/commit/3e409338959097c6518998c9b72757db257f5f6f) | Параметр проведен через публичные C/CXX API и языковые binding, включая Python; PR #3870 принят upstream ([описание и мотивировка](https://github.com/k2-fsa/sherpa-onnx/pull/3870)). |
| 18.08.2026, v1.13.6 | Первый релиз с публичным Python API. На PyPI есть cp313 и cp314 wheels без отметки yanked для всех платформ проекта ([release](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.6), [PyPI JSON](https://pypi.org/pypi/sherpa-onnx/1.13.6/json)). |

Граница проходит между двумя релизами: **1.13.5** дает C++-ядро и CLI, а в
**1.13.6** доступны C, CXX wrapper, Python и остальные binding.

## Публичный контракт 1.13.6

### Python

```python
pyannote = sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
    model=segmentation_model,
    window_shift_ratio=0.2,
)
```

Конструктор имеет сигнатуру `(model, window_shift_ratio=0.1)`. Объект
предоставляет изменяемое поле `.window_shift_ratio`. Прежний вызов только с
`model` остается допустимым и сохраняет прежнее поведение
([binding](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/python/csrc/offline-speaker-diarization.cc#L50-L59)).
Нативная валидация принимает только `(0, 1]`; ноль в Python не включает default
([конфигурация ядра](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-config.cc#L24-L39)).

Параметр читается при создании segmentation-модели. Метод
`OfflineSpeakerDiarization.set_config()` обновляет только clustering, поэтому
для каждой точки сетки нужен новый diarizer
([реализация](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L87-L94)).

### C и C++

- Публичная C-структура получила
  `SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig.window_shift_ratio`.
  Значение должно лежать в `(0, 1]`; только C-конвертер трактует `<= 0` как
  незаданное и подставляет `0,1`, чтобы не сломать вызывающий код с
  zero-initialized struct
  ([C header](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/c-api/c-api.h#L3839-L3846),
  [конвертер](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/c-api/c-api.cc#L3145-L3156)).
- Публичный CXX wrapper получил поле `float window_shift_ratio = 0.1f`
  ([cxx-api.h](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/c-api/cxx-api.h#L1911-L1917)).
- Внутренний C++-контракт ядра существовал уже в 1.13.5, но не давал
  приложению на Python поддерживаемого пути. Подменять им binding больше нет
  смысла.

## Совместимость wheel и зависимостей

Проект требует Python `>=3.13` и заявляет Linux, Windows/WSL2 и macOS
([pyproject.toml](../../pyproject.toml), [PRD](../PRD.md#42-требования-к-окружению),
[таблица платформ](../../README.md#платформы)). У 1.13.6 матрица нужных wheels
совпадает с 1.13.5.

| Среда проекта | `sherpa-onnx` 1.13.6 | `sherpa-onnx-core` 1.13.6 | Вывод |
|---|---|---|---|
| CPython 3.13/3.14, Linux x86-64 и WSL2 | `manylinux2014_x86_64` / `manylinux_2_17_x86_64` | `py3-none-manylinux2014_x86_64` | Готовый комплект есть |
| CPython 3.13/3.14, Linux AArch64 | `manylinux2014_aarch64` / `manylinux_2_17_aarch64` | `py3-none-manylinux2014_aarch64` | Готовый комплект есть |
| CPython 3.13/3.14, Windows x86-64 | `win_amd64` | `py3-none-win_amd64` | Готовый комплект есть |
| CPython 3.13/3.14, Windows ARM64 | `win_arm64` | `py3-none-win_arm64` | Готовый комплект есть; проектом отдельно не проверен |
| CPython 3.13/3.14, macOS x86-64 | `macosx_10_15_x86_64` или `universal2` | те же platform tags | Готовый комплект есть; macOS в проекте не проходил ручную приемку |
| CPython 3.13/3.14, macOS ARM64 | `macosx_11_0_arm64` или `universal2` | те же platform tags | Готовый комплект есть; macOS в проекте не проходил ручную приемку |

Linux-поставка рассчитана на glibc 2.17 и новее: `musllinux` wheels в релизе
нет. Поэтому таблица не подтверждает готовую бинарную установку на Alpine или
другом Linux с musl.

Источник файлов основной поставки: [PyPI 1.13.6
JSON](https://pypi.org/pypi/sherpa-onnx/1.13.6/json), нативной части:
[PyPI core 1.13.6 JSON](https://pypi.org/pypi/sherpa-onnx-core/1.13.6/json).
Метаданные основного пакета объявляют `Requires-Python >=3.7` и единственную
зависимость `sherpa-onnx-core==1.13.6`; core дополнительных `Requires-Dist` не
объявляет. Универсального `abi3` wheel нет: при сохранении открытой границы
Python `>=3.13` готовая бинарная установка на будущем CPython 3.15 потребует
нового upstream wheel. Исходный архив опубликован, но локальная нативная сборка
для 3.15 в это исследование не входила.

Для обновления проекта нужно синхронно поменять три места: пин
`sherpa-onnx==1.13.6`, uv override на `sherpa-onnx-core==1.13.6` и lock-файл.
Текущий комментарий рядом с override уже требует проверку `uv tree --locked
--package sherpa-onnx` ([pyproject.toml](../../pyproject.toml)). Других новых
Python-зависимостей между 1.13.5 и 1.13.6 нет
([метаданные 1.13.5](https://pypi.org/pypi/sherpa-onnx/1.13.5/json),
[метаданные 1.13.6](https://pypi.org/pypi/sherpa-onnx/1.13.6/json)).

### Риски совместимости

- Для Python изменение обратно совместимо по исходному коду: новый аргумент
  необязательный, default равен прежнему зашитому значению. Upstream отдельно
  проверил одинаковый шаг в сэмплах для стандартного окна при `0,1`
  ([PR #3769](https://github.com/k2-fsa/sherpa-onnx/pull/3769)).
- C-структура стала больше. Это изменение layout и потенциальный ABI-разрыв для
  стороннего бинарника, собранного с header 1.13.5 и загружающего библиотеку
  1.13.6 без пересборки. Проект использует согласованные Python/core wheels и
  такого бинарника не имеет. Zero-initialized C-код после пересборки сохраняет
  default `0,1` ([PR #3870](https://github.com/k2-fsa/sherpa-onnx/pull/3870)).
- Остальные изменения 1.13.6 относятся к сборке Android/Java, SPM,
  Flutter/Dart и примерам VAD+ASR. Релиз не заявляет удаления Python API или
  смены формата моделей ([release notes](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v1.13.6)).
- Временное окружение CPython 3.13.14 на Linux x86-64 установило согласованную
  пару `sherpa-onnx==1.13.6` и `sherpa-onnx-core==1.13.6`. Импорт и создание
  конфигураций со всеми пятью значениями сетки прошли. Для Windows и macOS пока
  подтверждена только upstream-поставка; при обновлении проекта нужен отдельный
  smoke test Windows x86-64 и повтор Linux-проверки из lock-файла.

## Что именно меняет параметр

При загрузке модели движок вычисляет `window_shift = int(ratio * window_size)`
и при debug печатает фактическое значение. Затем запись режется на окна по этому
шагу; для записи длиннее одного окна число segmentation-запусков равно
`floor((n - W) / S) + 1` плюс одно дополненное нулями окно, если есть остаток
([расчет шага](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.cc#L92-L123),
[цикл окон](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L278-L337)).

Для Pyannote Segmentation 3.0 с окном `160000` сэмплов при 16 кГц сетка дает:

| Ratio | Шаг, сэмплов | Шаг, с | Окон на ровных 300 с | Теоретическая доля окон к `0,1` |
|---:|---:|---:|---:|---:|
| 0,10 | 16 000 | 1,0 | 291 | 1,00 |
| 0,15 | 24 000 | 1,5 | 195 | 0,67 |
| 0,20 | 32 000 | 2,0 | 146 | 0,50 |
| 0,25 | 40 000 | 2,5 | 117 | 0,40 |
| 0,50 | 80 000 | 5,0 | 59 | 0,20 |

Это не прогноз полного ускорения. После segmentation движок создает отдельное
задание embedding для каждой достаточно длинной пары `(окно, локальный
говорящий)`. Их число зависит от речи и перекрытий
([формирование пар](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L408-L484),
[цикл embedding](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L510-L564)).
Upstream получил на одном английском файле 1,48× при `0,15` и 2,00× при `0,20`,
но сам ограничил вывод этим материалом; переносить коэффициенты на русские
созвоны нельзя ([измерение в PR #3769](https://github.com/k2-fsa/sherpa-onnx/pull/3769)).

## Точная граница следующего прототипа

Прототип должен отвечать только на вопрос, действительно ли сетка
`0,1 / 0,15 / 0,2 / 0,25 / 0,5` уменьшает работу текущей связки без
неприемлемой потери разметки говорящих.

1. Использовать обычные wheels 1.13.6 и текущую связку Pyannote FP32 +
   WeSpeaker, 8 потоков, порог `0,89`, автоматическое число голосовых кластеров.
   Не смешивать в эту сетку INT8, TitaNet, новый порог или параллельный ASR.
2. Добавить ratio только в benchmark-конфигурацию. Для каждой ячейки создавать
   новый diarizer; production CLI, config cascade и default `0,1` не менять.
3. Сначала выполнить публичный smoke test: импорт, сигнатура, `validate()` для
   пяти значений и один файл длиннее 10 секунд. Затем отдельная benchmark-задача
   может использовать те же три контрольных фрагмента, что предыдущий CPU-тест,
   с прежними правилами хранения данных.
4. В каждой ячейке сохранить счетчики механизма:
   - фактический `window_shift` в сэмплах из debug-журнала;
   - вычисленное число segmentation-окон по длине входа и формуле upstream;
   - финальный `num_total_chunks` progress callback как число embedding-заданий.
     Название callback вводит в заблуждение: реализация вызывает его внутри
     цикла по `(окно, локальный говорящий)`. В segmentation-цикле вызова нет
     ([исходник](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.6/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L517-L564));
   - время segmentation, embedding, clustering, total и RTF из штатного
     debug-профиля, а также wall time, CPU и peak RSS процесса.
5. Чтобы не принять ускорение ценой поломки результата, сохранить прежние
   контрольные показатели: число голосовых кластеров `3 / 2 / 2`, mapped
   speaker purity там, где есть опорная разметка, слова без говорящего,
   максимальный остаток сверх ожидаемых кластеров, число малых кластеров и
   неизменность распознанного текста. Для кандидата после сетки нужна отдельная
   слуховая проверка коротких ответов, смен говорящего и перекрывающейся речи.

Не входят в прототип: выбор нового production default, пользовательский
параметр CLI, повторная настройка clustering, сравнение embedding-моделей и
параллельный запуск ASR. Эти решения возможны только после сетки. Ускорение
должно сопровождаться ожидаемым снижением числа segmentation-окон и
embedding-заданий. По одному wall time нельзя связать ускорение с шагом окна.

## Рекомендация

Выбрать **обычное обновление до 1.13.6** и сначала проверить его в отдельном
прототипе. Обновление дает проекту штатную ручку для сокращения основной
CPU-работы; свежесть зависимости сама по себе здесь не аргумент. Малый backport
дублировал бы уже выпущенный upstream-код и потребовал бы собственного нативного
wheel.
Запрос upstream уже фактически выполнен PR #3870. Production default оставить
`0,1`, пока сетка не подтвердит одновременно снижение работы, приемлемую
скорость и сохранение качества разметки говорящих.
