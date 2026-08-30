# Ускорение извлечения голосовых эмбеддингов на x86 CPU

**Дата:** 2026-08-30

**Статус:** исследовательская записка. Production-код и конфигурация не
менялись.

**Метод:** проверены `sherpa-onnx==1.13.5`, `onnxruntime==1.28.0`, официальные
скрипты экспорта и исходники владельцев моделей. Три официальных release asset
прочитаны потоком без сохранения на диск: WeSpeaker ResNet34 LM, CAMPPlus en и
TitaNet small. Их входы, выходы и метаданные разобраны через ORT 1.28.0; для
каждого выполнен CPU smoke-test с batch size 2.

## Вопрос и краткий ответ

Текущий путь не использует batch: sherpa-onnx последовательно создает stream и
вызывает один `Session::Run` для каждой пары "окно, локальный говорящий".
Публичный `SpeakerEmbeddingExtractor` также принимает один stream за вызов.

Из трех направлений достойны прототипа два:

1. статическая INT8-квантизация действующего WeSpeaker в формате S8S8 QDQ;
2. несколько независимых stream поверх **одной** ORT session с ограниченным
   общим числом потоков.

Настоящий batch технически лучше всего подготовлен у TitaNet: граф принимает
вектор длин. Но TitaNet пока допустим только при явно известном числе говорящих,
а sherpa-onnx все равно фиксирует batch size 1. Поэтому batch TitaNet -
условный следующий прототип; действующую конфигурацию он не ускоряет.

Несколько отдельных session не стоит прототипировать первым: это не уменьшает
работу, создает отдельные пулы потоков и арены памяти, повышает риск
oversubscription и
роста RSS. Для WeSpeaker и CAMPPlus batch с простым дополнением также не готов:
их графы не принимают длины или маску.

## Что именно выполняется последовательно

Приложение создает один `OfflineSpeakerDiarization` и задает одно и то же
`num_threads` отдельно для segmentation и embedding
([адаптер проекта](../../src/local_transcriber/speaker_diarizer.py)). Внутри
sherpa-onnx стадии идут строго `segmentation → embeddings → clustering`, а
`ComputeEmbeddings` проходит список пар обычным циклом: создает stream,
добавляет все неперекрывающиеся фрагменты одного локального говорящего,
завершает input и синхронно вызывает `Compute`
([реализация Pyannote в sherpa-onnx 1.13.5](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h)).

Публичный C++ и Python API экспортирует `create_stream`, `is_ready` и
`compute(stream)`. Метода `compute_batch(streams)` нет
([C++ API](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/speaker-embedding-extractor.h),
[Python binding](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/python/csrc/speaker-embedding-extractor.cc)).
Python binding освобождает GIL на `compute`, поэтому внешний параллельный вызов
в принципе возможен; готовый диаризатор промежуточные stream наружу не отдает.

Для embedding создается одна ORT session. Sherpa-onnx передает заданное число и
в `intra_op_num_threads`, и в `inter_op_num_threads`
([настройка session](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/session.cc)).
ORT по умолчанию исполняет граф последовательно; `inter_op` начинает
распараллеливать узлы только при `ORT_PARALLEL`, тогда как `intra_op`
распараллеливает оператор
([официальная документация ORT](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)).
Значит, в текущем CPU-пути основная ручка - `intra_op`.

## Контракты трех графов

| Модель | Вход и выход официального экспорта | Следствие для batch |
|---|---|---|
| WeSpeaker ResNet34 LM | Release asset имеет один `float32`-вход `[B, T, 80]` и выход `[B, 256]` ([asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx)). Скрипт, которым sherpa-onnx добавляет метаданные, проверяет символические `B`, `T`, размерность 80 и единственный выход. Метаданные включают `framework=wespeaker`, `sample_rate`, `output_dim`, `normalize_samples=0` ([проверка sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/scripts/wespeaker/add_meta_data.py), [экспорт WeSpeaker](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/bin/export_onnx.py)). | Равные `T` можно сложить по `B`. Длины или маску граф не принимает; дополнение участвует в pooling и меняет эмбеддинг. |
| CAMPPlus en | Release asset имеет `float32`-вход `[N, T, 80]` и выход с динамической первой осью `[*, 512]` ([asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx)). Метаданные: `framework=3d-speaker`, `output_dim=512`, `normalize_samples=1`, `feature_normalize_type=global-mean` ([экспорт sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/scripts/3dspeaker/export-onnx.py)). | Несмотря на опечатку экспортного скрипта (`embedding`/`embeddings`), release asset принимает batch равной длины и возвращает все строки. Для разных длин граф по-прежнему не принимает маску или длины. |
| NeMo TitaNet small | Release asset имеет входы `float32 [N, 80, T]` и `int64 [N]`; выходы — logits `[N, 16681]` и embedding `[N, 192]` ([asset](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx), [контракт C++-обертки](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/speaker-embedding-extractor-nemo-model.h)). Метаданные содержат `framework=nemo`, `output_dim`, `feat_dim`, sample rate, параметры окна и нормализации ([экспорт](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/scripts/nemo/speaker-verification/export-onnx.py)). | Граф принимает batch и вектор длин. Но sherpa-onnx создает обе входные формы с `N=1` и возвращает только первую строку ([реализация](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/speaker-embedding-extractor-nemo-impl.h)). |

Smoke-test с синтетическими `float32`-признаками и одним intra-op thread дал
для WeSpeaker `(2, 256)`, для CAMPPlus `(2, 512)` и для TitaNet `(2, 192)`.
У WeSpeaker и CAMPPlus результат batch size 2 совпал с конкатенацией двух
последовательных `Run` (`max_abs_diff=0`). У TitaNet элемент длиной 100 кадров
также совпал, но элемент длиной 80, дополненный до 100 кадров, отличался от
отдельного непаддированного вызова (`max_abs_diff=0,0269`). Это не тест качества
на голосе, а подтверждение, что допустимый порог для variable-length batch нужно
определить в прототипе, а не считать эквивалентность гарантированной контрактом.

WeSpeaker ResNet34 состоит из Conv2d-блоков и statistics pooling
([исходник WeSpeaker](https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/resnet.py));
CAMPPlus сочетает Conv2d, TDNN/Conv1d и statistics pooling
([исходник 3D-Speaker](https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/models/campplus/DTDNN.py));
TitaNet использует depth-wise separable Conv1d, squeeze-and-excitation и
attention statistics pooling
([документация NeMo](https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr/speaker_recognition/models.html)).
Этого недостаточно, чтобы утверждать: каждый оператор будет квантизован и
ускорен.

В [официальном выпуске speaker-recognition models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
есть только FP32-файлы WeSpeaker, CAMPPlus и TitaNet; готовых INT8 embedding
assets нет. Статический INT8 здесь означает собственный производный артефакт с
отдельной проверкой лицензии, метаданных и качества.

## Оценка способов ускорения

| Способ | Техническая доступность | Вердикт |
|---|---|---|
| Настоящий batch | Release-графы WeSpeaker и CAMPPlus поддерживают batch равной длины; TitaNet принимает batch и второй вход с длинами. Sherpa-onnx для всех трех жестко формирует `N=1` и не имеет batch API. | Прототипировать только TitaNet и только условно для режима с известным числом говорящих. Для WeSpeaker/CAMPPlus без маски возможны лишь группы с одинаковым `T`; ожидаемая наполняемость неизвестна. |
| Несколько stream, одна session | Stream независимы, `Compute` использует локальные тензоры, а ORT разрешает нескольким потокам одновременно вызывать `Run` одной session ([архитектура ORT](https://onnxruntime.ai/docs/reference/high-level-design.html), [контракт 1.28.0](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/session/inference_session.h)). Текущий цикл sherpa-onnx этого не делает. | Достойно узкого прототипа на WeSpeaker. У одной session общий intra-op pool; нужно сравнить внутренний и внешний параллелизм. Полный pool для каждого worker не нужен. |
| Несколько session | Можно создать несколько extractor, но по умолчанию каждая session получает собственный intra-op pool и CPU arena. ORT отдельно предлагает общий pool против конкуренции session pools и общий allocator против роста памяти ([thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html), [C API guide](https://onnxruntime.ai/docs/get-started/with-c.html#share-allocator-s-between-sessions)). | Не брать как самостоятельный кандидат. Допустима одна контрольная ячейка против shared-session, чтобы подтвердить отказ по реальному времени и RSS. |
| Статический INT8 | ORT предоставляет `quantize_static`, использует calibration data и рекомендует static quantization для CNN. Для CPU первым выбором служит S8S8 QDQ; квантизация может ухудшить точность и даже скорость ([официальное руководство](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)). | Главный кандидат для действующего WeSpeaker. CAMPPlus и TitaNet квантизовать только после продуктового решения использовать соответствующую модель. |

Batch сокращает число `Session::Run` и может лучше загрузить SIMD, но число
эмбеддингов и основная модельная арифметика сохраняются; padding может даже
добавить работу. Несколько stream или session лишь перекрывают независимые
вызовы. Только INT8 меняет арифметику и объем движения весов. Увеличение шага
окна или отказ от части эмбеддингов действительно сокращают работу. Их нужно
рассматривать в отдельном исследовании шага окна.

## ISA, oversubscription и память

AVX2, VNNI и AVX-512 служат уровнями runtime dispatch одного формата модели.
MLAS в ORT 1.28.0 проверяет CPUID во время запуска и выбирает отдельные AVX2,
AVX-VNNI, AVX-512 и
AVX512-VNNI kernels
([runtime dispatch](https://github.com/microsoft/onnxruntime/blob/v1.28.0/onnxruntime/core/mlas/lib/platform.cpp)).
Один QDQ-граф поэтому нужно измерять на разных CPU. Отдельная сборка под каждый
из них не нужна.

Официальное руководство ORT предупреждает:

- S8S8 QDQ остается первым выбором CPU. Если он дает значимую потерю
  точности, ORT советует проверить U8U8;
- отдельный путь U8S8 на AVX2 и AVX-512 без VNNI использует `VPMADDUBSW` и
  может дать saturation; если до него дойдет матрица, этот риск снижают
  `reduce_range` или переход на U8U8;
- на x86 с VNNI этой проблемы нет, а INT8 обычно выигрывает больше;
- на старом CPU без подходящих инструкций quantize/dequantize overhead способен
  сделать модель медленнее
  ([разделы Data type selection и FAQ](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)).

Доступный Core i7-6820HQ официально заявляет только SSE4.1/4.2 и AVX2
([Intel ARK](https://www.intel.com/content/www/us/en/products/sku/88970/intel-core-i76820hq-processor-8m-cache-up-to-3-60-ghz/specifications.html)).
Современный класс должен выбираться по фактическим CPUID-флагам. Надписи
"11-е поколение" недостаточно: минимум нужен `avx512_vnni` или `avx_vnni`. Для доступного
Zen 4 AMD документирует AVX-512 и AVX512_VNNI как путь ускоренного integer GEMM
([AMD AOCL](https://docs.amd.com/r/en-US/68552-AOCL-api-guide/Hardware-Utilization)).

В параллельном прототипе нужно измерять RSS. Уже последовательный тест показал,
что TitaNet требует примерно на 147–149 МиБ больше RSS, чем WeSpeaker, а FP32
WeSpeaker на старом Intel использовал в среднем 6,84 логического ядра из восьми
([локальный benchmark](../benchmarks/2026-08-30-lightweight-diarization-cpu.md)).
Конкурентные `Run` одной session делят общий intra-op pool, но все равно
конкурируют за его очереди и добавляют вызывающие потоки. Несколько session уже
создают отдельные пулы и дополнительно размножают арены и подготовленные веса.
Точный рост не следует считать линейным - его нужно измерить.

## Что остается неизвестным

- Реальный процент Conv/MatMul, который ORT сможет перевести в INT8, и выигрыш
  каждого графа на двух ISA-классах.
- Сохранятся ли обязательные пользовательские метаданные после квантизации; без
  `framework`, `output_dim` и параметров обработки признаков sherpa-onnx модель
  не загрузит.
- Точность статического INT8 после калибровки и необходимость исключить отдельные
  узлы через quantization debugging.
- Распределение `T` у пар "окно, локальный говорящий". Без него нельзя оценить
  полезность exact-length buckets для WeSpeaker/CAMPPlus.
- Выигрыш shared-session concurrency после ограничения общего числа потоков,
  его пиковый RSS и стабильность повторных конкурентных вызовов TitaNet.

## Рекомендация и минимальная граница прототипа

### 1. INT8 WeSpeaker - первый кандидат

На отдельном стенде, не меняя загрузчик проекта:

- выполнить `quant_pre_process`, затем `quantize_static` в S8S8 QDQ;
- проверить opset, вход/выход, все пользовательские метаданные и загрузку через
  `SpeakerEmbeddingExtractor`;
- сравнить FP32 и INT8 по реальному времени embedding-стадии, процессорному
  времени, пиковому RSS и
  расстоянию между эмбеддингами;
- только после технического выигрыша повторить действующие проверки числа
  голосовых кластеров, purity и слов без говорящего.

При значительной потере точности S8S8 проверить U8U8. Отдельный U8S8 с
`reduce_range` в первую матрицу не входит.

### 2. Shared-session streams - второй кандидат

Изолированный стенд или минимальная upstream-поправка должен сохранить одну
модель и одну ORT session, подготовить независимые stream и сравнить:

```text
outer workers / shared intra-op pool: 1 / P, 2 / P, P / 1
inter-op threads:                     1
```

Здесь `P` - число физических ядер. Первая ячейка показывает текущий внутренний
параллелизм, вторая проверяет конкурентные `Run` на общем pool, третья переносит
параллелизм наружу и не создает intra-op pool. Измерять нужно полный набор
независимых embedding-вызовов. Latency одного удачного stream для решения
недостаточно. Отдельные session оставить одной отрицательной контрольной ячейкой
`P workers / 1 thread`.

### 3. Batch TitaNet - условный кандидат

Если отдельное решение разрешит TitaNet для режима с известным числом
говорящих, прямой ORT-стенд должен сначала доказать на размерах batch `1 / 4`,
что batch переменной длины с дополнением и `x_lens` дает те же эмбеддинги в
заданном допуске, что и последовательные вызовы. Затем измерить пропускную
способность и RSS. До этого менять
sherpa-onnx API ради TitaNet преждевременно. CAMPPlus и batch без mask в эту
границу не входят.

Минимальная CPU-матрица для обоих непосредственных кандидатов:

| Класс | Конкретный доступный ориентир | Зачем |
|---|---|---|
| AVX2, без VNNI/AVX-512 | Intel Core i7-6820HQ, 4C/8T | Проверить нижнюю границу INT8 и конкуренцию на старом CPU. |
| VNNI + AVX-512 | AMD Ryzen 7 8845H из предыдущих замеров, с обязательной фиксацией `lscpu`/CPUID | Проверить специализированный INT8 kernel и масштабирование на 8C/16T. |

Фактический целевой Core i5 11-го поколения добавить как приемочную машину,
когда он появится; перед запуском записать точную модель и CPUID. Двух машин
достаточно, чтобы решить судьбу кандидата на двух ISA-классах. Переносить
полученный коэффициент ускорения на весь x86 нельзя.
