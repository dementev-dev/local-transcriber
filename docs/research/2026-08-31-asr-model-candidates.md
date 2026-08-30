# Кандидаты на ASR-модель для русских встреч

**Дата:** 2026-08-31

**Статус:** кабинетное исследование по первичным источникам; локально новые модели еще не запускались

**Область исследования:** длинные русские встречи с английскими IT-терминами, локальная обработка и обязательные пословные таймкоды

## Исследовательский вопрос

Есть ли среди Nemotron 3.5 ASR Streaming 0.6B, Qwen3-ASR 0.6B, Canary 180M Flash и Cohere Transcribe кандидат, который лучше текущей GigaAM v3 E2E RNN-T подходит для локальной расшифровки длинных русских встреч с английскими IT-терминами?

## Контекст

Текущий профиль проекта, `gigaam-v3-e2e-rnnt`, уже встроен через `onnx-asr` и выбран по итогам проектного benchmark. Он дает быстрый русский E2E-текст с пунктуацией. Основная известная слабость - нестабильное распознавание английских терминов и названий. Подробные результаты приведены ниже.

Опубликованные WER разных моделей нельзя складывать в общий рейтинг. Корпуса, языки, нормализация и режимы распознавания различаются. Обоснованный вывод о преимуществе другой модели возможен только после прогона на проектных записях.

## Известные факты

### Сводка

| Модель | Русский | Таймкоды в практичном локальном пути | Размер локальных весов | Путь интеграции | Ключевое ограничение |
|---|---|---|---:|---|---|
| GigaAM v3 E2E RNN-T INT8 | Да, основной язык | Да, текущий адаптер `onnx-asr` | около 227 MB файлов ONNX | Уже встроена | Английские термины распознаются нестабильно |
| Nemotron 3.5 ASR Streaming 0.6B Q8 | Да, `ru-RU` в основном tier | Да, NeMo-Speech.cpp JSON содержит `words[]` | 742 MB | Новый backend поверх NeMo-Speech.cpp | Нет сопоставимых CPU-замеров |
| Qwen3-ASR 0.6B Q8 | Да, один из 30 языков | ASR не дает; нужен отдельный Qwen3-ForcedAligner 0.6B | 811 MB только ASR | Новый OpenVINO или native путь плюс aligner | Таймкоды требуют второго модельного этапа |
| Canary 180M Flash Q8 | Нет, только `en/de/es/fr` | В Handy и ONNX-порте нет | 208 MB GGUF, около 214 MB ONNX | Raw ONNX близок к текущему адаптеру | Нет русского; доступные порты теряют таймкоды |
| Cohere Transcribe Q5 | Нет, 14 языков без русского | Нет у upstream и Handy | 1.77 GB | Новый runtime и внешний VAD | Нет русского и таймкодов |

Размер файла не равен peak RAM или VRAM. Для новых моделей эти значения надо измерить отдельно.

### Режим работы и контракт результата

| Модель | Практический режим | Длинные записи | Контекст терминов | Результат |
|---|---|---|---|---|
| GigaAM v3 E2E RNN-T | Offline-файлы | Проект режет запись через Silero VAD | Текущий адаптер не передает словарь или контекст | Русский текст с регистром, пунктуацией, нормализацией и текущими word timestamps; confidence от модели не сохраняется |
| Nemotron 3.5 | Offline и настоящий cache-aware streaming | NeMo-Speech.cpp принимает файл целиком или поток | Word boosting в NeMo-Speech.cpp | Пунктуация и регистр; auto language tag; `words[]` с временем и confidence; полная ITN требует отдельной grammar |
| Qwen3-ASR 0.6B | Upstream: offline и vLLM streaming; Handy: только offline | Handy принимает до примерно 87 минут; aligner заявлен до пяти минут, официальный wrapper режет timestamp-вход по 180 секунд | Официальный runtime принимает свободный `context`; в текущем `transcribe.cpp` входа нет | Текст и определенный язык; timestamps только через отдельный ForcedAligner, без confidence; строгий ITN-контракт не описан |
| Canary 180M Flash | Offline, короткие реплики | Вход короче 40 секунд; long-form через chunk-and-stitch | Сопоставимый механизм не заявлен | Переключаемые пунктуация и регистр; upstream дает экспериментальные word/segment timestamps через вспомогательный CTC-aligner, доступные порты его теряют |
| Cohere Transcribe | Monolingual offline; incremental streaming не документирован | Transformers автоматически режет и собирает длинное аудио | Сопоставимый механизм не заявлен | Переключаемая пунктуация; язык надо указать; timestamps, diarization и auto language ID отсутствуют |

Источники по режимам и результату: [Nemotron model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), [NeMo-Speech.cpp API](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/api.md), [Qwen runtime](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py), [Qwen transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr.md), [Canary model card](https://huggingface.co/nvidia/canary-180m-flash), [Canary implementation](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/aed_multitask_models.py), [Cohere model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

Сопоставимого word-confidence между моделями нет. NeMo-Speech.cpp возвращает поле `confidence`, но его калибровка не заявлена. Текущий GigaAM-адаптер выбрасывает token log probabilities, Qwen ForcedAligner возвращает только текст и время, Canary и Cohere не документируют confidence как часть результата.

### Поставка и лицензии

| Модель | Полный локальный состав | Лицензия и доступ | Условие поставки |
|---|---|---|---|
| GigaAM | Около 227 MB ONNX плюс уже используемые `onnx-asr`, ONNX Runtime и Silero VAD | MIT | При распространении нужно сохранить текст лицензии |
| Nemotron 3.5 | 742 MB Q8 GGUF плюс NeMo-Speech.cpp; VAD и полная ITN требуют дополнительных моделей или файлов | Веса OpenMDW-1.1, runtime Apache-2.0 | При распространении весов надо сохранить лицензию и notices |
| Qwen3-ASR 0.6B | 811 MB Q8 только для ASR; timestamp-профилю нужен еще Qwen3-ForcedAligner 0.6B, готового GGUF-пути для него нет | Apache-2.0, upstream не gated | Профиль с таймкодами включает второй набор весов; сохраняются license/notices Apache-2.0 |
| Canary 180M Flash | 208 MB Q8 GGUF или около 214 MB ONNX дают текст без таймкодов; полный `.nemo` 737 MB содержит вспомогательный aligner и требует NeMo/PyTorch | CC-BY-4.0 | При поставке нужна атрибуция; timestamp-профиль перестает быть компактным |
| Cohere Transcribe | 1.77 GB Q5 GGUF; upstream BF16 занимает 4.13 GB | Apache-2.0; официальный Hugging Face repo gated | Автоматическая загрузка официальных весов требует пользовательского доступа |

Размеры и условия: [GigaAM ONNX](https://huggingface.co/istupakov/gigaam-v3-onnx/tree/main), [GigaAM](https://github.com/salute-developers/GigaAM), [Nemotron files](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/main), [OpenMDW 1.1](https://openmdw.ai/license/1-1/), [Qwen ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B), [Qwen ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), [Canary](https://huggingface.co/nvidia/canary-180m-flash), [Cohere](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026).

## Результаты по моделям

### GigaAM v3 E2E RNN-T

На трех реальных встречах общей длительностью 43:37 текущая GigaAM v3 E2E RNN-T INT8 показала `11.54x RTFx`, `26.9%` сравнительного WER и 290 знаков пунктуации на 1000 слов. Внешние транскрипты были неточным ориентиром, поэтому WER полезен только для сравнения прогонов на этих же файлах. [Проектный benchmark](../benchmarks/2026-08-11-gigaam-model-comparison.md#сводка-по-трём-записям)

Сильные стороны baseline:

- зрелая интеграция через `onnx-asr`, Silero VAD и обязательные token/word timestamps ([текущий адаптер](../../src/local_transcriber/backends/onnx_asr.py));
- русский E2E-текст с регистром, пунктуацией и нормализацией;
- компактные INT8-веса и измеренная CPU-скорость.

Известная слабость: английские термины и названия передаются нестабильно. В benchmark встречались варианты `Fine BI`, `FNB`, `API`, `AP`, `пи`, а `edge cases` часто превращался в кириллическую фонетическую запись. Ценность нового кандидата для проекта зависит прежде всего от улучшения этого среза без потери обычной русской речи.

### Nemotron 3.5 ASR Streaming 0.6B

Точная модель: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b), checkpoint `nemotron-3.5-asr-streaming-0.6b-v1`, выпущен 4 июня 2026 года. Это Cache-Aware FastConformer-RNNT на 600 млн параметров с 32 готовыми locale и еще восемью после адаптации. `ru-RU`, `en-US` и `en-GB` входят в основной transcription-ready tier. Модель поддерживает offline-распознавание и streaming с задержкой от 80 ms до 1.12 s, пунктуацию, регистр и auto language ID. [Model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)

#### Практическая совместимость

Официальный [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) работает на CPU, CUDA, Vulkan и Metal. Он принимает файлы и поток, а JSON/SRT/VTT-вывод содержит слова с началом, концом и confidence. Q8 GGUF занимает 742 MB. Для первого прототипа достаточно subprocess и JSON; долгоживущий процесс или C API понадобятся позже, если загрузка модели на каждый файл окажется дорогой. Runtime пока молодой: v0.1.0 выпущен 19 августа 2026 года, его API еще может меняться. [CLI и формат вывода](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/cli.md#subtitles-and-structured-output), [релиз v0.1.0](https://github.com/NVIDIA/NeMo-Speech.cpp/releases/tag/v0.1.0)

Runtime также поддерживает per-request word boosting для cache-aware RNN-T. [Матрица customization](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/asr/customization.md#feature-matrix)

#### Ограничения опубликованных данных

- NVIDIA приводит для FLEURS Russian WER `9.17%` с `ru-RU` и `10.03%` с auto при контексте 1.12 s. Эти числа подтверждают рабочий русский, но не превосходство над GigaAM на встречах.
- В обучении были code-switched samples, однако отдельного теста для внутрифразовой смеси `ru/en` нет. [Обсуждение модели](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/discussions/2)
- H100 throughput из model card относится к высокой конкурентности. CPU-скорость и peak RSS на сопоставимом desktop-процессоре не опубликованы.
- Пунктуация и регистр встроены. Полная inverse text normalization требует отдельной grammar в NeMo-Speech.cpp; по числам, датам и сокращениям паритет с GigaAM пока не доказан. [HTTP API](https://github.com/NVIDIA/NeMo-Speech.cpp/blob/main/docs/api.md#post-v1audiotranscriptions)

Текущий `onnx-asr` Nemotron 3.5 не поддерживает. Alias в каталоге проекта недостаточен, нужен адаптер нового runtime.

#### Вывод

Nemotron ближе остальных новых кандидатов к полному контракту проекта. Он не доказал превосходство по качеству и CPU-скорости, но уже имеет практический путь к русскому тексту, длинным файлам и пословным таймкодам.

### Qwen3-ASR 0.6B

Точная upstream-модель: [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). Handy использует Q8 GGUF размером 811 MB, собранный из зафиксированной revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`. Несмотря на имя `0.6B`, Hugging Face показывает около 0.9 млрд параметров всего checkpoint: к Qwen3-0.6B добавлены аудиоэнкодер и projector. [Описание порта transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr-0.6b.md), [файлы upstream](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/main)

Модель поддерживает 30 языков, включая русский и английский, 22 китайских диалекта и автоматическое определение языка. Один checkpoint рассчитан на offline и streaming. [Model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)

#### Контекст запроса

Qwen использует audio-language-model архитектуру. Официальный runtime принимает свободную строку `context` на каждый запрос и помещает ее в системное сообщение. Так можно передать названия продуктов или тему встречи. В основном `transcribe.cpp` этот вход пока отсутствует; его добавляет открытый [PR #144](https://github.com/handy-computer/transcribe.cpp/pull/144). Наиболее удобный CPU-порт пока теряет эту возможность upstream. [Qwen inference](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py)

Отдельного результата по внутрифразовой смеси `ru/en` нет. Официальный русскоязычный пример monolingual; cross-language demo опубликован для 1.7B и другой группы языков. [Qwen3-ASR blog](https://qwen.ai/blog?id=qwen3asr)

Официальные aggregate WER для Qwen3-ASR 0.6B составляют `12.75` на CommonVoice, `15.84` на MLC-SLM и `7.57` на FLEURS. В эти наборы входит русский, но отдельный русский результат не опубликован. На тех же aggregate-срезах модель не всегда лучше Whisper large-v3: например, FLEURS `7.57` против `5.27`. Высокая отметка качества в Handy не доказывает выигрыш на русских встречах. [Evaluation table](https://huggingface.co/Qwen/Qwen3-ASR-0.6B#evaluation)

Официальные `2000x` относятся к vLLM BF16 при concurrency 128 и CUDA Graph. Это серверный GPU-throughput, не скорость одного файла на CPU. В transcribe.cpp на Ryzen 7 PRO 4750U Q8 показала около `4.1-4.6x` на двух коротких английских клипах, Vulkan около `7.3-8.7x`. Железо и аудио отличаются от проектного benchmark, поэтому прямое сравнение с `11.54x` GigaAM некорректно. Оно лишь объясняет отметку Handy о сравнительно низкой скорости. [Технический отчет](https://arxiv.org/abs/2601.21337), [локальные замеры порта](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr-0.6b.md#performance)

#### Таймкоды

Qwen3-ASR возвращает текст и язык без timestamps. Официальный путь использует отдельную [`Qwen3-ForcedAligner-0.6B`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), которая поддерживает русский и английский, заявлена для аудио до пяти минут и выдает word/character alignment. Актуальный Python wrapper режет timestamp-вход на трехминутные чанки. Для часовой встречи потребуется второй набор весов и дополнительный inference.

Handy/transcribe.cpp работает с Qwen offline и не предоставляет VAD, streaming или timestamps. Официальный streaming доступен только через vLLM, без batching и timestamps; реализация повторно подает накопленное аудио на каждом шаге, то есть не дает cache-aware поведения Nemotron. [Исходный streaming path](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_asr.py), [ограничения transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/qwen3-asr.md)

#### Варианты интеграции

- **OpenVINO 2026.3:** ранняя поддержка Qwen3-ASR на CPU/GPU уже заявлена. Проект использует совместимую ветку OpenVINO, но текущий backend построен на `WhisperPipeline`. Нужны новый `ASRPipeline`, экспорт модели и проверка, какие сегменты возвращает API. Отдельный вопрос таймкодов остается. [OpenVINO 2026.3 release](https://github.com/openvinotoolkit/openvino/releases/tag/2026.3.0), [supported models](https://github.com/openvinotoolkit/openvino.genai/blob/master/site/docs/supported-models/index.mdx)
- **Официальный Python runtime:** Transformers работает offline и поддерживает batch, но добавляет PyTorch/Transformers и не решает timestamps без aligner.
- **transcribe.cpp:** дает готовые GGUF и CPU/Vulkan/Metal/CUDA, но потребует нового backend, внешнего VAD и отдельного aligner.

#### Вывод

Qwen отвечает языковым требованиям, но не полному контракту результата: обязательные таймкоды добавляют второй модельный этап и усложняют long-form pipeline. Опубликованных данных недостаточно, чтобы оценить, оправдает ли качество текста эту цену.

### Модели без русского языка

#### Canary 180M Flash

[`nvidia/canary-180m-flash`](https://huggingface.co/nvidia/canary-180m-flash) содержит 182 млн параметров и поддерживает только английский, немецкий, испанский и французский. Handy показывает Q8 GGUF 208 MB. Модель умеет ASR на этих языках и перевод между английским и тремя остальными, но автоматического определения языка нет. [Документация Handy](https://handy.computer/docs/models), [порт transcribe.cpp](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/canary-180m-flash.md)

Upstream Canary умеет экспериментальные word/segment timestamps через `transcribe(..., timestamps=True)`. За этим API стоит вспомогательный CTC-aligner внутри полного `.nemo`. Для аудио длиннее 10 секунд NVIDIA рекомендует long-form inference с чанками по 10 секунд. Handy не портирует aligner; готовый ONNX-export для `onnx-asr` содержит только encoder/decoder и возвращает текст без нужных token timestamps. Это нарушает текущий контракт `local-transcriber`, который требует слова со временем. [Canary timestamp usage](https://huggingface.co/nvidia/canary-180m-flash), [NeMo implementation](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/aed_multitask_models.py), [заметки порта](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/canary-180m-flash.md)

Canary компактна и быстро работает на чистом `en/de/es/fr`, но русский для нее полностью out-of-domain. Она может быть интересна для отдельного English-first профиля, который не входит в область этого исследования.

#### Cohere Transcribe

[`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) имеет 2 млрд параметров и поддерживает 14 языков, среди которых нет русского. Автоопределения языка, timestamps и diarization нет; incremental audio streaming не документирован. Модель требует language tag, склонна распознавать non-speech и нуждается во внешнем VAD. [Strengths and limitations](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026#strengths-and-limitations)

Handy использует Q5 GGUF размером 1.77 GB. Это примерно в восемь раз больше ONNX-весов GigaAM. На чистом английском Cohere показывает сильные результаты, но они ничего не говорят о русском. В рассматриваемом русском сценарии тяжелый backend без timestamps не дает полезного профиля. [GGUF-порт](https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/cohere-transcribe-03-2026.md)

## Рабочие гипотезы

- Word boosting в Nemotron может улучшить продуктовые названия и IT-термины, которые GigaAM распознает нестабильно.
- Свободный `context` в официальном runtime Qwen может дать похожий выигрыш, если выбранный локальный путь сохранит этот вход.
- Мультиязычность Nemotron и Qwen может помочь с английскими вставками, но сама по себе не доказывает устойчивость внутрифразового `ru/en` code-switch.

Все три гипотезы следуют из возможностей моделей. Результатов на проектных русских встречах для них пока нет.

## Рекомендация

Предварительная рекомендация - пока сохранить GigaAM текущим baseline. Nemotron выглядит основным кандидатом для локального сравнения, поскольку поддерживает русский и уже имеет runtime с пословными таймкодами. Qwen стоит рассматривать вторым кандидатом на качество текста; полный timestamp-профиль имеет смысл только при заметном выигрыше распознавания.

Canary 180M Flash и Cohere Transcribe не стоит включать в сравнение русского профиля, поскольку русский язык у них не поддерживается. Рекомендация сама по себе не меняет конфигурацию или модель по умолчанию.

## Что неизвестно до локального прогона

- Выигрывают ли Nemotron или Qwen у GigaAM на длинной русской речи, внутрифразовых `ru/en` переключениях и IT-терминах. Опубликованные WER этого не показывают.
- Каковы CPU-скорость, время загрузки и peak RSS полного рабочего профиля. Для Qwen сюда входят ASR и ForcedAligner, если нужны таймкоды.
- Сохраняются ли конец записи, пунктуация, числа и названия продуктов после long-form разбиения.
- Дают ли Nemotron word boosting и Qwen `context` воспроизводимый выигрыш на терминах без ухудшения обычной русской речи.
- Проходит ли результат обязательный контракт пословных таймкодов на всей записи. Для Nemotron путь уже существует в NeMo-Speech.cpp; для Qwen его практичность еще не подтверждена.

Подробный корпус, команды, режимы и критерии приемки в эту записку не входят. Если сравнительный эксперимент будет принят в работу, для него понадобится отдельная benchmark spec.
