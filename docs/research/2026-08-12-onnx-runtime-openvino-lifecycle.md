# Куда движутся ONNX Runtime и OpenVINO: жизненный цикл и переносимость моделей

**Дата:** 2026-08-12

**Статус:** исследование для карты диаризации. Не архитектурное решение и не
основание для консолидации всех движков распознавания на ONNX Runtime.

## Вопрос и границы

Исследование отвечает на два связанных вопроса:

1. Насколько устойчивы ONNX Runtime (ORT), его аппаратные Execution Provider
   (EP), Windows ML и нативный стек OpenVINO/OpenVINO GenAI?
2. Что эти пути практически дают текущим моделям проекта — GigaAM E2E RNN-T,
   OpenVINO Whisper и связке диаризации PyAnnote + WeSpeaker — на Intel, AMD,
   Apple Silicon и в браузере?

Терминология следует [`CONTEXT.md`](../../CONTEXT.md): ORT, OpenVINO и OpenVINO
GenAI — **движки распознавания**. Обновление движка само по себе не меняет
поддерживаемую модель или модель по умолчанию. Архитектурная точка отсчёта —
независимые бэкенды из [ADR-003](../adr/003-pluggable-backends.md) и принятый
ONNX CPU-путь из [ADR-006](../adr/006-onnx-asr-backend.md).

Исследование дополняет [срез обновлений движков](2026-08-10-engine-model-updates.md)
и [разведку диаризации](../benchmarks/2026-08-12-diarization-feasibility.md).
Оно основано на первичных источниках: официальной документации, release notes,
репозиториях владельцев и фактических метаданных PyPI на 2026-08-12.

Вне границ документа:

- решение о консолидации проекта на одном runtime;
- выбор нового устройства или модели по умолчанию;
- обещание производительности без model-specific benchmark;
- разработка браузерной версии `local-transcriber`.

## Краткий ответ

- **Переносимый фундамент проекта — ONNX-артефакт плюс ORT CPU EP.** ORT core
  активно развивается и имеет наиболее широкую поставку для CPython 3.13:
  Windows и Linux x86-64/ARM64, macOS ARM64.
- **Жизненный цикл ядра ORT не переносится автоматически на каждый EP.**
  DirectML уже в sustained engineering, OpenVINO EP активен, но отстаёт от
  ORT и OpenVINO, CoreML остаётся Preview, а удалённый ROCm EP сменяется
  MIGraphX.
- **Долгосрочный Intel-путь — нативный OpenVINO/OpenVINO GenAI.** Он имеет
  собственную release/LTS policy, развивает Whisper и NPU и уже предоставляет
  word-level timestamps. OpenVINO EP полезен как мост для ONNX-моделей, но не
  даёт автоматически последние возможности нативного стека.
- **Новый Windows-слой — Windows ML, а не DirectML.** Windows ML остаётся ORT,
  но добавляет обнаружение устройств и управляемый каталог vendor EP. Для AMD
  там доступен MIGraphX; Python-приложению всё равно нужны bootstrap, загрузка
  и явная регистрация EP.
- **На AMD и Apple готовый пакет ещё не означает ускорение конкретной модели.**
  Linux AMD имеет wheel MIGraphX для CPython 3.13, Apple Silicon — CoreML EP в
  обычном ORT wheel; полный offload GigaAM и моделей диаризации не подтверждён
  ни для одного из этих путей.
- **Браузеры используют ту же архитектурную идею:** ORT Web даёт единый API,
  WASM — переносимый CPU baseline, WebGPU/WebNN — опциональные ускорители с
  ограниченным набором операторов и fallback. Сам ONNX-файл не устраняет
  различия preprocessing, decoding и доступных kernels.
- **Главная находка для карты диаризации:** OpenVINO GenAI уже умеет возвращать
  пословные таймкоды. Их отсутствие в результате `local-transcriber` — пробел
  проектного `Backend`/`TranscribeResult`, а не ограничение OpenVINO.

Общий принцип: поддержка пути доказана только тогда, когда подтверждены
поставка, создание сессии, фактическое размещение графа, сохранение выходного
контракта и end-to-end стоимость. Наличие wheel или имени EP закрывает только
первый из этих пунктов.

## Как устроены исследуемые слои

Сравниваемые названия относятся к разным уровням и не являются
взаимозаменяемыми пакетами.

| Слой | Роль | Что фиксирует приложение |
|---|---|---|
| ONNX | Формат графа, операторов и типов данных | Артефакт модели и opset ([ONNX About](https://onnx.ai/about)) |
| ONNX Runtime | Движок, который загружает ONNX-граф и распределяет узлы между EP | API сессии, версия ORT и порядок EP ([архитектура ORT](https://onnxruntime.ai/docs/reference/high-level-design.html)) |
| Execution Provider | Адаптер ORT к CPU, GPU или NPU; получает только поддержанные узлы/подграфы | Аппаратный runtime, provider options и CPU fallback ([архитектура EP](https://onnxruntime.ai/docs/execution-providers/)) |
| Windows ML | Windows-поставка ORT с каталогом, установкой и обновлением vendor EP | Windows App SDK, deployment mode и политика выбора EP ([обзор](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)) |
| OpenVINO | Runtime, компилятор и device plugins для CPU/GPU/NPU; читает в том числе ONNX | API OpenVINO, устройство и поддержанные форматы ([поддержанные модели](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)) |
| OpenVINO GenAI | Высокоуровневые pipelines поверх OpenVINO, включая Whisper и общий ASR API | OpenVINO IR, pipeline API и согласованные версии компонентов ([GenAI PyPI](https://pypi.org/project/openvino-genai/2026.3.0.0/)) |
| ORT Web | Отдельная JavaScript/WebAssembly-поставка ORT для браузера | JS API, WASM runtime и browser EP ([обзор ORT Web](https://onnxruntime.ai/docs/tutorials/web/)) |

Один ONNX-артефакт можно исполнять обычным ORT CPU EP, передавать его
поддержанные подграфы OpenVINO EP или загружать напрямую в OpenVINO. Результат
различается по покрытию операторов, квантованию, fallback и производительности
([ORT partitioning](https://onnxruntime.ai/docs/execution-providers/),
[OpenVINO EP coverage](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html),
[чтение ONNX в OpenVINO](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)).

### Лестница доказательства

Для каждой пары «модель × устройство × движок» используются пять уровней:

1. **Поставка:** существует совместимый wheel/runtime.
2. **Загрузка:** все модельные сессии создаются без ошибки.
3. **Размещение:** profiler показывает, какие узлы действительно исполняет EP,
   а какие ушли в CPU fallback.
4. **Контракт:** текст, таймкоды, сегменты и эмбеддинги остаются допустимыми.
5. **Пригодность:** end-to-end скорость, память и качество проходят проектную
   приёмку.

Ниже «подтверждено» означает прямое upstream-обещание или локальный результат;
«вывод» следует из архитектуры, но не проверен на конкретной модели;
«эксперимент» означает, что неизвестен хотя бы один уровень после поставки.

## Жизненный цикл движков и аппаратных путей

### ONNX Runtime core

ORT core активно развивается. Версии 1.26, 1.27 и 1.28 вышли 8 мая, 19 июня и
25 июля 2026 года; в них продолжалось развитие plugin EP API, ядра,
безопасности и аппаратных провайдеров
([1.26.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0),
[1.27.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.27.0),
[1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).

Официальные страницы расходятся в обещанном cadence: servicing-документ говорит
о full releases примерно раз в квартал, roadmap — о ежемесячных релизах и
промежуточных patch-релизах. Публичной LTS/EOL policy нет, поэтому текущий
почти месячный темп нельзя считать гарантией
([servicing](https://onnxruntime.ai/docs/reference/releases-servicing.html),
[roadmap](https://onnxruntime.ai/roadmap),
[support policy](https://github.com/microsoft/onnxruntime/blob/main/SUPPORT.md)).

С ORT 1.23 новые EP рекомендуется делать отдельными plugins. В 1.24–1.28 API
получил prepacking, EP Context, zero-copy I/O, profiling и model packages
([инструкция для нового EP](https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html),
[1.24.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1),
[1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)). Это
укрепляет ORT как общий движок, но одновременно отделяет lifecycle конкретного
ускорителя от lifecycle ядра.

### Нативный OpenVINO и OpenVINO GenAI

OpenVINO публикует несколько регулярных релизов в год. Каждый поддерживается до
следующего, а последняя версия года становится LTS: security updates выходят
два года либо до двух следующих LTS, исправления новых bugs — один год.
Preview-компоненты этой гарантией не покрываются
([release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html),
[release policy](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/release-policy.html)).

OpenVINO GenAI — pipeline-библиотека поверх OpenVINO и OpenVINO Tokenizers.
Их `major.minor.patch` должны совпадать; разъезд версий может привести к
ABI/import errors. PyPI wheel нельзя смешивать с C++ archive другого ABI
([правила совместимости](https://pypi.org/project/openvino-genai/2026.3.0.0/)).

Whisper остаётся активным направлением:

- OpenVINO 2026.0 добавил word-level timestamps в `WhisperPipeline` на CPU,
  GPU и NPU; 2026.3 добавил язык в результат
  ([2026.0](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-0-0),
  [2026.3](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-3-0));
- OpenVINO 2026.3 ввёл общий `ASRPipeline` и Qwen3-ASR, расширив speech API за
  пределы Whisper ([2026.3](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-3-0));
- удалён только ранее deprecated stateless Whisper decoder; рекомендуемый путь
  использует stateful model
  ([deprecations](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#deprecation-and-support)).

NPU — полноценное устройство OpenVINO, но требует отдельного driver, работает
со static shapes, а совместимость compiled blobs между версиями не
гарантируется. `WhisperPipeline` поддерживает NPU, однако целевой Core i5 11-го
поколения NPU не имеет: для него OpenVINO означает CPU/iGPU
([NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html),
[Whisper on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html#whisper-inference-on-npu),
[AUTO priority](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html)).

### Аппаратные пути ORT

| Путь | Состояние на 2026-08-12 | Практическое следствие |
|---|---|---|
| CPU EP | Часть ORT core, production baseline | Самая широкая поставка; аппаратного ускорителя не обещает |
| DirectML EP | Sustained engineering; feature development перешёл в Windows ML | Поддерживается, но не подходит как новый долгосрочный GPU default ([DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)) |
| Windows ML | Production-поставка ORT для Windows с управляемым каталогом EP | Стратегический Windows-слой, но требует platform-specific bootstrap ([deployment](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app)) |
| OpenVINO EP | Активен; deprecated только часть старых provider options | Мост к Intel-ускорению, но готовый wheel отстаёт от ORT/OpenVINO ([OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)) |
| MIGraphX EP | Активный AMD-путь; прежний ROCm EP удалён из ORT 1.23 | Долгосрочнее ROCm EP, но зависит от ROCm/GPU/OS ([ORT 1.23](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0)) |
| CoreML EP | Preview | Доступен в macOS ORT wheel, но требует проверки partitioning ([CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)) |
| Native WebGPU EP | Новый plugin поверх Dawn/D3D12/Vulkan/Metal | Кросс-вендорный кандидат; browser WebGPU использует другой runtime path ([WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html)) |

#### DirectML и Windows ML

DirectML EP использует DirectML 1.15.2, поддерживает ONNX только до opset 20 и
не допускает parallel execution одной session. Последний
`onnxruntime-directml` на дату среза — 1.24.4, тогда как ORT core уже 1.28.0.
Исправления DML всё ещё входят в ORT, то есть sustained engineering означает
поддержку без прежнего feature cadence, а не удаление
([DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html),
[PyPI](https://pypi.org/project/onnxruntime-directml/),
[ORT 1.28](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).

Windows ML не меняет формат модели и не заменяет ORT: runtime содержит
`onnxruntime.dll`, DirectML и Windows ML API. Новый слой добавляет обнаружение
устройств, каталог vendor EP, их установку, регистрацию и обновление. DirectML
остаётся встроенным legacy EP; MIGraphX, VitisAI, OpenVINO, QNN и
NvTensorRtRtx поставляются через каталог или вместе с приложением
([обзор](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview),
[состав runtime](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app),
[каталог EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)).

Python-пакет называется `onnxruntime-windowsml`. Версия
`1.27.1.202607110137` имеет статус `Production/Stable`, требует Python 3.11+ и
публикует `cp313` wheels для Windows x86-64 и ARM64
([PyPI](https://pypi.org/project/onnxruntime-windowsml/)). Отдельная ONNX
Runtime GenAI Windows ML library 0.x остаётся Preview; её статус не относится
к обычному ONNX-инференсу
([GenAI Preview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-genai-onnx-models)).

Для Python поддержан только framework-dependent unpackaged deployment: нужны
Windows App SDK Runtime и bootstrap packages. Динамический каталог аппаратных
EP требует Windows 11 24H2 build 26100+
([get started](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started),
[deployment](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app)).
Приложение должно скачать выбранный EP через `ensure_ready_async()` и
зарегистрировать библиотеку в ORT; `EnsureAndRegisterCertifiedAsync()` не
регистрирует EP в Python environment
([инициализация EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers)).

#### OpenVINO EP

OpenVINO EP не объявлен deprecated или maintenance-only. Intel продолжает
публиковать пакет, а ORT 1.26 и 1.28 содержат его изменения. Но последний
готовый `onnxruntime-openvino` 1.24.1 включает OpenVINO 2025.4.1 на Linux и
требует отдельный OpenVINO на Windows. Нативный OpenVINO уже достиг 2026.3, ORT
core — 1.28
([PyPI](https://pypi.org/project/onnxruntime-openvino/1.24.1/),
[матрица совместимости](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html),
[ORT 1.26](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0),
[ORT 1.28](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).

Следствие: EP остаётся рабочим мостом для ONNX-моделей, но не является способом
автоматически получить последние Whisper/NPU-возможности OpenVINO. Deprecated
provider options, заменённые `load_config`, не означают deprecation самого EP.

## Практическая поставка по платформам

Срез сделан по фактическим wheel, а не только по classifiers.

| Платформа и устройство | Готовый runtime/EP | Статус | `cp313` | Текущий CLI без новой интеграции |
|---|---|---|---|---|
| Intel x86 CPU | ORT CPU; OpenVINO CPU | Production | Да | Оба пути уже есть |
| Intel GPU/NPU | Нативный OpenVINO | Production; часть NPU-функций Preview | Да, Windows/Linux x86-64 | OpenVINO GPU есть; NPU потребует нового device profile |
| Windows, AMD CPU | ORT CPU | Production baseline | Да, `win_amd64` | Да |
| Windows, AMD GPU | DirectML | Sustained engineering | Да, `onnxruntime-directml` | Нужен новый provider/device UX |
| Windows 11 24H2+, AMD GPU | Windows ML + MIGraphX | Windows ML production; EP зависит от driver/device | Да, `onnxruntime-windowsml` | Нужны bootstrap и регистрация EP |
| Linux, AMD CPU | ORT CPU | Production baseline | Да, manylinux x86-64 | Да |
| Linux, AMD GPU | MIGraphX | Активная замена удалённого ROCm EP | Да, `onnxruntime-migraphx 1.27.1` | Нужны ROCm stack и provider integration |
| Apple Silicon, CPU | ORT CPU | Production baseline | Да, macOS 14 ARM64 | Да |
| Apple Silicon, GPU/ANE | CoreML EP | Preview | Да, в обычном ORT wheel | Нужны provider integration и profiling |
| Apple Silicon, CPU | OpenVINO GenAI Whisper | Production package; CPU-only на macOS | Да | Текущий dependency marker исключает macOS |

Обычный `onnxruntime` 1.28.0 поставляет `cp313` wheels для Windows x86-64 и
ARM64, Linux x86-64 и ARM64, macOS 14 ARM64
([files](https://pypi.org/project/onnxruntime/1.28.0/#files)). Для сравнения,
`onnxruntime-directml` 1.24.4 ограничен Windows x86-64, а
`onnxruntime-openvino` 1.24.1 — Windows/Linux x86-64
([DirectML files](https://pypi.org/project/onnxruntime-directml/1.24.4/#files),
[OpenVINO EP files](https://pypi.org/project/onnxruntime-openvino/1.24.1/#files)).

### AMD

На AMD x86 CPU поддерживаемая опора — ORT CPU EP. OpenVINO 2026.3 официально
перечисляет Intel и ARM/Apple CPU, но не AMD x86; наличие x86 wheel само по себе
не является обещанием поддержки AMD
([OpenVINO requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html)).

Под Windows DirectML поддерживает AMD GCN первого поколения и новее, но его
ограниченный lifecycle делает Windows ML + MIGraphX более перспективным путём.
Текущий Windows ML MIGraphX требует совместимый GPU/driver и не поддерживает
GenAI scenarios; применимость этой формулировки к GigaAM RNN-T не определена и
должна проверяться экспериментом
([Windows ML EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)).

Под Linux прежний ROCm EP удалён из ORT 1.23. Пакет `onnxruntime-rocm`
1.22.2.post3 всё ещё имеет `cp313`, но закреплён на ветке до удаления EP.
Активный `onnxruntime-migraphx` 1.27.1 публикует
`cp313-manylinux_2_34_x86_64`; реальные ограничения теперь лежат в ROCm/GPU/OS
и покрытии графа
([ROCm PyPI JSON](https://pypi.org/pypi/onnxruntime-rocm/json),
[MIGraphX PyPI JSON](https://pypi.org/pypi/onnxruntime-migraphx/json),
[MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)).

### Apple Silicon

ORT CPU — готовый baseline. `onnx-asr` документирует CoreML в обычном
`onnxruntime` package. CoreML EP может использовать CPU, GPU и Apple Neural
Engine через `MLComputeUnits`, но забирает только поддержанные подграфы.
Dynamic shapes могут быть дорогими; offload внутри `Loop`/`Scan`/`If` по
умолчанию выключен. `ProfileComputePlan` позволяет увидеть размещение
([onnx-asr installation](https://istupakov.github.io/onnx-asr/installation/),
[CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)).

OpenVINO/OpenVINO GenAI имеют `cp313-macosx_11_0_arm64` wheels и поддерживают
Apple Silicon, но на macOS исполняются только на CPU. GPU plugin рассчитан на
Intel GPU, NPU plugin — на Intel NPU
([OpenVINO requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html),
[OpenVINO GenAI files](https://pypi.org/project/openvino-genai/2026.3.0.0/)).

## Возможности текущих моделей

Таблица применяет одну и ту же лестницу доказательства к трем модельным путям.

| Модель и требуемый контракт | Переносимый baseline | Intel accelerator | AMD accelerator | Apple accelerator | Browser |
|---|---|---|---|---|---|
| GigaAM v3 E2E RNN-T: текст + token timestamps | **Подтверждено:** `onnx-asr` + ORT CPU на x86/ARM | OpenVINO EP или конверсия в IR — **эксперимент** | DirectML/WinML MIGraphX/Linux MIGraphX — **эксперимент** | CoreML — **эксперимент** | Нужен порт Python preprocessing/decoder и проверка kernels |
| OpenVINO GenAI Whisper: текст + word timestamps | Нативный OpenVINO CPU на поддержанных платформах | **Подтверждено:** OpenVINO CPU/GPU/NPU | AMD GPU не поддержан; AMD CPU не входит в official hardware | **Подтверждено:** только OpenVINO CPU | Это другой runtime/model artifact; не подтверждено |
| PyAnnote segmentation + WeSpeaker embeddings: интервалы + кластеры | **Подтверждено локально:** sherpa-onnx + ORT CPU на одной записи | OpenVINO EP/native — **эксперимент** | DML/MIGraphX — **эксперимент**, для sherpa может потребоваться rebuild | CoreML — **эксперимент** | sherpa имеет WASM demo, но выбранная пара моделей не подтверждена |

### GigaAM E2E RNN-T

`onnx-asr` работает на x86/ARM CPU и перечисляет CoreML, DirectML, ROCm и
WebGPU. GigaAM создаёт обычные ORT-сессии encoder/decoder/joint, поэтому смена
EP архитектурно возможна
([onnx-asr](https://istupakov.github.io/onnx-asr/),
[installation](https://istupakov.github.io/onnx-asr/installation/),
[model card](https://huggingface.co/istupakov/gigaam-v3-onnx)). Но это не
доказывает operator coverage или полный offload конкретного E2E RNN-T.

Таймкоды формирует `onnx-asr.with_timestamps()` из тензорных выходов модели.
Если EP сохраняет эти выходы, `TimestampedResult` должен сохраниться — это
**вывод**, который требует golden test. Mixed precision, graph transforms и
CPU fallback могут менять численные результаты
([timestamps API](https://istupakov.github.io/onnx-asr/usage/),
[архитектура пакета](https://github.com/istupakov/onnx-asr/tree/v0.12.0)).

Официальный ONNX helper исходного GigaAM проверяет только text parity и теряет
emission frames. Поэтому проверять нужно именно контракт `onnx-asr`, а не
произвольный GigaAM ONNX export
([GigaAM](https://github.com/salute-developers/GigaAM),
[ONNX parity test](https://github.com/salute-developers/GigaAM/blob/main/tests/test_onnx.py),
[ONNX helper](https://github.com/salute-developers/GigaAM/blob/main/gigaam/onnx_utils.py)).

### OpenVINO Whisper

OpenVINO GenAI подтверждает Whisper tiny/base/small/medium/large-v3 и
Distil-Whisper. Word timestamps доступны на CPU/GPU/NPU, stateful model
обязателен
([ASR guide](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/),
[supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)).
Это наиболее доказанный accelerator-путь из рассматриваемых, но только для
поддержанного OpenVINO hardware. На Apple Silicon он остаётся CPU-путём; на AMD
GPU не работает.

Проектный OpenVINO backend уже запрашивает timestamps, но сводит результат к
chunk-сегментам. Поэтому для диаризации нужно сначала определить и протянуть
word-level контракт через `Backend`/`TranscribeResult`.

### PyAnnote + WeSpeaker для диаризации

Локальная разведка доказала, что связка PyAnnote segmentation + WeSpeaker
embeddings создаёт интервалы на CPU ORT для одной записи. Это закрывает базовую
совместимость, но не upstream-гарантию пары и не переносимость на другие EP
([разведка](../benchmarks/2026-08-12-diarization-feasibility.md)).

Официальный sherpa recipe перечисляет PyAnnote с 3D-Speaker или NeMo
embeddings, а WeSpeaker публикует собственные ONNX-модели. Поэтому на новом EP
нужно отдельно проверять обе сессии и полный pipeline
([sherpa models](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html),
[WeSpeaker models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)).

Итоговые сегменты создаёт sherpa после двух ONNX-моделей и clustering. Ускорение
одной сессии не означает ускорение pipeline; численные изменения эмбеддингов
могут изменить кластеры даже при совпадающем текстовом контракте
([C API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/speaker_diarization.html)).

## Почему ONNX Runtime Web работает между браузерами

Браузерная переносимость появляется не из ONNX-файла отдельно, а из сочетания
трёх решений:

1. ONNX задаёт общий сериализованный граф.
2. ORT Web даёт один JavaScript `InferenceSession` API.
3. WASM служит широким CPU baseline, а WebGPU/WebNN подключаются как
   ускорители с fallback на WASM.

На 2026-08-12 официальный browser matrix выглядит так
([матрица](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)):

- WASM работает в Chrome/Edge, Safari и Firefox на основных desktop/mobile
  платформах и имеет наиболее полное покрытие операторов;
- WebGPU поддерживается Chromium на Windows/macOS/Android, остаётся experimental
  в ORT Web и имеет собственный operator subset;
- WebNN experimental и в официальной матрице требует feature flag в
  Chrome/Edge Windows; неподдержанные узлы могут уйти в WASM
  ([WebNN guide](https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html));
- WebGL находится в maintenance mode.

Один model artifact не означает одинаковую работоспособность. WebGPU имеет
отдельную [таблицу операторов](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md),
а preprocessing и decoding остаются кодом приложения. Большие модели упираются
примерно в 2 GB для ArrayBuffer/Protobuf и 4 GB WebAssembly memory; external
data нужно загружать отдельно
([large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)).
WASM threading требует `crossOriginIsolated`; proxy worker несовместим с
WebGPU, а dynamic shapes и CPU fallback ограничивают graph capture
([environment flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html),
[WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)).

`onnx-asr` заявляет WebGPU для **native Python package**. Это не browser port:
GigaAM потребует JavaScript preprocessing/decoder, загрузки нескольких
артефактов и проверки kernels. У sherpa-onnx есть отдельная однопоточная WASM
speaker-diarization demo, но она не доказывает работу проектной пары PyAnnote +
WeSpeaker через ORT Web WebGPU
([onnx-asr installation](https://istupakov.github.io/onnx-asr/installation/),
[sherpa JS diarization](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/javascript.html)).

Native WebGPU EP также не равен browser WebGPU: Python plugin использует Dawn
поверх D3D12/Vulkan/Metal, ORT Web — browser JSEP/WASM path
([native WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html),
[plugin PyPI JSON](https://pypi.org/pypi/onnxruntime-ep-webgpu/json)).

Полезный для CLI вывод из браузерной архитектуры — не новый продукт, а строгая
политика capabilities:

- всегда сохранять переносимый CPU baseline;
- обнаруживать ускоритель во время запуска;
- различать наличие API, успешную сессию, размещение графа и сохранение
  контракта;
- измерять end-to-end pipeline, а не отдельное имя provider.

## Что это меняет для `local-transcriber`

1. **CPU ORT остаётся переносимым baseline.** Он не зависит от затухающего EP и
   обеспечивает самый широкий CPython/platform coverage для GigaAM и
   диаризации.
2. **Нативный OpenVINO остаётся отдельным долгосрочным Intel ASR-путём.** Его
   не следует заменять OpenVINO EP только ради единого ORT API: EP отстаёт и не
   даёт автоматически pipeline-возможности OpenVINO GenAI.
3. **Пословная диаризация на `openvino-*` технически достижима.** Upstream уже
   возвращает word timestamps; карта должна решить контракт и fallback, а не
   считать отсутствие таймкодов свойством движка.
4. **AMD/Apple acceleration нельзя добавлять по факту наличия wheel.** Сначала
   нужны model-specific smoke/profile/golden tests; только затем device UX и
   dependency markers.
5. **Диаризацию не нужно связывать с немедленным выбором аппаратного EP.**
   Переносимый CPU-вариант может быть специфицирован независимо; ускорение двух
   моделей — отдельная работа.
6. **Консолидация на ORT из исследования не следует.** FasterWhisper сохраняет
   CUDA и языковое покрытие, нативный OpenVINO — актуальный Intel ASR API, ORT —
   переносимый ONNX-путь.

Для текущей карты это даёт два входа:

- [«Выбрать единицу привязки спикера к тексту»](https://git.dementev.space/ddmitry/local-transcriber/issues/14)
  должен назвать timestamp-aware изменение `Backend`/`TranscribeResult`;
- [«UX диаризации: флаг, число участников, зависимость и поведение на OpenVINO»](https://git.dementev.space/ddmitry/local-transcriber/issues/16)
  должен определить поведение там, где конкретный backend/model не отдаёт
  нужных таймкодов.

Реализация и приёмка WinML, MIGraphX, CoreML и browser-путей остаются за
пунктом назначения карты.

## Минимальная экспериментальная матрица

Будущий platform experiment должен использовать один 5–10-минутный fixture с
перекрывающейся речью и зафиксированным CPU output.

| Объект | Сравнение | Что фиксировать |
|---|---|---|
| GigaAM E2E RNN-T | ORT CPU против DirectML, WinML MIGraphX, Linux MIGraphX, CoreML | Создание всех сессий, node placement, CPU fallback, текст, token timestamps, численное расхождение |
| PyAnnote segmentation | Те же EP отдельно от остального pipeline | Placement, интервалы и расхождение выходных тензоров |
| WeSpeaker embeddings | Те же EP отдельно | Placement, cosine drift и влияние на clustering |
| Полная диаризация | CPU baseline против каждого прошедшего EP | Число спикеров, границы, стабильность кластеров, время и память |
| OpenVINO Whisper | Intel CPU/GPU/NPU и Apple CPU | Word timestamps, проектный adapter, время и память |
| Browser, только если станет целью | WASM baseline против WebGPU/WebNN | Размер артефактов, kernels/fallback, preprocessing/decoder и память |

Путь можно предлагать пользователю только после прохождения всех пяти уровней
доказательства; выигрыш отдельной ONNX-сессии не считается выигрышем
end-to-end транскрипции или диаризации.

## Открытые вопросы после исследования

### Внутри карты диаризации

- Какой точный word/token timestamp contract нужен formatter и объединению с
  интервалами спикеров?
- Как представить capability backend/model и какое fallback-поведение выбрать,
  если пословных таймкодов нет?

### Отдельные будущие работы

- Проходят ли GigaAM E2E RNN-T, PyAnnote и WeSpeaker все пять уровней на
  DirectML, Windows ML MIGraphX, Linux MIGraphX и CoreML?
- Насколько устойчива локально работающая пара PyAnnote + WeSpeaker между EP,
  если upstream recipe её не фиксирует?
- Окупает ли Windows ML bootstrap/catalog преимущество над низкофрикционным,
  но legacy DirectML?
- Даёт ли OpenVINO EP выигрыш моделям диаризации на целевом Intel Core i5 после
  учёта fallback, загрузки и памяти?
- Нужен ли отдельный browser experiment, или браузерная ветка остаётся только
  архитектурным примером переносимого baseline и опциональных ускорителей?
