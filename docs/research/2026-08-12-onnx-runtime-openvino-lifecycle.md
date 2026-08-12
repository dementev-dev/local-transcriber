# Куда движутся ONNX Runtime и OpenVINO: сравнение жизненного цикла

**Дата:** 2026-08-12

**Статус:** исследование для карты диаризации. Не архитектурное решение и не
основание для консолидации всех движков распознавания на ONNX Runtime.

## Вопрос

Насколько устойчивы ONNX Runtime, его DirectML и OpenVINO Execution Provider,
а также нативный стек OpenVINO/OpenVINO GenAI; какой из путей с большей
вероятностью сохранит Intel-ускорение и поддержку Whisper/NPU; что из этого
практически доступно проекту на CPython 3.13.

Исследование опирается только на первичные источники: официальную документацию,
release notes, репозитории владельцев и метаданные PyPI.

## Границы в контексте проекта

Терминология следует [`CONTEXT.md`](../../CONTEXT.md): ONNX Runtime, OpenVINO и
OpenVINO GenAI здесь — **движки распознавания**, их обновление само по себе не
меняет поддерживаемую модель или модель по умолчанию. Архитектурная точка
отсчёта — отдельные pluggable backends из [ADR-003](../adr/003-pluggable-backends.md)
и принятый ONNX CPU-путь из [ADR-006](../adr/006-onnx-asr-backend.md).

Исследование дополняет [срез обновлений движков](2026-08-10-engine-model-updates.md)
и отвечает на инфраструктурный вопрос, открытый
[разведкой диаризации](../benchmarks/2026-08-12-diarization-feasibility.md) и
[бэклогом](../backlog.md#диаризация--разделение-говорящих). Оно не пересматривает
качество моделей и не принимает решение о полной консолидации на ORT.

## Краткий вывод

Расширенный portability-срез не меняет исходный lifecycle-вывод, но уточняет его: ORT CPU остаётся наиболее ровным baseline на AMD Windows/Linux и Apple Silicon; Windows ML — уже production-поставка ORT для Windows с cp313, хотя vendor EP требуют bootstrap/registration; DirectML остаётся legacy; Linux AMD движется к MIGraphX; Apple accelerator-путь — CoreML Preview с обязательной проверкой partitioning. Наличие wheel/provider не подтверждает совместимость GigaAM или двух diarization graphs и тем более полный offload.

Браузер показывает тот же устойчивый pattern: ONNX — artifact, ORT Web — отдельный runtime, WASM — portable baseline, WebGPU/WebNN — optional accelerators с operator subset и fallback. Это portability evidence, а не предложение browser product или консолидации всего проекта на ORT.

1. **ONNX Runtime — активно развиваемый, production-стабильный движок, но не
   все его EP имеют одинаковый жизненный цикл.** Версии 1.26, 1.27 и 1.28
   вышли 8 мая, 19 июня и 25 июля 2026 года, то есть три minor-релиза примерно
   за одиннадцать недель; 1.28 продолжает развивать plugin EP API, ядро,
   безопасность и аппаратные EP ([1.26.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0),
   [1.27.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.27.0),
   [1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).
2. **DirectML EP поддерживается, но переведён в sustained engineering.** Новая
   функциональность Windows-пути перенесена в WinML; Microsoft рекомендует
   WinML для новых Windows-развёртываний, а DirectML EP оставляет для legacy и
   специальных сценариев ([официальная страница DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html),
   [Windows-путь ORT](https://onnxruntime.ai/docs/get-started/with-windows.html)).
   Поэтому `onnxruntime-directml` нельзя считать перспективным
   кросс-вендорным GPU-дефолтом проекта, хотя пакет не заброшен.
3. **OpenVINO и OpenVINO GenAI — основной активно развиваемый Intel-стек.** В
   2026 году регулярные релизы вышли 23 февраля, 7 апреля, 28 мая и 4 августа;
   OpenVINO публикует формальную release/LTS policy, где регулярная версия
   поддерживается до следующей, а последняя версия года становится LTS с двумя
   годами security updates ([release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html),
   [release policy](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/release-policy.html)).
4. **Для Intel-ускорения более долгоживущая ставка — сам OpenVINO, а не
   конкретная обвязка ORT OpenVINO EP.** EP остаётся активным мостом из ORT к
   OpenVINO, но зависит сразу от двух release train и его готовые wheel заметно
   отстают от обоих ядер. Нативный OpenVINO одновременно является runtime для
   CPU/GPU/NPU, имеет собственную LTS policy и служит основанием OpenVINO GenAI
   ([OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html),
   [GenAI как расширение runtime](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html)).
   Это не означает, что проекту нужно переносить ONNX-путь на OpenVINO или
   консолидироваться на ORT: ONNX-модели сохраняют переносимость, а выбор
   движка распознавания остаётся отдельным решением по качеству и контракту.
5. **CPython 3.13 не блокирует ни один из трёх исследованных PyPI-пакетов на
   целевой Windows x86-64**, но матрицы платформ радикально различаются:
   `onnxruntime` кроссплатформенный, DirectML только Windows x86-64,
   OpenVINO EP только Windows/Linux x86-64
   ([onnxruntime 1.28.0 files](https://pypi.org/project/onnxruntime/1.28.0/#files),
   [DirectML 1.24.4 files](https://pypi.org/project/onnxruntime-directml/1.24.4/#files),
   [OpenVINO EP 1.24.1 files](https://pypi.org/project/onnxruntime-openvino/1.24.1/#files)).

## Что именно является чем

Слои нельзя сравнивать как взаимозаменяемые пакеты:

| Слой | Роль | Что фиксирует приложение |
|---|---|---|
| ONNX | Формат графа, операторов и типов данных; операторы исполняются внешней реализацией | Артефакт модели и его opset ([ONNX About](https://onnx.ai/about)) |
| ONNX Runtime | Движок выполнения ONNX-графа, который разбивает его между EP и CPU fallback | API сессии, версия ORT и набор EP ([архитектура ORT](https://onnxruntime.ai/docs/reference/high-level-design.html)) |
| OpenVINO | Intel runtime, компилятор и device plugins для CPU/GPU/NPU; умеет принимать в том числе ONNX-графы | API OpenVINO и поддерживаемые устройства/форматы ([поддержанные модели](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)) |
| OpenVINO EP | Адаптер внутри ORT: получает поддержанные подграфы, переводит и компилирует их для OpenVINO | Одновременно контракты ORT, EP и совместимой версии OpenVINO ([EP architecture](https://onnxruntime.ai/docs/execution-providers/), [OpenVINO EP](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)) |
| OpenVINO GenAI | Высокоуровневые генеративные pipelines поверх OpenVINO runtime, включая Whisper и общий ASR API | Формат моделей OpenVINO IR, pipeline API и согласованные версии OpenVINO/Tokenizers/GenAI ([GenAI PyPI](https://pypi.org/project/openvino-genai/2026.3.0.0/)) |

Следствие: **модель в ONNX не означает ONNX Runtime**, а **OpenVINO EP не
является форматом модели**. Один ONNX-артефакт можно исполнять CPU EP в ORT,
передавать поддержанные подграфы OpenVINO EP либо загружать в OpenVINO
напрямую; однако покрытие операторов, квантование, fallback и производительность
у этих путей различаются ([ORT EP partitioning](https://onnxruntime.ai/docs/execution-providers/),
[OpenVINO EP support coverage](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html),
[прямое чтение ONNX в OpenVINO](https://docs.openvino.ai/2026/openvino-workflow/model-preparation/convert-model-onnx.html)).

## ONNX Runtime и execution providers

### Ядро ORT

Официальные страницы расходятся в обещанном cadence: servicing-документ всё
ещё говорит о full releases «примерно ежеквартально», тогда как roadmap — о
ежемесячных релизах и patch-релизах между ними. Формального LTS/EOL-окна в
публичной support policy нет. Поэтому для планирования надёжнее опираться на
фактические публикации и backward-compatibility policy, а не превращать
текущий почти месячный темп в гарантию
([releases and servicing](https://onnxruntime.ai/docs/reference/releases-servicing.html),
[roadmap](https://onnxruntime.ai/roadmap),
[support policy](https://github.com/microsoft/onnxruntime/blob/main/SUPPORT.md)).

ORT 1.23 начал переход к независимо подключаемым plugin EP и прямо рекомендует
новые EP реализовывать как plugins, а не добавлять внутрь ядра. В 1.24–1.28
plugin API последовательно получал prepacking, EP Context, zero-copy I/O,
profiling и model packages ([инструкция для нового EP](https://onnxruntime.ai/docs/execution-providers/add-execution-provider.html),
[релиз 1.24.1](https://github.com/microsoft/onnxruntime/releases/tag/v1.24.1),
[релиз 1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).
Это сильный сигнал продолжения ORT как общего движка, но одновременно сигнал,
что жизненный цикл конкретного аппаратного backend всё больше принадлежит его
поставщику, а не ядру ORT.

### DirectML EP

Официальная формулировка однозначна: DirectML находится в **sustained
engineering**, поддержка продолжается, но feature development перешёл в WinML.
Документация также фиксирует DirectML 1.15.2 и покрытие только до ONNX opset 20;
модели с более высоким требованием официально не поддерживаются
([DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)).

Это согласуется с поставкой: последний `onnxruntime-directml` на дату среза —
1.24.4 от 17 марта, тогда как ядро ORT уже 1.28.0. При этом ORT 1.28 всё ещё
содержит исправление DML readback, то есть sustained engineering означает не
«удалён», а «исправления без прежнего темпа новых возможностей»
([DirectML на PyPI](https://pypi.org/project/onnxruntime-directml/),
[ORT 1.28, DML fix](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).

Для проекта DirectML остаётся возможным Windows-only экспериментом на AMD,
Intel и NVIDIA GPU, но его стратегический successor — WinML, который требует
Windows-специфической интеграции. Это слабее текущего требования ADR-003 о
плаггируемых бэкендах и кроссплатформенном ONNX CPU-пути.

### OpenVINO EP

Официальная документация не объявляет OpenVINO EP deprecated или maintenance-only.
Наоборот, Intel публикует готовые пакеты, принимает issues/PR, заявляет CPU,
интегрированные и дискретные GPU и NPU, а ORT 1.26 и 1.28 содержат OpenVINO EP
development updates ([страница пакета](https://pypi.org/project/onnxruntime-openvino/),
[ORT 1.26](https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0),
[ORT 1.28](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0)).

Но готовая поставка имеет свой темп. Последний wheel `onnxruntime-openvino`
1.24.1 от 26 февраля 2026 года включает OpenVINO 2025.4.1 на Linux и требует
отдельной установки OpenVINO на Windows. Официальная таблица совместимости
покрывает только три версии OpenVINO: ORT-EP 1.22/2025.1,
1.23/2025.3 и 1.24.1/2025.4.1
([PyPI](https://pypi.org/project/onnxruntime-openvino/1.24.1/),
[матрица совместимости](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)).
На дату среза нативный OpenVINO уже 2026.3, а ядро ORT — 1.28. Значит, EP
активен, но готовый Python-путь не является способом автоматически получить
самые новые возможности OpenVINO/NPU.

Начиная с ORT 1.23 часть старых provider options OpenVINO EP deprecated в
пользу `load_config` с нативными OpenVINO properties. Это локальная миграция
конфигурации, а не deprecation самого EP
([deprecation notice](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)).

## OpenVINO, OpenVINO GenAI, Whisper и Intel NPU

OpenVINO имеет явно описанный цикл: несколько регулярных релизов в год,
поддержка каждого до следующего и ежегодный LTS. LTS получает security updates
два года либо до двух следующих LTS, а исправления новых bugs — один год;
preview-компоненты этой гарантией не покрываются
([release policy](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/release-policy.html)).

OpenVINO GenAI — не конкурирующий runtime, а библиотека pipelines поверх
OpenVINO и OpenVINO Tokenizers. Их `major.minor.patch` должны совпадать:
разъезд может привести к ABI/import errors; PyPI wheel нельзя смешивать с C++
archives другого ABI ([официальные правила совместимости](https://pypi.org/project/openvino-genai/2026.3.0.0/)).

Whisper — активный, а не legacy use case OpenVINO GenAI:

- OpenVINO 2026.0 добавил word-level timestamps в WhisperPipeline на CPU, GPU
  и NPU; в 2026.3 результаты также содержат определённый/заданный язык, а NPU
  отдаёт word timestamps по умолчанию
  ([2026.0 release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-0-0),
  [2026.3 release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-3-0));
- OpenVINO 2026.3 ввёл общий `ASRPipeline` и поддержку Qwen3-ASR, то есть
  speech API расширяется за пределы Whisper
  ([2026.3 release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-3-0));
- удалён только ранее deprecated **stateless decoder** Whisper; рекомендуемый
  путь — stateful model, а не отказ от Whisper
  ([deprecation section](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#deprecation-and-support)).

NPU является первым классом устройств OpenVINO: NPU plugin доступен в
дистрибутивах, целевая аппаратная платформа начинается с Intel Core Ultra,
Compiler-In-Plugin появился preview в 2026.0 и стал предпочитаемым компилятором
в 2026.1. При этом NPU требует отдельный driver, поддерживает только static
shapes, а совместимость предкомпилированных blobs между версиями OpenVINO не
гарантируется ([NPU device](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)).
WhisperPipeline официально работает на NPU с tiny/base/small/large без
специальных ограничений pipeline, но документация рекомендует актуальный NPU
driver и даёт workaround для memory failures
([Whisper on NPU](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html#whisper-inference-on-npu)).

Это не делает NPU актуальным для целевого Intel Core i5 11-го поколения: NPU
появился только в Core Ultra. Для текущей целевой машины долгоживущий
OpenVINO-путь означает прежде всего CPU/iGPU; NPU — будущий аппаратный профиль,
который нужно выбирать явно, тем более что OpenVINO AUTO пока исключает NPU из
дефолтного приоритета
([NPU hardware](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html),
[AUTO priority](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html)).

## PyPI: версии, платформы и CPython 3.13

Срез сделан по фактически опубликованным wheel, а не по classifiers страницы.
У всех трёх пакетов отсутствует source distribution, поэтому неподдержанная
комбинация платформы и Python не сможет штатно собраться через обычный
`pip install` без самостоятельной сборки из репозитория.

| Пакет | Последняя версия на 2026-08-12 | Последняя публикация | wheel для CPython 3.13 | Платформы cp313 |
|---|---:|---:|---|---|
| `onnxruntime` | 1.28.0 | 2026-07-25 | Да | Windows x86-64 и ARM64; Linux x86-64 и ARM64 (glibc 2.27/2.28+); macOS 14+ ARM64 ([files](https://pypi.org/project/onnxruntime/1.28.0/#files)) |
| `onnxruntime-directml` | 1.24.4 | 2026-03-17 | Да | Только Windows x86-64; нет Linux, macOS и Windows ARM64 wheel ([files](https://pypi.org/project/onnxruntime-directml/1.24.4/#files)) |
| `onnxruntime-openvino` | 1.24.1 | 2026-02-26 | Да | Windows x86-64 и Linux x86-64 с glibc 2.28+; нет ARM64 и macOS wheel ([files](https://pypi.org/project/onnxruntime-openvino/1.24.1/#files)) |

Практический вывод для ADR-003 сохраняется: обычный `onnxruntime` остаётся
широким zero-config CPU-путём. DirectML и OpenVINO EP нельзя подставить как одну
безусловную зависимость на всех платформах; они требуют platform markers и
отдельных проверок доступного provider. На Windows OpenVINO EP дополнительно
требует совместимый `openvino`, а на Linux wheel уже включает конкретный
OpenVINO 2025.4.1
([OpenVINO EP installation](https://pypi.org/project/onnxruntime-openvino/1.24.1/)).

## Windows ML: следующий Windows-слой над ORT

### Что доступно сейчас, а что пока preview

> Artifact caveat: Microsoft Learn ещё указывает Python 3.10–3.13, тогда как текущий `onnxruntime-windowsml` требует Python 3.11+ и уже имеет cp314. Для установки источником истины служит фактическая wheel matrix PyPI; cp313 подтверждён для x64 и ARM64 ([PyPI JSON](https://pypi.org/pypi/onnxruntime-windowsml/json)).

**Подтверждено.** Windows ML — не новый формат модели и не замена ONNX Runtime: Microsoft описывает его как Windows-фреймворк локального инференса, **powered by ONNX Runtime**. Runtime содержит `onnxruntime.dll`, DirectML и Windows ML API; обычный инференс по-прежнему создаёт ORT `InferenceSession`. Windows ML добавляет обнаружение устройств, каталог совместимых vendor EP, их установку, регистрацию и обновление. DirectML остаётся встроенным legacy GPU EP; MIGraphX/VitisAI/OpenVINO/QNN/NvTensorRtRtx поставляются через каталог или вместе с приложением ([обзор](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview), [состав и deployment](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app), [API](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/api-reference)).

Для Python реальная поставка называется `onnxruntime-windowsml`, а не анонсированная ранее `onnxruntime-winml`. На 2026-08-12 последняя версия — `1.27.1.202607110137`, статус PyPI `Production/Stable`, `Requires-Python >=3.11`; есть `cp313` wheels для `win_amd64` и `win_arm64` ([PyPI](https://pypi.org/project/onnxruntime-windowsml/)). Это уже доступный продуктовый путь. Отдельная **ONNX Runtime GenAI Windows ML library 0.x** прямо обозначена как Preview; её нельзя переносить на статус обычного ONNX-инференса ([GenAI Preview](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/run-genai-onnx-models)).

Python поддержан на 3.10–3.13, x64/ARM64, но только как framework-dependent unpackaged app: нужны Python с python.org/winget, Windows App SDK Runtime и bootstrap-пакеты. Self-contained deployment для Python не предусмотрен. Базовый runtime может работать на поддерживаемых Windows 10, но динамический каталог аппаратных EP требует Windows 11 24H2, build 26100+ ([getting started](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/get-started), [deployment](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/distributing-your-app), [поддерживаемые EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)).

**Существенный caveat для CLI.** Каталог скачивает и обновляет EP, однако Python-приложение должно перечислить подходящие EP, вызвать `ensure_ready_async()` и зарегистрировать библиотеку через `onnxruntime.register_execution_provider_library`. Microsoft отдельно предупреждает, что `EnsureAndRegisterCertifiedAsync()` не регистрирует EP в Python ORT environment ([инициализация EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/initialize-execution-providers), [выбор EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers)). Поэтому:

- `onnxruntime-windowsml` + встроенный DirectML — практически применимая замена ORT-дистрибутива для `onnx-asr`; upstream 0.12 документирует именно эту установку и тот же providers API ([installation](https://istupakov.github.io/onnx-asr/installation/));
- vendor EP из каталога (например, MIGraphX) не появится в существующем `local-transcriber` без Windows App SDK bootstrap/registration glue;
- device policy (`MAX_PERFORMANCE`, `PREFER_NPU` и другие) — пожелание к выбору, а не доказательство полного offload конкретной модели. Нужны capability discovery, session profiling и проверка fallback.

## AMD и Apple: практические пути без кастомной сборки

### AMD CPU и GPU

На AMD x86 CPU обычный `onnxruntime` использует portable CPU EP: это готовый baseline на Windows и Linux. OpenVINO 2026.3 официально перечисляет Intel и ARM/Apple CPU, но **не перечисляет AMD x86 в supported hardware**; наличие x86 wheel ещё не является обещанием поддержки AMD CPU ([OpenVINO system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html)). Поэтому OpenVINO на AMD x86 — только экспериментальный путь, не поддерживаемая опора проекта.

На AMD GPU под Windows есть два готовых пути:

1. `onnxruntime-directml` 1.24.4 (`cp313-win_amd64`): DirectML официально поддерживает AMD GCN первого поколения и новее, но находится в sustained engineering, ограничен ONNX opset 20 и не гарантирует полный offload ([DirectML EP](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)).
2. Windows ML 2.x: встроенный DirectML или загружаемый MIGraphX. Каталог MIGraphX доступен только на Windows 11 24H2+ и при совместимых GPU/driver; Microsoft отдельно отмечает, что текущий MIGraphX EP не поддерживает GenAI scenarios ([Windows ML EP](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers)). Для GigaAM RNN-T граница понятия GenAI в этой таблице не определена — нужен локальный тест, а не перенос ограничения по аналогии.

На AMD GPU под Linux прежний ROCm EP удалён из source tree начиная с ORT 1.23; Microsoft рекомендует MIGraphX или VitisAI ([ORT 1.23](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.0)). Старый `onnxruntime-rocm` всё ещё публикуется на PyPI (последний `1.22.2.post3`, включая `cp313`), но остаётся на ветке до удаления EP и потому не является долгоживущим направлением ([PyPI JSON](https://pypi.org/pypi/onnxruntime-rocm/json)). Активный `onnxruntime-migraphx` уже имеет версию `1.27.1` и `cp313-manylinux_2_34_x86_64`; это готовый wheel, хотя он требует совместимых ROCm/GPU/OS и отличается от core ORT 1.28 ([PyPI JSON](https://pypi.org/pypi/onnxruntime-migraphx/json), [MIGraphX EP](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)). Следовательно, Python 3.13 больше не блокирует установку; реальными неизвестными остаются системный ROCm stack и совместимость конкретных графов.

### Apple Silicon

`onnxruntime` 1.28.0 публикует `cp313` wheel для macOS 14 ARM64; это готовый CPU baseline. Тот же macOS build доступен `onnx-asr`, который документирует `CPUExecutionProvider` и `CoreMLExecutionProvider` в обычном пакете ([ORT Python install](https://onnxruntime.ai/docs/get-started/with-python.html), [onnx-asr installation](https://istupakov.github.io/onnx-asr/installation/)).

CoreML EP может задействовать CPU, GPU и Apple Neural Engine через `MLComputeUnits`. Он забирает поддерживаемые subgraphs, допускает динамические shapes, но предупреждает об их возможной цене; для `Loop`/`Scan`/`If` offload внутри тела по умолчанию выключен. Параметры `RequireStaticInputShapes`, `EnableOnSubgraphs` и `ProfileComputePlan` позволяют проверить фактическое размещение. Сам EP в общей таблице ORT всё ещё помечен Preview ([CoreML EP](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html), [таблица EP](https://onnxruntime.ai/docs/execution-providers/)). Следовательно, «CoreML доступен» не означает «GigaAM целиком работает на ANE».

OpenVINO 2026.3 и OpenVINO GenAI 2026.3 имеют `cp313-macosx_11_0_arm64` wheels и официально поддерживают Apple silicon, но на macOS OpenVINO выполняет inference только на CPU; GPU plugin поддерживает только Intel GPU, NPU plugin — Intel NPU ([system requirements](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino/system-requirements.html), [openvino-genai PyPI](https://pypi.org/project/openvino-genai/2026.3.0.0/)). Значит OpenVINO Whisper на Apple Silicon технически поставляется без сборки, но не использует GPU/ANE. Текущий marker проекта, исключающий OpenVINO extra на macOS, остаётся отдельным integration constraint.

### Сводная матрица runtime/device

| Платформа и устройство | Готовый runtime/EP | Статус на 2026-08-12 | `cp313` | Без кастомной сборки в текущем CLI |
|---|---|---|---|---|
| Windows, AMD CPU | ORT CPU EP | production baseline | да, `win_amd64` | да |
| Windows, AMD GPU | ORT DirectML | sustained engineering | да, `onnxruntime-directml` | пакет есть; нужен новый provider/device UX |
| Windows 11 24H2+, AMD GPU | Windows ML + DirectML/MIGraphX | Windows ML production; catalog EP зависит от driver/device | да, `onnxruntime-windowsml` | DirectML близко к готовому; MIGraphX требует bootstrap/registration |
| Linux, AMD CPU | ORT CPU EP | production baseline | да, `manylinux x86_64` | да |
| Linux, AMD GPU | ORT MIGraphX | active replacement for removed ROCm EP | да, `onnxruntime-migraphx 1.27.1` | wheel есть; нужны ROCm compatibility и model smoke-test |
| Apple Silicon, CPU | ORT CPU EP | production baseline | да, macOS 14 ARM64 | да |
| Apple Silicon, GPU/ANE | CoreML EP | preview; partial partitioning possible | да, в обычном ORT wheel | пакет есть; требуется provider integration и profiling |
| Apple Silicon, CPU | OpenVINO GenAI Whisper | production package; CPU-only на macOS | да | upstream да; текущий project extra/marker — нет |

## Матрица моделей: что действительно переносится

Легенда: **подтверждено** — есть прямое upstream-обещание/поставка; **вывод** — следует из одинакового ONNX/ORT контракта, но нет проверки данной модели; **эксперимент** — session/model compatibility и offload неизвестны.

| Модель/контракт | ORT CPU (AMD Win/Linux, Apple) | AMD GPU Windows (DML/WinML) | AMD GPU Linux (MIGraphX) | Apple GPU/ANE (CoreML) | OpenVINO native |
|---|---|---|---|---|---|
| GigaAM v3 E2E RNN-T ONNX + token timestamps | **Подтверждено** upstream `onnx-asr` на x86/Arm CPU и готовыми cp313 wheels. `with_timestamps()` — API `onnx-asr` ([usage](https://istupakov.github.io/onnx-asr/usage/), [model card](https://huggingface.co/istupakov/gigaam-v3-onnx)) | `onnx-asr` заявляет DirectML/WebGPU support и документирует Windows ML package; **эксперимент** для конкретного E2E RNN-T: session creation, доля DML/MIGraphX, точность/timestamps | cp313 MIGraphX wheel есть; provider допустим как произвольная строка, но не first-class/tested; **эксперимент** для graph coverage и output | CoreML заявлен `onnx-asr`; **эксперимент** для dynamic encoder/decoder, control flow и доли ANE | OpenVINO EP package lagging; native IR не является тем же artifact; **эксперимент/конверсия** |
| OpenVINO GenAI Whisper + word timestamps | не тот runtime/model artifact | OpenVINO GPU не работает на AMD GPU; CPU path возможен лишь там, где CPU официально поддержан | то же | OpenVINO CPU на Apple silicon **подтверждён**; GPU/ANE нет | **Подтверждено** для Whisper tiny/base/small/medium/large-v3 и Distil-Whisper; word timestamps доступны CPU/GPU/NPU, stateful model обязателен ([ASR guide](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/speech-recognition/), [supported models](https://openvinotoolkit.github.io/openvino.genai/docs/supported-models/)) |
| sherpa-onnx pyannote segmentation + WeSpeaker embeddings | **Подтверждено локальной разведкой** на CPU ORT для одной записи; cp313 wheels есть на Windows/Linux/macOS ARM64 ([разведка](../benchmarks/2026-08-12-diarization-feasibility.md), [PyPI 1.13.5](https://pypi.org/project/sherpa-onnx/1.13.5/)) | sherpa имеет DirectML build option, но готовый Python wheel/provider и именно эти две модели на DML не подтверждены: **эксперимент/возможно rebuild** | готовая sherpa Python поставка с MIGraphX не подтверждена; **не готово** | upstream Python provider vocabulary обычно ограничивает `cpu,cuda,coreml`; наличие CoreML в wheel не доказывает поддержку diarization graphs: **эксперимент** | модели ONNX теоретически читаются OpenVINO, но полное/частичное покрытие и численная стабильность clustering inputs не подтверждены: **эксперимент** |

Почему timestamps должны пережить смену EP — это **вывод**, а не готовая совместимость: `onnx-asr` сохраняет в Python preprocessing и greedy decoding, а EP исполняет ONNX encoder/decoder; одинаковые тензорные выходы должны дать тот же `TimestampedResult` ([описание архитектуры](https://github.com/istupakov/onnx-asr/tree/v0.12.0), [timestamps API](https://istupakov.github.io/onnx-asr/usage/)). Но mixed precision, unsupported ops/fallback и provider-specific graph transforms требуют golden-output проверки. Для diarization выходной контракт сегментов создаётся sherpa pipeline после двух ONNX-моделей и clustering ([C API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/speaker_diarization.html)); ускорение одной модели не должно считаться ускорением всего pipeline.

Отдельный support gap: официальный recipe sherpa подтверждает PyAnnote
segmentation с 3D-Speaker или NeMo embeddings, но не с выбранным в разведке
WeSpeaker. Локальный CPU-прогон уже доказал, что эта пара создаёт интервалы на
одной записи; неизвестны upstream-гарантия контракта и переносимость на другие
EP, а не базовая совместимость CPU-пути
([разведка](../benchmarks/2026-08-12-diarization-feasibility.md),
[sherpa models](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html),
[WeSpeaker pretrained models](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md)).

GigaAM timestamp caveat: upstream GigaAM `transcribe(..., word_timestamps=True)` возвращает слова со start/end, но его официальный ONNX helper экспортирует encoder/decoder/joint и проверяет только text parity; helper возвращает `List[str]` и теряет emission frames ([GigaAM repository](https://github.com/salute-developers/GigaAM), [ONNX parity test](https://github.com/salute-developers/GigaAM/blob/main/tests/test_onnx.py), [ONNX helper](https://github.com/salute-developers/GigaAM/blob/main/gigaam/onnx_utils.py)). Timestamped contract текущего проекта даёт именно `onnx-asr.with_timestamps()`, а не произвольный GigaAM ONNX export. Поэтому golden test должен сравнивать project/onnx-asr contract, не только распознанный текст.

Минимальная будущая экспериментальная матрица без заявления производительности:

1. Один фиксированный 5–10-минутный fixture с overlap и эталоном текущего CPU output.
2. Для GigaAM E2E RNN-T: ORT CPU против DML, WinML MIGraphX, CoreML и MIGraphX Linux; фиксировать session creation, provider assignment/profile, CPU fallback, transcript/token timestamps и численное расхождение.
3. Для diarization: отдельно pyannote segmentation и WeSpeaker embeddings, затем полный pipeline; фиксировать provider assignment каждой сессии, сегменты/число спикеров и стабильность embeddings/clustering.
4. Для OpenVINO Whisper: CPU на Apple и поддерживаемом Intel, GPU/NPU только на Intel; проверять word timestamps и project adapter отдельно.

## Почему ONNX Runtime Web стал общим браузерным слоем

ONNX остаётся переносимым serialized graph/IR, а `onnxruntime-web` — отдельным JavaScript/WebAssembly runtime. Общность браузеров даёт не ONNX-файл сам по себе, а единый ORT JS API поверх разных EP: `wasm` как default CPU baseline, `webgpu`, `webnn` и legacy `webgl`. ORT распределяет поддерживаемые nodes/subgraphs на accelerator, а неподдерживаемые может оставить WASM, если он указан вторым provider ([web overview](https://onnxruntime.ai/docs/tutorials/web/), [session options](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)). Это та же архитектурная идея, что native ORT, но runtime binaries и provider kernels другие.

На 2026-08-12 официальный browser matrix таков ([matrix](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)):

- WASM: Chrome/Edge, Safari, Firefox на основных desktop/mobile платформах; полный набор ONNX operators, CPU baseline;
- WebGPU: Chromium 113+ на Windows, Chromium на macOS/Android; в ORT Web всё ещё обозначен experimental, operator subset;
- WebNN: experimental и не включён по умолчанию; официальный matrix требует feature flag в Chrome/Edge Windows; unsupported ops fall back to WASM ([WebNN guide](https://onnxruntime.ai/docs/tutorials/web/ep-webnn.html));
- WebGL: maintenance mode, operator subset.

Один ONNX artifact и близкий `InferenceSession` contract можно использовать native и web, но «один artifact» не означает одинаковую работоспособность. Для WebGPU опубликована отдельная operator table ([WebGPU operators](https://github.com/microsoft/onnxruntime/blob/main/js/web/docs/webgpu-operators.md)); accelerator EP поддерживают лишь subset, а WASM — все операторы. Большие модели ограничены браузером: около 2 GB для ArrayBuffer/Protobuf, 4 GB WebAssembly memory; external data нужно передавать URL/Blob явно ([large models](https://onnxruntime.ai/docs/tutorials/web/large-models.html)). WASM threading включается только при `crossOriginIsolated`; proxy worker не совместим с WebGPU и CSP-restricted environment ([env flags](https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html)). Dynamic shapes и CPU fallback также исключают некоторые оптимизации WebGPU graph capture ([WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)).

`onnx-asr` 0.12 заявляет WebGPU support для **native Python package** и называет `onnxruntime-webgpu` beta; это не browser JavaScript port `onnx-asr` ([installation](https://istupakov.github.io/onnx-asr/installation/)). Для GigaAM E2E RNN-T browser compatibility не подтверждена: нужны JS preprocessing/decoder либо порт Python-логики, загрузка нескольких model artifacts, проверка WebGPU operator coverage и WASM fallback. У sherpa-onnx есть отдельная WebAssembly speaker-diarization сборка и JS example, но она однопоточная и не доказывает, что выбранные проектом pyannote + WeSpeaker models работают через ORT Web WebGPU ([sherpa JS diarization](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/javascript.html), [build option](https://github.com/k2-fsa/sherpa-onnx/blob/master/CMakeLists.txt)).

Native WebGPU и browser WebGPU нельзя смешивать: native Python EP использует Dawn поверх D3D12/Vulkan/Metal и теперь поставляется plugin-пакетом `onnxruntime-ep-webgpu` (0.2.1, universal wheels), тогда как ORT Web использует browser JSEP/WASM path ([native WebGPU EP](https://onnxruntime.ai/docs/execution-providers/WebGPU-ExecutionProvider.html), [plugin PyPI JSON](https://pypi.org/pypi/onnxruntime-ep-webgpu/json)).

Практический урок для CLI — не новый browser product, а более строгая portability-модель:

- сохранять portable CPU baseline;
- считать accelerator опциональной capability, обнаруживаемой при запуске;
- различать «API/EP существует», «model session создалась», «graph offloaded» и «контракт/качество сохранены»;
- измерять долю fallback и end-to-end pipeline, а не обещать устройство по имени provider.

## Что это меняет для карты диаризации

1. Разведка `sherpa-onnx` на обычном CPU ORT не опирается на затухающий
   компонент: ядро ORT активно и имеет самый широкий CPython/platform coverage.
   Это поддерживает текущий вариант диаризации как optional post-processing,
   но ничего не говорит о качестве DirectML/OpenVINO EP на конкретных двух
   моделях.
2. Формулировка разведочного замера «у OpenVINO GenAI потокенных таймкодов нет»
   требует уточнения. Upstream с 2026.0 предоставляет word-level timestamps;
   сейчас их не экспортирует проектный OpenVINO backend. Следовательно,
   невозможность пословной привязки на `--device openvino-*` — **интеграционный
   пробел local-transcriber**, а не долгосрочное ограничение движка
   ([OpenVINO 2026.0](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-0-0)).
3. Не следует связывать UX диаризации с немедленным выбором EP. Сначала можно
   определить пользовательский контракт — флаг, `num_speakers`, зависимость,
   формат и честное поведение при отсутствии word timestamps. Ускорение
   диаризации через OpenVINO EP или DirectML должно пройти отдельную
   совместимость и benchmark на обеих моделях `sherpa-onnx`.
4. Не следует принимать решение о полной консолидации проекта на ORT. Нативный
   OpenVINO GenAI развивается как ASR-платформа, в том числе по таймкодам и NPU,
   а FasterWhisper сохраняет отдельные достоинства CUDA и языкового покрытия,
   уже зафиксированные ADR-003/006.

## Рекомендация по жизненному циклу

Для portability evidence приоритеты такие: CPU baseline должен оставаться обязательным; accelerator — opt-in capability с явной диагностикой provider/device/fallback; platform wheel и provider name считаются только предпосылкой, пока model-specific smoke/golden test не подтвердил session creation, placement и выходной контракт. Windows ML, MIGraphX, CoreML и native WebGPU следует оценивать отдельными экспериментами, а не добавлять в UX как обещанные устройства заранее.

Это не решение о консолидации backend-ов на ORT и не предложение browser-направления.

Ниже — **интерпретация источников для local-transcriber**, а не опубликованный
roadmap Microsoft или Intel. Она исходит из фактов о lifecycle, wheel-матрицах
и текущем контракте проекта; реальную пригодность каждого ускорителя должен
подтвердить проектный benchmark.

- Сохранять **ONNX-модели диаризации + обычный CPU ORT** как базовый переносимый
  путь: это наименее связанный с одним вендором слой и единственная из трёх
  поставок с wheel на Windows/Linux ARM64 и macOS ARM64.
- Рассматривать **OpenVINO EP как опциональное ускорение этих же ONNX-моделей на
  Intel**, но не обещать его до проверки operator coverage, фактического
  provider assignment, качества и скорости. Его wheel активен, однако отстаёт
  от текущих ORT/OpenVINO и требует собственной матрицы версий.
- Рассматривать **нативный OpenVINO/OpenVINO GenAI как основной долгосрочный
  Intel ASR-путь**, особенно для Whisper и будущего NPU. Для пословной
  диаризации сначала проверить и протянуть уже существующие upstream word
  timestamps через проектный `Backend` contract.
- Не закладывать новый DirectML backend проекта: текущий EP поддерживается, но
  feature development официально ушёл в WinML. Если кросс-вендорное Windows GPU
  ускорение станет отдельной целью, исследовать WinML как новый
  Windows-специфический backend, а не считать `onnxruntime-directml`
  долгоживущим default.

## Новые вопросы карты

- Нужен ли отдельный portability experiment: GigaAM E2E RNN-T и обе diarization-модели на DirectML, WinML MIGraphX, Linux MIGraphX и CoreML с node placement/profile и golden outputs?
- Насколько устойчива локально работающая пара PyAnnote + WeSpeaker между
  платформами и EP, если upstream recipe её не фиксирует?
- Должен ли Windows UX показывать не только выбранный provider, но и фактический accelerator/fallback после capability discovery?
- Стоит ли поддерживать Windows ML bootstrap/catalog как отдельный integration layer или оставить низкофрикционный DirectML до появления подтверждённого выигрыша?
- Какой минимальный browser experiment проверит GigaAM preprocessing/decoder/timestamps и pyannote+embedding WASM/WebGPU, не превращая карту в browser roadmap?

- Какой точный контракт word timestamps возвращают `WhisperPipeline` и новый
  `ASRPipeline` 2026.3, и как без потери совместимости добавить их в проектный
  `Backend`/`TranscribeResult`?
- Дают ли `sherpa-onnx` segmentation и embedding models полный offload в
  OpenVINO EP, или часть графа уходит в CPU EP; меняются ли границы и
  эмбеддинги численно?
- Есть ли выигрыш OpenVINO EP на целевом Intel Core i5 11-го поколения после
  учёта второго runtime, загрузки модели и памяти, или CPU ORT уже оптимальнее?
- Нужен ли UX явного отказа/огрубления диаризации на backend без word
  timestamps, либо backend contract должен сначала стать timestamp-aware?
- Следует ли разделить extra диаризации на переносимый CPU-вариант и
  Intel-ускорение с platform marker, чтобы не ухудшить zero-config установку на
  ARM/macOS?
