# Обновления движков и моделей на 2026-08-10

Исследование выполнено по первичным источникам: официальным release notes, changelog, PyPI и model cards Hugging Face. Текущее состояние проекта зафиксировано в [`pyproject.toml`](../../pyproject.toml) и [`uv.lock`](../../uv.lock).

## Краткий вывод

В проекте уже используется актуальный `faster-whisper` 1.2.1, однако связанные движки и модельный каталог можно обновить. Наиболее полезная последовательность: согласованно поднять OpenVINO и OpenVINO GenAI до 2026.3, разрешить `onnx-asr` 0.12 и проверить VAD, обновить CTranslate2 до 4.8.1, затем добавить `large-v3-turbo` для FasterWhisper и OpenVINO. ONNX Runtime 1.28 нельзя безусловно фиксировать, пока проект поддерживает Python 3.10, поскольку эта версия ORT требует Python 3.11 или новее ([ONNX Runtime 1.28.0 на PyPI](https://pypi.org/project/onnxruntime/1.28.0/)).

## Движки

| Компонент | Сейчас в проекте | Актуальная стабильная версия | Существенные изменения и риски |
|---|---:|---:|---|
| faster-whisper | 1.2.1 | 1.2.1 | Проект уже на последнем релизе. В 1.2.1 обновлён Silero VAD до v6, доработаны retry Hugging Face Hub и `clip_timestamps`; в 1.2.0 добавлена поддержка `distil-large-v3.5` ([официальные releases](https://github.com/SYSTRAN/faster-whisper/releases), [PyPI](https://pypi.org/project/faster-whisper/)). Для актуального GPU-стека документация указывает CUDA 12 и cuDNN 9; для старых комбинаций требуются специальные версии CTranslate2 ([официальная установка](https://github.com/SYSTRAN/faster-whisper#gpu)). |
| CTranslate2 | 4.7.1 | 4.8.1 | В 4.8.0 `PACKED_GEMM` включён по умолчанию для Intel MKL; 4.8.1 исправляет heap overflow при загрузке модели и аварийное деление на ноль в Whisper `align()` при отсутствии кадров ([официальный changelog](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md), [релиз 4.8.1](https://github.com/OpenNMT/CTranslate2/releases/tag/v4.8.1), [PyPI](https://pypi.org/project/ctranslate2/)). Публичного breaking API для используемого пути не заявлено, но из-за нативных библиотек нужно проверить Windows CPU, Linux CUDA и проектный CUDA bootstrap. Требование cuDNN 9 появилось в CTranslate2 4.5.0 ([changelog](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md#450)). |
| OpenVINO | 2026.0.0 | 2026.3.0 | В 2026.3 добавлены общий `ASRPipeline`, Qwen3-ASR и метрики задержки стадий распознавания; в ветке 2026.x также появились word-level timestamps и определённый язык в результатах Whisper ([официальные release notes](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html), [PyPI](https://pypi.org/project/openvino/)). Начиная с 2026.0 удалена поддержка устаревшего stateless Whisper decoder, поэтому модели должны быть stateful; та же версия требует минимум AVX2, использует manylinux_2_28 и больше не поддерживает CentOS 7 ([release notes 2026.0](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-0-0)). |
| openvino-genai | 2026.0.0.0 | 2026.3.0.0 | OpenVINO GenAI необходимо обновлять вместе с OpenVINO: официальная документация требует совпадения `major.minor.patch` у OpenVINO, OpenVINO Tokenizers и GenAI, иначе возможны ABI/import errors. PyPI-пакеты собраны с `_GLIBCXX_USE_CXX11_ABI=0`, а архивы C++ — с ABI=1, поэтому смешивать PyPI GenAI с OpenVINO из C++ archive нельзя ([официальная страница PyPI](https://pypi.org/project/openvino-genai/), [официальные releases](https://github.com/openvinotoolkit/openvino.genai/releases)). |
| onnx-asr | 0.11.0 | 0.12.0 | Обновление сейчас **явно заблокировано** зависимостью `onnx-asr[cpu,hub]>=0.11.0,<0.12.0` в [`pyproject.toml`](../../pyproject.toml). В 0.12 добавлены GigaAM Multilingual CTC/Large CTC, свёрточные ONNX preprocessors с ускорением CUDA EP и ORT fallback; также исправлены сегменты нулевой длины после VAD, что напрямую относится к используемому проектом `.with_vad()` ([релиз 0.12.0](https://github.com/istupakov/onnx-asr/releases/tag/v0.12.0), [официальные release notes](https://istupakov.github.io/onnx-asr/release-notes/), [PyPI](https://pypi.org/project/onnx-asr/)). Breaking Python API не объявлен, но проект должен проверить загрузку квантованных моделей, VAD и форму возвращаемых сегментов. |
| ONNX Runtime | 1.24.3 | 1.28.0 | В 1.28 обновлены ONNX и protobuf и включены security fixes, в том числе проверка границ в `WhisperDecoderSubgraph` ([официальный релиз 1.28.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0), [PyPI](https://pypi.org/project/onnxruntime/1.28.0/)). Версия 1.28 требует Python 3.11+, тогда как проект допускает Python 3.10, поэтому нужен условный lock/constraint либо осознанное повышение минимальной версии Python ([метаданные PyPI 1.28.0](https://pypi.org/project/onnxruntime/1.28.0/)). Для GPU-пакетов ORT 1.27+ используется CUDA 13.0 и cuDNN 9; major-версии CUDA и cuDNN должны совпадать с runtime ([официальная матрица CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)). |

## Модели и варианты

### Whisper large-v3-turbo

`whisper-large-v3-turbo` — мультиязычная модель на 99 языков с 809 млн параметров вместо 1550 млн у `large-v3`; число decoder layers сокращено с 32 до 4, что заметно ускоряет вывод ценой небольшой потери качества ([официальная model card OpenAI](https://huggingface.co/openai/whisper-large-v3-turbo)). Это наиболее полезный новый вариант для русского и смешанного аудио.

FasterWhisper уже содержит стандартные aliases `large-v3-turbo` и `turbo` в собственной таблице моделей ([официальный `utils.py`](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/utils.py)); готовая CTranslate2-конверсия опубликована как [`dropbox-dash/faster-whisper-large-v3-turbo`](https://huggingface.co/dropbox-dash/faster-whisper-large-v3-turbo). Проект использует собственную ограниченную таблицу aliases, поэтому модель нужно добавить явно либо перейти на стандартное разрешение имён FasterWhisper ([текущая реализация проекта](../../src/local_transcriber/backends/faster_whisper.py)).

Для OpenVINO опубликованы официальные варианты [`whisper-large-v3-turbo-fp16-ov`](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-fp16-ov) и [`whisper-large-v3-turbo-int8-ov`](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov). Их model cards требуют OpenVINO 2026.1 или новее, поэтому текущий OpenVINO 2026.0 недостаточен; после согласованного обновления до 2026.3 можно добавить общий alias `turbo` для FasterWhisper и OpenVINO.

### GigaAM Multilingual

`onnx-asr` 0.12 добавляет модели `gigaam-multilingual-ctc` и `gigaam-multilingual-large-ctc` ([официальный релиз 0.12.0](https://github.com/istupakov/onnx-asr/releases/tag/v0.12.0), [документация использования](https://istupakov.github.io/onnx-asr/usage/)). Конвертированные репозитории опубликованы как [`gigaam-multilingual-ctc-onnx`](https://huggingface.co/istupakov/gigaam-multilingual-ctc-onnx) и [`gigaam-multilingual-large-ctc-onnx`](https://huggingface.co/istupakov/gigaam-multilingual-large-ctc-onnx), исходная модель — [`ai-sage/GigaAM-Multilingual`](https://huggingface.co/ai-sage/GigaAM-Multilingual). Они полезны для смешанной речи на поддерживаемых пяти языках, но требуют снятия ограничения `<0.12.0`.

### GigaAM v3 E2E

Текущий ONNX-репозиторий GigaAM v3 содержит не только CTC/RNNT, но и E2E-варианты с пунктуацией и нормализацией текста ([официальная model card `istupakov/gigaam-v3-onnx`](https://huggingface.co/istupakov/gigaam-v3-onnx)). E2E-вариант может улучшить читаемость русского текста, но меняет семантику вывода относительно текущего `gigaam-v3-ctc`; перед заменой default нужны сравнительные тесты транскрипции и форматирования.

### Distil-Whisper large-v3.5

`distil-large-v3.5` поддерживается faster-whisper начиная с 1.2.0 ([официальные releases](https://github.com/SYSTRAN/faster-whisper/releases)); доступны [исходная модель](https://huggingface.co/distil-whisper/distil-large-v3.5) и [CTranslate2-конверсия](https://huggingface.co/distil-whisper/distil-large-v3.5-ct2). Модель предназначена только для английского языка, поэтому она не подходит как русский default, но может быть отдельной English-only опцией.

### Qwen3-ASR как R&D

OpenVINO 2026.3 добавляет раннюю поддержку Qwen3-ASR через новый общий `ASRPipeline` ([официальные release notes 2026.3](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-3-0)). Текущий backend проекта построен вокруг `WhisperPipeline` и Whisper-совместимых сегментов ([текущая реализация](../../src/local_transcriber/backends/openvino.py)), поэтому Qwen3-ASR не является drop-in обновлением модели: это отдельная R&D-задача с новым интерфейсом, модельным контрактом и тестами качества.

## Приоритет действий

1. Согласованно обновить `openvino` до 2026.3.0 и `openvino-genai` до 2026.3.0.0, не смешивая источники сборок; проверить существующие stateful Whisper-модели, CPU/GPU и минимальные платформенные требования ([совместимость GenAI](https://pypi.org/project/openvino-genai/), [release notes OpenVINO](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)).
2. Убрать блокирующее ограничение `<0.12.0`, поднять `onnx-asr` до 0.12.0 и прогнать интеграционные тесты VAD, сегментов и квантованных GigaAM-моделей ([релиз 0.12.0](https://github.com/istupakov/onnx-asr/releases/tag/v0.12.0)).
3. Обновить CTranslate2 с 4.7.1 до 4.8.1 и проверить Windows CPU/Linux CUDA, включая CUDA bootstrap ([changelog](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md)).
4. Добавить alias `turbo` и репозитории `large-v3-turbo` для FasterWhisper и OpenVINO после обновления OpenVINO ([OpenAI model card](https://huggingface.co/openai/whisper-large-v3-turbo), [OpenVINO FP16](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-fp16-ov), [OpenVINO INT8](https://huggingface.co/OpenVINO/whisper-large-v3-turbo-int8-ov)).
5. Не фиксировать ONNX Runtime 1.28 для всех окружений, пока поддерживается Python 3.10; сначала выбрать условные зависимости либо официально поднять Python floor ([PyPI 1.28.0](https://pypi.org/project/onnxruntime/1.28.0/)).
6. Рассматривать GigaAM Multilingual, GigaAM v3 E2E и Qwen3-ASR как отдельные эксперименты с качеством и совместимостью, а `distil-large-v3.5` — только как English-only профиль ([GigaAM Multilingual](https://huggingface.co/ai-sage/GigaAM-Multilingual), [GigaAM v3 ONNX](https://huggingface.co/istupakov/gigaam-v3-onnx), [OpenVINO 2026.3](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html#openvino-2026-3-0), [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3.5)).
