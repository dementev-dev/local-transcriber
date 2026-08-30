# Облегчённые модели диаризации: Pyannote INT8 и TitaNet small

**Дата:** 2026-08-29

**Статус:** исследовательская записка. Код, зависимости и рабочая конфигурация
не менялись.

**Метод:** использованы только первичные публичные источники: документация,
исходный код и выпуски sherpa-onnx, карточки владельцев моделей и исходная
публикация TitaNet. Приватные медиа и опорные транскрипты Hypescribe не
открывались.

## Краткий вывод

Sherpa-onnx официально документирует все четыре сочетания Pyannote
Segmentation 3.0 FP32/INT8 с 3D-Speaker и `nemo_en_titanet_small.onnx`. В
опубликованном разработчиками примере производительности переход FP32 → INT8
сокращает
RTF с 0,297 до 0,241 с 3D-Speaker и с 0,119 до 0,110 с TitaNet small. Но это
одна 56,861-секундная китайская контрольная запись, известные четыре говорящих,
CPU-провайдер и по одному потоку на сегментацию и эмбеддинги. Данных о
конкретном CPU, повторностях, разбросе и качестве на русском материале страница
не приводит. Поэтому цифры подтверждают совместимость и потенциал, но не
заменяют локальный тест. Условия и результаты взяты с [официальной
страницы sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html).

Официальные файлы выпусков доступны под следующими именами:

- `sherpa-onnx-pyannote-segmentation-3-0.tar.bz2` содержит
  `model.onnx` (FP32) и `model.int8.onnx`;
- `nemo_en_titanet_small.onnx` поставляется отдельным ONNX-файлом.

GitHub API не публикует `digest` ни для одного из этих файлов. Ниже зафиксированы
SHA-256 байтов, скачанных 2026-08-29 с официальных URL, и точные размеры,
совпавшие с GitHub API. Эти хеши пригодны для закрепления текущей поставки, но
получены в этой проверке и не заявлены разработчиками как контрольные суммы.

## Официальная поддержка и точные артефакты

### Pyannote Segmentation 3.0: FP32 и INT8 в одном архиве

Официальная документация sherpa-onnx называет модель
`sherpa-onnx-pyannote-segmentation-3-0`, указывает, что она конвертирована из
[`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0),
ссылается на [официальный каталог скриптов
конвертации](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/pyannote/segmentation)
и показывает оба файла, `model.onnx` и `model.int8.onnx`, в одном архиве
([документация и команда загрузки](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#download-the-model)).

Точное имя и URL:

```text
sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
```

На момент проверки [GitHub Releases
API](https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/speaker-segmentation-models)
сообщает размер файла `6 958 444` байта,
`created_at=2024-10-08T12:54:09Z`, `updated_at=2024-10-08T12:54:10Z` и
`digest=null`; сам [официальный выпуск](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models)
содержит этот файл, а документация даёт тот же URL загрузки.

Проверенное содержимое архива:

| Файл | Размер | SHA-256 |
|---|---:|---|
| `model.onnx` | 5 992 913 байт | `220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079` |
| `model.int8.onnx` | 1 540 506 байт | `d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d` |

Архив также содержит `LICENSE`, `README.md` и скрипты экспорта/проверки. Это
согласуется с опубликованным sherpa-onnx листингом: примерно 5,7 МБ для FP32 и
1,5 МБ для INT8
([официальный листинг](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#download-the-model)).

### NeMo TitaNet small: отдельная модель эмбеддингов

Официальная страница примеров загружает `nemo_en_titanet_small.onnx` из выпуска
моделей распознавания говорящих и использует его как `--embedding.model` вместе с
обоими вариантами Pyannote
([FP32](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#nemo-model-onnx),
[INT8](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#nemo-model-int8-onnx)).
Отдельная [страница моделей NeMo в sherpa-onnx](https://k2-fsa.github.io/sherpa/onnx/nemo/index.html)
перечисляет это точное имя среди поддерживаемых моделей эмбеддингов говорящих.

Точное имя и URL:

```text
nemo_en_titanet_small.onnx
https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx
```

Опечатка `recongition` является частью официальной метки выпуска и URL; исправлять
её на `recognition` нельзя. На момент проверки [GitHub Releases
API](https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/speaker-recongition-models)
сообщает размер файла
`40 257 283` байта, `created_at=2024-10-14T07:03:02Z`,
`updated_at=2024-10-14T07:03:03Z` и `digest=null`; [выпуск
sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)
предупреждает, что лицензия у каждой модели своя и её нужно искать в исходном
репозитории модели.

## Контекст опубликованных результатов

### Пример производительности sherpa-onnx

Все четыре числа ниже опубликованы в одном примере sherpa-onnx. В нём используется
`0-four-speakers-zh.wav` длительностью 56,861 с, явно задано
`--clustering.num-clusters=4`, а распечатанная конфигурация показывает CPU
провайдер и `num_threads=1` отдельно для сегментации и эмбеддингов. Страница не
указывает модель CPU, не описывает прогрев, число повторов или дисперсию и не
публикует метрику качества против эталонной разметки
([полный пример](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#sherpa-onnx-pyannote-segmentation-3-0)).

| Segmentation | Embedding | Время | RTF |
|---|---|---:|---:|
| Pyannote FP32 `model.onnx` | 3D-Speaker ERes2Net base | 16,870 с | 0,297 |
| Pyannote INT8 `model.int8.onnx` | 3D-Speaker ERes2Net base | 13,679 с | 0,241 |
| Pyannote FP32 `model.onnx` | NeMo TitaNet small | 6,756 с | 0,119 |
| Pyannote INT8 `model.int8.onnx` | NeMo TitaNet small | 6,231 с | 0,110 |

По опубликованным RTF мы пересчитали относительное ускорение. INT8 ускоряет
сочетание с 3D-Speaker примерно на 18,9%, а сочетание с TitaNet small —
примерно на 7,6%. Замена 3D-Speaker на TitaNet small при сохранении FP32 даёт
около 59,9%, а сочетание TitaNet small + INT8 относительно исходного FP32 +
3D-Speaker — около 63,0%. Здесь сравнивается только время выполнения на
указанной записи. По этим цифрам нельзя судить о DER, переносимости ускорения
на другой CPU или качестве на русском языке
([исходные времена и RTF](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#sherpa-onnx-pyannote-segmentation-3-0)).

### Что означает «small» в публикации TitaNet

Исходная статья определяет TitaNet-S как вариант с 256 каналами и 6,4 млн
параметров. Для очищенного набора VoxCeleb1 в задаче верификации говорящих она
сообщает EER 1,15% при сравнении по косинусному сходству. Для диаризации авторы
используют эталонный детектор голосовой активности (oracle SAD), кластеризацию
NME-SC, допуск 0,25 с и исключают перекрытия из подсчёта DER. При
известном числе говорящих DER TitaNet-S равен 6,37 / 2,00 / 2,22 / 1,11% на
NIST-SRE-2000 / AMI-Lapel / AMI-MixHeadset / CH109; при оценённом числе —
5,49 / 2,30 / 1,97 / 1,42%. Это тест архитектуры и исходной контрольной точки
NeMo в другом конвейере диаризации, не тест ONNX-экспорта внутри
sherpa-onnx
([первичная публикация TitaNet](https://arxiv.org/abs/2110.04410)).

Карточка NVIDIA NGC подтверждает назначение `titanet_small` для верификации
говорящих, диаризации и извлечения 192-мерных эмбеддингов, но её
раздел Performance содержит внутренне противоречивую фразу «TitaNet-L» рядом с
6,4 млн параметров. Поэтому для чисел TitaNet-S выше использована исходная
статья вместо этой фразы карточки
([официальная карточка NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small)).

## Происхождение моделей и лицензии

### Pyannote Segmentation 3.0

Исходная карточка Pyannote описывает 10-секундный mono-вход 16 кГц и powerset
выход из семи классов: тишина, три одиночных говорящих и три пары перекрывающихся
говорящих. Модель обучил Séverin Baroudi в `pyannote.audio` 3.0.0 на сочетании
AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE и VoxConverse
([карточка исходной модели](https://huggingface.co/pyannote/segmentation-3.0)).

Исходный репозиторий помечает модель лицензией MIT и требует принять условия
доступа к его файлам на Hugging Face. Публичный архив sherpa-onnx включает
собственный `LICENSE` с текстом MIT; в проверенном архиве copyright указан как
`Copyright (c) 2022 CNRS`, тогда как текущий LICENSE исходного репозитория
указывает 2023. Тип лицензии совпадает, но при дальнейшей поставке следует
сохранять именно `LICENSE` из закреплённого архива
([лицензия и условия исходной модели](https://huggingface.co/pyannote/segmentation-3.0),
[текущий LICENSE upstream](https://huggingface.co/pyannote/segmentation-3.0/blob/main/LICENSE)).

### NeMo TitaNet small

Sherpa-onnx публикует [скрипт экспорта](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/speaker-verification/export-onnx.py),
который загружает `EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_small")`,
экспортирует `nemo_en_titanet_small.onnx` и добавляет ONNX metadata с framework
`nemo`, language `English`, source URL NGC, sample rate и размерностью embedding.
[README каталога экспорта](https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/speaker-verification)
также прямо называет NGC `titanet_small` источником модели.

Карточка NGC называет актуальную опубликованную версию `1.19.0`, обновлённую
7 июня 2023 года, и описывает обучение на VoxCeleb 1/2, RIR noise, Fisher,
Switchboard и LibriSpeech. Реализация NeMo связывает имя `titanet_small` с
checkpoint `titanet-s.nemo` версии `1.19.0`
([карточка NVIDIA NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small),
[официальный реестр pretrained-моделей NeMo](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/label_models.py)).

NGC говорит, что использование модели регулируется лицензией NeMo; ссылка из
карточки ведёт на Apache License 2.0. Сам файл выпуска sherpa-onnx — одиночный
ONNX-файл без соседнего LICENSE, а описание выпуска перекладывает проверку
лицензии на пользователя. Поэтому при включении модели в приложение нужно
зафиксировать происхождение NVIDIA NeMo/TitaNet-S и приложить лицензию Apache-2.0,
а не считать общий LICENSE sherpa-onnx лицензией весов
([секция Licence карточки NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/titanet_small),
[LICENSE NeMo](https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE),
[предупреждение release sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)).

Происхождение нельзя воспроизвести полностью: текущий `run.sh` sherpa-onnx
устанавливает NeMo из ветки `main` без закрепления коммита или метки, затем
вызывает экспорт
для `titanet_small`. Он документирует процесс, но сам по себе не доказывает,
из какого коммита NeMo был создан файл от 2024-10-14. Для побайтовой
воспроизводимости следует считать официально скачанный ONNX плюс SHA-256 ниже
канонической единицей, пока разработчики не опубликуют данные закреплённой сборки
([официальный `run.sh`](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/speaker-verification/run.sh)).

## Проверенные SHA-256

Файлы скачаны 2026-08-29 напрямую с двух URL выше в `/tmp`; размер каждого
сверен с полем `size` GitHub Releases API. Архив дополнительно проверен командой
`bzip2 -t`. Поле `digest` в API было `null` для обоих файлов, поэтому ни один из
следующих хешей нельзя называть опубликованной контрольной суммой.

```text
24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488  sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e  nemo_en_titanet_small.onnx
```

Внутри проверенного Pyannote-архива:

```text
220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079  model.onnx
d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d  model.int8.onnx
```

Команда проверки после загрузки:

```bash
sha256sum \
  sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 \
  nemo_en_titanet_small.onnx
```

## Результат локальной проверки

[Прогон на Core i7-6820HQ](../benchmarks/2026-08-30-lightweight-diarization-cpu.md)
не подтвердил опубликованное ускорение INT8: сочетание с WeSpeaker выиграло
только 2,1%, а с TitaNet стало на 5,7% медленнее. TitaNet ускорил обработку в
среднем на 24,3%, но потребовал примерно на 147–149 МиБ больше RSS и не прошёл
приёмку общего автоматического порога.

## Практический вердикт

1. Для минимального эксперимента можно заменить только путь Pyannote
   `model.onnx` на `model.int8.onnx`: это официально поставляемый вариант того же
   архива и не требует менять модель эмбеддингов
   ([официальный пример INT8](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#d-speaker-model-int8-onnx)).
2. Для большего потенциального ускорения sherpa-onnx официально принимает
   `nemo_en_titanet_small.onnx` с обоими вариантами Pyannote, но опубликованный
   пример производительности не измеряет DER и не относится к русской речи
   ([официальные примеры NeMo](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html#nemo-model-onnx)).
3. Для воспроизводимой поставки следует закрепить точные URL, размеры и SHA-256,
   хранить MIT LICENSE из Pyannote-архива и явно приложить сведения об авторстве
   и лицензии Apache-2.0 для TitaNet/NeMo. До локальной калибровки нельзя
   переносить ни коэффициент ускорения, ни порог кластеризации, ни качество с
   опубликованной контрольной записи на целевые русскоязычные созвоны.
