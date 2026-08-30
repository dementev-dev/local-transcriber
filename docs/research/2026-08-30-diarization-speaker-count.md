# Оценка числа говорящих для быстрых эмбеддингов

**Дата:** 2026-08-30

**Статус:** исследовательская записка. Production-код и конфигурация не
менялись.

## Вопрос

Как отделить оценку числа говорящих от порога `FastClustering`, чтобы
использовать быстрые CAMPPlus или TitaNet и не запускать сегментацию и
извлечение эмбеддингов второй раз?

## Краткий вывод

Нужна двухшаговая схема: один раз получить ровно те эмбеддинги, которые
подготовил конвейер sherpa-onnx, отдельно оценить по ним число голосовых
кластеров `N`, затем передать `N` штатному `FastClustering`. Для первого
прототипа рекомендован **только счетчик NME**, без полной spectral clustering:
ядро алгоритма из статьи реализовать на уже доступном NumPy, а итоговые метки
по-прежнему получить через `FastClustering(num_clusters=N)`. Простой eigengap с
фиксированным прореживанием нужен как дешевый контрольный вариант.

Публичного промежуточного интерфейса для этого в sherpa-onnx 1.13.5 нет.
`SpeakerEmbeddingExtractor` и `FastClustering` публичны, но
`OfflineSpeakerDiarization.process()` возвращает только готовые интервалы.
Матрица эмбеддингов и соответствие строк локальным говорящим остаются внутри
реализации. После `set_config()` повторный `process()` снова выполнит модельный
инференс. Python-адаптер проекта сам по себе не может дать дешевую повторную
кластеризацию; будущему прототипу нужен узкий исследовательский интерфейс на
уровне sherpa-onnx.

## Что подтверждено про sherpa-onnx

### `FastClustering` уже умеет второй шаг

В версии 1.13.5 алгоритм нормирует строки, строит все попарные cosine
dissimilarity, выполняет complete-linkage AHC и затем выбирает один из двух
способов разрезать одно и то же дерево:

- при `num_clusters > 0` - ровно на заданное число кластеров;
- иначе - по высоте `threshold`.

Это видно непосредственно в
[`fast-clustering.cc`](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/fast-clustering.cc)
и закреплено в комментарии к
[`FastClusteringConfig`](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/fast-clustering-config.h):
при известном числе кластеров порог игнорируется. Python binding публично
принимает C-contiguous матрицу `float32` размера `M x D` и возвращает метку для
каждой строки
([исходник binding](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/python/csrc/fast-clustering.cc)).
Значит, передавать оцененное `N` в существующий кластеризатор не требует новой
реализации распределения эмбеддингов по кластерам.

Порог `FastClustering` не принадлежит модели эмбеддингов. Это высота
complete-linkage дерева в шкале cosine dissimilarity. Официальный пример
CAMPPlus для speaker verification лишь вычисляет cosine score и не задает
универсальной границы решения
([`infer_sv.py`](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/speakerlab/bin/infer_sv.py)).
Официальный diarization-рецепт 3D-Speaker отдельно задает `pval=0.012` для
spectral clustering и `mer_cos=0.8` для последующего слияния центров
([`infer_diarization.py`](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/speakerlab/bin/infer_diarization.py)).
Эти величины имеют другой смысл и не могут быть перенесены в sherpa как
`порог модели`.

### Публичных деталей недостаточно для повтора всего конвейера

Публичный `SpeakerEmbeddingExtractor` позволяет создать stream, подать в него
аудио и получить один embedding
([Python binding](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/python/csrc/speaker-embedding-extractor.cc),
[C/C++ API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/speaker_embedding.html)).
Это пригодно для самостоятельно выбранного фрагмента, но не выдает материал,
который выбрал diarization-конвейер.

Внутри `OfflineSpeakerDiarization` после сегментации:

1. из перекрывающихся окон строятся пары `окно x локальный говорящий`;
2. речь с одновременными голосами исключается;
3. для каждой пары собирается свой stream и вычисляется embedding;
4. вся матрица передается в `FastClustering`;
5. метки строк переводятся обратно в разметку говорящих.

Последовательность и временная матрица видны в
[`OfflineSpeakerDiarizationPyannoteImpl::Process`](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L87-L204),
а сборка строк - в
[`GetChunkSpeakerSampleIndexes` и `ComputeEmbeddings`](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L376-L528).
Это private-методы реализации.

Python-класс `OfflineSpeakerDiarization` экспортирует только `sample_rate`,
`set_config` и `process`; `process` возвращает готовый результат, а не признаки
([binding 1.13.5](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/python/csrc/offline-speaker-diarization.cc#L73-L128)).
Внутренний `SetConfig` заменяет только объект кластеризации, тогда как каждый
`Process` заново вызывает segmentation и `ComputeEmbeddings`
([реализация](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/sherpa-onnx/csrc/offline-speaker-diarization-pyannote-impl.h#L79-L175)).
Актуальная публичная C API также документирует только создание, настройку и
получение итоговых сегментов
([официальная страница](https://k2-fsa.github.io/sherpa/onnx/c-api/html/speaker_diarization.html)).

Итог: публичный API позволяет **кластеризовать имеющуюся матрицу повторно**.
Извлечь или переиграть промежуточную матрицу штатного diarization-конвейера
через него нельзя.

## Варианты оценки `N`

### 1. Отдельный порог для каждой embedding-модели

Это текущий `FastClustering` в автоматическом режиме. Он дешев, не добавляет
зависимостей и на одном проходе сразу оценивает число кластеров и назначает
метки. На порог влияют архитектура модели эмбеддингов, длительность и состав
выбранных фрагментов, акустика и распределение голосов. Локальная калибровка
уже не нашла общего рабочего порога для CAMPPlus,
а TitaNet не прошел автоматическую сетку
([исходное исследование](2026-08-14-diarization-cpu-cost.md),
[повторный CPU-тест](../benchmarks/2026-08-30-lightweight-diarization-cpu.md)).

Вводить еще один `официальный порог модели` оснований нет: upstream-рецепты
публикуют параметры конкретного clustering pipeline, а не переносимую границу
для complete-linkage sherpa. Вариант остается контрольным, но не решает тикет.

### 2. Spectral clustering с простым eigengap

Официальный CAMPPlus-конвейер 3D-Speaker строит cosine affinity, оставляет в
каждой строке фиксированную долю ближайших соседей, симметризует матрицу,
вычисляет ненормированный Laplacian и несколько наименьших собственных пар.
Число говорящих - позиция максимального eigengap в заданных пределах, после
чего собственные векторы кластеризуются k-means
([`SpectralCluster`](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/speakerlab/process/cluster.py#L21-L106)).

Для нашей двухшаговой схемы нужны только собственные значения и оценка `N`;
k-means можно не выполнять. Это одна spectral decomposition и потому хороший
минимальный контрольный вариант. Однако фиксированный `pval` остается новым
порогом.
Официальная реализация задает его для своего способа нарезки; пары
`окно x локальный говорящий` sherpa устроены иначе. Вариант отделяет `N` от
порога AHC, но сохраняет калибруемый параметр.

Если брать реализацию 3D-Speaker как зависимость, она импортирует NumPy, SciPy,
scikit-learn, fastcluster, UMAP и HDBSCAN, причем последние два обязательны уже
при импорте модуля
([исходник](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/speakerlab/process/cluster.py#L3-L18),
[requirements](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/requirements.txt)).
Для production это несоразмерно счетчику.

### 3. NME-SC

NME-SC создан именно для совместной оценки числа кластеров и параметра
прореживания без dev-set. Для каждого числа соседей `p` алгоритм бинаризует
cosine affinity, симметризует ее, строит ненормированный Laplacian и считает
eigengap. Нормированный gap равен максимальному gap, деленному на наибольшее
собственное значение; выбирается `p`, минимизирующий отношение `p / gap`, а `N`
задает позиция максимального gap
([статья авторов, раздел III](https://arxiv.org/abs/2003.02405)).
Авторы ограничивали поиск `p` диапазоном до четверти числа фрагментов и число
говорящих - восемью; результаты статьи получены на x-vector и англоязычных
корпусах, поэтому перенос на наши эмбеддинги еще предстоит проверить.

Поддерживаемая реализация NeMo принимает готовые эмбеддинги, строит cosine
affinity и отдельно получает `est_num_of_spk`; известное число говорящих затем
подменяет эту оценку перед spectral clustering
([`offline_clustering.py`](https://github.com/NVIDIA-NeMo/Speech/blob/d47d7a3c306790637aff39c2d3f9eceb83ca8879/nemo/collections/asr/parts/utils/offline_clustering.py#L1087-L1161)).
Для ускорения NME она по умолчанию ограничивает матрицу анализа 512 строками,
перебирает разреженную сетку из 30 значений `p` и предупреждает, что меньше 20
может ухудшить оценку
([класс `NMESC`](https://github.com/NVIDIA-NeMo/Speech/blob/d47d7a3c306790637aff39c2d3f9eceb83ca8879/nemo/collections/asr/parts/utils/offline_clustering.py#L801-L958)).
Для коротких записей NeMo добавляет случайные anchor embeddings и повторяет
оценку. Это отдельное расширение за пределами минимального ядра статьи
([исходник](https://github.com/NVIDIA-NeMo/Speech/blob/d47d7a3c306790637aff39c2d3f9eceb83ca8879/nemo/collections/asr/parts/utils/offline_clustering.py#L536-L595)).

Полная NME-SC после оценки `N` еще раз раскладывает Laplacian, строит spectral
embeddings и запускает k-means. Для этого тикета такая замена итогового
кластеризатора не нужна: локальные тесты уже показывают полезный путь
`FastClustering` с известным числом говорящих. Нужна только функция
`embeddings -> N`.

### 4. Оценить `N`, затем передать известное число кластеров

Это не отдельный счетчик. Это рекомендуемая граница ответственности:

```text
segmentation -> embeddings +-> count-only NME -> N
                           +-> FastClustering(features, num_clusters=N)
                                     -> штатная финализация sherpa
```

Плюсы:

- порог complete-linkage больше не определяет число голосовых кластеров;
- CAMPPlus и TitaNet используют один модельный проход;
- итоговое назначение меток остается в уже принятом FastClustering;
- на одной матрице можно независимо сравнить несколько счетчиков.

Минус один, но архитектурный: без нового промежуточного интерфейса текущий
`process()` не позволяет вставить счетчик между вычислением матрицы и
кластеризацией.

## Зависимости, лицензии и цена

Штатная установка NeMo требует среди прочего NumPy, scikit-learn и PyTorch
([`pyproject.toml`](https://github.com/NVIDIA-NeMo/Speech/blob/d47d7a3c306790637aff39c2d3f9eceb83ca8879/pyproject.toml)).

| Вариант | Новые runtime-зависимости | Дополнительная работа после эмбеддингов | Ограничение |
|---|---|---|---|
| Порог `FastClustering` | нет | уже существующие попарные расстояния и AHC | порог калибруется для модели и материала |
| Простой eigengap | SciPy в официальной реализации; либо только уже доступный NumPy | affinity `M x M`, одно собственное разложение | фиксированный параметр прореживания |
| Полная NME-SC NeMo | `nemo-toolkit`, PyTorch, scikit-learn и другие зависимости NeMo | до нескольких десятков eigenvalue-разложений, затем еще spectral embedding и k-means | самый тяжелый и меняет итоговый кластеризатор |
| Счетчик NME + FastClustering | только NumPy при чистой реализации ядра статьи | NME на ограниченной матрице, затем штатный AHC по известному `N` | нужен промежуточный интерфейс и локальная проверка переноса |

`FastClustering` уже хранит `M(M-1)/2` расстояний и строит AHC. Простой
eigengap добавляет еще одну плотную affinity-матрицу. NME повторяет eigenvalue-
анализ для набора `p`; ограничение размера и разреженный поиск контролируют
цену, но их влияние на качество нужно измерить. Полная spectral clustering
также хранит собственные векторы и запускает k-means, поэтому для одной оценки
`N` избыточна. В проектных замерах основной расход по-прежнему приходится на
извлечение эмбеддингов, а не на clustering
([профиль стоимости](2026-08-14-diarization-cpu-cost.md),
[CPU-тест](../benchmarks/2026-08-30-lightweight-diarization-cpu.md)); это не
заменяет отдельного замера NME на целевой матрице.

sherpa-onnx, 3D-Speaker и NeMo Speech распространяют код под Apache-2.0
([sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx/blob/v1.13.5/LICENSE),
[3D-Speaker](https://github.com/modelscope/3D-Speaker/blob/065629c313eaf1a01c65c640c46d77e61e9607b4/LICENSE),
[NeMo Speech](https://github.com/NVIDIA-NeMo/Speech/blob/d47d7a3c306790637aff39c2d3f9eceb83ca8879/LICENSE)).
Авторская реализация NME-SC опубликована под MIT и сама направляет пользователей
к поддерживаемой версии NeMo
([репозиторий авторов](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering/tree/ae97f9be9c33b554205c7192c5137ce8f456c1b7)).
Файл NeMo с clustering дополнительно сохраняет BSD-уведомление scikit-learn.
NumPy имеет BSD-3-Clause
([лицензия 2.4.3](https://github.com/numpy/numpy/blob/v2.4.3/LICENSE.txt)) и уже
присутствует в [lock-файле проекта](../../uv.lock). Лицензионного запрета на
прототип нет;
прямое заимствование кода потребует сохранить его уведомления. Новые модели
для счетчика не нужны.

## Ограничения вывода

- NME-SC проверялась авторами на другой нарезке и x-vector. Строки sherpa - это
  локальные говорящие перекрывающихся Pyannote-окон после исключения overlap;
  переносимость нельзя считать доказанной.
- Правильное `N` не гарантирует качественную разметку говорящих. Для TitaNet в
  режиме известного числа кластеров уже наблюдался малый кластер, поэтому нужны
  прежние proxy-метрики и слуховая проверка.
- У sherpa есть отдельный путь для единственного окна, который обходится без
  общей матрицы эмбеддингов. Прототип не должен ломать этот special case.
- Python binding `FastClustering` нормирует переданную матрицу на месте;
  сравниваемым вариантам нужно выдавать отдельные C-contiguous копии.
- Цена NME на числе строк, которое реально создает sherpa, здесь не измерялась.

## Точная граница будущего прототипа

Production-код, CLI, модели по умолчанию и ADR не менять. Сделать отдельный
исследовательский стенд на закрепленном sherpa-onnx 1.13.5:

1. Узко разделить внутренний `Process` на подготовку и финализацию либо добавить
   исследовательский callback после `ComputeEmbeddings`. За один проход
   сохранить в памяти матрицу `float32`, список валидных пар
   `окно x локальный говорящий` и локальные frame labels; аудио и текст не
   сериализовать.
2. На **одной и той же** матрице сравнить три счетчика:
   текущий порог как контроль, простой eigengap с фиксированным прореживанием и
   count-only NME. Ограничить `N` диапазоном `1..8`.
3. Минимальный NME реализовать по статье на NumPy: cosine affinity,
   top-`p` binarization, симметризация, ненормированный Laplacian,
   `numpy.linalg.eigvalsh`, выбор `p` и `N` по NME. Для первого прогона взять
   детерминированное ограничение 512 строк и sparse search из 20 значений;
   anchor embeddings, Torch, spectral embeddings и k-means не включать.
4. Для каждого оцененного `N` вызвать публичный `FastClustering` на копии той же
   матрицы с `num_clusters=N`, затем выполнить существующую финализацию без
   повторной сегментации или извлечения эмбеддингов.
5. На тех же трех контрольных фрагментах измерить: оцененное `N`, общее и
   содержательное число кластеров, mapped speaker purity, слова без говорящего,
   время счетчика/кластеризации и peak RSS. Прогон с CAMPPlus и TitaNet должен
   завершиться слуховой проверкой.
6. Если простой eigengap и NME одинаково проходят проверку, выбрать более
   дешевый eigengap. Если проходит только NME, отдельно решить, переносить ли
   чистую NumPy-реализацию в production. Если оба ошибаются, быстрые embedding-
   модели оставить только для явно заданного числа говорящих.

Этот прототип отвечает ровно на два неизвестных: можно ли стабильно оценить
`N` по внутренним эмбеддингам CAMPPlus/TitaNet и укладывается ли count-only NME
в приемлемую CPU-цену. Он не выбирает новую модель по умолчанию, не заменяет
FastClustering и не расширяет продуктовый контракт.
