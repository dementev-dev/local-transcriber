# ADR-001: Preload libcublas из pip-пакета для GPU на Linux/WSL2

**Статус**: Принято
**Дата**: 2026-03-18

## Контекст

ctranslate2 (backend faster-whisper) в runtime делает `dlopen("libcublas.so.12")`,
но не бандлит эту библиотеку в свой wheel — ожидает её в системе.
Без установленного CUDA toolkit `uv run transcribe --device cuda` падает с ошибкой.

Ключевые факты:
- `libcuda.so.1` приходит от NVIDIA driver (всегда есть, если GPU есть)
- `libcublas.so.12` отсутствует в wheel ctranslate2 на обеих платформах
- **cuDNN не нужен** — в `libctranslate2.so` ноль символов cudnn (проверено `nm -D` и `strings`)
- На Windows ctranslate2 делает `os.add_dll_directory` в своём `__init__.py`,
  но только для DLL внутри пакета; `cublas64_12.dll` тоже не бандлится

## Решение

### nvidia-cublas-cu12 как pip-зависимость

В `pyproject.toml`:
```
"nvidia-cublas-cu12>=12.4; sys_platform == 'linux' and platform_machine == 'x86_64'"
```

Нижняя граница `>=12.4` — ctranslate2 собран с CUDA 12.4. Пакет доступен только
для Linux x86_64 (на Windows и macOS не устанавливается по platform marker).

### ctypes.CDLL preload вместо LD_LIBRARY_PATH

`_cuda_bootstrap.py` загружает `libcublas.so.12` по полному пути через
`ctypes.CDLL(path, mode=RTLD_GLOBAL)` **до** первого `import ctranslate2`.

Почему не `os.environ["LD_LIBRARY_PATH"]`: на Linux/glibc динамический линкер (`ld.so`)
кеширует пути поиска при старте процесса и **не перечитывает** `LD_LIBRARY_PATH`
из environ в рамках уже запущенного процесса.

Почему `RTLD_GLOBAL`: без этого флага символы cublas не видны другим `.so`,
загруженным позже (в т.ч. `libctranslate2.so`).

Динамический линкер кеширует загруженные библиотеки по soname — когда ctranslate2
потом вызовет `dlopen("libcublas.so.12")`, линкер вернёт уже загруженный handle.

### strict_device для явного --device

`--device cuda` / `--device cpu` → `strict_device=True` → CUDA-ошибка = raise, без fallback.
`--device auto` → `strict_device=False` → текущее поведение с fallback на CPU.

Мотивация: silent fallback для часового файла = 60 минут вместо 5.

## Последствия и tradeoffs

**~400 MB на CPU-only Linux x86_64**: nvidia-cublas-cu12 ставится на все Linux x86_64,
включая машины без GPU. Bootstrap при этом preload'ит libcublas (overhead ~1 ms),
но она не используется, т.к. ctranslate2 не получит запрос на CUDA device.

Альтернатива — `[project.optional-dependencies]` + `uv sync --extra cuda`,
но тогда теряется zero-config UX. Для v1 оставляем как обязательную зависимость.

**Windows GPU**: nvidia-cublas-cu12 недоступен как pip-пакет для Windows.
Единственный путь — системный CUDA toolkit (`choco install cuda` / `winget install -e --id Nvidia.CUDA`).
CLI выводит эту подсказку при CUDA-ошибке на `sys.platform == "win32"`.

**Namespace package**: `nvidia.cublas` — namespace package (`__file__` is `None`),
для определения директории используется `__path__[0]`, а не `__file__`.

## Отклонённые альтернативы

| Альтернатива | Почему отклонена |
|---|---|
| `nvidia-cudnn-cu12` в зависимостях | ctranslate2 не использует cuDNN — ноль символов, проверено через `nm -D` |
| Self-reexec с `LD_LIBRARY_PATH` | ctypes.CDLL решает задачу без перезапуска процесса; self-reexec создаёт проблемы с сигналами, tty, fd |
| `os.environ["LD_LIBRARY_PATH"] += ...` | glibc кеширует пути при старте, не перечитывает environ |
| `DeviceResolution` dataclass + `resolve_device()` | Один `bool strict_device` решает ту же задачу проще |
| ctranslate2 preflight (`get_supported_compute_types`) | Латентность; try/catch при загрузке модели не хуже |
| Bootstrap console_scripts entrypoint | Не нужен — достаточно вызова в начале `transcriber.py` |
