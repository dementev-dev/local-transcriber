# Issue Tracker

Задачи проекта ведутся в Gitea-репозитории `ddmitry/local-transcriber`.

- Основной remote: `origin`
- Gitea: `https://git.dementev.space`
- CLI: `tea`
- Внешние pull request не входят в очередь triage

## Доступ

Перед операциями с задачами проверить наличие `tea`.

Если команда недоступна, остановиться и предложить пользователю установку:

```powershell
winget install --id Gitea.tea --exact
```

Не переключаться автоматически на GitHub Issues или локальные markdown-задачи.

Проверить настроенные подключения:

```powershell
tea login list
```

Если подходящего подключения нет, предложить пользователю настроить его через
`tea login add`. Не запрашивать и не выводить токены в переписке или логах.

## Прокси

Рабочее окружение использует корпоративный прокси (`HTTP_PROXY` и `HTTPS_PROXY`),
через который `git.dementev.space` недоступен: запрос к API завершается ошибкой
`EOF`. Хост нужно добавить в `NO_PROXY`.

Разделитель — **запятая**, не точка с запятой: `tea` написан на Go, а Go
разбирает `NO_PROXY` по запятым, и хост после `;` не распознаётся.

На текущую сессию:

```powershell
$env:NO_PROXY = "$env:NO_PROXY,git.dementev.space"
```

Постоянно, в пользовательских переменных окружения (значение подхватят только
новые процессы):

```powershell
[Environment]::SetEnvironmentVariable("NO_PROXY", "$env:NO_PROXY,git.dementev.space", "User")
```

## Работа с задачами

Из рабочего дерева использовать Gitea remote `origin`:

```powershell
tea issues list --remote origin
tea issues create --remote origin
tea issues edit <index> --remote origin
tea labels list --remote origin
```

За пределами рабочего дерева явно указывать репозиторий
`ddmitry/local-transcriber` и настроенный Gitea login.

## Wayfinding operations

Навык `wayfinder` ведёт карту как issue с меткой `wayfinder:map`, а её тикеты —
как отдельные issue с метками `wayfinder:research`, `wayfinder:prototype`,
`wayfinder:grilling` и `wayfinder:task`.

### Принадлежность карте

Gitea 1.27 не имеет подзадач в API: среди эндпоинтов `issues/{index}` есть
`dependencies` и `blocks`, но родительских связей нет. Поэтому принадлежность
тикета карте выражается двумя способами сразу: меткой `wayfinder:<тип>` и первой
строкой тела со ссылкой на карту.

```markdown
Часть карты: [<заголовок карты>](<url>) (#<номер>)
```

### Блокировки

Блокировки — нативные зависимости Gitea, они отображаются в интерфейсе. Тикет
разблокирован, когда закрыты все блокирующие его тикеты.

```powershell
tea api --remote origin -X POST `
  repos/ddmitry/local-transcriber/issues/<блокируемый>/dependencies `
  -d '{"index": <блокирующий>, "owner": "ddmitry", "repo": "local-transcriber"}'
```

### Запросы фронтира

Фронтир — открытые, разблокированные и никому не назначенные тикеты карты.
Заявка на тикет — назначение его на себя до начала работы.

```powershell
tea issues list --remote origin --labels wayfinder:map
tea issues list --remote origin --labels wayfinder:research,wayfinder:prototype,wayfinder:grilling,wayfinder:task
tea api --remote origin repos/ddmitry/local-transcriber/issues/<номер>/dependencies
```

### Особенности `tea api`

Три вещи, на которых легко потерять время:

- **Путь без ведущего слэша.** `repos/{owner}/{repo}/...` работает,
  `/repos/...` возвращает `404 page not found`. Подстановка `{owner}` и `{repo}`
  из контекста репозитория при этом не срабатывает — писать владельца и имя явно.
- **Тело зависимости требует `owner` и `repo`.** Только `{"index": N}` даёт
  `repository does not exist [id: 0, uid: 0, owner_name: , name: ]`.
- **Код возврата не отражает HTTP-статус.** `tea api` завершается с нулевым
  кодом даже на 404, поэтому скрипты должны запрашивать `-i` и разбирать строку
  `HTTP/...` из stderr, иначе ошибки пройдут незамеченными.
