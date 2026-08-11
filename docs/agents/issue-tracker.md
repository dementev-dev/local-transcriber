# Issue Tracker

Задачи проекта ведутся в Gitea-репозитории `ddmitry/local-transcriber`.

- Основной remote: `origin`
- Gitea: `https://git.dementev.space`
- CLI: `tea`
- Remote `github` является зеркалом и не используется для управления задачами
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
