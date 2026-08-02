# Qwen Code CLI — установка, прямой доступ мимо прокси, цены

Ресёрч и настройка от **02.08.2026**. Задача: поставить [Qwen Code](https://docs.qwencloud.com/developer-guides/clients-and-developer-tools/qwen-code)
так, чтобы он ходил в Qwen Cloud **напрямую**, не через локальный прокси `127.0.0.1:10810`
(через который вынужденно ходит Claude Code).

Итог: **работает, прямой доступ подтверждён экспериментально.** Все выводы ниже — из фактических
запросов и грепа по бандлу, не из общих соображений.

---

## TL;DR

| Что | Значение |
|---|---|
| Пакет | `@qwen-code/qwen-code@0.21.3` глобально через npm |
| Бинарь | `qwen` (`C:\Program Files\nodejs\qwen.ps1`) |
| Конфиг | `~/.qwen/settings.json` + `~/.qwen/.env` |
| План | pay-as-you-go, endpoint `dashscope-intl` |
| Модель | `qwen3.7-max` |
| Прокси | **не нужен**, все endpoint'ы Qwen Cloud достижимы из РФ напрямую |

---

## 1. Установка

Официальный инсталлер (`install-qwen.bat --source bailian`) ставит Node.js + npm-пакет + пишет
`~/.qwen/source.json`. Node v24.18.0 уже стоял, поэтому взяли пакет напрямую:

```powershell
npm install -g @qwen-code/qwen-code@latest
qwen --version   # 0.21.3
```

`source.json` воспроизвели руками — это единственное, что скрипт делает сверх `npm i -g`
(значение используется только для трекинга источника установки):

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.qwen" | Out-Null
'{"source":"bailian"}' | Out-File "$env:USERPROFILE\.qwen\source.json" -Encoding utf8 -NoNewline
```

> npm предупреждает про непровёренные install-скрипты (`@qwen-code/audio-capture`, `sharp`) —
> на работу CLI это не влияет, `npm approve-scripts` не запускали.

---

## 2. Доступность Qwen Cloud из РФ — напрямую

Проверяли реальными запросами с `-Proxy $null`. **Прокси не требуется ни для одного endpoint'а:**

| Endpoint | Ответ |
|---|---|
| `dashscope-intl.aliyuncs.com/compatible-mode/v1` | 401 → достижим |
| `coding-intl.dashscope.aliyuncs.com/v1` | 200 |
| `token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | 401 → достижим |
| `chat.qwen.ai` (OAuth device flow) | 405 → достижим |
| `home.qwencloud.com` | 200 |

`401`/`405` здесь = сервер отвечает, просто нужен ключ/правильный метод. Это и есть доказательство
сетевой достижимости.

---

## 3. Конфигурация

Схему auth подтвердили грепом по бандлу, а не догадкой:

```bash
grep -roh "security[^,;]\{0,80\}selectedType" bundled/ chunks/
# → security.auth.selectedType
```

**`~/.qwen/settings.json`:**

```json
{
  "security": { "auth": { "selectedType": "openai" } },
  "$version": 4
}
```

(`$version` CLI дописывает сам при первом запуске.)

**`~/.qwen/.env`** — три профиля, активен ровно один, остальные закомментированы.
Переменные, которые читает CLI (тоже из грепа): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`,
а также `DASHSCOPE_API_KEY`, `BAILIAN_*_API_KEY`, `NO_PROXY`/`HTTPS_PROXY`.

| Профиль | Endpoint | Модель |
|---|---|---|
| **pay-as-you-go** (активен) | `dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen3.7-max` |
| Token Plan | `token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` | `qwen3.8-max-preview` |
| Coding Plan | `coding-intl.dashscope.aliyuncs.com/v1` | `qwen3-coder-plus` |

**Ключ у каждого плана свой** — от pay-as-you-go на Token Plan не подойдёт.

> `.env` лежит вне репы (`~/.qwen/`) и содержит живой ключ — в git не попадает.

---

## 4. Обход прокси — как доказали

Первый тест был некорректен: прокси на `:10810` живой, поэтому успешный ответ ничего не доказывал —
запрос мог просто пройти через него.

**Чистый тест:** подставили заведомо мёртвый порт. Если ответ придёт — значит трафик пошёл мимо.

```powershell
$env:HTTP_PROXY='http://127.0.0.1:9'; $env:HTTPS_PROXY='http://127.0.0.1:9'
qwen -p "Ответь ровно одним словом: 2+2?"
# → 4
```

Ответ пришёл ⇒ `NO_PROXY` из `.env` отрабатывает, `qwen` ходит напрямую.

Отдельно: `HTTP_PROXY`/`HTTPS_PROXY` **не заданы** на уровне User/Machine — они инжектятся только в
сессии Claude Code. Из обычного терминала `qwen` идёт напрямую и без `NO_PROXY`; строка в `.env`
нужна ровно для запусков из-под Claude Code (`!qwen`).

---

## 5. Доступность моделей по планам

Главная находка ресёрча — **`qwen3.8-max-preview` недоступна на pay-as-you-go**, только Token Plan.
Доки Qwen Cloud про pay-as-you-go пишут обтекаемо («самый широкий OpenAI-совместимый каталог»),
поимённого списка не дают — выяснилось только прицельным поиском.

| Модель | pay-as-you-go | Token Plan | Coding Plan |
|---|:---:|:---:|:---:|
| `qwen3.8-max-preview` | ✗ | ✓ | ✗ |
| `qwen3.7-max` | ✓ | ✓ | ✗ |
| `qwen3.7-plus` | ✓ | ✓ | ✓ |
| `qwen3.6-plus` | ✓ | ✓ | ✓ |
| `qwen3.6-flash` | ✓ | ✓ | ✗ |
| `qwen3-coder-plus` | ✗ | ✗ | ✓ |

Token Plan даёт ещё и не-Qwen модели: `glm-5.2`, `deepseek-v4-pro`, `deepseek-v4-flash-0731`;
Team Edition — дополнительно `kimi-k2.7-code`, `MiniMax-M2.5`, `qwen3.6-plus` и др.

---

## 6. Цены (на 02.08.2026)

| Модель | Input /1M | Output /1M | Контекст |
|---|---|---|---|
| `qwen3.7-max` | $2.50 (промо −50% → $1.25) | $7.50 (промо → $3.75) | 1M |
| `qwen3-coder-plus` | $1.00 | $5.00 | 1M |
| `qwen3.7-plus` | $0.40 (≤256K) / $1.20 (256K–1M) | $1.60 / $4.80 | 1M |
| `qwen3.7-flash` | $0.03 (≤32K) | $0.13 | — |

Прикидка на агентскую сессию ~50K вход / ~10K выход:

| Модель | За сессию |
|---|---|
| `qwen3.7-max` (лист) | ~$0.20 |
| `qwen3.7-max` (промо) | ~$0.10 |
| `qwen3-coder-plus` | ~$0.10 |
| `qwen3.7-plus` | ~$0.04 |
| `qwen3.7-flash` | ~$0.003 |

**Развенчали интуицию:** `qwen3-coder-plus` ($1/$5) сам по себе **дешевле** активной `qwen3.7-max`
($2.50/$7.50 по листу). Дорог не токен, а вход в Coding Plan — это фиксированная подписка
**от $50/мес**, которая не окупается на малых объёмах.

**Что помнить:** сейчас активна самая дорогая модель каталога, и промо −50% на неё когда-нибудь
кончится. Рабочей лошадкой разумнее `qwen3.7-plus` (в 5–6 раз дешевле, тот же ключ и endpoint,
меняется одна строка `OPENAI_MODEL`), а `qwen3.7-max` дёргать точечно.

---

## 7. Замеры

| Что | Результат |
|---|---|
| Прямой HTTP к `dashscope-intl` | 200, 10.0 с |
| `qwen -p` (CLI, односложный ответ) | 13.1 с |
| Токены на ответ «Париж» | `prompt=16`, **`completion=466`** |

**Важный практический вывод:** `qwen3.7-max` — reasoning-модель. На односложный ответ ушло
466 completion-токенов при выставленном `max_tokens=20` — **thinking-токены в лимит не укладываются,
но в счёт идут**. Output — самая дорогая статья, так что короткий ответ ≠ дешёвый запрос.
Отсюда же задержка 10–13 с на тривиальный вопрос. Для массовых прогонов — `qwen3.6-flash`
или отключение thinking там, где он не нужен.

---

## 8. Грабли

1. **PowerShell 5.1 не знает `-SkipHttpErrorCheck`** (это PS 7+). Проверка достижимости endpoint'ов
   падала с «A parameter cannot be found». Обход: `try/catch [System.Net.WebException]` и читать
   `[int]$_.Exception.Response.StatusCode` — тогда 401/404 видны как «достижим».
2. **Кракозябры в выводе** (`Ð Ð°Ð±Ð¾ÑÐ°ÐµÑ` вместо «Работает») — это консоль PS 5.1, а не API.
   Лечится `[Console]::OutputEncoding = [Text.Encoding]::UTF8`.
3. **Успешный запрос при живом прокси ничего не доказывает** про обход — нужен заведомо мёртвый
   прокси-порт как контроль (см. §4).
4. Доки Qwen Cloud **не перечисляют модели pay-as-you-go поимённо** — доступность приходится
   проверять эмпирически или прицельным поиском.

---

## Полезные команды

```powershell
# Интерактивный TUI
qwen

# Неинтерактивно (для скриптов)
qwen -p "запрос"

# Сменить модель / план — одна-три строки
notepad $env:USERPROFILE\.qwen\.env
```

## Ссылки

- [Qwen Code — доки Qwen Cloud](https://docs.qwencloud.com/developer-guides/clients-and-developer-tools/qwen-code)
- [Pricing](https://docs.qwencloud.com/developer-guides/getting-started/pricing)
- [API-ключи](https://home.qwencloud.com/api-keys)
- [Coding Plan](https://www.qwencloud.com/pricing/coding-plan)
