# slovo-llm

Локальный LLM-провайдер для проекта **slovo** на базе Ollama.
Запускает [Laguna XS.2](https://ollama.com/library/laguna-xs.2) — MoE-модель 33B/3B активных параметров, заточенную под агентское кодирование.

Используется как fallback / приватный inference рядом с основным провайдером (Claude API).

---

## Железо (целевая конфигурация)

- **CPU:** Intel Core i9-11900K (8C/16T)
- **RAM:** 64 GB DDR4
- **GPU:** NVIDIA RTX 4070 Ti SUPER, 16 GB GDDR6X
- **OS:** Windows 10 + Docker Desktop (WSL2 backend)

На таком железе Laguna XS.2 в кванте `q4_K_M` (23 GB) запускается со split'ом GPU+CPU
(25/41 слой на GPU, 16 на CPU при контексте 32K) и выдаёт **~11 tok/s** на агентских
задачах. Bottleneck — CPU-слои; GPU при этом загружена на 30–40 % и потребляет 40 Вт
из 285 Вт лимита. Подробный разбор и история оптимизаций — в `CLAUDE.md`.

---

## Требования

- Docker Desktop с включённым **WSL2 backend**
- **NVIDIA Container Toolkit** (включается в Docker Desktop → Settings → Resources → WSL Integration → enable GPU)
- Свежий драйвер NVIDIA (≥ 550.x)
- Свободно ≥ 30 GB на диске под образ модели

Проверка, что Docker видит GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Должна вывестись таблица с RTX 4070 Ti SUPER.

---

## Структура

```
slovo-llm/
├── .env                  # COMPOSE_PROJECT_NAME=slovo-llm
├── docker-compose.yml    # Ollama сервис
├── gpu-monitor.sh        # Real-time GPU+container monitoring (bash)
├── gpu-monitor.ps1       # Real-time GPU+container monitoring (PowerShell)
└── README.md             # этот файл
```

---

## Запуск

### 1. Поднять контейнер

```bash
cd slovo-llm
docker compose up -d
```

Проверить, что взлетело:

```bash
docker compose ps
docker compose logs -f ollama
```

### 2. Скачать модель

Образ Ollama стартует пустой — модели качаются отдельно.

```bash
docker exec -it ollama-laguna ollama pull laguna-xs.2:q4_K_M
```

Это ~23 GB, лучше запускать на ночь или с хорошим интернетом.

### 3. Прогреть модель

Чтобы первый запрос из slovo не висел минуту:

```bash
docker exec -it ollama-laguna ollama run laguna-xs.2:q4_K_M "hi"
```

### 4. Проверить из NestJS / curl

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "laguna-xs.2:q4_K_M",
  "messages": [{"role": "user", "content": "Напиши hello world на TypeScript"}],
  "stream": false
}'
```

---

## Конфигурация Ollama (через ENV)

| Переменная | Значение | Зачем |
|---|---|---|
| `OLLAMA_HOST` | `0.0.0.0:11434` | Слушать на всех интерфейсах внутри контейнера |
| `OLLAMA_MAX_LOADED_MODELS` | `1` | Не держать несколько моделей в памяти одновременно |
| `OLLAMA_NUM_PARALLEL` | `1` | Один параллельный слот. На single-user конфигурации `2` только удваивает KV-cache и выкидывает слои на CPU |
| `OLLAMA_KEEP_ALIVE` | `24h` | Не выгружать модель после запроса (важно для агентских сессий) |
| `OLLAMA_KV_CACHE_TYPE` | `q8_0` | Квантование KV-кэша. `q8_0` — баланс памяти и качества. На текущей конфигурации `q4_0` не даёт прироста скорости (упирается в compute buffers), а `f16` режет GPU layers — см. `CLAUDE.md` |
| `OLLAMA_FLASH_ATTENTION` | `1` | Включить Flash Attention — критично для скорости |
| `OLLAMA_CONTEXT_LENGTH` | `32768` | Дефолтный контекст 32K (Laguna умеет 128K, но это дороже по памяти) |

Лимит памяти контейнера: **40 GB** (оставляем системе + slovo dev-стеку ~24 GB).

---

## Доступные кванты Laguna XS.2

| Тег | Размер | Комментарий |
|---|---|---|
| `laguna-xs.2:q4_K_M` | 23 GB | **Рекомендуется** для этой машины |
| `laguna-xs.2:latest` | 23 GB | То же самое |
| `laguna-xs.2:q8_0` | 37 GB | Влезет в RAM, но GPU split не получится → медленно |
| `laguna-xs.2:nvfp4` | 22 GB | Только для RTX 50xx/Blackwell |
| `laguna-xs.2:bf16` | 67 GB | Не запустится |

---

## Мониторинг

В одном терминале:

```bash
nvidia-smi -l 1
```

В другом:

```bash
docker stats ollama-laguna
```

Для удобного live-мониторинга GPU + container memory:

```bash
# bash
./gpu-monitor.sh ollama-laguna

# PowerShell  
.\gpu-monitor.ps1
```

Смотрим:
- VRAM на GPU (должно быть занято ~14–15 GB из 16)
- RAM контейнера (должно быть ~10–15 GB поверх VRAM, остальные слои в системной памяти)
- Загрузка GPU при запросе — должна прыгать к 80–100%

---

## Подключение из slovo (NestJS)

Если slovo запускается на хосте (не в Docker) — обращаться к `http://localhost:11434`.

Если slovo тоже в Docker — добавить контейнер slovo в сеть `slovo-llm-net`:

```yaml
# в docker-compose.yml самого slovo
services:
  slovo-api:
    # ...
    networks:
      - slovo-llm-net
networks:
  slovo-llm-net:
    external: true
```

Тогда из slovo Ollama доступна по адресу `http://ollama-laguna:11434`.

### Пример NestJS-провайдера

```typescript
// llm/providers/ollama.provider.ts
import { Injectable } from '@nestjs/common';

@Injectable()
export class OllamaProvider {
  private readonly baseUrl = process.env.OLLAMA_URL ?? 'http://localhost:11434';
  private readonly model = 'laguna-xs.2:q4_K_M';

  async chat(messages: Array<{ role: string; content: string }>) {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.model,
        messages,
        stream: false,
        options: {
          temperature: 0.7,
          top_k: 20,
          num_ctx: 32768,
        },
      }),
    });
    return response.json();
  }
}
```

---

## Команды на каждый день

```bash
# Поднять
docker compose up -d

# Логи
docker compose logs -f ollama

# Остановить (модель выгрузится из памяти)
docker compose stop

# Снести контейнер (volume с моделями сохранится)
docker compose down

# Полный снос вместе с моделями (осторожно — потеряешь 23 GB закачки)
docker compose down -v

# Список установленных моделей
docker exec -it ollama-laguna ollama list

# Удалить модель из контейнера
docker exec -it ollama-laguna ollama rm laguna-xs.2:q4_K_M

# Обновить образ Ollama
docker compose pull
docker compose up -d
```

---

## Troubleshooting

**Контейнер не видит GPU:**
- Docker Desktop → Settings → Resources → WSL Integration → проверить, что включена интеграция с дистрибутивом и GPU
- Проверить драйвер: `nvidia-smi` на хосте → должна быть видна 4070 Ti SUPER
- Проверка: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

**OOM при запуске модели:**
- Уменьшить `OLLAMA_CONTEXT_LENGTH` до `16384` или `8192`
- Закрыть Chrome / IDE, проверить `docker stats`
- Убедиться что не запущен другой образ на GPU (`nvidia-smi`)

**Скорость <10 tok/s:**
- Проверить, что модель реально села на GPU: `nvidia-smi` во время генерации, GPU util должен скакать (на текущей конфигурации — 30–40 %)
- Проверить переменную `OLLAMA_FLASH_ATTENTION=1`
- В логах `docker compose logs ollama` найти `offloaded N/41 layers to GPU` — на текущей конфигурации должно быть 25/41
- Возможно, выбран `q8_0` веса — они не влезают в 16 GB VRAM, перейти на `q4_K_M`

**Healthcheck падает:**
- `start_period: 60s` — даём Ollama время подняться. Если не хватает, увеличить до 120s
- Проверить логи: `docker compose logs ollama`

---

## Ссылки

- [Laguna XS.2 на Ollama](https://ollama.com/library/laguna-xs.2)
- [Poolside blog: Laguna deeper dive](https://poolside.ai/blog/laguna-a-deeper-dive)
- [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Ollama env variables](https://github.com/ollama/ollama/blob/main/docs/faq.md)
