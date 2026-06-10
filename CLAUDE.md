# NeuroBridge API — backend (Python)

Backend BaaS-платформы биокомпьютинга. Frontend в `~/neurobridge`. 132 endpoint, world-class rewrite.

## ⚡ ПЕРЕД любой работой — прочитай vault:

1. `/Users/lucky/vault/Биокомпьютинг/Биокомпьютинг.md` — overview (если есть)
2. `/Users/lucky/vault/Биокомпьютинг/Recent Decisions.md` — последние решения
3. `/Users/lucky/vault/Биокомпьютинг/Open Questions.md` — что висит
4. Свежий `Сессия [дата] — context for next.md` (auto-инжектится через SessionStart hook)

## ⚡ В КОНЦЕ сессии:

1. Обнови `Recent Decisions.md` (новые сверху, с датой)
2. Обнови `Open Questions.md`
3. Создай `Сессия [сегодня] — context for next.md` в `~/vault/Биокомпьютинг/`. Прошлый → `Архив сессий/`
4. Обнови auto-memory `~/.claude/projects/-Users-lucky/memory/project_neurobridge.md`

## Vault — общий с frontend

`~/vault/Биокомпьютинг/` — оба repo (frontend + backend) пишут handoff'ы сюда же. Разделяем разделом «Backend changes» в Recent Decisions если нужно.

## Frontend repo

`~/neurobridge` — Next.js dashboard, тот же vault folder.

## API format compat

World-class rewrite изменил форматы — backend поддерживает оба варианта во время transition (см. memory `feedback_api_format_compat.md`).
