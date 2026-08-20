---
name: naming-conventions
description: This skill provides guidance for naming modules, classes, functions, variables, components, and other identifiers in Python and TypeScript/React code. Use this skill when writing new code, reviewing code, or any task involving identifier names.
---

## Steps

### 1. Match the Case Style to the Language

Each language has one correct case style. Never carry one language's style into
the other.

#### Python (PEP 8)

| Element                | Style                 | Example                 |
| ---------------------- | --------------------- | ----------------------- |
| Module and package     | `snake_case`          | `rag_pipeline.py`       |
| Function and method    | `snake_case`          | `build_qdrant_client()` |
| Variable and parameter | `snake_case`          | `document_repository`   |
| Class                  | `PascalCase`          | `QdrantEmbeddingIndex`  |
| Constant               | `UPPER_SNAKE_CASE`    | `EMBEDDING_DIM`         |
| Settings field         | `UPPER_SNAKE_CASE`    | `LLM_BASE_URL`          |
| Module-private         | `_leading_underscore` | `_FORWARDED_ROLES`      |
| Type variable          | `PascalCase`          | `DocumentT`             |

A single leading underscore marks a module-private name. This is correct
Python. Do not use a double leading underscore. Do not use a trailing
underscore unless you must avoid a keyword clash.

#### TypeScript and React

| Element               | Style              | Example                      |
| --------------------- | ------------------ | ---------------------------- |
| File                  | `snake_case.tsx`   | `chat_window.tsx`            |
| React component       | `PascalCase`       | `ChatWindow`                 |
| Hook                  | `useCamelCase`     | `useAgentRun`                |
| Variable and function | `camelCase`        | `pythonApiUrl`               |
| Type and interface    | `PascalCase`       | `RunAgentInput`              |
| Constant              | `camelCase`        | `defaultAgentUrl`            |
| CSS module class      | `camelCase`        | `styles.chatPanel`           |
| Environment variable  | `UPPER_SNAKE_CASE` | `NEXT_PUBLIC_PYTHON_API_URL` |

A file name uses `snake_case`. The exported component inside it uses
`PascalCase`. This repo already follows that pairing:
`chat_window.tsx` exports `ChatWindow`.

Never use `snake_case` for a TypeScript variable. Never use `camelCase` for a
Python function.

### 2. NO ABBREVIATIONS

Always use full, descriptive names. This rule applies to both languages.

| Bad (Abbreviated) | Good (Python)          | Good (TypeScript)      |
| ----------------- | ---------------------- | ---------------------- |
| `ct`              | `cancellation_token`   | `cancellationToken`    |
| `cfg`             | `configuration`        | `configuration`        |
| `ctx`             | `context`              | `context`              |
| `req` / `res`     | `request` / `response` | `request` / `response` |
| `msg`             | `message`              | `message`              |
| `conn`            | `connection`           | `connection`           |
| `repo`            | `repository`           | `repository`           |
| `db`              | `database`             | `database`             |
| `emb`             | `embedding`            | `embedding`            |
| `doc`             | `document`             | `document`             |
| `svc`             | Use a pattern name     | Use a pattern name     |
| `mgr`             | Use a pattern name     | Use a pattern name     |
| `util`            | Use a pattern name     | Use a pattern name     |

**Allowlist.** These domain terms are names, not abbreviations. Do not expand
them: `llm`, `rag`, `pii`, `api`, `url`, `uri`, `id`, `sse`, `orm`,
`baml`, `bm25`, `vad`, `json`, `http`, `ttl`.

### 3. Enums and Literals, NOT Bare Strings

Never compare against a bare string. Give the set of values a type.

**Python** — use `Literal` for settings and small closed sets. Use
`enum.Enum` when the value needs methods or a stable wire format.

| Bad                             | Good                                                   |
| ------------------------------- | ------------------------------------------------------ |
| `provider: str = "mock"`        | `provider: Literal["mock", "openai_compatible"]`       |
| `if status == "complete":`      | `if status is DocumentStatus.COMPLETE:`                |
| `role in ["user", "assistant"]` | `role in FORWARDED_ROLES` (a module-level `frozenset`) |

**TypeScript** — use a union type or `as const`. Do not use a TypeScript
`enum`; it emits runtime code and does not narrow as well as a union.

| Bad                             | Good                                            |
| ------------------------------- | ----------------------------------------------- |
| `status: string`                | `status: "idle" \| "running" \| "error"`        |
| `enum Role { User, Assistant }` | `const roles = ["user", "assistant"] as const;` |

**Enum guidelines:**

- Use a singular name: `DocumentStatus` not `DocumentStatuses`.
- Give every Python `Enum` member an explicit value. Never rely on `auto()`
  when the value is persisted or sent over the wire. A renumbered member
  silently corrupts every stored row.
- Prefer a `str` mixin (`class DocumentStatus(str, Enum)`) when the value
  crosses a JSON or database boundary. The value stays readable and stable.

### 4. NO MAGIC STRINGS

Never hardcode a configuration string in the middle of a function.

| String Type                      | Location                                        |
| -------------------------------- | ----------------------------------------------- |
| Any environment-driven value     | A field on `Settings` in `config.py`            |
| Collection, table, or queue name | A field on `Settings`                           |
| Model name or base URL           | A field on `Settings`                           |
| Fixed protocol or header value   | Module-level constant, `UPPER_SNAKE_CASE`       |
| A closed set of values           | `Literal` type or `Enum`                        |
| User-facing text (React)         | A constant near the component, or a labels prop |
| Route path shared by two modules | Module-level constant                           |

Read configuration from the `settings` object. Never read `os.environ`
directly outside `config.py`.

### 5. Name Async Functions for What They Return

Do not add an `async` or `_async` suffix. Python and TypeScript both make
async visible in the signature.

| Bad                    | Good             |
| ---------------------- | ---------------- |
| `get_document_async()` | `get_document()` |
| `fetchDataAsync()`     | `fetchData()`    |

Name a generator for the stream it yields: `run_chat_agent()` yields events,
`stream_chunks()` yields chunks.

### 6. Module-Level Functions Over Utility Classes

Python has module-level functions. Use them. Do not create a class whose only
job is to hold static methods.

| Scenario                       | Good                                             | Bad                                  |
| ------------------------------ | ------------------------------------------------ | ------------------------------------ |
| Pure helper                    | `def to_snake_case(value: str) -> str:`          | `class StringUtils:` with statics    |
| Client construction            | `def build_redis_client(settings) -> Redis:`     | `class RedisFactory` with one method |
| Behavior that needs state      | A class with `__init__` dependencies             | A module global                      |
| Behavior that needs a resource | A class the caller constructs once in `lifespan` | A function that builds it per call   |

In TypeScript, export a plain function. Do not wrap it in a class or a
namespace.
