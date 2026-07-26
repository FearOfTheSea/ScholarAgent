# Folder Guide

## Directory structure

```mermaid
flowchart TD
    root["ScholarAgent"] --> source["src/scholar_agent"]
    root --> tests["tests"]
    root --> documentation["docs"]
    source --> domain["domain"]
    source --> application["application"]
    source --> infrastructure["infrastructure"]
    source --> presentation["presentation"]
    source --> configuration["config"]
    source --> shared["shared"]
    application --> input_ports["input_ports"]
    application --> output_ports["output_ports"]
    application --> use_cases["use_cases"]
    infrastructure --> adapters["adapters"]
    infrastructure --> di["di"]
    presentation --> api["api"]
    presentation --> web["web"]
```

| Folder | Responsibility | May depend on |
| --- | --- | --- |
| `domain` | Business concepts and repository contracts | Standard library only |
| `application` | Use cases, ports, DTOs, validators | `domain` |
| `infrastructure` | Port implementations and DI setup | `application`, `domain`, configuration |
| `presentation` | HTTP and Streamlit delivery adapters | `application`, composition root |
| `config` | Environment-backed settings | Pydantic settings |
| `shared` | Stable cross-cutting primitives | Standard library where possible |
| `tests` | Boundary-focused verification | Public package interfaces |
| `docs` | Architecture and contributor guidance | None |

Each source folder includes a short README that documents its local purpose and
dependency boundary.

