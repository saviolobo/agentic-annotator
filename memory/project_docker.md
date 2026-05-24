---
name: project-docker
description: Docker is available locally for this project; Redis Stack runs via docker-compose
metadata:
  type: project
---

Docker is available on the user's machine. Redis Stack will be run via docker-compose for guidelines-mcp vector search.

**Why:** User confirmed Docker available when asked before Phase 2 started.
**How to apply:** Include Redis Stack in docker-compose.yml from Phase 2 Step 2 onwards; do not use file-backed fallback for vector search.
