# 🚀 VibeRAG 🚀

[![VibeRAG Banner](media/vibeRAGbanner.jpg)](https://github.com/stinkgen/vibeRAG)

### _{ Seamlessly Fuse Your Knowledge with AI }_

---

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Docker Powered](https://img.shields.io/badge/Docker-Powered-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue?logo=python&logoColor=yellow)](https://www.python.org/)
[![FastAPI Backend](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React Frontend](https://img.shields.io/badge/React-Frontend-blue?logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Strictly%20Typed-blue?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)


---

**VibeRAG** ain't your grandpa's document tool. It's a Retrieval-Augmented Generation system packing a FastAPI backend punch, a slick React/TypeScript frontend, and Milvus vector muscle – all wrapped in a UI dripping with properly vibed aesthetic. Talk to your docs, generate content, make knowledge *work*.

## Core Features 🔥

*   **Doc Devourer:** Ingests PDFs, Markdown, TXT – chunks 'em, embeds 'em. Done.
*   **Milvus Vector Core:** Blazing-fast similarity search for finding the *exact* context you need.
*   **RAG Chat Engine:** Have actual *conversations* with your knowledge base. Uses LLMs (OpenAI/Ollama) fused with retrieved context. Remembers previous turns (no goldfish memory here).
*   **Source Linking:** Know *where* the info came from. Clickable sources on RAG responses.
*   **Real-time Streaming:** WebSocket magic for instant response delivery. No waiting.
*   **Persistent Memory (Client-Side):** Chat history sticks around in your browser's `localStorage`.
*   **Agentic Content Gen:**
    *   Whip up presentation outlines based on your docs.
    *   Generate structured research reports from knowledge + web results.
*   **Web Crawler Integration:** Pulls real-time info from the web using Google Custom Search to augment stored knowledge.
*   **Config Dashboard:** Tweak LLM settings, API keys, check provider status on the fly.

## The Tech Stack 🛠️

*   **Backend:** Python | FastAPI | Uvicorn
*   **Frontend:** TypeScript | React | Vite | `react-markdown` for slick rendering
*   **Vector DB:** Milvus
*   **LLM Providers:** OpenAI | Ollama
*   **Containerization:** Docker | Docker Compose
*   **API/WS Proxy:** Node.js/Express + `ws` (running *in* the frontend container)

## Ignition Sequence 👾 (Running with Docker Compose)

This is the way. Launches the whole matrix: backend, frontend, Milvus cluster.

**1. Gear Up:**

*   Docker & Docker Compose installed.
*   Git installed.

**2. Clone the Signal:**

```bash
git clone https://github.com/stinkgen/vibeRAG.git
cd vibeRAG
```

**3. Dial In the Environment (`.env.local`):**

*   Copy the template: `cp .env.example .env.local`
*   **CRITICAL:** Edit `.env.local` (`nano .env.local` or your weapon of choice).
    *   **REQUIRED:** Plug in your `OPENAI_API_KEY`.
    *   **REQUIRED (for Web Search):** Add `GOOGLE_SEARCH_API_KEY` and `GOOGLE_SEARCH_ENGINE_ID`.
    *   **Verify:** `OLLAMA_HOST` *must* point to your Ollama instance (defaults to `http://host.docker.internal:11434` for host machine access from Docker).
    *   Adjust `FRONTEND_PORT` / `BACKEND_PORT` only if the defaults (3000/8000) clash on your system.

**4. LAUNCH!**

```bash
docker compose up --build -d
```

*   `--build`: Rebuilds images if code changed (USE THIS after pulling updates or editing Dockerfiles/dependencies).
*   `-d`: Runs detached (in the background). Give it a minute for Milvus to boot.

**5. Jack In:**

*   **Frontend UI:** `http://localhost:3000` (or your custom `FRONTEND_PORT`)
*   **Backend API (Direct):** `http://localhost:8000` (or your custom `BACKEND_PORT`)

**6. Pull the Plug:**

```bash
docker compose down
```

*   Want to wipe the Milvus data too? `docker compose down -v`

**7. System Diagnostics (Logs):**

```bash
docker compose logs -f          # Tail ALL service logs
docker compose logs -f backend  # Backend specific logs
docker compose logs -f frontend # Frontend Node.js proxy logs
```

## Dev Mode & Hot-Reloading (Backend Only) ⚡

The `docker-compose.yml` is set up to mount your local `./backend/src` into the running backend container. Uvicorn's `--reload` flag means backend Python changes should auto-reload the server. Keep an eye on `docker compose logs -f backend` to see it happen.

*(Frontend hot-reloading inside Docker isn't configured – the current setup builds a production bundle.)*

## System Architecture Blueprint 🗺️

```plaintext
vibeRAG/
├── backend/             # FastAPI Microservice (Python)
│   ├── src/             # Core source code (modules: generation, retrieval, etc.)
│   ├── Dockerfile
│   └── requirements.txt # Python deps
├── frontend/            # React Frontend & Node.js Proxy
│   ├── public/          # Static assets
│   ├── src/             # React/TS source code (components, config, etc.)
│   ├── Dockerfile       # Builds React app -> Copies into Node.js server image
│   ├── package.json     # Node.js deps (React, Express, ws, etc.)
│   └── server.js        # Node.js server: Serves React build, proxies API/WebSocket calls
├── .env.example         # Template for environment variables
├── .env.local           # YOUR local secrets & config (GITIGNORED!)
├── .gitignore           # Tells Git what to ignore (node_modules, .env.local, etc.)
├── docker-compose.yml   # Defines and orchestrates all Docker services (backend, frontend, milvus)
├── README.md            # You are here.
└── ...                  # Config files, maybe future scripts
```

## Glitches & Gremlins (Known Issues) 🐛

*   **Startup Race Condition:** Frontend might load before the backend API is fully ready after `docker compose up`. If things seem borked, a quick browser refresh usually fixes it. Backend health checks are planned.
*   **PDF Download:** The `jspdf` presentation download feature needs more testing across different setups. Might be flaky.
*   **Web Search:** Needs to be revalidated following major changes to message handling.

## The Upgrade Path (Roadmap) 🌌

This rig is constantly evolving. Here's the upgrade manifest:

*   **User and Admin Profiles:** Make vibeRAG multi-user for maximum greatness!
*   **Better Config and LLM Management:** Add more flexible LLM router and configuration.
*   **Agent Command Center:** Dedicated UI for managing specialized AI agents (beyond just Presentation/Research). Think tuning, orchestration, the whole nine yards.
*   **Smarter Agents:** Jacking up the Presentation/Research agents with deeper reasoning, planning, and more tool integrations.
*   **Sensory Overload:** Adding Text-to-Speech (TTS), Speech-to-Text (STT), and Image Generation. Interact with data in new ways.
*   **Real Memory:** Swapping the temporary backend chat store for a persistent DB (Redis? Postgres? We'll see).
*   **Laser-Guided RAG:** Implementing backend support for advanced search `filters`.
*   **Instant Dev Feedback:** Configuring frontend Docker for proper hot-reloading.
*   **Security Lockdown:** Implementing real authentication/authorization.
*   **Battle Hardening:** Production-ready logging, monitoring, performance optimization.
*   **Chrome & Polish:** Continuous UI/UX refinement. Max VIBE.
*   **Bulletproof Bootup:** Better health checks for smoother startups.

## Freedom Protocol (License) 📜

MIT License. Go nuts. Build something cool.

## 📸 Screenshots

<details>
<summary>Click to view screenshots</summary>

[![VibeRAG Screenshot 1](media/vibeRAG_screen1.jpg)](media/vibeRAG_screen1.jpg)
*Document Manager - List View*

[![VibeRAG Screenshot 2](media/vibeRAG_screen2.jpg)](media/vibeRAG_screen2.jpg)
*Document Manager - Card View*

[![VibeRAG Screenshot 3](media/vibeRAG_screen3.jpg)](media/vibeRAG_screen3.jpg)
*Chat Interface*

[![VibeRAG Screenshot 4](media/vibeRAG_screen4.jpg)](media/vibeRAG_screen4.jpg)
*Configuration Page*

</details>

## 🚀 Getting Started