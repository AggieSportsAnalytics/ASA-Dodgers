# LA Dodgers Scouting — Frontend

Vite + React app for uploading pitch clips, running ReleaseFrame analysis, and viewing pitcher profiles.

## Prerequisites

- Node.js 18+
- [ReleaseFrame API](../ReleaseFrame) running (default `http://localhost:8000`)

## Setup

```bash
npm install
cp .env.example .env.local   # optional — edit for ngrok proxy
npm run dev
```

Open http://localhost:5173

## API URL

- **Local API:** leave proxy off; set the API base in the Upload page to `http://localhost:8000`, or use defaults in storage.
- **Ngrok:** set `VITE_RELEASEFRAME_USE_PROXY=1` and `VITE_PROXY_API_TARGET` in `.env.local`, restart `npm run dev`. Requests go through the Vite proxy at `/api/analyze`.

## Build

```bash
npm run build
npm run preview
```

## Scripts

| Script | Description |
|--------|-------------|
| `npm run dev` | Dev server on port 5173 |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview production build |
