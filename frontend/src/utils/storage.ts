import type { AnalysisRecord } from "../types";

const HISTORY_KEY = "releaseframe_history_v1";
const API_URL_KEY = "releaseframe_api_url_v1";

const ENV_PROXY = (import.meta as { env?: Record<string, string> }).env;
const USE_PROXY = ENV_PROXY?.VITE_RELEASEFRAME_USE_PROXY === "1";
const PROXY_LABEL =
  ENV_PROXY?.VITE_RELEASEFRAME_PROXY_LABEL ??
  ENV_PROXY?.VITE_PROXY_API_TARGET ??
  "";

export function getApiUrl(): string {
  if (USE_PROXY) {
    return "/api/analyze";
  }
  try {
    const saved = window.localStorage.getItem(API_URL_KEY);
    if (saved && saved.trim()) {
      const base = saved.trim().replace(/\/+$/, "");
      return `${base}/analyze`;
    }
  } catch {
    // ignore
  }
  return "http://localhost:8000/analyze";
}

export function getApiBase(): string {
  if (USE_PROXY) return PROXY_LABEL || "(proxy)";
  try {
    return window.localStorage.getItem(API_URL_KEY) || "";
  } catch {
    return "";
  }
}

export function setApiBase(base: string): void {
  try {
    window.localStorage.setItem(API_URL_KEY, base);
  } catch {
    // ignore
  }
}

function readRaw(): AnalysisRecord[] {
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as AnalysisRecord[]) : [];
  } catch {
    return [];
  }
}

function writeRaw(records: AnalysisRecord[]): boolean {
  try {
    window.localStorage.setItem(HISTORY_KEY, JSON.stringify(records));
    return true;
  } catch {
    return false;
  }
}

export function historyRecordMergeKey(r: AnalysisRecord): string {
  return r.id || `${r.videoName}::${r.createdAt}::${r.releaseFrame}`;
}

export async function loadHistoryMerged(): Promise<AnalysisRecord[]> {
  return readRaw();
}

export function addRecords(
  records: AnalysisRecord[],
): { history: AnalysisRecord[]; persisted: boolean } {
  const cur = readRaw();
  const byKey = new Map<string, AnalysisRecord>();
  for (const r of cur) byKey.set(historyRecordMergeKey(r), r);
  for (const r of records) byKey.set(historyRecordMergeKey(r), r);
  const next = [...byKey.values()].sort(
    (a, b) =>
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
  );
  const persisted = writeRaw(next);
  return { history: next, persisted };
}

export function deleteRecord(
  mergeKey: string,
): { history: AnalysisRecord[]; persisted: boolean } {
  const next = readRaw().filter((r) => historyRecordMergeKey(r) !== mergeKey);
  const persisted = writeRaw(next);
  return { history: next, persisted };
}

export function clearHistory(): boolean {
  return writeRaw([]);
}

export function bindHistoryStorageSync(
  onChange: (records: AnalysisRecord[]) => void,
): () => void {
  const handler = (e: StorageEvent) => {
    if (e.key === HISTORY_KEY) onChange(readRaw());
  };
  window.addEventListener("storage", handler);
  return () => window.removeEventListener("storage", handler);
}
