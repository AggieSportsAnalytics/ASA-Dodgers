import type { AnalysisRecord } from "../types";

const UNTAGGED = "untagged";

export function pitcherKeyFromRecord(r: AnalysisRecord): string | null {
  const p = r.player;
  if (!p) return null;
  if (p.id?.trim()) return p.id.trim();
  if (p.label?.trim()) return p.label.trim();
  if (p.jerseyNumber?.trim()) return `#${p.jerseyNumber.trim()}`;
  return null;
}

export function effectivePlayerLabelFromRecord(
  r: AnalysisRecord,
): string | null {
  const p = r.player;
  if (!p) return null;
  if (p.label?.trim()) return p.label.trim();
  if (p.jerseyNumber?.trim()) return `#${p.jerseyNumber.trim()}`;
  return null;
}

export function getPitcherDisplayTitle(key: string): string {
  if (!key || key === UNTAGGED) return "Untagged pitcher";
  return key;
}

export function distinctPitcherKeysFromRecords(
  records: AnalysisRecord[],
): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const r of records) {
    const k = pitcherKeyFromRecord(r);
    if (k && !seen.has(k)) {
      seen.add(k);
      out.push(k);
    }
  }
  return out;
}

export function recordsMatchingPitcherKey(
  records: AnalysisRecord[],
  key: string,
): AnalysisRecord[] {
  return records.filter((r) => pitcherKeyFromRecord(r) === key);
}
