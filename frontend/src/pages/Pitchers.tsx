import React, { useMemo, useState } from "react";
import { ArrowUpDown, ListFilter } from "lucide-react";
import type { AnalysisRecord } from "../types";
import { ArmSlotBadge } from "../components/ArmSlotBadge";
import { angleToSlot, type ArmSlotId, ARM_SLOT_ORDER, ARM_SLOT_SHORT } from "../lib/armSlot";
import {
  distinctPitcherKeysFromRecords,
  recordsMatchingPitcherKey,
  getPitcherDisplayTitle,
} from "../lib/player";

type Variant = "pitching" | "batting";

type PitchersProps = {
  variant?: Variant;
  records: AnalysisRecord[];
  onOpenProfile?: (pitcherKey: string) => void;
};

type Row = {
  key: string;
  name: string;
  count: number;
  avgAngle: number | null;
  avgSlot: ArmSlotId | "unknown";
};

const Pitchers: React.FC<PitchersProps> = ({
  variant = "pitching",
  records,
  onOpenProfile,
}) => {
  const [query, setQuery] = useState("");
  const [sortDir, setSortDir] = useState<"asc" | "desc" | null>(null);
  const [slotFilter, setSlotFilter] = useState<Set<ArmSlotId>>(
    () => new Set(ARM_SLOT_ORDER),
  );
  const [sortOpen, setSortOpen] = useState(false);
  const [filterOpen, setFilterOpen] = useState(false);

  const rosterLabel = variant === "batting" ? "Batters" : "Pitchers";
  const title =
    variant === "batting" ? "Batter Profiles" : "Pitcher Profiles";

  const rows: Row[] = useMemo(() => {
    const keys = distinctPitcherKeysFromRecords(records);
    return keys.map((key) => {
      const subset = recordsMatchingPitcherKey(records, key);
      const angles = subset
        .map((r) => r.angles.atRelease)
        .filter((v): v is number => typeof v === "number");
      const avg =
        angles.length > 0
          ? angles.reduce((s, v) => s + v, 0) / angles.length
          : null;
      return {
        key,
        name: getPitcherDisplayTitle(key),
        count: subset.length,
        avgAngle: avg,
        avgSlot: avg !== null ? angleToSlot(avg) : "unknown",
      };
    });
  }, [records]);

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    let out = rows.filter((r) => {
      if (q && !r.name.toLowerCase().includes(q)) return false;
      if (r.avgSlot === "unknown") return slotFilter.size === ARM_SLOT_ORDER.length;
      return slotFilter.has(r.avgSlot);
    });
    if (sortDir) {
      const dir = sortDir === "asc" ? 1 : -1;
      out = [...out].sort((a, b) => {
        const va = a.avgAngle ?? -1;
        const vb = b.avgAngle ?? -1;
        return (va - vb) * dir;
      });
    }
    return out;
  }, [rows, query, slotFilter, sortDir]);

  const toggleSlot = (s: ArmSlotId) => {
    setSlotFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  };

  return (
    <section className="ps-pitchers-page" aria-labelledby="pitchers-h">
      <h1 id="pitchers-h" className="ps-page-title">
        {title}
      </h1>

      <div className="ps-stats-toolbar">
        <div className="ps-dropdown-wrap">
          <button
            type="button"
            className={`ps-btn ps-btn--primary ps-btn--icon${filterOpen ? " is-active" : ""}`}
            onClick={() => {
              setFilterOpen((o) => !o);
              setSortOpen(false);
            }}
          >
            <ListFilter size={16} strokeWidth={2.5} />
            Filter
          </button>
          {filterOpen && (
            <div className="ps-dropdown ps-dropdown--filter">
              <div className="ps-dropdown__group">
                <span className="ps-dropdown__label">Avg slot</span>
                <div className="ps-slot-chips">
                  {ARM_SLOT_ORDER.map((s) => (
                    <label
                      key={s}
                      className={`ps-slot-chip${slotFilter.has(s) ? " is-on" : ""}`}
                    >
                      <input
                        type="checkbox"
                        checked={slotFilter.has(s)}
                        onChange={() => toggleSlot(s)}
                      />
                      {ARM_SLOT_SHORT[s]}
                    </label>
                  ))}
                </div>
              </div>
              <button
                type="button"
                className="ps-dropdown__reset"
                onClick={() => setSlotFilter(new Set(ARM_SLOT_ORDER))}
              >
                Reset
              </button>
            </div>
          )}
        </div>
        <div className="ps-dropdown-wrap">
          <button
            type="button"
            className={`ps-btn ps-btn--primary ps-btn--icon${sortOpen ? " is-active" : ""}`}
            onClick={() => {
              setSortOpen((o) => !o);
              setFilterOpen(false);
            }}
          >
            <ArrowUpDown size={16} strokeWidth={2.5} />
            Sort
          </button>
          {sortOpen && (
            <div className="ps-dropdown ps-dropdown--sort">
              <button
                type="button"
                className={`ps-dropdown__item${sortDir === "asc" ? " is-active" : ""}`}
                onClick={() => {
                  setSortDir("asc");
                  setSortOpen(false);
                }}
              >
                <span className="ps-dropdown__item-title">Arm Angle:</span>
                <span className="ps-dropdown__item-sub">Lowest to Highest</span>
              </button>
              <button
                type="button"
                className={`ps-dropdown__item${sortDir === "desc" ? " is-active" : ""}`}
                onClick={() => {
                  setSortDir("desc");
                  setSortOpen(false);
                }}
              >
                <span className="ps-dropdown__item-title">Arm Angle:</span>
                <span className="ps-dropdown__item-sub">Highest to Lowest</span>
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="ps-table-card">
        <div className="ps-table-card__head">
          <h2 className="ps-table-card__title">{rosterLabel}</h2>
          <label className="ps-search">
            <span>Search:</span>
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </label>
        </div>
        <div className="ps-table-wrap">
          <table className="ps-table">
            <thead>
              <tr>
                <th className="ps-col-hash">#</th>
                <th>Name</th>
                <th>Avg Slot</th>
                <th>Avg Arm Angle</th>
              </tr>
            </thead>
            <tbody>
              {visible.length === 0 ? (
                <tr>
                  <td colSpan={4} className="ps-table__empty">
                    {records.length === 0
                      ? "Empty — no analyses yet."
                      : "No pitchers match your filter."}
                  </td>
                </tr>
              ) : (
                visible.map((r, i) => {
                  const clickable = Boolean(onOpenProfile);
                  return (
                    <tr
                      key={r.key}
                      className={clickable ? "ps-row-clickable" : undefined}
                      onClick={
                        clickable
                          ? () => onOpenProfile?.(r.key)
                          : undefined
                      }
                      onKeyDown={
                        clickable
                          ? (e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                onOpenProfile?.(r.key);
                              }
                            }
                          : undefined
                      }
                      tabIndex={clickable ? 0 : undefined}
                      role={clickable ? "button" : undefined}
                      aria-label={
                        clickable ? `Open profile for ${r.name}` : undefined
                      }
                    >
                      <td className="ps-col-hash">{i + 1}</td>
                      <td>{r.name}</td>
                      <td>
                        <ArmSlotBadge slot={r.avgSlot} angleDeg={r.avgAngle} />
                      </td>
                      <td>
                        {r.avgAngle !== null
                          ? `${r.avgAngle.toFixed(1)}°`
                          : "--"}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
};

export default Pitchers;
