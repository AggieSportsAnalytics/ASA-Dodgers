import React, { useEffect, useMemo, useRef, useState } from "react";
import { ArrowUpDown, ListFilter, MessageSquare, Send, Trash2 } from "lucide-react";
import type { AnalysisRecord } from "../types";
import { ArmSlotBadge } from "../components/ArmSlotBadge";
import { ImageModal } from "../components/ImageModal";
import {
  ARM_SLOT_LABEL,
  ARM_SLOT_ORDER,
  ARM_SLOT_SHORT,
  angleToSlot,
  type ArmSlotId,
} from "../lib/armSlot";
import { effectivePlayerLabelFromRecord } from "../lib/player";
import {
  clearHistory,
  deleteRecord,
  historyRecordMergeKey,
} from "../utils/storage";

type StatsProps = {
  records: AnalysisRecord[];
  onUpdate: (next: AnalysisRecord[]) => void;
  onGoUpload?: () => void;
};

type SortMode = "asc" | "desc";

const formatDate = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

const Stats: React.FC<StatsProps> = ({ records, onUpdate, onGoUpload }) => {
  const [filterOpen, setFilterOpen] = useState(false);
  const [sortOpen, setSortOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [activeSlots, setActiveSlots] = useState<Set<ArmSlotId>>(
    () => new Set(ARM_SLOT_ORDER),
  );
  const [minDeg, setMinDeg] = useState<string>("");
  const [maxDeg, setMaxDeg] = useState<string>("");
  const [sortMode, setSortMode] = useState<SortMode | null>(null);
  const [activeImage, setActiveImage] = useState<{
    src: string;
    caption: string;
  } | null>(null);

  const filterRef = useRef<HTMLDivElement | null>(null);
  const sortRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (filterRef.current && !filterRef.current.contains(e.target as Node)) {
        setFilterOpen(false);
      }
      if (sortRef.current && !sortRef.current.contains(e.target as Node)) {
        setSortOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  const filtered = useMemo(() => {
    const min = minDeg.trim() === "" ? null : Number(minDeg);
    const max = maxDeg.trim() === "" ? null : Number(maxDeg);
    return records.filter((r) => {
      const deg = r.angles.atRelease;
      if (deg === null) return false;
      if (min !== null && !Number.isNaN(min) && deg < min) return false;
      if (max !== null && !Number.isNaN(max) && deg > max) return false;
      const slot = angleToSlot(deg);
      if (slot === "unknown") return false;
      return activeSlots.has(slot);
    });
  }, [records, activeSlots, minDeg, maxDeg]);

  const sorted = useMemo(() => {
    if (!sortMode) return filtered;
    const dir = sortMode === "asc" ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const va = a.angles.atRelease ?? -1;
      const vb = b.angles.atRelease ?? -1;
      return (va - vb) * dir;
    });
  }, [filtered, sortMode]);

  const metrics = useMemo(() => {
    const angles = filtered
      .map((r) => r.angles.atRelease)
      .filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
    const avg =
      angles.length > 0
        ? Math.round(angles.reduce((s, v) => s + v, 0) / angles.length)
        : null;
    const slotSet = new Set(
      filtered.map((r) => angleToSlot(r.angles.atRelease)),
    );
    slotSet.delete("unknown");
    const distinctSlots = slotSet.size;
    const slotLabel =
      [...slotSet]
        .sort((a, b) => ARM_SLOT_ORDER.indexOf(a as ArmSlotId) - ARM_SLOT_ORDER.indexOf(b as ArmSlotId))
        .map((s) => ARM_SLOT_SHORT[s as ArmSlotId])
        .join(", ") || "—";
    const min = angles.length ? Math.min(...angles) : null;
    const max = angles.length ? Math.max(...angles) : null;
    return { count: filtered.length, avg, distinctSlots, slotLabel, min, max };
  }, [filtered]);

  const toggleSlot = (s: ArmSlotId) => {
    setActiveSlots((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s);
      else next.add(s);
      return next;
    });
  };

  const onDeleteRow = (r: AnalysisRecord) => {
    const { history } = deleteRecord(historyRecordMergeKey(r));
    onUpdate(history);
  };

  const onClearAll = () => {
    if (!records.length) return;
    if (!window.confirm("Clear all saved stats?")) return;
    clearHistory();
    onUpdate([]);
  };

  const resetFilter = () => {
    setActiveSlots(new Set(ARM_SLOT_ORDER));
    setMinDeg("");
    setMaxDeg("");
  };

  return (
    <section className="ps-stats-page" aria-labelledby="stats-h">
      <h1 id="stats-h" className="ps-page-title">
        Track Slots + Release Angles
      </h1>

      <div className="ps-stats-summary">
        <h2 className="ps-section-eyebrow">Stats Summary</h2>
        <div className="ps-summary-grid">
          <div className="ps-summary-card">
            <span className="ps-summary-card__label">In filter</span>
            <span className="ps-summary-card__value">{metrics.count}</span>
            <span className="ps-summary-card__sub">rows matching filter</span>
          </div>
          <div className="ps-summary-card">
            <span className="ps-summary-card__label">Avg arm angle</span>
            <span className="ps-summary-card__value">
              {metrics.avg !== null ? `${metrics.avg}°` : "--"}
            </span>
            <span className="ps-summary-card__sub">at release</span>
          </div>
          <div className="ps-summary-card">
            <span className="ps-summary-card__label">Distinct slots</span>
            <span className="ps-summary-card__value">
              {metrics.distinctSlots}
            </span>
            <span className="ps-summary-card__sub">{metrics.slotLabel}</span>
          </div>
          <div className="ps-summary-card">
            <span className="ps-summary-card__label">Angle range</span>
            <span className="ps-summary-card__value ps-summary-card__value--sm">
              {metrics.min !== null && metrics.max !== null
                ? `${metrics.min.toFixed(1)}° – ${metrics.max.toFixed(1)}°`
                : "--"}
            </span>
            <span className="ps-summary-card__sub">min – max</span>
          </div>
        </div>
      </div>

      <div className="ps-stats-toolbar">
        <div className="ps-dropdown-wrap" ref={filterRef}>
          <button
            type="button"
            className={`ps-btn ps-btn--primary ps-btn--icon${filterOpen ? " is-active" : ""}`}
            aria-expanded={filterOpen}
            onClick={() => {
              setFilterOpen((o) => !o);
              setSortOpen(false);
            }}
          >
            <ListFilter size={16} strokeWidth={2.5} />
            Filter
          </button>
          {filterOpen && (
            <div className="ps-dropdown ps-dropdown--filter" role="dialog">
              <div className="ps-dropdown__group">
                <span className="ps-dropdown__label">Arm Angle Range</span>
                <label className="ps-field">
                  <span>Min:</span>
                  <input
                    type="number"
                    value={minDeg}
                    onChange={(e) => setMinDeg(e.target.value)}
                    placeholder="0"
                  />
                </label>
                <label className="ps-field">
                  <span>Max:</span>
                  <input
                    type="number"
                    value={maxDeg}
                    onChange={(e) => setMaxDeg(e.target.value)}
                    placeholder="180"
                  />
                </label>
              </div>
              <div className="ps-dropdown__group">
                <span className="ps-dropdown__label">Slot types</span>
                <div className="ps-slot-chips">
                  {ARM_SLOT_ORDER.map((s) => (
                    <label
                      key={s}
                      className={`ps-slot-chip${
                        activeSlots.has(s) ? " is-on" : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={activeSlots.has(s)}
                        onChange={() => toggleSlot(s)}
                      />
                      {ARM_SLOT_LABEL[s]}
                    </label>
                  ))}
                </div>
              </div>
              <button
                type="button"
                className="ps-dropdown__reset"
                onClick={resetFilter}
              >
                Reset
              </button>
            </div>
          )}
        </div>

        <div className="ps-dropdown-wrap" ref={sortRef}>
          <button
            type="button"
            className={`ps-btn ps-btn--primary ps-btn--icon${sortOpen ? " is-active" : ""}`}
            aria-expanded={sortOpen}
            onClick={() => {
              setSortOpen((o) => !o);
              setFilterOpen(false);
            }}
          >
            <ArrowUpDown size={16} strokeWidth={2.5} />
            Sort
          </button>
          {sortOpen && (
            <div className="ps-dropdown ps-dropdown--sort" role="menu">
              <button
                type="button"
                role="menuitem"
                className={`ps-dropdown__item${sortMode === "asc" ? " is-active" : ""}`}
                onClick={() => {
                  setSortMode("asc");
                  setSortOpen(false);
                }}
              >
                <span className="ps-dropdown__item-title">Arm Angle:</span>
                <span className="ps-dropdown__item-sub">Lowest to Highest</span>
              </button>
              <button
                type="button"
                role="menuitem"
                className={`ps-dropdown__item${sortMode === "desc" ? " is-active" : ""}`}
                onClick={() => {
                  setSortMode("desc");
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
          <h2 className="ps-table-card__title">Stats History</h2>
          <button
            type="button"
            className="ps-btn ps-btn--primary ps-btn--sm"
            onClick={onClearAll}
            disabled={!records.length}
          >
            Clear all saved
          </button>
        </div>
        <div className="ps-table-wrap">
          <table className="ps-table">
            <thead>
              <tr>
                <th className="ps-col-hash">#</th>
                <th className="ps-col-video">Video</th>
                <th className="ps-col-player">Player</th>
                <th className="ps-col-slot">Slot</th>
                <th className="ps-col-angle">Angle</th>
                <th className="ps-col-frame">Frame</th>
                <th className="ps-col-date">Date Saved</th>
                <th className="ps-col-action" />
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={8} className="ps-table__empty">
                    {records.length === 0 ? (
                      <>
                        Empty —{" "}
                        {onGoUpload ? (
                          <button
                            type="button"
                            className="ps-link"
                            onClick={onGoUpload}
                          >
                            run an analysis
                          </button>
                        ) : (
                          "run an analysis"
                        )}{" "}
                        to add rows.
                      </>
                    ) : (
                      "No rows match your filter."
                    )}
                  </td>
                </tr>
              ) : (
                sorted.map((r, i) => {
                  const deg = r.angles.atRelease;
                  const label = effectivePlayerLabelFromRecord(r);
                  return (
                    <tr key={historyRecordMergeKey(r)}>
                      <td className="ps-col-hash">{i + 1}</td>
                      <td className="ps-col-video">
                        <div className="ps-video-cell">
                          {r.annotatedImage ? (
                            <button
                              type="button"
                              className="ps-video-cell__thumb ps-video-cell__thumb--btn"
                              onClick={() =>
                                setActiveImage({
                                  src: r.annotatedImage!,
                                  caption: r.videoName,
                                })
                              }
                              aria-label="View annotated release frame"
                            >
                              <img src={r.annotatedImage} alt="" />
                            </button>
                          ) : (
                            <span className="ps-video-cell__thumb" aria-hidden />
                          )}
                          <span
                            className="ps-video-cell__name"
                            title={r.videoName}
                          >
                            {r.videoName}
                          </span>
                        </div>
                      </td>
                      <td className="ps-col-player">{label ?? "--"}</td>
                      <td className="ps-col-slot">
                        <ArmSlotBadge angleDeg={deg} />
                      </td>
                      <td className="ps-col-angle">
                        {deg !== null ? `${deg.toFixed(1)}°` : "--"}
                      </td>
                      <td className="ps-col-frame">{r.releaseFrame}</td>
                      <td className="ps-col-date">{formatDate(r.createdAt)}</td>
                      <td className="ps-col-action">
                        <button
                          type="button"
                          className="ps-icon-btn"
                          aria-label="Delete row"
                          onClick={() => onDeleteRow(r)}
                        >
                          <Trash2 size={16} />
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <RodgerWidget open={chatOpen} onOpenChange={setChatOpen} />

      <ImageModal
        src={activeImage?.src ?? null}
        caption={activeImage?.caption}
        onClose={() => setActiveImage(null)}
      />
    </section>
  );
};

type RodgerMsg = { from: "you" | "rodger"; text: string };

const RodgerWidget: React.FC<{
  open: boolean;
  onOpenChange: (next: boolean) => void;
}> = ({ open, onOpenChange }) => {
  const [draft, setDraft] = useState("");
  const [thread, setThread] = useState<RodgerMsg[]>([]);

  const send = () => {
    const text = draft.trim();
    if (!text) return;
    setDraft("");
    setThread((t) => [
      ...t,
      { from: "you", text },
      {
        from: "rodger",
        text:
          "Thanks for the question — assistant responses aren’t wired to a backend yet, but your saved stats are visible in the table.",
      },
    ]);
  };

  if (!open) {
    return (
      <button
        type="button"
        className="ps-rodger-fab"
        aria-label="Open Rodger assistant"
        onClick={() => onOpenChange(true)}
      >
        <MessageSquare size={18} />
        Rodger
      </button>
    );
  }

  return (
    <aside className="ps-rodger" aria-label="Rodger assistant">
      <header className="ps-rodger__head">
        <div className="ps-rodger__title">
          <MessageSquare size={16} />
          Rodger
        </div>
        <button
          type="button"
          className="ps-rodger__close"
          aria-label="Close assistant"
          onClick={() => onOpenChange(false)}
        >
          ×
        </button>
      </header>
      <p className="ps-rodger__hint">
        <a className="ps-rodger__cta">Ask questions about your saved stats.</a>
      </p>
      <div className="ps-rodger__thread">
        {thread.map((m, i) => (
          <div
            key={i}
            className={`ps-rodger__bubble ps-rodger__bubble--${m.from}`}
          >
            {m.text}
          </div>
        ))}
      </div>
      <form
        className="ps-rodger__form"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          className="ps-rodger__input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Type a question…"
        />
        <button
          type="submit"
          className="ps-rodger__send"
          aria-label="Send message"
        >
          <Send size={16} />
        </button>
      </form>
    </aside>
  );
};

export default Stats;
