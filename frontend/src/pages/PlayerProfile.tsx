import React, { useMemo, useState } from "react";
import { ArrowLeft, Trash2 } from "lucide-react";
import type { AnalysisRecord } from "../types";
import { ArmSlotBadge } from "../components/ArmSlotBadge";
import { ImageModal } from "../components/ImageModal";
import { ARM_SLOT_ORDER, ARM_SLOT_SHORT, angleToSlot, type ArmSlotId } from "../lib/armSlot";
import {
  getPitcherDisplayTitle,
  recordsMatchingPitcherKey,
} from "../lib/player";
import { deleteRecord, historyRecordMergeKey } from "../utils/storage";

type Props = {
  pitcherKey: string;
  records: AnalysisRecord[];
  onUpdate: (next: AnalysisRecord[]) => void;
  onBack: () => void;
};

const formatDate = (iso: string) =>
  new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "long",
    day: "numeric",
  });

const PlayerProfile: React.FC<Props> = ({
  pitcherKey,
  records,
  onUpdate,
  onBack,
}) => {
  const [activeImage, setActiveImage] = useState<{
    src: string;
    caption: string;
  } | null>(null);

  const display = getPitcherDisplayTitle(pitcherKey);
  const rows = useMemo(
    () =>
      [...recordsMatchingPitcherKey(records, pitcherKey)].sort(
        (a, b) =>
          new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      ),
    [records, pitcherKey],
  );

  const metrics = useMemo(() => {
    const angles = rows
      .map((r) => r.angles.atRelease)
      .filter((v): v is number => typeof v === "number" && !Number.isNaN(v));
    const avg =
      angles.length > 0
        ? angles.reduce((s, v) => s + v, 0) / angles.length
        : null;
    const slotSet = new Set(
      rows.map((r) => angleToSlot(r.angles.atRelease)),
    );
    slotSet.delete("unknown");
    const slotLabel =
      [...slotSet]
        .sort(
          (a, b) =>
            ARM_SLOT_ORDER.indexOf(a as ArmSlotId) -
            ARM_SLOT_ORDER.indexOf(b as ArmSlotId),
        )
        .map((s) => ARM_SLOT_SHORT[s as ArmSlotId])
        .join(", ") || "—";
    return {
      count: rows.length,
      avgAngle: avg,
      distinctSlots: slotSet.size,
      slotLabel,
      avgSlot: avg !== null ? angleToSlot(avg) : ("unknown" as const),
    };
  }, [rows]);

  const onDelete = (r: AnalysisRecord) => {
    const { history } = deleteRecord(historyRecordMergeKey(r));
    onUpdate(history);
  };

  return (
    <section className="ps-stats-page" aria-labelledby="profile-h">
      <button type="button" className="ps-back-link" onClick={onBack}>
        <ArrowLeft size={16} />
        All Pitchers
      </button>

      <h1 id="profile-h" className="ps-page-title">
        {display}
      </h1>

      <div className="ps-summary-grid">
        <div className="ps-summary-card">
          <span className="ps-summary-card__label">Pitches</span>
          <span className="ps-summary-card__value">{metrics.count}</span>
          <span className="ps-summary-card__sub">saved analyses</span>
        </div>
        <div className="ps-summary-card">
          <span className="ps-summary-card__label">Avg arm angle</span>
          <span className="ps-summary-card__value">
            {metrics.avgAngle !== null
              ? `${Math.round(metrics.avgAngle)}°`
              : "--"}
          </span>
          <span className="ps-summary-card__sub">at release</span>
        </div>
        <div className="ps-summary-card">
          <span className="ps-summary-card__label">Avg slot</span>
          <span className="ps-summary-card__value ps-summary-card__value--sm">
            <ArmSlotBadge
              slot={metrics.avgSlot}
              angleDeg={metrics.avgAngle}
            />
          </span>
          <span className="ps-summary-card__sub">across saved frames</span>
        </div>
        <div className="ps-summary-card">
          <span className="ps-summary-card__label">Distinct slots</span>
          <span className="ps-summary-card__value">
            {metrics.distinctSlots}
          </span>
          <span className="ps-summary-card__sub">{metrics.slotLabel}</span>
        </div>
      </div>

      <div className="ps-table-card">
        <div className="ps-table-card__head">
          <h2 className="ps-table-card__title">Pitches</h2>
          <span className="ps-search">
            <span>
              {rows.length} {rows.length === 1 ? "row" : "rows"}
            </span>
          </span>
        </div>
        <div className="ps-table-wrap">
          <table className="ps-table">
            <thead>
              <tr>
                <th className="ps-col-hash">#</th>
                <th className="ps-col-video">Video</th>
                <th className="ps-col-slot">Slot</th>
                <th className="ps-col-angle">Angle</th>
                <th className="ps-col-frame">Frame</th>
                <th className="ps-col-date">Date Saved</th>
                <th className="ps-col-action" />
              </tr>
            </thead>
            <tbody>
              {rows.length === 0 ? (
                <tr>
                  <td colSpan={7} className="ps-table__empty">
                    No saved analyses for this pitcher yet.
                  </td>
                </tr>
              ) : (
                rows.map((r, i) => {
                  const deg = r.angles.atRelease;
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
                            <span
                              className="ps-video-cell__thumb"
                              aria-hidden
                            />
                          )}
                          <span
                            className="ps-video-cell__name"
                            title={r.videoName}
                          >
                            {r.videoName}
                          </span>
                        </div>
                      </td>
                      <td className="ps-col-slot">
                        <ArmSlotBadge angleDeg={deg} />
                      </td>
                      <td className="ps-col-angle">
                        {deg !== null ? `${deg.toFixed(1)}°` : "--"}
                      </td>
                      <td className="ps-col-frame">{r.releaseFrame}</td>
                      <td className="ps-col-date">
                        {formatDate(r.createdAt)}
                      </td>
                      <td className="ps-col-action">
                        <button
                          type="button"
                          className="ps-icon-btn"
                          aria-label="Delete row"
                          onClick={() => onDelete(r)}
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

      <ImageModal
        src={activeImage?.src ?? null}
        caption={activeImage?.caption}
        onClose={() => setActiveImage(null)}
      />
    </section>
  );
};

export default PlayerProfile;
