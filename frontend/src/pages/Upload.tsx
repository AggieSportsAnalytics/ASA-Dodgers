import React, { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Upload as UploadIcon, Play } from "lucide-react";
import type { AnalysisRecord, AngleSet } from "../types";
import { ArmSlotBadge } from "../components/ArmSlotBadge";
import { ImageModal } from "../components/ImageModal";
import { ARM_SLOT_SHORT, angleToSlot } from "../lib/armSlot";
import { addRecords, getApiUrl } from "../utils/storage";

type Variant = "pitching" | "batting";

type UploadProps = {
  variant?: Variant;
  onSaved: (next: AnalysisRecord[]) => void;
};

type AnalysisResponse = {
  release_frame: number;
  arm_angles: {
    before: number | null;
    at: number | null;
    after: number | null;
  };
  annotated_image?: string | null;
  player?: {
    label?: string;
    id?: string;
    source?: "manual" | "detected";
    confidence?: number;
    jersey_number?: string;
  } | null;
};

type ClipState = {
  file: File;
  status: "idle" | "analyzing" | "done" | "error";
  record?: AnalysisRecord;
  saved?: boolean;
  error?: string;
};

function bytesToMb(b: number) {
  return (b / 1_048_576).toFixed(2);
}

const Upload: React.FC<UploadProps> = ({ variant = "pitching", onSaved }) => {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [clips, setClips] = useState<ClipState[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [batchAnalyzing, setBatchAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [modalImage, setModalImage] = useState<{
    src: string;
    caption: string;
  } | null>(null);

  const previewUrl = useMemo(() => {
    const f = clips[activeIdx]?.file;
    return f ? URL.createObjectURL(f) : null;
  }, [clips, activeIdx]);

  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const totalSelectedLabel = useMemo(() => {
    if (clips.length === 0) return "";
    const names = clips
      .map((c) => `${c.file.name} (${bytesToMb(c.file.size)} MB)`)
      .join(", ");
    return names;
  }, [clips]);

  const onPickFiles = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) {
      setClips([]);
      setActiveIdx(0);
    } else {
      const next: ClipState[] = Array.from(files).map((f) => ({
        file: f,
        status: "idle",
      }));
      setClips(next);
      setActiveIdx(0);
    }
    setError(null);
    e.target.value = "";
  };

  const clearSelection = () => {
    setClips([]);
    setActiveIdx(0);
    setError(null);
  };

  const analyzeOne = async (idx: number): Promise<void> => {
    const clip = clips[idx];
    if (!clip) return;
    setClips((prev) =>
      prev.map((c, i) =>
        i === idx ? { ...c, status: "analyzing", error: undefined } : c,
      ),
    );

    try {
      const form = new FormData();
      form.append("file", clip.file);
      form.append("handedness", "auto");
      const apiUrl = getApiUrl();
      const headers: Record<string, string> = {};
      if (apiUrl.includes("ngrok"))
        headers["ngrok-skip-browser-warning"] = "1";
      const res = await fetch(apiUrl, {
        method: "POST",
        headers,
        body: form,
      });
      if (!res.ok) {
        const t = await res.text().catch(() => "");
        throw new Error(t || `Analysis failed (${res.status})`);
      }
      const data = (await res.json()) as AnalysisResponse;
      const angles: AngleSet = {
        beforeRelease: data.arm_angles?.before ?? null,
        atRelease: data.arm_angles?.at ?? null,
        afterRelease: data.arm_angles?.after ?? null,
      };
      const p = data.player ?? null;
      const jn = p?.jersey_number?.trim();
      const label = p?.label?.trim() || (jn ? `#${jn}` : "");
      const record: AnalysisRecord = {
        id: crypto.randomUUID(),
        videoName: clip.file.name,
        createdAt: new Date().toISOString(),
        releaseFrame: data.release_frame,
        angles,
        annotatedImage: data.annotated_image ?? null,
        player:
          p && (label || jn)
            ? {
                label: label || `#${jn!}`,
                ...(p.id?.trim() ? { id: p.id.trim() } : {}),
                source:
                  p.source === "manual" || p.source === "detected"
                    ? p.source
                    : "detected",
                ...(typeof p.confidence === "number"
                  ? { confidence: p.confidence }
                  : {}),
                ...(jn ? { jerseyNumber: jn } : {}),
              }
            : null,
      };
      const { history, persisted } = addRecords([record]);
      if (persisted) onSaved(history);
      setClips((prev) =>
        prev.map((c, i) =>
          i === idx
            ? { ...c, status: "done", record, saved: persisted }
            : c,
        ),
      );
      if (!persisted) {
        setError(
          "Analyzed, but couldn’t persist to local history (storage may be full).",
        );
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      setClips((prev) =>
        prev.map((c, i) =>
          i === idx ? { ...c, status: "error", error: msg } : c,
        ),
      );
      setError(msg);
    }
  };

  const analyzeActive = async () => {
    setError(null);
    if (clips.length === 0) {
      setError("Pick a video first.");
      return;
    }
    setBatchAnalyzing(true);
    try {
      await analyzeOne(activeIdx);
    } finally {
      setBatchAnalyzing(false);
    }
  };

  const analyzeAll = async () => {
    setError(null);
    if (clips.length === 0) {
      setError("Pick at least one video first.");
      return;
    }
    setBatchAnalyzing(true);
    try {
      for (let i = 0; i < clips.length; i++) {
        await analyzeOne(i);
      }
    } finally {
      setBatchAnalyzing(false);
    }
  };

  const goPrev = () => setActiveIdx((i) => Math.max(0, i - 1));
  const goNext = () =>
    setActiveIdx((i) => Math.min(clips.length - 1, i + 1));

  const activeClip = clips[activeIdx];
  const showResults = clips.length > 0;
  const heading =
    variant === "batting"
      ? "Analyze Batting Mechanics"
      : "Analyze Pitching Mechanics";

  // Stats panel content
  const rec = activeClip?.record;
  const atAngle = rec?.angles.atRelease ?? null;
  const slot = atAngle !== null ? angleToSlot(atAngle) : "unknown";
  const slotLabel =
    slot !== "unknown" ? ARM_SLOT_SHORT[slot] : null;

  let statusLine = "";
  if (activeClip) {
    if (activeClip.status === "idle") statusLine = "Ready";
    else if (activeClip.status === "analyzing")
      statusLine = `${activeIdx + 1} of ${clips.length} analyzing...`;
    else if (activeClip.status === "done") statusLine = "Finished Analyzing";
    else if (activeClip.status === "error")
      statusLine = activeClip.error || "Analysis failed";
  }

  return (
    <section className="ps-upload-page" aria-labelledby="upload-h">
      <h1 id="upload-h" className="ps-page-title">
        {heading}
      </h1>

      <input
        ref={fileInputRef}
        id="ps-video-input"
        type="file"
        accept="video/mp4,video/quicktime,video/*"
        multiple
        className="ps-visually-hidden"
        onChange={onPickFiles}
        disabled={variant === "batting"}
      />
      <label
        htmlFor="ps-video-input"
        className={`ps-upload-card${
          variant === "batting" ? " ps-upload-card--inactive" : ""
        }`}
        aria-disabled={variant === "batting"}
      >
        <span className="ps-upload-card__icon" aria-hidden>
          <UploadIcon size={32} strokeWidth={2.25} />
        </span>
        <span className="ps-upload-card__text">
          <span className="ps-upload-card__title">
            Choose Video to Upload (.mp4, .mov)
          </span>
          <span className="ps-upload-card__sub">
            Select one or more clips.
          </span>
        </span>
      </label>

      {variant === "batting" && (
        <p className="ps-upload-note">
          Batting analysis isn’t wired up yet — switch to{" "}
          <strong>Pitching</strong> to run a real analysis.
        </p>
      )}

      {showResults && (
        <div className="ps-upload-results">
          <div className="ps-upload-results__main">
            <div className="ps-upload-meta">
              <div className="ps-upload-meta__count">
                {clips.length} file(s) selected
              </div>
              <div className="ps-upload-meta__names" title={totalSelectedLabel}>
                {totalSelectedLabel}
              </div>
              <button
                type="button"
                className="ps-btn ps-btn--primary ps-btn--sm"
                onClick={clearSelection}
                disabled={batchAnalyzing}
              >
                Clear selection
              </button>
            </div>

            <div className="ps-video-preview">
              {activeClip?.record?.annotatedImage ? (
                <button
                  type="button"
                  className="ps-video-preview__frame-btn"
                  onClick={() =>
                    setModalImage({
                      src: activeClip.record!.annotatedImage!,
                      caption: activeClip.file.name,
                    })
                  }
                  aria-label="Open annotated release frame"
                >
                  <img
                    className="ps-video-preview__frame"
                    src={activeClip.record.annotatedImage}
                    alt="Annotated release frame"
                  />
                  <span className="ps-video-preview__badge">
                    Release frame · joints annotated
                  </span>
                </button>
              ) : previewUrl ? (
                <video
                  className="ps-video-preview__video"
                  src={previewUrl}
                  controls
                />
              ) : (
                <div className="ps-video-preview__placeholder" aria-hidden>
                  <Play size={48} strokeWidth={1.5} />
                </div>
              )}
            </div>

            {clips.length > 1 && (
              <div className="ps-pager" role="navigation" aria-label="Clip pager">
                <button
                  type="button"
                  className="ps-pager__btn"
                  onClick={goPrev}
                  disabled={activeIdx === 0}
                  aria-label="Previous clip"
                >
                  <ChevronLeft size={18} />
                </button>
                <div className="ps-pager__dots" aria-hidden>
                  {clips.map((_, i) => (
                    <span
                      key={i}
                      className={`ps-pager__dot${
                        i === activeIdx ? " ps-pager__dot--active" : ""
                      }`}
                    />
                  ))}
                </div>
                <button
                  type="button"
                  className="ps-pager__btn"
                  onClick={goNext}
                  disabled={activeIdx >= clips.length - 1}
                  aria-label="Next clip"
                >
                  <ChevronRight size={18} />
                </button>
              </div>
            )}

            <div className="ps-upload-actions">
              <button
                type="button"
                className="ps-btn ps-btn--primary"
                onClick={analyzeActive}
                disabled={batchAnalyzing || clips.length === 0}
              >
                Analyze Video
              </button>
              <button
                type="button"
                className="ps-btn ps-btn--primary"
                onClick={analyzeAll}
                disabled={batchAnalyzing || clips.length === 0}
              >
                Analyze All
              </button>
            </div>

            {error && <p className="ps-error">{error}</p>}
          </div>

          <aside className="ps-upload-stats" aria-label="Active clip stats">
            <h2 className="ps-upload-stats__title">Stats</h2>
            <div className="ps-upload-stats__card">
              <div className="ps-upload-stats__head">
                <span className="ps-upload-stats__head-title">
                  Release Frame + Angle
                </span>
                {rec && (
                  <span
                    className={`ps-pill ps-pill--${
                      activeClip?.saved ? "saved" : "unsaved"
                    }`}
                  >
                    {activeClip?.saved ? "SAVED" : "SAVING…"}
                  </span>
                )}
              </div>
              {statusLine && (
                <p className="ps-upload-stats__status">{statusLine}</p>
              )}

              <dl className="ps-kv">
                <div className="ps-kv__row">
                  <dt>PITCHER:</dt>
                  <dd>
                    {rec?.player?.label?.trim() ? rec.player.label : "—"}
                  </dd>
                </div>
                <div className="ps-kv__row">
                  <dt>SLOT TYPE:</dt>
                  <dd>
                    {slotLabel ? (
                      <ArmSlotBadge angleDeg={atAngle} />
                    ) : (
                      "—"
                    )}
                  </dd>
                </div>
                <div className="ps-kv__row">
                  <dt>ARM ANGLE:</dt>
                  <dd>
                    {atAngle !== null ? (
                      <span className="ps-value-pill">
                        {Math.round(atAngle)}°
                      </span>
                    ) : (
                      "—"
                    )}
                  </dd>
                </div>
                <div className="ps-kv__row">
                  <dt>RELEASE FRAME:</dt>
                  <dd>
                    {rec ? (
                      <span className="ps-value-pill">{rec.releaseFrame}</span>
                    ) : (
                      "—"
                    )}
                  </dd>
                </div>
              </dl>
            </div>

            {rec && (
              <p className="ps-upload-stats__autosave">
                {activeClip?.saved
                  ? "Saved automatically · view in Stats"
                  : "Saving…"}
              </p>
            )}
          </aside>
        </div>
      )}

      <ImageModal
        src={modalImage?.src ?? null}
        caption={modalImage?.caption}
        onClose={() => setModalImage(null)}
      />
    </section>
  );
};

export default Upload;
