import React from "react";
import { Upload as UploadIcon } from "lucide-react";

type Props = {
  onSwitchToPitching?: () => void;
};

const BattingPlaceholder: React.FC<Props> = ({ onSwitchToPitching }) => {
  return (
    <section className="ps-upload-page" aria-labelledby="batting-h">
      <h1 id="batting-h" className="ps-page-title">
        Analyze Batting Mechanics
      </h1>

      <div className="ps-upload-card ps-upload-card--inactive" aria-disabled>
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
      </div>

      <p className="ps-upload-note">
        Batting analysis isn’t available yet.{" "}
        {onSwitchToPitching ? (
          <button
            type="button"
            className="ps-link"
            onClick={onSwitchToPitching}
          >
            Return to pitching
          </button>
        ) : null}
      </p>
    </section>
  );
};

export default BattingPlaceholder;
