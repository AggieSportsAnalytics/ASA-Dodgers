import React from "react";

export type PitchSightMode = "pitching" | "batting";

const assetBase = import.meta.env.BASE_URL;

type PitchSightHeaderProps = {
  mode: PitchSightMode;
  onModeChange: (mode: PitchSightMode) => void;
};

const PitchSightHeader: React.FC<PitchSightHeaderProps> = ({
  mode,
  onModeChange,
}) => {
  return (
    <header className="ps-header" role="banner">
      <div className="ps-header__left">
        <h1
          className="ps-header__title ps-header__title--brand"
          aria-label="LA Dodgers Scouting"
        >
          <img
            src={`${assetBase}image.png`}
            alt=""
            aria-hidden
            className="ps-header__brand-logo"
          />
          <span className="ps-header__title-em">Scouting</span>
        </h1>
      </div>

      <div
        className="ps-header__tabs"
        role="tablist"
        aria-label="Analysis mode"
      >
        <button
          type="button"
          role="tab"
          aria-selected={mode === "pitching"}
          className={`ps-header__tab${
            mode === "pitching" ? " ps-header__tab--active" : ""
          }`}
          onClick={() => onModeChange("pitching")}
        >
          Pitching
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={mode === "batting"}
          className={`ps-header__tab${
            mode === "batting" ? " ps-header__tab--active" : ""
          }`}
          onClick={() => onModeChange("batting")}
        >
          Batting
        </button>
      </div>
    </header>
  );
};

export default PitchSightHeader;
