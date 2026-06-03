import React from "react";
import { BarChart3, Upload as UploadIcon } from "lucide-react";

export type AppPage = "upload" | "stats" | "pitchers";

type SidebarProps = {
  sport: "pitching" | "batting";
  activePage: AppPage;
  onNavigate: (page: AppPage) => void;
};

const Sidebar: React.FC<SidebarProps> = ({
  sport,
  activePage,
  onNavigate,
}) => {
  const rosterLabel = sport === "batting" ? "All Batters" : "All Pitchers";

  const item = (
    page: AppPage,
    label: string,
    icon: React.ReactNode | null,
  ) => {
    const active = activePage === page;
    return (
      <button
        type="button"
        onClick={() => onNavigate(page)}
        aria-current={active ? "page" : undefined}
        className={`ps-nav-item${active ? " ps-nav-item--active" : ""}`}
      >
        {icon ? <span className="ps-nav-item__icon">{icon}</span> : null}
        <span className="ps-nav-item__label">{label}</span>
      </button>
    );
  };

  return (
    <nav aria-label="Main navigation" className="ps-sidebar">
      <div className="ps-sidebar__inner">
        {item("upload", "Upload", <UploadIcon size={18} strokeWidth={2.25} />)}
        {item("stats", "Stats", <BarChart3 size={18} strokeWidth={2.25} />)}
        {item("pitchers", rosterLabel, null)}
      </div>
    </nav>
  );
};

export default Sidebar;
