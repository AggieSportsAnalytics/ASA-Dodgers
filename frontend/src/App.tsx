import React, { useCallback, useEffect, useState } from "react";
import PitchSightHeader from "./components/PitchSightHeader";
import Sidebar, { type AppPage as SidebarPage } from "./components/Sidebar";
import Upload from "./pages/Upload";
import Stats from "./pages/Stats";
import Pitchers from "./pages/Pitchers";
import PlayerProfile from "./pages/PlayerProfile";
import BattingPlaceholder from "./pages/BattingPlaceholder";
import type { AnalysisRecord } from "./types";
import { bindHistoryStorageSync, loadHistoryMerged } from "./utils/storage";

type Mode = "pitching" | "batting";
type AppPage = SidebarPage | "pitcher-profile";

const App: React.FC = () => {
  const [mode, setMode] = useState<Mode>("pitching");
  const [page, setPage] = useState<AppPage>("upload");
  const [pitcherKey, setPitcherKey] = useState<string | null>(null);
  const [history, setHistory] = useState<AnalysisRecord[]>([]);

  useEffect(() => {
    let cancelled = false;
    void loadHistoryMerged().then((records) => {
      if (!cancelled) setHistory(records);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => bindHistoryStorageSync(setHistory), []);

  const handleModeChange = useCallback((next: Mode) => {
    setMode(next);
    setPage("upload");
    setPitcherKey(null);
  }, []);

  const navigate = useCallback((next: SidebarPage) => {
    setPage(next);
    setPitcherKey(null);
  }, []);

  const openProfile = useCallback((key: string) => {
    setPitcherKey(key);
    setPage("pitcher-profile");
  }, []);

  const sidebarActive: SidebarPage =
    page === "pitcher-profile" ? "pitchers" : page;

  const renderMain = () => {
    if (mode === "batting") {
      if (page === "stats") {
        return (
          <Stats
            records={history}
            onUpdate={setHistory}
            onGoUpload={() => setPage("upload")}
          />
        );
      }
      if (page === "pitchers") {
        return (
          <Pitchers
            variant="batting"
            records={history}
            onOpenProfile={openProfile}
          />
        );
      }
      if (page === "pitcher-profile" && pitcherKey) {
        return (
          <PlayerProfile
            pitcherKey={pitcherKey}
            records={history}
            onUpdate={setHistory}
            onBack={() => navigate("pitchers")}
          />
        );
      }
      return (
        <BattingPlaceholder onSwitchToPitching={() => handleModeChange("pitching")} />
      );
    }

    if (page === "stats") {
      return (
        <Stats
          records={history}
          onUpdate={setHistory}
          onGoUpload={() => setPage("upload")}
        />
      );
    }
    if (page === "pitchers") {
      return (
        <Pitchers
          variant="pitching"
          records={history}
          onOpenProfile={openProfile}
        />
      );
    }
    if (page === "pitcher-profile" && pitcherKey) {
      return (
        <PlayerProfile
          pitcherKey={pitcherKey}
          records={history}
          onUpdate={setHistory}
          onBack={() => navigate("pitchers")}
        />
      );
    }
    return <Upload variant="pitching" onSaved={setHistory} />;
  };

  return (
    <div className="ps-app">
      <PitchSightHeader mode={mode} onModeChange={handleModeChange} />
      <div className="ps-app__body">
        <Sidebar sport={mode} activePage={sidebarActive} onNavigate={navigate} />
        <main className="ps-main">
          <div className="ps-main__inner">{renderMain()}</div>
        </main>
      </div>
    </div>
  );
};

export default App;
