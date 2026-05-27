export type AngleSet = {
  beforeRelease: number | null;
  atRelease: number | null;
  afterRelease: number | null;
};

export type PlayerAttribution = {
  label: string;
  id?: string;
  source: "manual" | "detected";
  confidence?: number;
  jerseyNumber?: string;
};

export type AnalysisRecord = {
  id: string;
  videoName: string;
  createdAt: string;
  releaseFrame: number;
  angles: AngleSet;
  annotatedImage?: string | null;
  player?: PlayerAttribution | null;
};
