export type ArmSlotId =
  | "ott"
  | "high_three_quarter"
  | "three_quarter"
  | "sidearm"
  | "submarine";

export const ARM_SLOT_ORDER: ArmSlotId[] = [
  "ott",
  "high_three_quarter",
  "three_quarter",
  "sidearm",
  "submarine",
];

export const ARM_SLOT_SHORT: Record<ArmSlotId, string> = {
  ott: "OTT",
  high_three_quarter: "H 3/4",
  three_quarter: "3/4",
  sidearm: "SA",
  submarine: "SUB",
};

export const ARM_SLOT_LABEL: Record<ArmSlotId, string> = {
  ott: "Over the top",
  high_three_quarter: "High 3/4",
  three_quarter: "3/4",
  sidearm: "Sidearm",
  submarine: "Submarine",
};

/**
 * Map a release arm angle (degrees) to a discrete slot.
 * Convention: higher degree = more vertical / over the top.
 */
export function angleToSlot(angleDeg: number | null): ArmSlotId | "unknown" {
  if (angleDeg === null || Number.isNaN(angleDeg)) return "unknown";
  const a = Math.abs(angleDeg);
  if (a >= 135) return "ott";
  if (a >= 100) return "high_three_quarter";
  if (a >= 60) return "three_quarter";
  if (a >= 25) return "sidearm";
  return "submarine";
}

export type SlotFilter = {
  slots: Set<ArmSlotId>;
  min: number | null;
  max: number | null;
};

export function defaultSlotFilter(): SlotFilter {
  return {
    slots: new Set<ArmSlotId>(ARM_SLOT_ORDER),
    min: null,
    max: null,
  };
}

export function recordMatchesSlotFilter(
  angles: { atRelease: number | null },
  filter: SlotFilter,
): boolean {
  const deg = angles.atRelease;
  if (deg === null) return filter.slots.size === ARM_SLOT_ORDER.length;
  if (filter.min !== null && deg < filter.min) return false;
  if (filter.max !== null && deg > filter.max) return false;
  const slot = angleToSlot(deg);
  if (slot === "unknown") return false;
  return filter.slots.has(slot);
}
