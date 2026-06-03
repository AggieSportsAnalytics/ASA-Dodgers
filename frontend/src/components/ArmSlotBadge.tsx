import React from "react";
import { ARM_SLOT_SHORT, angleToSlot, type ArmSlotId } from "../lib/armSlot";

type Props = {
  angleDeg: number | null | undefined;
  slot?: ArmSlotId | "unknown";
};

export const ArmSlotBadge: React.FC<Props> = ({ angleDeg, slot }) => {
  const resolved =
    slot ?? angleToSlot(typeof angleDeg === "number" ? angleDeg : null);
  if (resolved === "unknown") {
    return <span className="ps-slot-pill ps-slot-pill--unknown">—</span>;
  }
  const label = ARM_SLOT_SHORT[resolved];
  return (
    <span className={`ps-slot-pill ps-slot-pill--${resolved}`}>{label}</span>
  );
};

export default ArmSlotBadge;
