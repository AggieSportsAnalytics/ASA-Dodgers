import React, { useEffect } from "react";
import { X } from "lucide-react";

type Props = {
  src: string | null;
  alt?: string;
  caption?: string;
  onClose: () => void;
};

export const ImageModal: React.FC<Props> = ({
  src,
  alt = "",
  caption,
  onClose,
}) => {
  useEffect(() => {
    if (!src) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [src, onClose]);

  if (!src) return null;

  return (
    <div
      className="ps-modal"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div className="ps-modal__card" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className="ps-modal__close"
          onClick={onClose}
          aria-label="Close"
        >
          <X size={20} />
        </button>
        <img className="ps-modal__img" src={src} alt={alt} />
        {caption ? <p className="ps-modal__caption">{caption}</p> : null}
      </div>
    </div>
  );
};

export default ImageModal;
