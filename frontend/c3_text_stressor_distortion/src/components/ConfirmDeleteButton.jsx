import React, { useState } from "react";
import { Loader2, Trash2, X } from "lucide-react";

export default function ConfirmDeleteButton({ onConfirm, label = "Delete", compact = false }) {
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);

  async function handleConfirm() {
    if (deleting) return;
    setDeleting(true);
    try {
      await onConfirm();
    } finally {
      setDeleting(false);
      setConfirming(false);
    }
  }

  if (!confirming) {
    return (
      <button
        type="button"
        className={compact ? "icon-button" : "delete-trigger"}
        onClick={() => setConfirming(true)}
        aria-label={label}
      >
        <Trash2 size={compact ? 15 : 16} />
        {!compact && <span>Delete</span>}
      </button>
    );
  }

  return (
    <span className="delete-confirm" role="group" aria-label="Confirm deletion">
      <span>Delete?</span>
      <button type="button" className="delete-cancel" onClick={() => setConfirming(false)} aria-label="Cancel deletion">
        <X size={14} />
      </button>
      <button type="button" className="delete-approve" onClick={handleConfirm} disabled={deleting}>
        {deleting ? <Loader2 size={13} className="spin" /> : "Yes"}
      </button>
    </span>
  );
}
