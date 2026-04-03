"use client";

import { ReactNode } from "react";

export default function MetaBadge({
  children,
  tone = "neutral",
  icon,
}: {
  children: ReactNode;
  tone?: "neutral" | "good" | "warn" | "bad";
  icon?: ReactNode;
}) {
  const toneClasses =
    tone === "good"
      ? "bg-emerald-500/15 text-emerald-200 ring-emerald-500/30"
      : tone === "warn"
        ? "bg-amber-500/15 text-amber-200 ring-amber-500/30"
        : tone === "bad"
          ? "bg-rose-500/15 text-rose-200 ring-rose-500/30"
          : "bg-sky-500/15 text-sky-200 ring-sky-500/30";

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1 ring-1 ${toneClasses}`}
    >
      {icon ? <span className="text-sm">{icon}</span> : null}
      <span className="text-sm font-medium">{children}</span>
    </div>
  );
}

