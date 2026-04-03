"use client";

import { useMemo } from "react";

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function hexToRgb(hex: string) {
  const cleaned = hex.replace("#", "").trim();
  const full = cleaned.length === 3 ? cleaned.split("").map((c) => c + c).join("") : cleaned;
  const num = parseInt(full, 16);
  return { r: (num >> 16) & 255, g: (num >> 8) & 255, b: num & 255 };
}

function rgbToHex(r: number, g: number, b: number) {
  const toHex = (x: number) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function scoreToColor(score: number) {
  // 300 -> red, 600 -> yellow, 850 -> green
  const red = "#ef4444"; // 0
  const yellow = "#eab308"; // 0.5
  const green = "#22c55e"; // 1

  const min = 300;
  const mid = 600;
  const max = 850;

  const clamped = Math.max(min, Math.min(max, score));
  if (clamped <= mid) {
    const t = (clamped - min) / (mid - min);
    const c0 = hexToRgb(red);
    const c1 = hexToRgb(yellow);
    return rgbToHex(lerp(c0.r, c1.r, t), lerp(c0.g, c1.g, t), lerp(c0.b, c1.b, t));
  }

  const t = (clamped - mid) / (max - mid);
  const c0 = hexToRgb(yellow);
  const c1 = hexToRgb(green);
  return rgbToHex(lerp(c0.r, c1.r, t), lerp(c0.g, c1.g, t), lerp(c0.b, c1.b, t));
}

export default function RadialScoreGauge({
  score,
  loading,
}: {
  score: number | null;
  loading: boolean;
}) {
  const { strokeDashoffset, color, normalized } = useMemo(() => {
    if (score === null) {
      return { strokeDashoffset: 0, color: "#475569", normalized: 0 };
    }

    const min = 300;
    const max = 850;
    const clamped = Math.max(min, Math.min(max, score));
    const normalized01 = (clamped - min) / (max - min);

    const radius = 74;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - normalized01);

    return { strokeDashoffset: offset, color: scoreToColor(score), normalized: normalized01 };
  }, [score]);

  const radius = 74;
  const circumference = 2 * Math.PI * radius;

  return (
    <div className="relative mx-auto flex aspect-square w-[240px] items-center justify-center">
      {/* Background track */}
      <svg viewBox="0 0 240 240" className="absolute inset-0 h-full w-full">
        <circle
          cx="120"
          cy="120"
          r={radius}
          stroke="rgba(148,163,184,0.25)"
          strokeWidth="16"
          fill="none"
        />

        <circle
          cx="120"
          cy="120"
          r={radius}
          stroke={loading ? "rgba(148,163,184,0.55)" : color}
          strokeWidth="16"
          strokeLinecap="round"
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={loading ? circumference * 0.2 : strokeDashoffset}
          transform="rotate(-90 120 120)"
          style={{
            transition: "stroke-dashoffset 700ms ease, stroke 300ms ease",
          }}
        />
      </svg>

      {/* Center label */}
      <div className="relative z-10 flex flex-col items-center gap-1 text-center">
        <div className="text-xs font-medium tracking-wide text-zinc-500 dark:text-zinc-400">
          CNB Credit Score
        </div>
        {loading ? (
          <div className="h-10 w-[96px] animate-pulse rounded bg-zinc-200 dark:bg-zinc-800" />
        ) : (
          <div className="text-4xl font-semibold tabular-nums text-zinc-900 dark:text-zinc-50">
            {score}
          </div>
        )}
        <div className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
          300–850
        </div>
      </div>
    </div>
  );
}

