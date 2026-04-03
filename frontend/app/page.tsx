"use client";

import { useMemo, useState } from "react";
import { Loader2, Sparkles, Zap } from "lucide-react";

import RadialScoreGauge from "./components/RadialScoreGauge";
import MetaBadge from "./components/MetaBadge";

type PredictResponse = {
  prob_default: number;
  cnb_credit_score: number;
  category: string;
  action: string;
  apr_estimate: string;
  version?: string;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:5000";

function scoreTone(score: number) {
  if (score >= 750) return "good";
  if (score >= 650) return "warn";
  return "bad";
}

type UpiOption = "1-5" | "6-10" | "10-20" | "20+";
type PaymentOption = "before" | "sometimes_before" | "on" | "post";
type AppActivityOption = "lt_1" | "1_3" | "gt_3";

function mapUpiVelocity(opt: UpiOption): number {
  switch (opt) {
    case "1-5":
      return 0.2;
    case "6-10":
      return 0.5;
    case "10-20":
      return 0.8;
    case "20+":
      return 1.0;
  }
}

function mapPaymentReliability(opt: PaymentOption): number {
  switch (opt) {
    case "before":
      return 1.0;
    case "sometimes_before":
      return 0.7;
    case "on":
      return 0.4;
    case "post":
      return 0.1;
  }
}

function mapAppActivity(opt: AppActivityOption): number {
  switch (opt) {
    case "lt_1":
      return 0.2;
    case "1_3":
      return 0.6;
    case "gt_3":
      return 1.0;
  }
}

function mapCibilToExternalTrust(cibil: number): number {
  // Treat higher CIBIL as LOWER risk for the model.
  // We map CIBIL in [300, 900] to a 0–1 score where:
  //  - 300  -> 1.0 (worst)
  //  - 900  -> 0.0 (best)
  const normalized = (cibil - 300) / 600;
  const risk = Math.max(0, Math.min(1, normalized));
  return 1 - risk;
}

export default function Page() {
  // Layman UI inputs. These are mapped to the technical model keys inside `handlePredict()`.
  const [cibilScore, setCibilScore] = useState<number>(750);
  const [upiHabit, setUpiHabit] = useState<UpiOption>("6-10");
  const [paymentTrack, setPaymentTrack] = useState<PaymentOption>("on");
  const [appActivity, setAppActivity] = useState<AppActivityOption>("1_3");

  // Optional manual overrides for the 0–1 features. When set, these take precedence over mappings.
  const [extOverride, setExtOverride] = useState<string>("");
  const [upiOverride, setUpiOverride] = useState<string>("");
  const [payOverride, setPayOverride] = useState<string>("");
  const [appOverride, setAppOverride] = useState<string>("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [score, setScore] = useState<number | null>(null);
  const [category, setCategory] = useState<string | null>(null);
  const [action, setAction] = useState<string | null>(null);
  const [aprEstimate, setAprEstimate] = useState<string | null>(null);
  const [modelVersion, setModelVersion] = useState<string>("1.1.0-injected");

  const tone = useMemo(() => (score === null ? "neutral" : scoreTone(score)), [score]);

  async function handlePredict() {
    setError(null);
    setLoading(true);

    try {
      console.log("Fetching from:", `${API_BASE_URL}/api/predict`);

      // Resolve final numeric values for each feature (manual override takes precedence when valid).
      const parsedExt = extOverride.trim() === "" ? NaN : Number(extOverride);
      const parsedUpi = upiOverride.trim() === "" ? NaN : Number(upiOverride);
      const parsedPay = payOverride.trim() === "" ? NaN : Number(payOverride);
      const parsedApp = appOverride.trim() === "" ? NaN : Number(appOverride);

      const clamp01 = (v: number) =>
        Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : NaN;

      const extFromOverride = clamp01(parsedExt);
      const upiFromOverride = clamp01(parsedUpi);
      const payFromOverride = clamp01(parsedPay);
      const appFromOverride = clamp01(parsedApp);

      const extFromMapped = clamp01(mapCibilToExternalTrust(cibilScore));
      const upiFromMapped = clamp01(mapUpiVelocity(upiHabit));
      const payFromMapped = clamp01(mapPaymentReliability(paymentTrack));
      const appFromMapped = clamp01(mapAppActivity(appActivity));

      const extValue = Number.isFinite(extFromOverride)
        ? extFromOverride
        : extFromMapped;
      const upiValue = Number.isFinite(upiFromOverride)
        ? upiFromOverride
        : upiFromMapped;
      const payValue = Number.isFinite(payFromOverride)
        ? payFromOverride
        : payFromMapped;
      const appValue = Number.isFinite(appFromOverride)
        ? appFromOverride
        : appFromMapped;
      const res = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          // Keep model keys exactly as expected.
          EXT_SOURCE_1: extValue,
          UPI_VELOCITY: upiValue,
          BILL_PAY_CONSISTENCY: payValue,
          APP_USAGE_DAYS: appValue,
        }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`API error (${res.status}): ${text || res.statusText}`);
      }

      const data = (await res.json()) as PredictResponse;
      setScore(Number(data.cnb_credit_score));
      setCategory(data.category);
      setAction(data.action);
      setAprEstimate(data.apr_estimate);
      if (data.version) setModelVersion(data.version);
    } catch (e) {
      setScore(null);
      setCategory(null);
      setAction(null);
      setAprEstimate(null);
      setError(e instanceof Error ? e.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-50">
      <header className="mx-auto flex max-w-6xl items-center justify-between px-5 py-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-black/5 ring-1 ring-black/10 dark:bg-white/5 dark:ring-white/10">
            <Sparkles className="h-5 w-5 text-zinc-700 dark:text-zinc-200" />
          </div>
          <div>
            <div className="text-lg font-semibold tracking-tight">CreditNeverBefore</div>
            <div className="text-xs text-zinc-500 dark:text-zinc-400">
              Behavioral credit scoring
            </div>
          </div>
        </div>

        <div className="text-xs text-zinc-500 dark:text-zinc-400">
          Dev: GenAI Manish Mahto
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 pb-10">
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Left: inputs */}
          <section className="rounded-2xl bg-white/80 p-5 shadow-sm ring-1 ring-black/5 backdrop-blur dark:bg-white/5 dark:ring-white/10">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Input Profile
              </h2>
              <div className="flex items-center gap-2 rounded-xl bg-zinc-900/5 px-3 py-1 text-xs text-zinc-600 dark:bg-white/5 dark:text-zinc-300">
                <Zap className="h-3.5 w-3.5" />
                Instant scoring
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-sm font-medium">Existing CIBIL Score</span>
                  <span className="text-xs text-zinc-500 dark:text-zinc-400 tabular-nums">
                    {cibilScore}
                  </span>
                </div>
                <div className="mb-2 text-xs text-zinc-500 dark:text-zinc-400">
                  Maps to a trusted credit signal ({mapCibilToExternalTrust(cibilScore).toFixed(2)})
                </div>

                <input
                  type="range"
                  min={300}
                  max={900}
                  step={1}
                  value={cibilScore}
                  onChange={(e) => setCibilScore(Number(e.target.value))}
                  className="w-full"
                />
                <div className="mt-1 flex justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
                  <span>300</span>
                  <span>900</span>
                </div>

                <div className="mt-3">
                  <label className="flex items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                    <span>Manual override (0.0 – 1.0)</span>
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={extOverride}
                      onChange={(e) => setExtOverride(e.target.value)}
                      className="w-24 rounded-md border border-zinc-200 bg-white px-2 py-1 text-right text-xs text-zinc-900 outline-none focus:border-zinc-400 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-50"
                      placeholder={mapCibilToExternalTrust(cibilScore)
                        .toFixed(2)
                        .toString()}
                    />
                  </label>
                </div>
              </div>

              <SegmentedGroup<UpiOption>
                title="Digital Spend Habit"
                subtitle="How often you use UPI for your daily needs."
                value={upiHabit}
                onChange={setUpiHabit}
                options={[
                  { value: "1-5", label: "1-5 times/mo", weight: 0.2 },
                  { value: "6-10", label: "6-10 times/mo", weight: 0.5 },
                  { value: "10-20", label: "10-20 times/mo", weight: 0.8 },
                  { value: "20+", label: "More than 20", weight: 1.0 },
                ]}
              />

              <div>
                <label className="flex items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                  <span>Digital habit override (0.0 – 1.0)</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={upiOverride}
                    onChange={(e) => setUpiOverride(e.target.value)}
                    className="w-24 rounded-md border border-zinc-200 bg-white px-2 py-1 text-right text-xs text-zinc-900 outline-none focus:border-zinc-400 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-50"
                    placeholder={mapUpiVelocity(upiHabit).toFixed(2).toString()}
                  />
                </label>
              </div>

              <SegmentedGroup<PaymentOption>
                title="On-Time Payment Score"
                subtitle="Your track record with bill deadlines."
                value={paymentTrack}
                onChange={setPaymentTrack}
                options={[
                  { value: "before", label: "Before due date", weight: 1.0 },
                  { value: "sometimes_before", label: "Sometimes before", weight: 0.7 },
                  { value: "on", label: "On due date", weight: 0.4 },
                  { value: "post", label: "Post due date", weight: 0.1 },
                ]}
              />

              <div>
                <label className="flex items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                  <span>Reliability override (0.0 – 1.0)</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={payOverride}
                    onChange={(e) => setPayOverride(e.target.value)}
                    className="w-24 rounded-md border border-zinc-200 bg-white px-2 py-1 text-right text-xs text-zinc-900 outline-none focus:border-zinc-400 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-50"
                    placeholder={mapPaymentReliability(paymentTrack).toFixed(2).toString()}
                  />
                </label>
              </div>

              <SegmentedGroup<AppActivityOption>
                title="App Activity"
                subtitle="Time spent on this platform to manage your expenses."
                value={appActivity}
                onChange={setAppActivity}
                options={[
                  { value: "lt_1", label: "Less than a month", weight: 0.2 },
                  { value: "1_3", label: "1-3 months", weight: 0.6 },
                  { value: "gt_3", label: "Beyond 3 months", weight: 1.0 },
                ]}
              />

              <div>
                <label className="flex items-center justify-between gap-2 text-xs text-zinc-500 dark:text-zinc-400">
                  <span>Activity override (0.0 – 1.0)</span>
                  <input
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={appOverride}
                    onChange={(e) => setAppOverride(e.target.value)}
                    className="w-24 rounded-md border border-zinc-200 bg-white px-2 py-1 text-right text-xs text-zinc-900 outline-none focus:border-zinc-400 dark:border-zinc-800 dark:bg-zinc-900/60 dark:text-zinc-50"
                    placeholder={mapAppActivity(appActivity).toFixed(2).toString()}
                  />
                </label>
              </div>

              <button
                type="button"
                onClick={handlePredict}
                disabled={loading}
                className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-70 dark:bg-zinc-50 dark:text-zinc-950 dark:hover:bg-zinc-200"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
                Calculate My Score
              </button>

              {error ? (
                <div className="rounded-xl bg-rose-500/10 p-3 text-sm text-rose-200 ring-1 ring-rose-500/20">
                  {error}
                </div>
              ) : null}
            </div>
          </section>

          {/* Right: results */}
          <section className="rounded-2xl bg-white/80 p-5 shadow-sm ring-1 ring-black/5 backdrop-blur dark:bg-white/5 dark:ring-white/10">
            <div className="mb-4 flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold">Result</h2>
                <p className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
                  Behavioral score + credit decision signals.
                </p>
              </div>
              <div className="text-right text-xs text-zinc-500 dark:text-zinc-400">
                {loading ? "Scoring..." : "Ready"}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-1">
              <RadialScoreGauge score={score} loading={loading} />

              <div className="space-y-3">
                <div className="flex flex-wrap gap-2">
                  {category ? (
                    <MetaBadge
                      tone={tone as any}
                      icon={category === "Excellent" ? "★" : category === "Good" ? "✓" : "!"}
                    >
                      {category}
                    </MetaBadge>
                  ) : (
                    <MetaBadge tone="neutral">Awaiting score</MetaBadge>
                  )}

                  {action ? (
                    <MetaBadge tone={tone as any}>
                      {action}
                    </MetaBadge>
                  ) : null}
                </div>

                <div>
                  <div className="mb-1 text-sm font-medium">APR Estimate</div>
                  <input
                    readOnly
                    value={aprEstimate ?? ""}
                    placeholder="—"
                    className="w-full cursor-not-allowed rounded-xl border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 outline-none dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-50"
                  />
                </div>
              </div>
            </div>

            <div className="mt-6 text-xs text-zinc-500 dark:text-zinc-400">
              Model Version: {modelVersion}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}

function Range({
  label,
  subtitle,
  value,
  onChange,
}: {
  label: string;
  subtitle?: string;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="text-sm font-medium">{label}</span>
        <span className="text-xs text-zinc-500 dark:text-zinc-400">{value.toFixed(2)}</span>
      </div>
      {subtitle ? (
        <div className="mb-2 text-xs text-zinc-500 dark:text-zinc-400">{subtitle}</div>
      ) : null}
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
      <div className="mt-1 flex justify-between text-[11px] text-zinc-500 dark:text-zinc-400">
        <span>0.0</span>
        <span>1.0</span>
      </div>
    </div>
  );
}

function SegmentedGroup<T extends string>({
  title,
  subtitle,
  value,
  onChange,
  options,
}: {
  title: string;
  subtitle: string;
  value: T;
  onChange: (v: T) => void;
  options: Array<{ value: T; label: string; weight: number }>;
}) {
  return (
    <div>
      <div className="mb-2">
        <div className="text-sm font-medium">{title}</div>
        <div className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
          {subtitle}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-2 lg:grid-cols-2">
        {options.map((opt) => {
          const active = opt.value === value;
          return (
            <button
              key={opt.value}
              type="button"
              onClick={() => onChange(opt.value)}
              className={[
                "rounded-xl border px-3 py-3 text-left transition",
                "focus:outline-none focus:ring-2 focus:ring-zinc-400/60 dark:focus:ring-zinc-500/60",
                active
                  ? "border-zinc-900 bg-zinc-900 text-white dark:border-white dark:bg-white dark:text-zinc-950"
                  : "border-zinc-200 bg-white/60 text-zinc-800 hover:bg-white dark:border-zinc-800 dark:bg-white/5 dark:text-zinc-100 dark:hover:bg-white/10",
              ].join(" ")}
            >
              <div className="text-xs font-semibold">{opt.label}</div>
              <div className="mt-1 text-[11px] text-zinc-500 dark:text-zinc-400">
                Weight: {opt.weight.toFixed(1)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

