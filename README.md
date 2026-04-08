# CreditNeverBefore
Full-stack demo for **behavioral credit scoring**: a **Next.js** frontend collects user-friendly inputs and calls a **Python (Flask)** backend that scores features and returns a 300–850-style score plus decision hints.
**Disclaimer:** Educational / portfolio project only—not financial advice or a production credit system.
---
## Architecture
```text
Browser (Next.js, port 3001)
    │  POST JSON (AGE, EXT_SOURCE_1, UPI_VELOCITY, …)
    ▼
/api/predict
    ├── Local dev:  http://localhost:5000/api/predict  (Flask)
    └── Production: same path on Vercel (serverless Python)
The frontend never runs the model; all inference happens in the Python API (api/index.py + api/model_logic.py).

Frontend
Stack
Next.js 14 (App Router), React 18, TypeScript
Tailwind CSS for layout and styling
lucide-react for icons
next-themes (via app/providers.tsx) for theme support
What it does
Renders a single main screen (app/page.tsx): sliders and segmented controls for age, “CIBIL-style” trust, UPI habit, payment behavior, and app activity.
Maps those UI choices to the exact feature names the model expects: AGE, EXT_SOURCE_1, UPI_VELOCITY, BILL_PAY_CONSISTENCY, APP_USAGE_DAYS.
Optional numeric overrides (0–1) for power users; when valid, they replace the mapped values.
Calls the predict API with fetch, then shows score (radial gauge), category, action, and APR estimate, plus model version from the response.
API URL logic
If NEXT_PUBLIC_API_BASE_URL is set, requests go to
{NEXT_PUBLIC_API_BASE_URL}/api/predict (trailing slashes stripped).
On localhost, if that env is unset, the app uses http://localhost:5000/api/predict so you can run Flask locally.
Otherwise (e.g. deployed on Vercel), it uses the relative path /api/predict (same origin as the Next app).
Run locally
npm install
npm run dev
Default dev URL: http://localhost:3001 (see package.json scripts).

Frontend file map
Path	Role
app/page.tsx
Main UI, feature mapping, fetch to /api/predict
app/layout.tsx
Root layout, metadata, font
app/globals.css
Global styles
app/providers.tsx
Theme / client providers
app/components/RadialScoreGauge.tsx
Score visualization
app/components/MetaBadge.tsx
Category / action chips
Backend
Stack
Flask HTTP server
flask-cors — CORS enabled for browser calls (including Next dev server)
Pure Python scoring in api/model_logic.py (exported score()), loaded by api/index.py
Endpoint
POST /api/predict

Content-Type: application/json
Body: JSON object with model inputs (see below). DAYS_BIRTH can be sent instead of AGE (age is derived).
Example keys (aligned with training / FEATURES in api/index.py):

Key	Meaning (high level)
AGE
Age in years (or use DAYS_BIRTH for compatibility)
EXT_SOURCE_1
External trust signal (0–1; UI maps from CIBIL-style slider)
UPI_VELOCITY
Digital spend / UPI habit (0–1)
BILL_PAY_CONSISTENCY
On-time payment behavior (0–1)
APP_USAGE_DAYS
App engagement in days (scaled as in training)
Success response (JSON):

prob_default — model default probability
cnb_credit_score — integer 300–850
category — e.g. Excellent / Good / Average / Poor
action — e.g. approval hint
apr_estimate — illustrative APR band
version — API/model label string
Errors return JSON with an error field and appropriate HTTP status (e.g. 400 for bad JSON).

Vercel
vercel.json rewrites /api/:path* → /api/index.py so the Flask app serves under /api/*.
The module exposes handler / app for the serverless runtime; python api/index.py is for local runs.
Run API locally
cd api
pip install -r requirements.txt
python index.py
Default port comes from the PORT env var, or 5000 if unset—matching the frontend’s localhost default.

Backend file map
Path	Role
api/index.py
Flask app, /api/predict, score buckets, JSON responses
api/model_logic.py
Generated / pure-Python score() used at inference
api/requirements.txt
flask, flask-cors
Monorepo layout
CreditNeverBefore/
├── app/                    # Frontend (Next.js)
├── api/                    # Backend (Flask + model)
├── vercel.json             # API routing & serverless config
├── package.json
└── …
Scripts (quick reference)
Command	Purpose
npm run dev
Next.js dev server (port 3001)
npm run build / npm run start
Production build / start
cd api && python index.py
Local Flask API
Author
Manish Mahto
mahatomanish.com
