# SentiVeraAI frontend

React + TypeScript with two separate web experiences in one application:

- `/patient` (or `/patient/login`): patient sign-in and registration, optional wellbeing check-in, text/voice conversations, and care-team sharing settings. No sensor plots, prediction scores, XAI panels, or generation traces.
- `/doctor` (or `/doctor/login`): doctor sign-in, linked patient list, recent conversations, saved response-generation steps, on-demand XAI, and patient-specific wearable controls and waveforms.

The backend enforces roles and patient ownership independently of the frontend. Doctor accounts must be provisioned by an administrator. Patients choose a doctor by email under **Your care team**, explicitly sharing their retained activity. They can change doctors or stop sharing. Revocation also disconnects their active wearable.

## Run

Start the API from `agent/backend` in your configured Python environment:

```bash
uvicorn api:app --reload --port 8005 --reload-exclude ".venv/*"
```

Create a doctor account from that directory (password is entered privately):

```bash
python manage_accounts.py doctor --email doctor@example.com --name "Dr. Perera"
```

Then, from `agent/frontend`:

```bash
npm install
npm run dev
```

Open `http://localhost:5173/patient` and `http://localhost:5173/doctor`. Use separate browser profiles or a private window to remain signed in as two different accounts simultaneously. Login cookies are shared between tabs in the same browser profile.

Patients create their own account on the patient sign-in page, then enter the doctor's provisioned email in care-team settings. A doctor sees only patients who have shared access with that account.

## Sessions and data

The API proxy uses `/api/*` at `http://localhost:8005`. Authentication uses a server-issued HttpOnly, SameSite cookie with a 12-hour lifetime. Passwords are stored as salted scrypt hashes. Logout revokes the server session. Browser-generated `session_id` values are no longer trusted; each account owns one continuing text/voice conversation across tabs and reloads. **New chat** clears that patient's retained conversation and live explanation/audio contexts.

Conversations retain the existing backend limits (`chat_history` in `config.yaml`): by default, 200 messages and a 24-hour inactivity expiry. Saved decision traces share that retention. On-demand XAI and voice audio use the latest 50 in-memory turn contexts and expire after a backend restart. Older traces remain readable even when XAI is unavailable. Existing anonymous demo conversations are not assigned to new accounts.

The current hardware runtime supports **one active wearable at a time**. A doctor selects a patient before connecting or uploading a recording; that signal only influences the selected patient's replies. Disconnect before moving the device to another patient. Patients can chat without a wearable.

For HTTPS deployment, set `AGENT_COOKIE_SECURE=1`, serve the UI and `/api` through the same origin, and configure the web server to fall back to `index.html` for `/patient` and `/doctor`. `AGENT_ACCOUNTS_DB` optionally sets the account database path. The default is `agent/backend/accounts.sqlite3`; account and chat database files are ignored by Git. Run one API worker while sensor, questionnaire, and XAI state remain in memory.

## Checks

```bash
npm run build
npm run lint
```

Backend authorization regression tests (from `agent/backend`):

```bash
python -m unittest tests.test_portal_access tests.test_chat_store tests.test_trace tests.test_multimodal_stress -v
```
