# SafeVision AI — Security & Admin Setup Guide

This covers what changed and what you need to do to wire it into your
actual project (since I only had `app.py`, `main.js`, and `index.html` —
`database/db_helper.py` and `utils/email_notifier.py` were reconstructed
from how `app.py` calls them).

## 1. Install new dependencies

```bash
pip install flask-limiter flask-wtf
```

Add both to `requirements.txt`.

## 2. Merge `database/db_helper.py`

If your real `db_helper.py` differs from the one here, just copy the
`migrate_schema()` function and the `_column_exists` / `_table_exists`
helpers into your real file, and call `migrate_schema(conn)` at the end
of your existing `init_db()`. It only **adds** columns/tables if they're
missing, so it's safe to run against a database that already has users
and detection logs in it — nothing is dropped or altered.

New columns on `users`: `is_admin`, `failed_login_attempts`,
`locked_until`, `reset_token`, `reset_token_expiry`, `created_at`.
New table: `admin_audit_log` (records role changes, user deletions, and
target-rule edits — visible via SQLite directly for now; no UI yet).

**First registered user becomes Admin automatically.** If your database
already has users, the migration promotes whichever one has the lowest
`id` (i.e. your earliest account) to Admin on next startup. Sign in with
that account to reach the new Admin Panel and promote/demote anyone else
from there.

## 3. Add the password-reset email function

`app.py` looks for `send_password_reset_email` in
`utils/email_notifier.py`, the same optional-import pattern already used
for `send_signup_notification`. Add this to that file, reusing whatever
SMTP/mail-sending code your existing `send_login_notification` already
uses:

```python
def send_password_reset_email(to_email, full_name, reset_url):
    """
    Sends the password reset link. reset_url already points at
    /reset_password/<token> — just drop it into your email template.
    """
    # reuse your existing mail-sending helper here
    ...
```

Until you add it, `/forgot_password` still works — it just prints the
reset link to the server console instead of emailing it, so you can keep
testing locally.

## 4. What's now protected

| Feature | Details |
|---|---|
| **Rate limiting** | `/login`: 5/min · `/register`: 5/hour · `/forgot_password`: 3/hour · `/api/reset_password`: 5/hour. Backed by in-memory storage — fine for one process; switch `storage_uri` in `utils/security.py` to Redis if you run multiple gunicorn workers. |
| **Account lockout** | 5 failed logins locks the account for 15 minutes, independent of the IP-based rate limit above (so a distributed attacker hitting one account from many IPs is still stopped). |
| **CSRF protection** | Flask-WTF `CSRFProtect` is on globally. All `fetch()` calls in `main.js` now go through a shared `apiFetch()` helper that attaches the `X-CSRFToken` header (read from the `<meta name="csrf-token">` tag). The plain HTML `/detect` upload form got a hidden `csrf_token` field instead. |
| **Session timeout** | 30 minutes of inactivity clears the session (`enforce_session_timeout()` runs on every request via `before_request`). Adjust `SESSION_IDLE_TIMEOUT_MINUTES` in `utils/security.py`. |
| **Forgot/reset password** | `/forgot_password` (POST, email) → emails a one-time link valid 30 minutes → `/reset_password/<token>` (new page) → `/api/reset_password` (POST) sets the new password. Deliberately returns the same generic message whether or not the email exists, so it can't be used to enumerate accounts. |
| **Admin vs Safety Inspector** | New `/api/admin/users` (list), `/api/admin/users/<id>/role` (promote/demote), `/api/admin/users/<id>` (delete) — all admin-only. `/api/targets` (the PPE mandate checklist) is now admin-only too; the UI disables the checkboxes/save button for non-admins and shows a note. |

## 5. Environment variables for production

```bash
export SAFEVISION_SECRET_KEY="a long random value — don't reuse the dev default"
export SAFEVISION_HTTPS=1   # once you're actually behind HTTPS, so cookies are Secure-only
```

## 6. Model accuracy & validation

That one's separate from the web-app security work above — see
`utils/model_validation.py`. It runs Ultralytics' built-in `model.val()`
against a labeled validation set and produces:

- Overall + per-class Precision / Recall / mAP50 / mAP50-95
- A confusion matrix image
- PR/F1 curves
- A plain-text `model_validation_report.txt` you can drop into an FYP report

**You need a labeled validation split to run it** (images + YOLO-format
label `.txt` files + a `data.yaml`) — that's the actual prerequisite, not
something the script can work around. If your original training run used
a `data.yaml` with a `val:` split, point `--data` at that same file:

```bash
python utils/model_validation.py --data path/to/data.yaml
```

If you don't have that file anymore or never held out a validation split
during training, let me know and we can figure out how to carve one out
of your existing labeled dataset (or, if you only have unlabeled
footage, what your options are for getting ground-truth labels).