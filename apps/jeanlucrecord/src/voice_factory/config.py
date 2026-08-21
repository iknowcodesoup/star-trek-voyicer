"""Environment-variable-name constants and path roots shared across cli.py
and app.py.

diarizer/diarize_worker.py keeps its own copy of HF_TOKEN_ENV_VAR: it runs in
a separate uv environment (see diarizer/pyproject.toml) and is out of scope
for this package, so it cannot import from here.
"""

from pathlib import Path

from dotenv import load_dotenv

# This file lives at src/voice_factory/config.py -- three parents up is the
# app root (apps/jeanlucrecord/), where work/, samples/, checkpoints/,
# docker/, and .env all live. PACKAGE_ROOT is the installed package's own
# directory, one level shallower, for anything that needs to walk relative to
# the package instead of the app (e.g. cli.py locating docker/*.sh).
#
# The nesting depth is load-bearing. Moving this file up or down a directory
# silently retargets APP_ROOT, and .env then fails to load with no error --
# the webhook and HF_TOKEN settings just go missing. Keep config.py directly
# inside the package, and see infrastructure/filesystem_layout.py, which
# resolves WORK_DIR the same way.
PACKAGE_ROOT = Path(__file__).resolve().parent
APP_ROOT = PACKAGE_ROOT.parent.parent

HF_TOKEN_ENV_VAR = "HF_TOKEN"

CORS_ALLOW_ORIGINS_ENV_VAR = "VOICE_FACTORY_CORS_ALLOW_ORIGINS"

# Where to report job changes, so the orchestrator does not have to poll for
# them. Unset, nothing here changes: jobs run exactly as before and the
# orchestrator falls back to asking.
WEBHOOK_URL_ENV_VAR = "VOICE_ORCHESTRATOR_WEBHOOK_URL"
WEBHOOK_TOKEN_ENV_VAR = "VOICE_WEBHOOK_TOKEN"
PROGRESS_INTERVAL_ENV_VAR = "VOICE_PROGRESS_INTERVAL_SECONDS"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0


def load_app_dotenv() -> None:
    # an already-exported variable wins, which is what override=False gives
    load_dotenv(APP_ROOT / ".env", override=False)
