from __future__ import annotations

import ipaddress
import json
import shutil
import threading
import webbrowser
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

from .actions import ActionDecision
from .schema_draft import SchemaSuggestionDecision
from .web_service import LocalWebStore


class DecisionPayload(BaseModel):
    decision: Literal["approved", "rejected", "proposed"]


def _loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_bind_host(host: str, *, allow_network: bool) -> None:
    if _loopback_host(host):
        return
    if not allow_network:
        raise ValueError(
            "Refusing to expose research data over the network. "
            "Use a loopback host (127.0.0.1/localhost) or explicitly pass "
            "--allow-network."
        )


def _api_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def create_app(
    data_root: str | Path,
    *,
    max_upload_mb: int = 1024,
    trusted_hosts: list[str] | None = None,
) -> FastAPI:
    store = LocalWebStore(data_root)
    app = FastAPI(
        title="Data Cleaning Assistant",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.max_upload_bytes = max_upload_mb * 1024 * 1024
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=trusted_hosts
        or ["127.0.0.1", "localhost", "[::1]", "testserver"],
    )

    @app.middleware("http")
    async def security_headers(_request, call_next):
        response = await call_next(_request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "object-src 'none'; "
            "base-uri 'none'; "
            "frame-ancestors 'none'"
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_APP_HTML)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "mode": "local-only"}

    @app.post("/api/sessions")
    async def create_session(file: UploadFile = File(...)):
        session_root: Path | None = None
        try:
            session_id, source_path, workspace = store.new_session_paths(
                file.filename or ""
            )
            session_root = workspace.parent
            total = 0
            with source_path.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > app.state.max_upload_bytes:
                        raise ValueError(
                            f"Upload exceeds the {max_upload_mb} MB local UI limit."
                        )
                    handle.write(chunk)
            if total == 0:
                raise ValueError("Uploaded dataset is empty.")

            store.start_from_path(source_path, workspace)
            return store.state(session_id).model_dump(mode="json")
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            if session_root is not None and session_root.exists():
                shutil.rmtree(session_root, ignore_errors=True)
            raise _api_error(exc) from exc
        finally:
            await file.close()

    @app.get("/api/sessions/{session_id}")
    async def session_state(session_id: str):
        try:
            return store.state(session_id).model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.get("/api/sessions/{session_id}/schema")
    async def schema_suggestions(session_id: str):
        try:
            return [
                item.model_dump(mode="json")
                for item in store.schema_suggestions(session_id)
            ]
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/schema/{suggestion_id}")
    async def schema_decision(
        session_id: str,
        suggestion_id: str,
        payload: DecisionPayload,
    ):
        try:
            state = store.decide_schema(
                session_id,
                suggestion_id,
                SchemaSuggestionDecision(payload.decision),
            )
            return state.model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/schema/complete")
    async def complete_schema(session_id: str):
        try:
            return store.complete_schema(session_id).model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.get("/api/sessions/{session_id}/actions")
    async def actions(session_id: str):
        try:
            return [
                item.model_dump(mode="json")
                for item in store.actions(session_id)
            ]
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/actions/{action_id}")
    async def action_decision(
        session_id: str,
        action_id: str,
        payload: DecisionPayload,
    ):
        try:
            state = store.decide_action(
                session_id,
                action_id,
                ActionDecision(payload.decision),
            )
            return state.model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/review/complete")
    async def complete_review(session_id: str):
        try:
            return store.complete_review(session_id).model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/apply")
    async def apply(session_id: str):
        try:
            return store.apply(session_id).model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.get("/api/sessions/{session_id}/verification")
    async def verification(session_id: str):
        try:
            session = store.load(session_id)
            if not session.verification_path:
                raise FileNotFoundError("Verification report is not available yet.")
            return json.loads(
                Path(session.verification_path).read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.get("/api/sessions/{session_id}/download")
    async def download(session_id: str):
        try:
            path = store.cleaned_path(session_id)
            return FileResponse(
                path,
                filename=path.name,
                media_type="application/octet-stream",
            )
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    return app


def run_local_web(
    *,
    data_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    allow_network: bool = False,
    max_upload_mb: int = 1024,
) -> None:
    validate_bind_host(host, allow_network=allow_network)

    import uvicorn

    trusted_hosts = ["*"] if allow_network and not _loopback_host(host) else None
    app = create_app(
        data_root,
        max_upload_mb=max_upload_mb,
        trusted_hosts=trusted_hosts,
    )
    if open_browser and _loopback_host(host):
        browser_host = "127.0.0.1" if host == "0.0.0.0" else host
        url = f"http://{browser_host}:{port}/"
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")


_APP_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Data Cleaning Assistant</title>
<style>
:root {
  color-scheme: light;
  --bg:#f5f7f9; --panel:#ffffff; --ink:#17212b; --muted:#65717d;
  --line:#dbe2e8; --accent:#1f6f5f; --accent2:#e8f3f0; --danger:#a33535;
  --warn:#8b6518; --shadow:0 12px 34px rgba(25,40,55,.08);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.48 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1060px;margin:0 auto;padding:40px 22px 70px}
header{display:flex;gap:24px;justify-content:space-between;align-items:flex-end;margin-bottom:28px}
h1{font-size:32px;letter-spacing:-.04em;margin:0 0 6px}
h2{font-size:21px;margin:0 0 14px}
h3{font-size:16px;margin:0 0 8px}
p{margin:6px 0;color:var(--muted)}
.badge{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:white;font-size:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:var(--shadow);margin:16px 0}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.grow{flex:1}
button,.button{border:0;border-radius:10px;padding:10px 14px;font:inherit;font-weight:650;cursor:pointer;text-decoration:none;display:inline-block}
button.primary,.button.primary{background:var(--accent);color:white}
button.secondary{background:#eef2f5;color:var(--ink)}
button.reject{background:#f7eaea;color:var(--danger)}
button:disabled{opacity:.45;cursor:not-allowed}
input[type=file]{display:block;width:100%;padding:16px;border:1px dashed #aeb9c3;border-radius:12px;background:#fafcfd}
.card{border:1px solid var(--line);border-radius:13px;padding:16px;margin:12px 0}
.card.approved{border-color:#98cabc;background:#f6fbf9}
.card.rejected{border-color:#e0b1b1;background:#fffafa}
.meta{font-size:12px;color:var(--muted);margin-bottom:8px}
.code{font:12px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;background:#f3f5f7;padding:10px;border-radius:9px;overflow:auto;white-space:pre-wrap}
.status{padding:10px 12px;border-radius:10px;background:var(--accent2);color:#24594e;margin:10px 0}
.error{background:#fdecec;color:#852d2d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.metric{border:1px solid var(--line);padding:12px;border-radius:10px}
.metric b{display:block;font-size:20px}
.hidden{display:none!important}
footer{margin-top:30px;color:var(--muted);font-size:12px}
@media(max-width:650px){header{display:block}h1{font-size:27px}.panel{padding:16px}}
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>Data Cleaning Assistant</h1>
    <p>Local-first review, cleaning, provenance, and verification.</p>
  </div>
  <span class="badge">Runs on this computer</span>
</header>

<section id="start" class="panel">
  <h2>Start a cleaning session</h2>
  <p>Select a CSV, XLSX, or Parquet dataset. The file stays with this local server.</p>
  <form id="uploadForm">
    <input id="fileInput" name="file" type="file" accept=".csv,.xlsx,.parquet" required>
    <div class="row" style="margin-top:14px">
      <button class="primary" type="submit">Analyze dataset</button>
      <span id="uploadNote" class="meta"></span>
    </div>
  </form>
</section>

<section id="session" class="hidden">
  <div id="sessionStatus" class="status"></div>
  <div id="summary" class="grid"></div>

  <section id="schemaPanel" class="panel hidden">
    <h2>Confirm the study dictionary</h2>
    <p>These are suggestions only. Approve a statement only when it matches the study's actual meaning.</p>
    <div id="schemaList"></div>
    <button id="schemaComplete" class="primary">Finish schema review</button>
  </section>

  <section id="actionsPanel" class="panel hidden">
    <h2>Review proposed cleaning</h2>
    <p>Each change remains read-only until you explicitly approve it and finish review.</p>
    <div id="actionsList"></div>
    <button id="reviewComplete" class="primary">Finish cleaning review</button>
  </section>

  <section id="applyPanel" class="panel hidden">
    <h2>Ready to create the cleaned copy</h2>
    <p>The source file remains unchanged. Approved actions will be applied to a new file and verified automatically.</p>
    <button id="applyButton" class="primary">Apply approved changes & verify</button>
  </section>

  <section id="donePanel" class="panel hidden">
    <h2>Verified cleaning complete</h2>
    <p id="doneText"></p>
    <div id="verificationMetrics" class="grid"></div>
    <div class="row" style="margin-top:14px">
      <a id="downloadLink" class="button primary" href="#">Download cleaned dataset</a>
      <button id="newSession" class="secondary">Start another session</button>
    </div>
  </section>
</section>

<footer>
No cloud upload, telemetry, or external assets are used by this interface. Keep the local server private unless you intentionally enable network access.
</footer>
</main>

<script>
let currentId = localStorage.getItem("dataCleanerSession");
const $ = id => document.getElementById(id);

function escapeText(value) {
  if (value == null) return "";
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
async function api(url, options={}) {
  const res = await fetch(url, options);
  let body = null;
  try { body = await res.json(); } catch (_) {}
  if (!res.ok) throw new Error(body?.detail || "Request failed");
  return body;
}
function setStatus(text, error=false) {
  const el = $("sessionStatus");
  el.textContent = text;
  el.className = "status" + (error ? " error" : "");
}
function decisionButtons(kind, id, decision) {
  return `
    <div class="row">
      <button class="secondary decision" data-kind="${kind}" data-id="${id}" data-decision="approved">Approve</button>
      <button class="reject decision" data-kind="${kind}" data-id="${id}" data-decision="rejected">Reject</button>
      <span class="meta">Current: ${escapeText(decision)}</span>
    </div>`;
}
function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}
function cardClass(decision) {
  return "card " + (decision === "approved" ? "approved" : decision === "rejected" ? "rejected" : "");
}
function renderSummary(state) {
  $("summary").innerHTML = `
    <div class="metric"><span>Stage</span><b>${escapeText(state.stage)}</b></div>
    <div class="metric"><span>Schema unresolved</span><b>${state.schema_unresolved}</b></div>
    <div class="metric"><span>Cleaning unresolved</span><b>${state.action_unresolved}</b></div>
    <div class="metric"><span>Approved actions</span><b>${state.approved_actions}</b></div>`;
}
async function refresh() {
  if (!currentId) return;
  const state = await api(`/api/sessions/${currentId}`);
  renderSummary(state);
  setStatus(`${state.source_name} · ${state.stage}`);
  ["schemaPanel","actionsPanel","applyPanel","donePanel"].forEach(id => $(id).classList.add("hidden"));

  if (state.stage === "schema_review") {
    $("schemaPanel").classList.remove("hidden");
    const items = await api(`/api/sessions/${currentId}/schema`);
    $("schemaList").innerHTML = items.map(item => `
      <article class="${cardClass(item.decision)}">
        <div class="meta">${escapeText(item.kind)} · ${escapeText(item.table)}${item.column ? "."+escapeText(item.column) : ""} · ${Math.round(item.confidence*100)}%</div>
        <h3>${escapeText(item.rationale)}</h3>
        <div class="code">${escapeText(pretty(item.proposed))}</div>
        ${decisionButtons("schema", item.id, item.decision)}
      </article>`).join("");
    $("schemaComplete").disabled = state.schema_unresolved !== 0;
  } else if (state.stage === "cleaning_review") {
    const items = await api(`/api/sessions/${currentId}/actions`);
    const readyForCompletion = state.action_unresolved === 0 && !state.review_completed;
    if (!state.review_completed) {
      $("actionsPanel").classList.remove("hidden");
      $("actionsList").innerHTML = items.map(item => {
        const preview = item.sample_changes.length ? item.sample_changes : item.sample_rows;
        return `
        <article class="${cardClass(item.decision)}">
          <div class="meta">${escapeText(item.type)} · ${escapeText(item.table)}${item.column ? "."+escapeText(item.column) : ""} · ${item.affected_count} affected</div>
          <h3>${escapeText(item.title)}</h3>
          <p>${escapeText(item.description)}</p>
          <div class="code">${escapeText(pretty(item.parameters))}</div>
          ${preview.length ? '<details><summary>Preview examples</summary><div class="code">'+escapeText(pretty(preview))+'</div></details>' : ''}
          ${decisionButtons("action", item.id, item.decision)}
        </article>`;
      }).join("");
      $("reviewComplete").disabled = !readyForCompletion;
    } else {
      $("applyPanel").classList.remove("hidden");
    }
  } else if (state.stage === "verified") {
    $("donePanel").classList.remove("hidden");
    $("doneText").textContent = "The cleaned copy passed source, audit, replay, and undo verification.";
    $("downloadLink").href = `/api/sessions/${currentId}/download`;
    const report = await api(`/api/sessions/${currentId}/verification`);
    const m = report.metrics;
    $("verificationMetrics").innerHTML = `
      <div class="metric"><span>Executed actions</span><b>${m.execution_count}</b></div>
      <div class="metric"><span>Cell changes</span><b>${m.cell_change_events}</b></div>
      <div class="metric"><span>Net changed cells</span><b>${m.net_different_cells ?? "—"}</b></div>
      <div class="metric"><span>Repeated touches</span><b>${m.repeated_touch_events}</b></div>`;
  } else if (["no_changes","no_approved_changes","no_executable_changes"].includes(state.stage)) {
    setStatus("Session complete: " + state.stage.replaceAll("_"," "));
    $("donePanel").classList.remove("hidden");
    $("doneText").textContent = "No cleaned file was written because there were no executable approved changes.";
    $("downloadLink").classList.add("hidden");
  } else if (state.stage === "verification_failed") {
    setStatus("Verification failed. The source file remains unchanged.", true);
  }

  document.querySelectorAll(".decision").forEach(button => {
    button.onclick = async () => {
      try {
        button.disabled = true;
        const kind = button.dataset.kind;
        const id = button.dataset.id;
        const decision = button.dataset.decision;
        await api(
          `/api/sessions/${currentId}/${kind === "schema" ? "schema" : "actions"}/${id}`,
          {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision})}
        );
        await refresh();
      } catch (err) { setStatus(err.message, true); }
    };
  });
}

$("uploadForm").onsubmit = async event => {
  event.preventDefault();
  const file = $("fileInput").files[0];
  if (!file) return;
  const data = new FormData();
  data.append("file", file);
  $("uploadNote").textContent = "Analyzing locally…";
  try {
    const state = await api("/api/sessions", {method:"POST", body:data});
    currentId = state.id;
    localStorage.setItem("dataCleanerSession", currentId);
    $("start").classList.add("hidden");
    $("session").classList.remove("hidden");
    await refresh();
  } catch (err) {
    $("uploadNote").textContent = err.message;
  }
};

$("schemaComplete").onclick = async () => {
  try {
    await api(`/api/sessions/${currentId}/schema/complete`, {method:"POST"});
    await refresh();
  } catch (err) { setStatus(err.message, true); }
};
$("reviewComplete").onclick = async () => {
  try {
    await api(`/api/sessions/${currentId}/review/complete`, {method:"POST"});
    await refresh();
  } catch (err) { setStatus(err.message, true); }
};
$("applyButton").onclick = async () => {
  try {
    $("applyButton").disabled = true;
    setStatus("Applying approved changes and verifying…");
    await api(`/api/sessions/${currentId}/apply`, {method:"POST"});
    await refresh();
  } catch (err) {
    $("applyButton").disabled = false;
    setStatus(err.message, true);
  }
};
$("newSession").onclick = () => {
  localStorage.removeItem("dataCleanerSession");
  location.reload();
};

(async () => {
  if (!currentId) return;
  try {
    $("start").classList.add("hidden");
    $("session").classList.remove("hidden");
    await refresh();
  } catch (_) {
    localStorage.removeItem("dataCleanerSession");
    currentId = null;
    $("start").classList.remove("hidden");
    $("session").classList.add("hidden");
  }
})();
</script>
</body>
</html>
"""
