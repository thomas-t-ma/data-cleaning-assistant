from __future__ import annotations

import ipaddress
import json
import shutil
import threading
import webbrowser
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .actions import ActionDecision
from .ai_provider import AIProviderError, OllamaProvider
from .schema_draft import SchemaSuggestionDecision
from .semantic_narrator import ActionNarrator, LocalSemanticNarrator
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


def _browser_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.version == 6:
        return f"[{host}]"
    return host


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
    if isinstance(exc, AIProviderError):
        return HTTPException(status_code=503, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


def create_app(
    data_root: str | Path,
    *,
    max_upload_mb: int = 1024,
    trusted_hosts: list[str] | None = None,
    narrator: ActionNarrator | None = None,
) -> FastAPI:
    store = LocalWebStore(data_root, narrator=narrator)
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
    async def create_session(file: Annotated[UploadFile, File()]):
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

    @app.post("/api/sessions/{session_id}/schema/complete")
    async def complete_schema(session_id: str):
        try:
            return store.complete_schema(session_id).model_dump(mode="json")
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

    @app.get("/api/sessions/{session_id}/actions")
    async def actions(session_id: str):
        try:
            return [
                item.model_dump(mode="json")
                for item in store.actions(session_id)
            ]
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.get("/api/sessions/{session_id}/families")
    async def families(session_id: str):
        try:
            return [
                item.model_dump(mode="json")
                for item in store.action_families(session_id)
            ]
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/families/{family_id}")
    async def family_decision(
        session_id: str,
        family_id: str,
        payload: DecisionPayload,
    ):
        try:
            state = store.decide_family(
                session_id,
                family_id,
                ActionDecision(payload.decision),
            )
            return state.model_dump(mode="json")
        except (FileNotFoundError, ValueError) as exc:
            raise _api_error(exc) from exc

    @app.post("/api/sessions/{session_id}/actions/{action_id}/narrate")
    async def narrate_action(session_id: str, action_id: str):
        try:
            return store.narrate_action(
                session_id,
                action_id,
            ).model_dump(mode="json")
        except (AIProviderError, FileNotFoundError, ValueError) as exc:
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
    narrator_model: str | None = None,
) -> None:
    validate_bind_host(host, allow_network=allow_network)

    import uvicorn

    if _loopback_host(host):
        trusted_hosts = list(
            dict.fromkeys(
                [host, _browser_host(host), "127.0.0.1", "localhost", "[::1]"]
            )
        )
    else:
        trusted_hosts = ["*"] if allow_network else None

    narrator = (
        LocalSemanticNarrator(
            OllamaProvider(),
            model=narrator_model,
        )
        if narrator_model
        else None
    )
    app = create_app(
        data_root,
        max_upload_mb=max_upload_mb,
        trusted_hosts=trusted_hosts,
        narrator=narrator,
    )
    if open_browser and _loopback_host(host):
        url = f"http://{_browser_host(host)}:{port}/"
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
  --bg:#f4f6f8; --panel:#ffffff; --ink:#17212b; --muted:#66737f;
  --line:#d9e0e6; --accent:#176b5b; --accent-soft:#eaf5f2;
  --danger:#9b3434; --danger-soft:#fff2f2; --warn:#82611d;
  --sheet-head:#edf1f4; --changed:#fff5d9; --after:#e5f5ed;
  --shadow:0 12px 34px rgba(25,40,55,.08);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:36px 22px 72px}
header{display:flex;gap:24px;justify-content:space-between;align-items:flex-end;margin-bottom:26px}
h1{font-size:32px;letter-spacing:-.04em;margin:0 0 5px}
h2{font-size:21px;margin:0 0 10px}
h3{font-size:17px;margin:0 0 7px}
h4{font-size:15px;margin:0 0 7px}
p{margin:6px 0;color:var(--muted)}
.badge{display:inline-flex;border:1px solid var(--line);border-radius:999px;padding:6px 10px;background:white;font-size:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:var(--shadow);margin:16px 0}
.family{border:1px solid var(--line);border-radius:16px;padding:18px;margin:16px 0;background:#fff}
.member{border-top:1px solid var(--line);padding-top:16px;margin-top:16px}
.card.approved,.family.approved{border-color:#92c9ba;background:#f8fcfb}
.card.rejected,.family.rejected{border-color:#ddb1b1;background:#fffafa}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.grow{flex:1}
button,.button{border:0;border-radius:10px;padding:10px 14px;font:inherit;font-weight:650;cursor:pointer;text-decoration:none;display:inline-block}
button.primary,.button.primary{background:var(--accent);color:white}
button.secondary{background:#edf1f4;color:var(--ink)}
button.reject{background:#f8eaea;color:var(--danger)}
button.ai{background:#eef0fb;color:#384884}
button:disabled{opacity:.45;cursor:not-allowed}
input[type=file]{display:block;width:100%;padding:16px;border:1px dashed #aeb9c3;border-radius:12px;background:#fafcfd}
.meta{font-size:12px;color:var(--muted);margin-bottom:8px}
.status{padding:10px 12px;border-radius:10px;background:var(--accent-soft);color:#24594e;margin:10px 0}
.error{background:#fdecec;color:#852d2d}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.metric{border:1px solid var(--line);padding:12px;border-radius:10px;background:#fff}
.metric b{display:block;font-size:20px}
.hidden{display:none!important}
.review-note{border-left:3px solid #b8c3cc;padding:4px 0 4px 11px;margin:10px 0;color:#47535d}
.caution{border-left:3px solid #d4a94f;background:#fffaf0;padding:9px 11px;margin:10px 0;border-radius:0 8px 8px 0;color:#5f4a1b}
.ai-note{background:#f5f6ff;border:1px solid #d7dcf5;border-radius:10px;padding:12px;margin-top:10px}
.sheet-wrap{overflow-x:auto;border:1px solid #cfd7de;border-radius:10px;margin:12px 0;background:white}
.sheet{border-collapse:collapse;width:100%;min-width:560px;font-size:13px}
.sheet th,.sheet td{border-right:1px solid #d8dee4;border-bottom:1px solid #d8dee4;padding:8px 10px;text-align:left;vertical-align:top;white-space:nowrap}
.sheet th{background:var(--sheet-head);font-weight:700}
.sheet th.target{background:#dcefe9}
.sheet .rownum{width:48px;text-align:right;color:#7b8791;background:#f6f8fa;font-variant-numeric:tabular-nums}
.sheet td.changed{background:var(--changed)}
.before{color:#8a3c3c;text-decoration:line-through;text-decoration-thickness:1px}
.arrow{color:#7d8790;margin:0 6px}
.after{color:#176b5b;font-weight:700;background:var(--after);padding:2px 4px;border-radius:4px}
.caption{font-size:12px;color:var(--muted);margin:4px 0 12px}
.schema-card{border:1px solid var(--line);border-radius:13px;padding:16px;margin:12px 0}
details summary{cursor:pointer;color:#53616c;font-size:13px}
footer{margin-top:30px;color:var(--muted);font-size:12px}
@media(max-width:650px){header{display:block}h1{font-size:27px}.panel{padding:16px}}
</style>
</head>
<body>
<main>
<header>
  <div>
    <h1>Data Cleaning Assistant</h1>
    <p>Review your data the way you would review a spreadsheet—not a configuration file.</p>
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
    <h2>Confirm what the fields mean</h2>
    <p>These suggestions help the cleaner understand the structure of this dataset. Check them against what you know about the study.</p>
    <div id="schemaList"></div>
    <button id="schemaComplete" class="primary">Finish field review</button>
  </section>

  <section id="actionsPanel" class="panel hidden">
    <h2>Review proposed cleanup</h2>
    <p>Related changes are grouped together. The spreadsheet snippets below show real examples from the uploaded file.</p>
    <div id="familiesList"></div>
    <button id="reviewComplete" class="primary">Finish cleaning review</button>
  </section>

  <section id="applyPanel" class="panel hidden">
    <h2>Ready to create the cleaned copy</h2>
    <p>The source file remains unchanged. Approved changes will be applied to a new file and verified automatically.</p>
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
No cloud upload, telemetry, or external assets are used by this interface. Local-AI explanations, when enabled, are advisory only and cannot change the cleaning plan.
</footer>
</main>

<script>
let currentId = localStorage.getItem("dataCleanerSession");
let currentState = null;
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
function cardClass(decision, base="") {
  return base + (decision === "approved" ? " approved" : decision === "rejected" ? " rejected" : "");
}
function decisionButtons(kind, id, decision, count=1) {
  const approveText = kind === "family" && count > 1 ? `Approve all ${count}` : "Approve";
  const rejectText = kind === "family" && count > 1 ? `Reject all ${count}` : "Reject";
  return `
    <div class="row" style="margin-top:10px">
      <button class="secondary decision" data-kind="${kind}" data-id="${id}" data-decision="approved">${approveText}</button>
      <button class="reject decision" data-kind="${kind}" data-id="${id}" data-decision="rejected">${rejectText}</button>
      <span class="meta">Current: ${escapeText(decision)}</span>
    </div>`;
}
function explanationHtml(exp) {
  return `
    <h3>${escapeText(exp.title)}</h3>
    <p>${escapeText(exp.summary)}</p>
    ${exp.context ? '<div class="review-note">'+escapeText(exp.context)+'</div>' : ''}
    ${exp.caution ? '<div class="caution"><strong>What to check:</strong> '+escapeText(exp.caution)+'</div>' : ''}`;
}
function renderSheet(preview) {
  if (!preview || !preview.columns?.length || !preview.rows?.length) return "";
  const targetSet = new Set(preview.target_columns || []);
  const headers = preview.columns.map(column =>
    `<th class="${targetSet.has(column) ? "target" : ""}">${escapeText(column)}</th>`
  ).join("");
  const body = preview.rows.map(row => {
    const cells = preview.columns.map(column => {
      const cell = row.cells[column] || {text:""};
      if (cell.changed && cell.before != null && cell.after != null) {
        return `<td class="changed"><span class="before">${escapeText(cell.before)}</span><span class="arrow">→</span><span class="after">${escapeText(cell.after)}</span></td>`;
      }
      return `<td class="${cell.changed ? "changed" : ""}">${escapeText(cell.text)}</td>`;
    }).join("");
    return `<tr><td class="rownum">${row.row_number}</td>${cells}</tr>`;
  }).join("");
  return `
    <div class="sheet-wrap">
      <table class="sheet">
        <thead><tr><th class="rownum">Row</th>${headers}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
    ${preview.caption ? '<div class="caption">'+escapeText(preview.caption)+'</div>' : ''}`;
}
function renderSummary(state) {
  $("summary").innerHTML = `
    <div class="metric"><span>Stage</span><b>${escapeText(state.stage.replaceAll("_"," "))}</b></div>
    <div class="metric"><span>Field questions left</span><b>${state.schema_unresolved}</b></div>
    <div class="metric"><span>Cleanup decisions left</span><b>${state.action_unresolved}</b></div>
    <div class="metric"><span>Approved changes</span><b>${state.approved_actions}</b></div>`;
}
function schemaCard(item) {
  return `
    <article class="${cardClass(item.decision, "schema-card")}">
      <div class="meta">${Math.round(item.confidence*100)}% confidence · ${escapeText(item.table)}${item.column ? " · "+escapeText(item.column) : ""}</div>
      ${explanationHtml(item.explanation)}
      ${(item.previews || []).map(renderSheet).join("")}
      ${decisionButtons("schema", item.id, item.decision)}
    </article>`;
}
function memberHtml(member, narratorEnabled) {
  return `
    <section class="member" id="member-${escapeText(member.action_id)}">
      <div class="meta">${escapeText(member.target_label)} · ${member.affected_count.toLocaleString()} affected</div>
      ${explanationHtml(member.explanation)}
      ${renderSheet(member.preview)}
      <div class="row">
        ${decisionButtons("action", member.action_id, member.decision)}
        ${narratorEnabled ? '<button class="ai narrate" data-action="'+escapeText(member.action_id)+'">Explain this field with local AI</button>' : ''}
      </div>
      <div class="ai-note hidden" id="narration-${escapeText(member.action_id)}"></div>
    </section>`;
}
function familyHtml(family, narratorEnabled) {
  const familyDecision = family.decision;
  const bulk = family.member_count > 1
    ? decisionButtons("family", family.id, familyDecision, family.member_count)
    : "";
  return `
    <article class="${cardClass(familyDecision, "family")}">
      <div class="meta">${family.affected_count.toLocaleString()} total affected · ${family.member_count} field${family.member_count===1?"":"s"}</div>
      ${family.member_count > 1 ? explanationHtml(family.explanation) : ""}
      ${bulk}
      ${family.members.map(member => memberHtml(member, narratorEnabled)).join("")}
    </article>`;
}
async function refresh() {
  if (!currentId) return;
  const state = await api(`/api/sessions/${currentId}`);
  currentState = state;
  renderSummary(state);
  setStatus(`${state.source_name} · ${state.stage.replaceAll("_"," ")}`);
  ["schemaPanel","actionsPanel","applyPanel","donePanel"].forEach(id => $(id).classList.add("hidden"));

  if (state.stage === "schema_review") {
    $("schemaPanel").classList.remove("hidden");
    const items = await api(`/api/sessions/${currentId}/schema`);
    $("schemaList").innerHTML = items.map(schemaCard).join("");
    $("schemaComplete").disabled = state.schema_unresolved !== 0;
  } else if (state.stage === "cleaning_review") {
    if (!state.review_completed) {
      $("actionsPanel").classList.remove("hidden");
      const families = await api(`/api/sessions/${currentId}/families`);
      $("familiesList").innerHTML = families.map(
        family => familyHtml(family, state.narrator_enabled)
      ).join("");
      $("reviewComplete").disabled = state.action_unresolved !== 0;
    } else {
      $("applyPanel").classList.remove("hidden");
    }
  } else if (state.stage === "verified") {
    $("donePanel").classList.remove("hidden");
    $("doneText").textContent = "The cleaned copy passed source, audit, replay, and undo verification.";
    $("downloadLink").classList.remove("hidden");
    $("downloadLink").href = `/api/sessions/${currentId}/download`;
    const report = await api(`/api/sessions/${currentId}/verification`);
    const m = report.metrics;
    $("verificationMetrics").innerHTML = `
      <div class="metric"><span>Executed changes</span><b>${m.execution_count}</b></div>
      <div class="metric"><span>Cell edits</span><b>${m.cell_change_events}</b></div>
      <div class="metric"><span>Cells different from source</span><b>${m.net_different_cells ?? "—"}</b></div>
      <div class="metric"><span>Cells touched more than once</span><b>${m.repeated_touch_events}</b></div>`;
  } else if (["no_changes","no_approved_changes","no_executable_changes"].includes(state.stage)) {
    setStatus("Session complete: " + state.stage.replaceAll("_"," "));
    $("donePanel").classList.remove("hidden");
    $("doneText").textContent = "No cleaned file was written because there were no executable approved changes.";
    $("downloadLink").classList.add("hidden");
  } else if (state.stage === "verification_failed") {
    setStatus("Verification failed. The source file remains unchanged.", true);
  }

  wireButtons();
}
function wireButtons() {
  document.querySelectorAll(".decision").forEach(button => {
    button.onclick = async () => {
      try {
        button.disabled = true;
        const kind = button.dataset.kind;
        const id = button.dataset.id;
        const decision = button.dataset.decision;
        if (
          kind === "family"
          && decision === "approved"
          && !window.confirm(
            "Approve every displayed change in this family? Review the examples first."
          )
        ) {
          button.disabled = false;
          return;
        }
        const path = kind === "schema"
          ? `schema/${id}`
          : kind === "family"
            ? `families/${id}`
            : `actions/${id}`;
        await api(
          `/api/sessions/${currentId}/${path}`,
          {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({decision})}
        );
        await refresh();
      } catch (err) { setStatus(err.message, true); }
    };
  });
  document.querySelectorAll(".narrate").forEach(button => {
    button.onclick = async () => {
      const actionId = button.dataset.action;
      const panel = $("narration-"+actionId);
      try {
        button.disabled = true;
        button.textContent = "Reading the field…";
        const narration = await api(
          `/api/sessions/${currentId}/actions/${actionId}/narrate`,
          {method:"POST"}
        );
        panel.innerHTML = `
          <strong>${escapeText(narration.title)}</strong>
          <p>${escapeText(narration.explanation)}</p>
          <p><strong>Dataset context:</strong> ${escapeText(narration.data_context)}</p>
          <p><strong>Human check:</strong> ${escapeText(narration.caution)}</p>`;
        panel.classList.remove("hidden");
        button.textContent = "Local AI explanation loaded";
      } catch (err) {
        panel.textContent = err.message;
        panel.classList.remove("hidden");
        button.disabled = false;
        button.textContent = "Try local AI explanation again";
      }
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
