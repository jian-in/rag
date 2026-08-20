const state = {
  query: "",
  scopes: ["vault", "profile", "episodes"],
  context: "",
  reviewStatus: "pending",
};

const scopeMeta = {
  vault: { short: "V", label: "知识穹顶" },
  profile: { short: "P", label: "身份底片" },
  episodes: { short: "E", label: "经历潮汐" },
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function requestJSON(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `请求失败 (${response.status})`);
  }
  return response.json();
}

function selectedScopes() {
  return $$('input[name="scope"]:checked').map((input) => input.value);
}

function setBusy(busy) {
  $(".shell").classList.toggle("is-loading", busy);
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.remove("show"), 2400);
}

function formatTime(value) {
  if (!value) return "等待首次同步";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : `同步于 ${date.toLocaleString("zh-CN", { hour12: false })}`;
}

async function refreshStatus() {
  const live = $("#live-state");
  try {
    const data = await requestJSON("/api/status");
    for (const scope of Object.keys(scopeMeta)) {
      const details = data.scopes[scope];
      $(`#count-${scope}`).textContent = details.chunk_count.toLocaleString("zh-CN");
      $(`#sync-${scope}`).textContent = formatTime(details.last_sync);
    }
    const modes = [];
    if (data.embedding.configured) modes.push(`向量 ${data.embedding.model}`);
    if (data.reranker.configured) modes.push(`重排 ${data.reranker.model}`);
    $("#provider-label").textContent = modes.length ? modes.join(" · ") : "本地词法检索模式";
    const pending = data.review?.pending || 0;
    $("#pending-count").textContent = `${pending} 条待审核`;
    live.className = "live-state ready";
    $("#live-label").textContent = data.ready ? "记忆入口在线" : "入口在线 · 等待同步";
  } catch (error) {
    live.className = "live-state error";
    $("#live-label").textContent = "记忆入口连接失败";
    throw error;
  }
}

function candidateField(tag, className, value, disabled = false) {
  const field = document.createElement(tag);
  field.className = className;
  field.value = value || "";
  field.disabled = disabled;
  return field;
}

function renderCandidates(candidates) {
  const container = $("#review-list");
  container.replaceChildren();
  if (!candidates.length) {
    const empty = document.createElement("div");
    empty.className = "review-empty";
    empty.textContent = state.reviewStatus === "pending"
      ? "现在没有待审核候选。Agent 提交的新记忆会先停在这里。"
      : "这个状态下暂时没有候选记录。";
    container.append(empty);
    return;
  }

  candidates.forEach((candidate) => {
    const editable = candidate.status === "pending";
    const card = document.createElement("article");
    card.className = `candidate-card status-${candidate.status}`;

    const meta = document.createElement("div");
    meta.className = "candidate-meta";
    const id = document.createElement("span");
    id.textContent = `#${candidate.id.slice(0, 8)}`;
    const source = document.createElement("span");
    source.textContent = candidate.source || "agent";
    const time = document.createElement("span");
    time.textContent = new Date(candidate.created_at).toLocaleString("zh-CN", { hour12: false });
    meta.append(id, source, time);

    const scope = candidateField("select", "candidate-scope", candidate.scope, !editable);
    Object.entries(scopeMeta).forEach(([value, details]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = details.label;
      option.selected = value === candidate.scope;
      scope.append(option);
    });
    const content = candidateField("textarea", "candidate-content", candidate.content, !editable);
    content.rows = 3;
    const reason = document.createElement("p");
    reason.className = "candidate-reason";
    reason.textContent = candidate.reason ? `提出理由：${candidate.reason}` : "未附提出理由";

    const controls = document.createElement("div");
    controls.className = "candidate-controls";
    const note = candidateField("input", "candidate-note", "", false);
    note.placeholder = editable ? "审核备注（可选）" : "忘记原因（可选）";

    if (editable) {
      controls.append(note);
      const save = document.createElement("button");
      save.type = "button";
      save.className = "quiet-button";
      save.textContent = "保存纠正";
      save.addEventListener("click", () => reviewAction(candidate, "update", scope, content, note));
      const reject = document.createElement("button");
      reject.type = "button";
      reject.className = "danger-button";
      reject.textContent = "拒绝";
      reject.addEventListener("click", () => reviewAction(candidate, "reject", scope, content, note));
      const approve = document.createElement("button");
      approve.type = "button";
      approve.className = "approve-button";
      approve.textContent = "纠正并批准";
      approve.addEventListener("click", () => reviewAction(candidate, "approve", scope, content, note));
      controls.append(save, reject, approve);
    } else if (candidate.status === "approved") {
      controls.append(note);
      const forget = document.createElement("button");
      forget.type = "button";
      forget.className = "danger-button";
      forget.textContent = "精准忘记";
      forget.addEventListener("click", () => reviewAction(candidate, "forget", scope, content, note));
      controls.append(forget);
    }

    card.append(meta, scope, content, reason);
    if (editable || candidate.status === "approved") card.append(controls);
    container.append(card);
  });
}

async function refreshCandidates(status = state.reviewStatus) {
  state.reviewStatus = status;
  $$(".review-filter").forEach((button) => {
    button.classList.toggle("active", button.dataset.reviewStatus === status);
  });
  try {
    const data = await requestJSON(`/api/candidates?status=${encodeURIComponent(status)}`);
    renderCandidates(data.candidates);
  } catch (error) {
    toast(error.message);
  }
}

async function reviewAction(candidate, action, scope, content, note) {
  if (action === "forget" && !window.confirm("只移除这条由平台托管的记忆块，并立即刷新索引。确定继续吗？")) {
    return;
  }
  const isUpdate = action === "update";
  const body = { action, note: note.value.trim() };
  if (isUpdate || action === "approve") {
    body.scope = scope.value;
    body.content = content.value.trim();
  }
  setBusy(true);
  try {
    const path = isUpdate
      ? `/api/candidates/${candidate.id}`
      : `/api/candidates/${candidate.id}/decision`;
    await requestJSON(path, {
      method: isUpdate ? "PATCH" : "POST",
      body: JSON.stringify(body),
    });
    await Promise.all([refreshStatus(), refreshCandidates()]);
    const labels = { update: "候选已纠正", approve: "候选已批准并建入索引", reject: "候选已拒绝", forget: "托管记忆已精准移除" };
    toast(labels[action]);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

function renderResults(results) {
  const container = $("#results");
  container.replaceChildren();
  $("#result-count").textContent = results.length ? `${results.length} 条相关片段` : "没有命中";
  if (!results.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.innerHTML = '<span class="empty-glyph" aria-hidden="true">∅</span><p>这次没有找到足够相关的记忆。可以换个说法，或先同步记忆舱。</p>';
    container.append(empty);
    return;
  }

  results.forEach((item, index) => {
    const article = document.createElement("article");
    article.className = "result-item";
    article.style.animationDelay = `${Math.min(index * 45, 260)}ms`;

    const stamp = document.createElement("span");
    stamp.className = `scope-stamp ${item.scope}`;
    stamp.textContent = scopeMeta[item.scope]?.short || "M";
    stamp.title = scopeMeta[item.scope]?.label || item.scope;

    const body = document.createElement("div");
    const title = document.createElement("h3");
    title.textContent = item.heading || item.title;
    const preview = document.createElement("p");
    preview.textContent = item.preview;
    const locator = document.createElement("span");
    locator.className = "result-locator";
    locator.textContent = `${scopeMeta[item.scope]?.label || item.scope} · ${item.locator}`;
    body.append(title, preview, locator);

    const action = document.createElement(item.obsidian_uri ? "a" : "span");
    action.className = "open-note";
    action.textContent = item.obsidian_uri ? "↗" : "·";
    if (item.obsidian_uri) {
      action.href = item.obsidian_uri;
      action.title = "在 Obsidian 中打开";
    }
    article.append(stamp, body, action);
    container.append(article);
  });
}

async function runSearch(event) {
  event?.preventDefault();
  const query = $("#query").value.trim();
  const scopes = selectedScopes();
  if (!query) return;
  if (!scopes.length) return toast("至少选择一座记忆舱");
  state.query = query;
  state.scopes = scopes;
  setBusy(true);
  try {
    const data = await requestJSON("/api/search", {
      method: "POST",
      body: JSON.stringify({ query, scopes, top_k: 10 }),
    });
    renderResults(data.results);
    $("#context-button").disabled = !data.results.length;
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function buildContext() {
  if (!state.query) return;
  setBusy(true);
  try {
    const data = await requestJSON("/api/context", {
      method: "POST",
      body: JSON.stringify({
        query: state.query,
        scopes: state.scopes,
        top_k: 10,
        char_budget: 6000,
      }),
    });
    state.context = data.context;
    $("#context-output").textContent = data.context || "没有足够的记忆可生成上下文。";
    $("#copy-button").disabled = !data.context;
    toast(`已整理 ${data.hits.length} 条记忆`);
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function syncMemories() {
  const scopes = selectedScopes();
  if (!scopes.length) return toast("至少选择一座记忆舱");
  setBusy(true);
  try {
    await requestJSON("/api/sync", {
      method: "POST",
      body: JSON.stringify({ scopes }),
    });
    await refreshStatus();
    toast("记忆舱同步完成");
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(false);
  }
}

async function copyContext() {
  if (!state.context) return;
  await navigator.clipboard.writeText(state.context);
  toast("Agent 上下文已复制");
}

$("#search-form").addEventListener("submit", runSearch);
$("#context-button").addEventListener("click", buildContext);
$("#sync-button").addEventListener("click", syncMemories);
$("#copy-button").addEventListener("click", copyContext);
$$(".review-filter").forEach((button) => {
  button.addEventListener("click", () => refreshCandidates(button.dataset.reviewStatus));
});
$("#review-refresh").addEventListener("click", () => refreshCandidates());
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && document.activeElement !== $("#query")) {
    event.preventDefault();
    $("#query").focus();
  }
});

Promise.all([refreshStatus(), refreshCandidates()]).catch((error) => toast(error.message));
