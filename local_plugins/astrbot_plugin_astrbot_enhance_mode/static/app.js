(() => {
  const state = {
    token: localStorage.getItem("enhance_rag_token") || "",
    page: 1,
    pageSize: 20,
    total: 0,
    hasMore: false,
    busyCount: 0,
    filters: {
      keyword: "",
      groupScope: "",
      roleId: "",
    },
  };

  const dom = {
    loginView: document.getElementById("login-view"),
    dashboardView: document.getElementById("dashboard-view"),
    loginForm: document.getElementById("login-form"),
    loginError: document.getElementById("login-error"),
    password: document.getElementById("password"),
    logoutBtn: document.getElementById("logout-btn"),
    cleanupBtn: document.getElementById("cleanup-btn"),
    refreshBtn: document.getElementById("refresh-btn"),
    searchBtn: document.getElementById("search-btn"),
    resetBtn: document.getElementById("reset-btn"),
    keywordInput: document.getElementById("keyword-input"),
    scopeInput: document.getElementById("scope-input"),
    roleInput: document.getElementById("role-input"),
    body: document.getElementById("memory-body"),
    prevBtn: document.getElementById("prev-btn"),
    nextBtn: document.getElementById("next-btn"),
    pageInfo: document.getElementById("page-info"),
    pageSize: document.getElementById("page-size"),
    drawer: document.getElementById("detail-drawer"),
    drawerClose: document.getElementById("drawer-close"),
    drawerContent: document.getElementById("drawer-content"),
    toast: document.getElementById("toast"),
    statTotal: document.getElementById("stat-total"),
    statScopes: document.getElementById("stat-scopes"),
    statGroups: document.getElementById("stat-groups"),
    statRoles: document.getElementById("stat-roles"),
    timezoneTag: document.getElementById("timezone-tag"),
    timeColumnTitle: document.getElementById("time-column-title"),
  };

  function init() {
    dom.loginForm.addEventListener("submit", (event) => {
      void onLogin(event);
    });
    dom.logoutBtn.addEventListener("click", () => {
      void logout();
    });
    dom.cleanupBtn.addEventListener("click", () => {
      void onCleanup();
    });
    dom.refreshBtn.addEventListener("click", () => {
      void fetchAll();
    });
    dom.searchBtn.addEventListener("click", () => {
      void onSearch();
    });
    dom.resetBtn.addEventListener("click", () => {
      void onReset();
    });
    dom.prevBtn.addEventListener("click", () => {
      if (state.page > 1) {
        state.page -= 1;
        void fetchMemories();
      }
    });
    dom.nextBtn.addEventListener("click", () => {
      if (state.hasMore) {
        state.page += 1;
        void fetchMemories();
      }
    });
    dom.pageSize.addEventListener("change", () => {
      state.pageSize = Number(dom.pageSize.value || 20);
      state.page = 1;
      void fetchMemories();
    });
    dom.drawerClose.addEventListener("click", () => {
      dom.drawer.classList.remove("open");
    });

    [dom.keywordInput, dom.scopeInput, dom.roleInput].forEach((input) => {
      input.addEventListener("keydown", (event) => {
        if (event.key !== "Enter") return;
        event.preventDefault();
        void onSearch();
      });
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        dom.drawer.classList.remove("open");
      }
    });

    updateControlState();

    if (state.token) {
      switchView("dashboard");
      fetchAll().catch((error) => {
        console.error(error);
        void logout();
      });
    } else {
      switchView("login");
    }
  }

  function switchView(target) {
    if (target === "dashboard") {
      dom.loginView.classList.remove("active");
      dom.dashboardView.classList.add("active");
      return;
    }
    dom.dashboardView.classList.remove("active");
    dom.loginView.classList.add("active");
    dom.password.focus();
  }

  function startBusy() {
    state.busyCount += 1;
    document.body.classList.add("is-busy");
    updateControlState();
  }

  function endBusy() {
    state.busyCount = Math.max(0, state.busyCount - 1);
    if (state.busyCount === 0) {
      document.body.classList.remove("is-busy");
    }
    updateControlState();
  }

  async function runWithBusy(task) {
    startBusy();
    try {
      return await task();
    } finally {
      endBusy();
    }
  }

  function updateControlState() {
    const isBusy = state.busyCount > 0;
    dom.searchBtn.disabled = isBusy;
    dom.cleanupBtn.disabled = isBusy;
    dom.refreshBtn.disabled = isBusy;
    dom.resetBtn.disabled = isBusy;
    dom.pageSize.disabled = isBusy;
    dom.prevBtn.disabled = isBusy || state.page <= 1;
    dom.nextBtn.disabled = isBusy || !state.hasMore;
  }

  function showToast(message, isError = false) {
    dom.toast.textContent = message;
    dom.toast.style.background = isError ? "#991b1b" : "#132d32";
    dom.toast.classList.add("show");
    setTimeout(() => dom.toast.classList.remove("show"), 2200);
  }

  async function api(path, options = {}) {
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };
    if (!options.skipAuth && state.token) {
      headers.Authorization = `Bearer ${state.token}`;
    }
    const response = await fetch(path, {
      method: options.method || "GET",
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async function onLogin(event) {
    event.preventDefault();
    dom.loginError.textContent = "";

    try {
      await runWithBusy(async () => {
        const payload = await api("/api/login", {
          method: "POST",
          body: { password: dom.password.value.trim() },
          skipAuth: true,
        });
        state.token = payload.token;
        localStorage.setItem("enhance_rag_token", state.token);
        dom.password.value = "";
        switchView("dashboard");
        showToast("Login success.");
        await fetchAll(false);
      });
    } catch (error) {
      dom.loginError.textContent = error.message || "Login failed";
    }
  }

  async function logout() {
    try {
      if (state.token) {
        await api("/api/logout", { method: "POST" });
      }
    } catch (error) {
      console.warn(error);
    } finally {
      state.token = "";
      localStorage.removeItem("enhance_rag_token");
      dom.drawer.classList.remove("open");
      switchView("login");
    }
  }

  function formatNumber(value) {
    return Number(value || 0).toLocaleString("en-US");
  }

  function renderStats(stats) {
    dom.statTotal.textContent = formatNumber(stats.total_memories);
    dom.statScopes.textContent = formatNumber(stats.group_scope_count);
    dom.statGroups.textContent = formatNumber(stats.group_id_count);
    dom.statRoles.textContent = formatNumber(stats.role_count);
    const timezoneName = String(stats.display_timezone || "").trim();
    dom.timezoneTag.textContent = timezoneName || "-";
    dom.timeColumnTitle.textContent = timezoneName ? `Time (${timezoneName})` : "Time";
  }

  function clipText(text, max = 90) {
    const source = String(text || "");
    return source.length > max ? `${source.slice(0, max)}...` : source;
  }

  function renderRows(items) {
    if (!Array.isArray(items) || items.length === 0) {
      dom.body.innerHTML =
        '<tr><td colspan="6" class="center muted">No records found for current filters.</td></tr>';
      return;
    }

    dom.body.innerHTML = "";
    for (const item of items) {
      const tr = document.createElement("tr");
      const rolePills = (item.related_role_ids || [])
        .map((role) => `<span class="pill mono">${escapeHtml(role)}</span>`)
        .join("");
      tr.innerHTML = `
        <td class="mono">${item.memory_id}</td>
        <td>${escapeHtml(clipText(item.content, 120))}</td>
        <td class="mono">${escapeHtml(item.group_scope || "-")}</td>
        <td>${rolePills || "<span class='muted'>-</span>"}</td>
        <td class="mono">${escapeHtml(item.memory_time_iso || "-")}</td>
        <td>
          <div class="action-group">
            <button class="ghost" data-action="detail" data-id="${item.memory_id}">Detail</button>
            <button class="danger" data-action="delete" data-id="${item.memory_id}">Delete</button>
          </div>
        </td>
      `;
      dom.body.appendChild(tr);
    }

    dom.body.querySelectorAll("button[data-action='detail']").forEach((btn) => {
      btn.addEventListener("click", () => {
        void openDetail(Number(btn.dataset.id));
      });
    });
    dom.body.querySelectorAll("button[data-action='delete']").forEach((btn) => {
      btn.addEventListener("click", () => {
        void deleteMemory(Number(btn.dataset.id));
      });
    });
  }

  async function fetchStats() {
    const payload = await api("/api/stats");
    renderStats(payload.data || {});
  }

  async function fetchMemories(withBusy = true) {
    const run = async () => {
      const params = new URLSearchParams();
      params.set("page", String(state.page));
      params.set("page_size", String(state.pageSize));
      if (state.filters.keyword) params.set("keyword", state.filters.keyword);
      if (state.filters.groupScope) params.set("group_scope", state.filters.groupScope);
      if (state.filters.roleId) params.set("role_id", state.filters.roleId);

      const payload = await api(`/api/memories?${params.toString()}`);
      const data = payload.data || {};
      state.total = Number(data.total || 0);
      state.hasMore = Boolean(data.has_more);
      renderRows(data.items || []);
      const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
      dom.pageInfo.textContent = `Page ${state.page}/${totalPages} · Total ${formatNumber(state.total)}`;
      updateControlState();
    };

    if (withBusy) {
      return runWithBusy(run);
    }
    return run();
  }

  async function fetchAll(withBusy = true) {
    const run = async () => {
      try {
        await Promise.all([fetchStats(), fetchMemories(false)]);
      } catch (error) {
        showToast(error.message || "Load failed", true);
        if (String(error.message || "").toLowerCase().includes("token")) {
          void logout();
        }
      }
    };

    if (withBusy) {
      return runWithBusy(run);
    }
    return run();
  }

  async function onSearch() {
    state.filters.keyword = dom.keywordInput.value.trim();
    state.filters.groupScope = dom.scopeInput.value.trim();
    state.filters.roleId = dom.roleInput.value.trim();
    state.page = 1;
    try {
      await fetchMemories();
    } catch (error) {
      showToast(error.message || "Search failed", true);
    }
  }

  async function onReset() {
    dom.keywordInput.value = "";
    dom.scopeInput.value = "";
    dom.roleInput.value = "";
    state.filters.keyword = "";
    state.filters.groupScope = "";
    state.filters.roleId = "";
    state.page = 1;

    try {
      await fetchMemories();
    } catch (error) {
      showToast(error.message || "Reset failed", true);
    }
  }

  async function openDetail(memoryId) {
    try {
      await runWithBusy(async () => {
        const payload = await api(`/api/memories/${memoryId}`);
        dom.drawerContent.textContent = JSON.stringify(payload.data || {}, null, 2);
        dom.drawer.classList.add("open");
      });
    } catch (error) {
      showToast(error.message || "Detail load failed", true);
    }
  }

  async function deleteMemory(memoryId) {
    const confirmed = window.confirm(`Delete memory ${memoryId}?`);
    if (!confirmed) return;

    try {
      await runWithBusy(async () => {
        await api(`/api/memories/${memoryId}`, { method: "DELETE" });
        showToast(`Deleted memory ${memoryId}`);
        await fetchAll(false);
      });
    } catch (error) {
      showToast(error.message || "Delete failed", true);
    }
  }

  async function onCleanup() {
    const confirmed = window.confirm(
      "Run cleanup to normalize legacy records and refresh timestamp metadata?"
    );
    if (!confirmed) return;

    try {
      await runWithBusy(async () => {
        const payload = await api("/api/cleanup", { method: "POST" });
        const data = payload.data || {};
        const scanned = Number(data.scanned || 0);
        const updated = Number(data.updated || 0);
        showToast(`Cleanup done: updated ${updated}/${scanned} records.`);
        await fetchAll(false);
      });
    } catch (error) {
      showToast(error.message || "Cleanup failed", true);
    }
  }

  function escapeHtml(text) {
    return String(text || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  init();
})();
