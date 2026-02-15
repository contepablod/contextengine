    const $ = (id) => document.getElementById(id);
    const DOC_STORE_KEY = "contextengine_docs";
    const CONTEXT_STORE_KEY = "contextengine_context_entries";
    const CONTEXT_FILTER_KEY = "contextengine_context_filters";
    const BLUEPRINT_STORE_KEY = "contextengine_blueprints";
    const LAST_DOC_KEY = "contextengine_last_doc";

    let lastAnswerText = "";
    let lastEvidence = [];
    let lastCitations = [];
    let lastDocId = null;
    let editingContextId = null;
    let editingContextType = null;
    let editingBlueprintId = null;
    let uploadStageTimers = [];
    let selectedFile = null;
    let isAuthenticated = false;

    function baseUrl() {
      const origin = window.location.origin || "";
      const host = window.location.hostname || "";
      const port = window.location.port || "";
      if ((host === "localhost" || host === "127.0.0.1") && port && port !== "8000") {
        return `${window.location.protocol}//${host}:8000`;
      }
      return origin.replace(/\/+$/, "");
    }

    function setupSectionNav() {
      const navLinks = Array.from(document.querySelectorAll(".nav-panel a[data-section]"));
      if (!navLinks.length) return;
      const sections = navLinks
        .map((link) => {
          const id = link.getAttribute("data-section");
          const section = id ? document.getElementById(id) : null;
          return section ? { id, section, link } : null;
        })
        .filter(Boolean);
      if (!sections.length) return;

      const activate = (id) => {
        navLinks.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("data-section") === id);
        });
      };

      const updateActive = () => {
        const scrollPos = window.scrollY + 140;
        let current = sections[0].id;
        sections.forEach((item) => {
          if (item.section.offsetTop <= scrollPos) {
            current = item.id;
          }
        });
        activate(current);
      };

      window.addEventListener("scroll", () => {
        window.requestAnimationFrame(updateActive);
      });
      window.addEventListener("resize", () => {
        window.requestAnimationFrame(updateActive);
      });
      updateActive();
    }

    async function apiFetch(url, options = {}) {
      const res = await fetch(url, { ...options, credentials: "include" });
      if (res.status === 401) {
        setAuthState(false);
      }
      return res;
    }

    function setAuthState(authenticated) {
      isAuthenticated = authenticated;
      const loginScreen = $("loginScreen");
      const appRoot = $("appRoot");
      if (loginScreen) {
        loginScreen.style.display = authenticated ? "none" : "flex";
      }
      if (appRoot) {
        appRoot.style.display = authenticated ? "block" : "none";
      }
      if (authenticated) {
        loadModels();
      }
    }

    async function checkAuth() {
      try {
        const res = await apiFetch(`${baseUrl()}/auth/me`);
        const data = await res.json();
        if (res.ok && data.authenticated) {
          setAuthState(true);
          showMessage("loginMessage", "", "");
          return;
        }
      } catch (err) {
        // fall through to unauthenticated
      }
      setAuthState(false);
    }

    async function login() {
      const username = $("loginUsername").value.trim();
      const password = $("loginPassword").value;
      if (!username || !password) {
        showMessage("loginMessage", "❌ Enter username and password", "error");
        return;
      }
      const btn = $("loginBtn");
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Signing in...";
      try {
        const res = await apiFetch(`${baseUrl()}/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Login failed");
        }
        $("loginPassword").value = "";
        showMessage("loginMessage", "✅ Logged in", "success");
        setAuthState(true);
      } catch (err) {
        showMessage("loginMessage", `❌ ${err.message || String(err)}`, "error");
        setAuthState(false);
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    async function logout() {
      const btn = $("logoutBtn");
      if (btn) {
        btn.disabled = true;
      }
      try {
        await apiFetch(`${baseUrl()}/auth/logout`, { method: "POST" });
      } catch (err) {
        // ignore
      } finally {
        if (btn) btn.disabled = false;
      }
      setAuthState(false);
    }

    function isPdfFile(file) {
      return file && file.name && file.name.toLowerCase().endsWith(".pdf");
    }

    function setSelectedFile(file) {
      selectedFile = file || null;
      const meta = $("dropZoneMeta");
      if (!meta) return;
      if (selectedFile) {
        meta.textContent = `Selected: ${selectedFile.name}`;
      } else {
        meta.textContent = "or click to browse";
      }
    }

    function setDocId(id) {
      const display = id || "Not uploaded";
      const badge = $("docId");
      badge.textContent = display;
      badge.className = id ? "badge badge-success" : "badge badge-primary";
      
      const isEnabled = !!id;
      $("askBtn").disabled = !isEnabled;
      $("summaryBtn").disabled = !isEnabled;

      lastDocId = id || null;
      if (id) {
        localStorage.setItem(LAST_DOC_KEY, id);
      } else {
        localStorage.removeItem(LAST_DOC_KEY);
      }

      const docs = loadDocs();
      renderDocSelect(docs, id);
      renderDocList(docs, id);
      updateDocUI(id);
      const activeDoc = id ? getDocById(id) : null;
      renderUploadResult(activeDoc);
    }

    function loadDocs() {
      try {
        return JSON.parse(localStorage.getItem(DOC_STORE_KEY) || "[]");
      } catch {
        return [];
      }
    }

    function saveDocs(docs) {
      localStorage.setItem(DOC_STORE_KEY, JSON.stringify(docs || []));
    }

    function upsertDoc(doc) {
      if (!doc || !doc.doc_id) return;
      const docs = loadDocs();
      const idx = docs.findIndex((d) => d.doc_id === doc.doc_id);
      const entry = {
        doc_id: doc.doc_id,
        filename: doc.filename || "unknown",
        uploaded_at: new Date().toISOString(),
        pages: doc.pages,
        chunks: doc.chunks,
        file_size_bytes: doc.file_size_bytes,
        namespace: doc.namespace,
        doc_type: doc.doc_type,
        extraction_method: doc.extraction_method,
        chunk_chars: doc.chunk_chars,
        overlap_chars: doc.overlap_chars,
        sections: doc.sections || [],
        display_name: doc.display_name,
      };
      if (idx >= 0) {
        docs[idx] = { ...docs[idx], ...entry };
      } else {
        docs.unshift(entry);
      }
      saveDocs(docs);
      renderDocSelect(docs, doc.doc_id);
      renderDocList(docs, doc.doc_id);
    }

    function renderDocSelect(docs, selectedId) {
      const select = $("docSelect");
      select.innerHTML = "";
      if (!docs.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No documents yet";
        select.appendChild(opt);
        return;
      }

      docs.forEach((d) => {
        const opt = document.createElement("option");
        opt.value = d.doc_id;
        const name = d.display_name || d.filename || d.doc_id;
        const label = `${name} (${d.doc_id.slice(0, 8)})`;
        opt.textContent = label;
        if (selectedId && d.doc_id === selectedId) {
          opt.selected = true;
        }
        select.appendChild(opt);
      });
    }

    function formatBytes(bytes) {
      if (!bytes && bytes !== 0) return "";
      const units = ["B", "KB", "MB", "GB"];
      let size = bytes;
      let idx = 0;
      while (size >= 1024 && idx < units.length - 1) {
        size /= 1024;
        idx += 1;
      }
      return `${size.toFixed(size >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
    }

    function getDocById(id) {
      const docs = loadDocs();
      return docs.find((d) => d.doc_id === id) || null;
    }

    function renderDocList(docs, activeId) {
      const list = $("docList");
      list.innerHTML = "";
      if (!docs.length) {
        list.innerHTML = "<div class='muted'>No documents yet</div>";
        return;
      }

      docs.forEach((doc) => {
        const row = document.createElement("div");
        row.className = "doc-row";

        const left = document.createElement("div");
        left.style.flex = "1";

        const title = document.createElement("div");
        title.className = "doc-title";
        title.textContent = doc.display_name || doc.filename || doc.doc_id;

        const meta = document.createElement("div");
        meta.className = "doc-meta";
        const metaParts = [];
        if (doc.pages) metaParts.push(`${doc.pages} pages`);
        if (doc.chunks) metaParts.push(`${doc.chunks} chunks`);
        if (doc.file_size_bytes) metaParts.push(formatBytes(doc.file_size_bytes));
        if (doc.doc_type) metaParts.push(`type: ${doc.doc_type}`);
        if (doc.extraction_method) metaParts.push(`extract: ${doc.extraction_method}`);
        if (doc.chunk_chars) metaParts.push(`chunk: ${doc.chunk_chars}/${doc.overlap_chars || 0}`);
        if (doc.uploaded_at) metaParts.push(new Date(doc.uploaded_at).toLocaleString());
        meta.textContent = metaParts.join(" • ") || "No metadata";

        left.appendChild(title);
        left.appendChild(meta);

        const actions = document.createElement("div");
        actions.className = "doc-actions";

        const useBtn = document.createElement("button");
        useBtn.className = "btn-secondary tiny-btn";
        useBtn.textContent = activeId === doc.doc_id ? "Active" : "Use";
        useBtn.disabled = activeId === doc.doc_id;
        useBtn.addEventListener("click", () => setDocId(doc.doc_id));

        const renameBtn = document.createElement("button");
        renameBtn.className = "btn-secondary tiny-btn";
        renameBtn.textContent = "Rename";
        renameBtn.addEventListener("click", () => {
          const name = prompt("New display name", doc.display_name || doc.filename || "");
          if (!name) return;
          doc.display_name = name.trim();
          saveDocs(docs);
          renderDocSelect(docs, activeId);
          renderDocList(docs, activeId);
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-secondary tiny-btn";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", async () => {
          if (!confirm("Delete this document from Pinecone?")) return;
          try {
            await deleteDocFromStore(doc.doc_id);
            const remaining = loadDocs().filter((d) => d.doc_id !== doc.doc_id);
            saveDocs(remaining);
            if (activeId === doc.doc_id) {
              setDocId(null);
            } else {
              renderDocSelect(remaining, activeId);
              renderDocList(remaining, activeId);
            }
          } catch (err) {
            showMessage("uploadMessage", `❌ ${err.message || String(err)}`, "error");
          }
        });

        actions.appendChild(useBtn);
        actions.appendChild(renameBtn);
        actions.appendChild(deleteBtn);

        row.appendChild(left);
        row.appendChild(actions);
        list.appendChild(row);
      });
    }

    function updateDocUI(docId) {
      const doc = docId ? getDocById(docId) : null;
      renderSectionOptions(doc);
      renderSuggestions(doc);
    }

    function renderSectionOptions(doc) {
      const select = $("sectionFilter");
      if (!select) return;
      select.innerHTML = "";
      const optAll = document.createElement("option");
      optAll.value = "";
      optAll.textContent = "All sections";
      select.appendChild(optAll);

      const sections = (doc && doc.sections) ? doc.sections : [];
      sections.forEach((section) => {
        const opt = document.createElement("option");
        opt.value = section;
        opt.textContent = section;
        select.appendChild(opt);
      });
    }

    function renderSuggestions(doc) {
      const root = $("suggestions");
      root.innerHTML = "";
      const docType = doc?.doc_type || "generic";
      const suggestions = {
        scholarly: [
          "What are the main contributions?",
          "Summarize the methodology.",
          "What are the limitations?",
        ],
        financial: [
          "What are the key risks?",
          "Summarize major changes in the balance sheet.",
          "What are the critical audit matters?",
        ],
        legal: [
          "Summarize key obligations.",
          "What are the termination conditions?",
          "Highlight compliance risks.",
        ],
        scan: [
          "Summarize the document.",
          "List key findings.",
          "What are the main conclusions?",
        ],
        generic: [
          "Summarize the document.",
          "What are the main conclusions?",
          "List key findings with citations.",
        ],
      };

      (suggestions[docType] || suggestions.generic).forEach((text) => {
        const btn = document.createElement("button");
        btn.className = "btn-secondary tiny-btn";
        btn.textContent = text;
        btn.addEventListener("click", () => {
          $("question").value = text;
          $("question").focus();
        });
        root.appendChild(btn);
      });
    }

    function persistSectionsFromEvidence(docId, evidence) {
      if (!docId || !evidence || !evidence.length) return;
      const docs = loadDocs();
      const idx = docs.findIndex((d) => d.doc_id === docId);
      if (idx === -1) return;
      const sections = new Set(docs[idx].sections || []);
      evidence.forEach((e) => {
        if (e.section) sections.add(e.section);
      });
      docs[idx].sections = Array.from(sections).slice(0, 50);
      saveDocs(docs);
      if (docId === lastDocId) {
        renderSectionOptions(docs[idx]);
      }
    }

    function showMessage(elementId, message, type) {
      const elem = $(elementId);
      if (!message) {
        elem.style.display = "none";
        return;
      }
      
      const className = `${type}-message`;
      elem.className = className;
      elem.innerHTML = message;
      elem.style.display = "block";
    }

    function updateUploadProgress(state) {
      const steps = ["upload", "extract", "embed", "ready"];
      steps.forEach((step) => {
        const el = document.querySelector(`.progress-step[data-step="${step}"]`);
        if (!el) return;
        el.classList.remove("active", "done", "error");
        if (state === "error") {
          if (step === "upload") el.classList.add("error");
          return;
        }
        if (state === step) {
          el.classList.add("active");
          return;
        }
        const stateIndex = steps.indexOf(state);
        const stepIndex = steps.indexOf(step);
        if (stateIndex > stepIndex) {
          el.classList.add("done");
        }
      });
      setUploadBarState(state);
    }

    function clearUploadStageTimers() {
      uploadStageTimers.forEach((timer) => clearTimeout(timer));
      uploadStageTimers = [];
    }

    function setUploadBar(value, label) {
      const bar = $("uploadBar");
      const fill = $("uploadBarFill");
      const text = $("uploadBarLabel");
      if (!bar || !fill) return;
      bar.style.display = "block";
      fill.style.width = `${Math.max(0, Math.min(100, value))}%`;
      bar.setAttribute("aria-valuenow", String(value));
      if (text) {
        text.textContent = label || "";
        text.style.display = label ? "block" : "none";
      }
    }

    function setUploadBarState(state) {
      const labels = {
        upload: "Uploading file...",
        extract: "Extracting text...",
        embed: "Embedding + upserting...",
        ready: "Ready",
        error: "Upload failed",
      };
      const targets = {
        upload: 20,
        extract: 55,
        embed: 85,
        ready: 100,
        error: 0,
      };
      if (!state) return;
      setUploadBar(targets[state] ?? 0, labels[state] || "");
      if (state === "ready" || state === "error") {
        clearUploadStageTimers();
        window.setTimeout(() => {
          const bar = $("uploadBar");
          const text = $("uploadBarLabel");
          if (bar) bar.style.display = "none";
          if (text) text.style.display = "none";
        }, 1400);
      }
    }

    function scheduleUploadStages() {
      clearUploadStageTimers();
      uploadStageTimers = [
        setTimeout(() => updateUploadProgress("extract"), 800),
        setTimeout(() => updateUploadProgress("embed"), 1800),
      ];
    }

    function loadContextEntries() {
      try {
        return JSON.parse(localStorage.getItem(CONTEXT_STORE_KEY) || "[]");
      } catch {
        return [];
      }
    }

    function saveContextEntries(entries) {
      localStorage.setItem(CONTEXT_STORE_KEY, JSON.stringify(entries || []));
    }

    function renderContextList(entries) {
      const list = $("contextList");
      if (!list) return;
      list.innerHTML = "";
      if (!entries.length) {
        list.innerHTML = "<div class='muted'>No context entries yet</div>";
        renderContextFilterOptions();
        return;
      }

      entries.forEach((entry) => {
        const row = document.createElement("div");
        row.className = "context-item";

        const left = document.createElement("div");
        left.style.flex = "1";

        const title = document.createElement("div");
        title.className = "doc-title";
        title.textContent = formatContextType(entry.context_type);

        const meta = document.createElement("div");
        meta.className = "context-meta";
        meta.textContent = entry.context_id ? `id: ${entry.context_id.slice(0, 8)}` : "";

        const text = document.createElement("div");
        text.className = "context-text";
        text.textContent = entry.text || "";

        left.appendChild(title);
        left.appendChild(meta);
        left.appendChild(text);

        const actions = document.createElement("div");
        actions.className = "context-actions";

        const editBtn = document.createElement("button");
        editBtn.className = "btn-secondary tiny-btn";
        editBtn.textContent = "Edit";
        editBtn.addEventListener("click", () => {
          editingContextId = entry.context_id;
          editingContextType = entry.context_type || "custom";
          const textElem = $("contextText");
          if (textElem) textElem.value = entry.text || "";
          if (
            editingContextType &&
            !["executive_brief", "technical_analysis", "legal_compliance"].includes(editingContextType)
          ) {
            const customElem = $("customContextType");
            if (customElem) customElem.value = editingContextType;
          }
          showMessage("blueprintMessage", "✏️ Editing context entry", "success");
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-secondary tiny-btn";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", async () => {
          if (!confirm("Delete this context entry?")) return;
          try {
            await deleteContextEntry(entry.context_id);
            const remaining = loadContextEntries().filter((e) => e.context_id !== entry.context_id);
            saveContextEntries(remaining);
            renderContextList(remaining);
          } catch (err) {
            showMessage("blueprintMessage", `❌ ${err.message || String(err)}`, "error");
          }
        });

        actions.appendChild(editBtn);
        actions.appendChild(deleteBtn);

        row.appendChild(left);
        row.appendChild(actions);
        list.appendChild(row);
      });

      renderContextFilterOptions();
    }

    function loadBlueprints() {
      try {
        return JSON.parse(localStorage.getItem(BLUEPRINT_STORE_KEY) || "[]");
      } catch {
        return [];
      }
    }

    function saveBlueprints(entries) {
      localStorage.setItem(BLUEPRINT_STORE_KEY, JSON.stringify(entries || []));
    }

    function formatBlueprintSnippet(blueprint) {
      if (!blueprint) return "";
      const text =
        typeof blueprint === "string"
          ? blueprint
          : JSON.stringify(blueprint, null, 2);
      return truncateText(text, 500);
    }

    function renderBlueprintList(entries) {
      const list = $("blueprintList");
      if (!list) return;
      list.innerHTML = "";
      if (!entries.length) {
        list.innerHTML = "<div class='muted'>No blueprints yet</div>";
        renderContextFilterOptions();
        return;
      }

      entries.forEach((entry) => {
        const row = document.createElement("div");
        row.className = "context-item";

        const left = document.createElement("div");
        left.style.flex = "1";

        const title = document.createElement("div");
        title.className = "doc-title";
        title.textContent = entry.blueprint_id || "Untitled blueprint";

        const meta = document.createElement("div");
        meta.className = "context-meta";
        const metaParts = [];
        if (entry.created_at) metaParts.push(new Date(entry.created_at).toLocaleString());
        meta.textContent = metaParts.join(" • ");

        const desc = document.createElement("div");
        desc.className = "context-text";
        desc.textContent = entry.description || "";

        const code = document.createElement("div");
        code.className = "blueprint-text";
        code.textContent = formatBlueprintSnippet(entry.blueprint);

        left.appendChild(title);
        left.appendChild(meta);
        left.appendChild(desc);
        left.appendChild(code);

        const actions = document.createElement("div");
        actions.className = "context-actions";

        const editBtn = document.createElement("button");
        editBtn.className = "btn-secondary tiny-btn";
        editBtn.textContent = "Edit";
        editBtn.addEventListener("click", () => {
          editingBlueprintId = entry.blueprint_id;
          $("blueprintId").value = entry.blueprint_id || "";
          $("blueprintDesc").value = entry.description || "";
          if (entry.blueprint) {
            $("blueprintJson").value = JSON.stringify(entry.blueprint, null, 2);
          }
          showMessage("blueprintMessage", "✏️ Editing blueprint", "success");
        });

        const deleteBtn = document.createElement("button");
        deleteBtn.className = "btn-secondary tiny-btn";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", async () => {
          if (!confirm("Delete this blueprint?")) return;
          try {
            await deleteContextEntry(entry.blueprint_id);
            const remaining = loadBlueprints().filter(
              (e) => e.blueprint_id !== entry.blueprint_id
            );
            saveBlueprints(remaining);
            renderBlueprintList(remaining);
          } catch (err) {
            showMessage("blueprintMessage", `❌ ${err.message || String(err)}`, "error");
          }
        });

        actions.appendChild(editBtn);
        actions.appendChild(deleteBtn);

        row.appendChild(left);
        row.appendChild(actions);
        list.appendChild(row);
      });

      renderContextFilterOptions();
    }

    function formatContextType(value) {
      const map = {
        executive_brief: "Executive Brief",
        technical_analysis: "Technical Analysis",
        legal_compliance: "Legal/Compliance",
      };
      if (!value) return "Custom";
      return map[value] || value;
    }

    function loadContextFilters() {
      try {
        return JSON.parse(localStorage.getItem(CONTEXT_FILTER_KEY) || "[]");
      } catch {
        return [];
      }
    }

    function saveContextFilters(filters) {
      localStorage.setItem(CONTEXT_FILTER_KEY, JSON.stringify(filters || []));
    }

    function normalizeFilterSelection(raw) {
      if (Array.isArray(raw)) return raw;
      if (raw && typeof raw === "object") {
        const map = {
          executive_brief: "executive_brief",
          technical_analysis: "technical_analysis",
          legal_compliance: "legal_compliance",
          custom: "custom",
        };
        return Object.keys(map).filter((k) => raw[k] !== false).map((k) => map[k]);
      }
      return [];
    }

    function collectContextFilterOptions() {
      const options = [];
      const seen = new Set();

      const blueprintEntries = loadBlueprints();
      blueprintEntries.forEach((entry) => {
        const value = entry.blueprint_id;
        if (!value || seen.has(value)) return;
        seen.add(value);
        options.push({
          value,
          label: value,
          title: entry.description || value,
        });
      });

      const contextEntries = loadContextEntries();
      contextEntries.forEach((entry) => {
        const value = entry.context_type;
        if (!value || seen.has(value)) return;
        seen.add(value);
        options.push({
          value,
          label: value,
          title: entry.text || value,
        });
      });

      return options;
    }

    function renderContextFilterOptions(filtersOverride) {
      const select = $("ctxFilterSelect");
      if (!select) return;
      const filters = normalizeFilterSelection(filtersOverride ?? loadContextFilters());
      const selectedSet = new Set(filters);
      const options = collectContextFilterOptions();

      select.innerHTML = "";
      if (!options.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No contexts yet";
        opt.disabled = true;
        opt.selected = true;
        select.appendChild(opt);
        return;
      }

      options.forEach((optData) => {
        const opt = document.createElement("option");
        opt.value = optData.value;
        opt.textContent = optData.label;
        if (optData.title) opt.title = optData.title;
        opt.selected = !filters.length || selectedSet.has(optData.value);
        select.appendChild(opt);
      });

      if (filters.length) {
        const selectedCount = Array.from(select.selectedOptions).length;
        if (!selectedCount) {
          Array.from(select.options).forEach((opt) => {
            opt.selected = true;
          });
        }
      }
    }

    function getActiveContextTypes() {
      const select = $("ctxFilterSelect");
      if (!select) return [];
      const selected = Array.from(select.selectedOptions).map((opt) => opt.value);
      const allValues = Array.from(select.options).map((opt) => opt.value);
      const allSelected = selected.length === allValues.length;
      const types = selected.filter((value) => value);

      saveContextFilters(types);
      if (!types.length || allSelected) return [];
      return types;
    }

    function applyContextFilters(filters) {
      renderContextFilterOptions(filters);
    }


    function truncateText(text, limit = 600) {
      if (!text) return "";
      if (text.length <= limit) return text;
      return text.slice(0, limit).trim() + "…";
    }

    function makeMetaChip(text) {
      const chip = document.createElement("span");
      chip.className = "chip chip-muted";
      chip.textContent = text;
      return chip;
    }

    async function deleteDocFromStore(docId) {
      const res = await apiFetch(`${baseUrl()}/delete-doc`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: docId }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || "Delete failed");
      }
    }

    async function deleteContextEntry(contextId) {
      const res = await apiFetch(`${baseUrl()}/delete-context`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ context_id: contextId }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || "Delete failed");
      }
    }

    function buildMarkdownExport() {
      const lines = [];
      lines.push("# Answer");
      lines.push("");
      lines.push(lastAnswerText || "—");
      lines.push("");

      if (lastCitations && lastCitations.length) {
        lines.push("## Citations");
        lines.push("");
        lastCitations.forEach((id) => {
          const ev = (lastEvidence || []).find(
            (e) => String(e.id || "").toLowerCase() === String(id).toLowerCase()
          );
          if (!ev) return;
          const meta = [];
          if (ev.source) meta.push(ev.source);
          if (ev.page_start) meta.push(`page ${ev.page_start}`);
          if (ev.section) meta.push(ev.section);
          lines.push(`- **${id}** (${meta.join(" • ")})`);
          const snippet = ev.snippet || ev.text || ev.content || "";
          if (snippet) {
            lines.push(`  - ${snippet.replace(/\n+/g, " ").trim()}`);
          }
        });
        lines.push("");
      }
      return lines.join("\n");
    }

    function downloadMarkdown() {
      const content = buildMarkdownExport();
      const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "answer.md";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }

    function renderEvidence(items, activeCitations = []) {
      const root = $("evidence");
      root.innerHTML = "";
      if (!items || !items.length) {
        root.innerHTML = "<p class='muted'>No supporting evidence found</p>";
        return;
      }

      const citedSet = new Set((activeCitations || []).map((c) => String(c).toLowerCase()));

      items.forEach((e, idx) => {
        const box = document.createElement("details");
        const evidenceId = e.id || `e${idx + 1}`;
        if (citedSet.has(String(evidenceId).toLowerCase())) {
          box.classList.add("cited");
        }
        const sum = document.createElement("summary");
        const score = typeof e.score === "number" ? e.score.toFixed(3) : (e.score || "?");
        const source = e.source || e.id || `Result ${idx + 1}`;
        
        const scoreSpan = document.createElement("span");
        scoreSpan.textContent = `(relevance: ${score})`;
        scoreSpan.style.fontSize = "12px";
        scoreSpan.style.color = "var(--text-secondary)";
        scoreSpan.style.marginLeft = "8px";
        
        sum.innerHTML = `📎 ${source} <span class="chip chip-muted">${evidenceId}</span>`;
        sum.appendChild(scoreSpan);
        box.appendChild(sum);

        const meta = document.createElement("div");
        meta.className = "evidence-meta";
        if (e.page_start) meta.appendChild(makeMetaChip(`page ${e.page_start}`));
        if (e.section) meta.appendChild(makeMetaChip(e.section));
        box.appendChild(meta);

        const fullText = e.text || e.snippet || e.content || "";
        const truncated = truncateText(fullText, 700);

        const pre = document.createElement("pre");
        pre.textContent = truncated;
        pre.dataset.full = fullText;
        pre.dataset.truncated = truncated;
        pre.dataset.expanded = "false";
        box.appendChild(pre);

        if (fullText && fullText.length > truncated.length) {
          const toggle = document.createElement("span");
          toggle.className = "toggle-link";
          toggle.textContent = "Show more";
          toggle.addEventListener("click", (evt) => {
            evt.preventDefault();
            const expanded = pre.dataset.expanded === "true";
            pre.textContent = expanded ? pre.dataset.truncated : pre.dataset.full;
            pre.dataset.expanded = expanded ? "false" : "true";
            toggle.textContent = expanded ? "Show more" : "Show less";
          });
          box.appendChild(toggle);
        }

        root.appendChild(box);
      });
    }

    function renderAnswerPayload(payload) {
      if (payload && (payload.J || payload.S || payload.O || payload.N)) {
        const title = payload.J ? renderMarkdown(String(payload.J)) : "";
        const summary = payload.S ? renderMarkdown(String(payload.S)) : "";
        const outlook = payload.O ? renderMarkdown(String(payload.O)) : "";
        const notes = payload.N ? renderMarkdown(String(payload.N)) : "";

        return (
          `<div class="answer-card">` +
          (title ? `<div class="answer-section"><div class="answer-title">Title</div><div>${title}</div></div>` : "") +
          (summary ? `<div class="answer-section"><div class="answer-title">Summary</div><div>${summary}</div></div>` : "") +
          (outlook ? `<div class="answer-section"><div class="answer-title">Outlook</div><div>${outlook}</div></div>` : "") +
          (notes ? `<div class="answer-section"><div class="answer-title">Notes</div><div>${notes}</div></div>` : "") +
          `</div>`
        );
      }

      if (payload && (payload.key_findings || payload.main_conclusions || payload.citations)) {
        const findings = Array.isArray(payload.key_findings) ? payload.key_findings : [];
        const conclusions = Array.isArray(payload.main_conclusions) ? payload.main_conclusions : [];
        const citations = Array.isArray(payload.citations) ? payload.citations : [];

        const renderList = (items) => {
          if (!items.length) return "";
          return "<ul>" + items.map((item) => `<li>${renderMarkdown(String(item))}</li>`).join("") + "</ul>";
        };

        return (
          `<div class="answer-card">` +
          (findings.length ? `<div class="answer-section"><div class="answer-title">Key Findings</div>${renderList(findings)}</div>` : "") +
          (conclusions.length ? `<div class="answer-section"><div class="answer-title">Main Conclusions</div>${renderList(conclusions)}</div>` : "") +
          (citations.length ? `<div class="answer-section"><div class="answer-title">Citations</div>${renderList(citations)}</div>` : "") +
          `</div>`
        );
      }

      const summary = payload.summary || payload.answer || payload.response;
      if (!summary) {
        return "<div class='muted'>No structured answer available</div>";
      }

      return (
        `<div class="answer-card">` +
        `<div class="answer-section">` +
        `<div class="answer-title">Answer</div>` +
        `<div>${renderMarkdown(String(summary))}</div>` +
        `</div>` +
        `</div>`
      );
    }

    function renderUploadResult(data) {
      const root = $("uploadOut");
      if (!data) {
        root.style.display = "none";
        root.innerHTML = "";
        return;
      }

      const rows = [
        ["doc_id", data.doc_id],
        ["filename", data.filename],
        ["size", data.file_size_bytes ? formatBytes(data.file_size_bytes) : ""],
        ["pages", data.pages],
        ["chunks", data.chunks],
        ["namespace", data.namespace],
        ["doc_type", data.doc_type],
        ["extract", data.extraction_method],
        ["chunk", data.chunk_chars && data.overlap_chars ? `${data.chunk_chars}/${data.overlap_chars}` : ""],
      ];

      root.innerHTML = rows
        .filter(([, value]) => value !== undefined && value !== null && value !== "")
        .map(
          ([label, value]) =>
            `<div class="upload-row"><div class="upload-label">${label}</div><div>${value}</div></div>`
        )
        .join("");
      root.style.display = "block";
    }

    function renderModelsPanel(data) {
      const panel = $("modelsPanel");
      if (!data) {
        panel.innerHTML = "<div class='muted'>Unable to load model configuration</div>";
        return;
      }

      const rows = [
        ["generation", data.generation_model],
        ["planning", data.planning_model],
        ["moderation", data.moderation_model],
        ["embedding", data.embedding_model],
        ["reranker", data.reranker_model],
        ["llm_rerank", String(data.enable_llm_rerank)],
        ["rerank_top_n", String(data.rerank_top_n)],
        ["doc_top_k", String(data.doc_top_k)],
        ["bm25_lexical", String(data.enable_bm25_lexical)],
      ];

      panel.innerHTML = rows
        .filter(([, value]) => value !== undefined && value !== null && value !== "")
        .map(
          ([label, value]) =>
            `<div class="upload-row"><div class="upload-label">${label}</div><div>${value}</div></div>`
        )
        .join("");
    }

    async function loadModels() {
      try {
        const res = await apiFetch(`${baseUrl()}/models`);
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Failed to load models");
        }
        renderModelsPanel(data);
      } catch (err) {
        renderModelsPanel(null);
      }
    }

    function renderMarkdown(text) {
      if (!text) return '';

      // Escape HTML first
      text = text.replace(/[&<>"']/g, function(match) {
        const escapes = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
        return escapes[match];
      });

      // Headers (must be done before other replacements)
      text = text.replace(/^### (.*$)/gim, '<h3>$1</h3>');
      text = text.replace(/^## (.*$)/gim, '<h2>$1</h2>');
      text = text.replace(/^# (.*$)/gim, '<h1>$1</h1>');

      // Code blocks (must be done before inline code)
      text = text.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

      // Inline code
      text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>');

      // Bold
      text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');

      // Italic
      text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
      text = text.replace(/_(.*?)_/g, '<em>$1</em>');

      // Blockquotes
      text = text.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

      // Lists (basic - convert to HTML lists)
      text = text.replace(/^\* (.*$)/gim, '<ul><li>$1</li></ul>');
      text = text.replace(/^\d+\. (.*$)/gim, '<ol><li>$1</li></ol>');

      // Clean up nested lists (merge consecutive ul/ol elements)
      text = text.replace(/<\/ul>\s*<ul>/g, '');
      text = text.replace(/<\/ol>\s*<ol>/g, '');

      // Line breaks and paragraphs
      text = text.replace(/\n\n+/g, '</p><p>');
      text = text.replace(/\n/g, '<br>');

      // Wrap in paragraph if not already wrapped
      if (!text.match(/^<h[1-6]|<ul|<ol|<pre|<blockquote|<p/)) {
        text = '<p>' + text + '</p>';
      }

      // Clean up empty paragraphs
      text = text.replace(/<p><\/p>/g, '');
      text = text.replace(/<p><br><\/p>/g, '');

      return text;
    }

    function extractJsonCandidate(raw) {
      if (!raw || typeof raw !== "string") return null;
      const trimmed = raw.trim();
      const jsonStart = trimmed.indexOf("{");
      const jsonEnd = trimmed.lastIndexOf("}");
      if (jsonStart === -1 || jsonEnd <= jsonStart) return null;
      return trimmed.slice(jsonStart, jsonEnd + 1);
    }

    function extractJsonStringValue(raw, key) {
      if (!raw || typeof raw !== "string") return null;
      const keyIdx = raw.indexOf(`"${key}"`);
      if (keyIdx === -1) return null;
      const colonIdx = raw.indexOf(":", keyIdx);
      if (colonIdx === -1) return null;
      const quoteIdx = raw.indexOf("\"", colonIdx + 1);
      if (quoteIdx === -1) return null;

      let i = quoteIdx + 1;
      let escaped = false;
      let buf = "";
      for (; i < raw.length; i++) {
        const ch = raw[i];
        if (escaped) {
          buf += ch;
          escaped = false;
          continue;
        }
        if (ch === "\\") {
          buf += ch;
          escaped = true;
          continue;
        }
        if (ch === "\"") {
          try {
            return JSON.parse(`"${buf}"`);
          } catch {
            return buf;
          }
        }
        buf += ch;
      }
      return null;
    }

    function extractJsonStringList(raw, key) {
      if (!raw || typeof raw !== "string") return [];
      const keyIdx = raw.indexOf(`"${key}"`);
      if (keyIdx === -1) return [];
      const bracketStart = raw.indexOf("[", keyIdx);
      if (bracketStart === -1) return [];
      const bracketEnd = raw.indexOf("]", bracketStart);
      if (bracketEnd === -1) return [];
      const inner = raw.slice(bracketStart + 1, bracketEnd);
      const items = [];
      const re = /"([^"]*)"/g;
      let m;
      while ((m = re.exec(inner)) !== null) {
        try {
          items.push(JSON.parse(`"${m[1]}"`));
        } catch {
          items.push(m[1]);
        }
      }
      return items;
    }

    function extractCitations(text) {
      if (!text) return [];
      const citationPattern = /\b\[?e(\d+)\]?\b/gi;
      const citations = [];
      let match;

      while ((match = citationPattern.exec(text)) !== null) {
        citations.push(`e${match[1]}`);
      }

      return [...new Set(citations)]; // Remove duplicates
    }

    function renderCitations(citationNumbers, evidenceItems) {
      const citationsDiv = $("citations");
      const citationsListDiv = $("citations-list");
      
      if (!citationNumbers || !citationNumbers.length) {
        citationsDiv.style.display = "none";
        return [];
      }
      
      citationsListDiv.innerHTML = "";
      
      const activeIds = [];
      citationNumbers.forEach(id => {
        let evidenceItem = (evidenceItems || []).find(
          (e) => String(e.id || "").toLowerCase() === String(id).toLowerCase()
        );
        if (!evidenceItem) {
          const num = parseInt(String(id).replace(/[^\d]/g, ""), 10);
          if (!Number.isNaN(num) && evidenceItems && evidenceItems[num - 1]) {
            evidenceItem = evidenceItems[num - 1];
          }
        }
        if (!evidenceItem) return;

        const citationDiv = document.createElement("div");
        citationDiv.className = "citation-item";
        
        const refDiv = document.createElement("div");
        refDiv.className = "citation-ref";
        refDiv.textContent = `[${id}]`;

        const meta = [];
        if (evidenceItem.source) meta.push(evidenceItem.source);
        if (evidenceItem.page_start) meta.push(`page ${evidenceItem.page_start}`);
        if (evidenceItem.section) meta.push(evidenceItem.section);
        if (typeof evidenceItem.score === "number") meta.push(`score ${evidenceItem.score.toFixed(3)}`);
        
        const metaDiv = document.createElement("div");
        metaDiv.className = "citation-meta";
        metaDiv.textContent = meta.join(" • ");

        const textDiv = document.createElement("div");
        textDiv.className = "citation-text";
        textDiv.textContent = evidenceItem.snippet || evidenceItem.text || evidenceItem.content || "No content available";
        
        citationDiv.appendChild(refDiv);
        if (meta.length) citationDiv.appendChild(metaDiv);
        citationDiv.appendChild(textDiv);
        citationsListDiv.appendChild(citationDiv);
        activeIds.push(id);
      });
      
      citationsDiv.style.display = "block";
      return activeIds;
    }

    async function uploadPdf() {
      console.log("uploadPdf called");
      $("uploadMessage").style.display = "none";
      $("answer").innerHTML = "—";
      $("citations").style.display = "none";
      $("evidence").innerHTML = "<p class='muted'>Upload a PDF to get started</p>";
      setDocId(null);

      const fileInput = $("file");
      const inputFile = fileInput ? fileInput.files[0] : null;
      const f = selectedFile || inputFile;
      console.log("Selected file:", f);
      if (!f) {
        showMessage("uploadMessage", "❌ Please select a PDF file", "error");
        return;
      }

      if (!isPdfFile(f)) {
        showMessage("uploadMessage", "❌ Please select a valid PDF file", "error");
        return;
      }

      if (inputFile && inputFile !== selectedFile) {
        setSelectedFile(inputFile);
      }

      updateUploadProgress("upload");
      scheduleUploadStages();

      const btn = $("uploadBtn");
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Uploading...";

      try {
        const fd = new FormData();
        fd.append("file", f);

        const url = `${baseUrl()}/upload`;
        console.log("Uploading to:", url);

        const res = await apiFetch(url, {
          method: "POST",
          body: fd,
        });

        console.log("Response status:", res.status);
        const data = await res.json();
        console.log("Response data:", data);
        renderUploadResult(data);

        if (!res.ok) {
          throw new Error(data?.detail || "Upload failed");
        }
        clearUploadStageTimers();
        updateUploadProgress("ready");

        setDocId(data.doc_id);
        upsertDoc({
          doc_id: data.doc_id,
          filename: data.filename,
          pages: data.pages,
          chunks: data.chunks,
          file_size_bytes: data.file_size_bytes,
          namespace: data.namespace,
          doc_type: data.doc_type,
          extraction_method: data.extraction_method,
          chunk_chars: data.chunk_chars,
          overlap_chars: data.overlap_chars,
        });
        setSelectedFile(null);
        showMessage("uploadMessage", `✅ PDF uploaded successfully! Doc ID: ${data.doc_id}`, "success");
      } catch (err) {
        console.error("Upload error:", err);
        showMessage("uploadMessage", `❌ Error: ${err.message || String(err)}`, "error");
        clearUploadStageTimers();
        updateUploadProgress("error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    async function resetKnowledge() {
      if (!confirm("This will delete all stored document chunks from Pinecone. Continue?")) {
        return;
      }

      const btn = $("resetKnowledgeBtn");
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Resetting...";

      try {
        const res = await apiFetch(`${baseUrl()}/reset-knowledge`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Reset failed");
        }

        saveDocs([]);
        renderDocSelect([], null);
        renderDocList([], null);
        setDocId(null);
        renderUploadResult(null);
        $("answer").innerHTML = "—";
        $("citations").style.display = "none";
        $("evidence").innerHTML = "<p class='muted'>Upload a PDF to get started</p>";
        showMessage("uploadMessage", "✅ Knowledge store cleared and doc list reset", "success");
      } catch (err) {
        showMessage("uploadMessage", `❌ Error: ${err.message || String(err)}`, "error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    function resolveContextType(typeKey) {
      if (typeKey === "custom") {
        const customElem = $("customContextType");
        const custom = customElem ? customElem.value.trim() : "";
        if (!custom) {
          showMessage("blueprintMessage", "❌ Enter a custom context type label", "error");
          return null;
        }
        return custom;
      }
      return typeKey;
    }

    async function uploadContext(typeKey, btnEl) {
      const textElem = $("contextText");
      if (!textElem) return;
      const text = textElem.value.trim();
      if (!text) {
        showMessage("blueprintMessage", "❌ Please enter context text", "error");
        return;
      }

      const contextType = resolveContextType(typeKey);
      if (!contextType) return;

      const btn = btnEl || document.activeElement;
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Saving...";

      try {
        if (editingContextId) {
          try {
            await deleteContextEntry(editingContextId);
          } catch (err) {
            console.warn("Failed to delete old context entry:", err);
          }
        }

        const res = await apiFetch(`${baseUrl()}/context`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, context_type: contextType }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Context upload failed");
        }

        const entries = loadContextEntries();
        const entry = {
          context_id: data.context_id,
          context_type: contextType,
          text,
          created_at: new Date().toISOString(),
        };
        const filtered = editingContextId
          ? entries.filter((e) => e.context_id !== editingContextId)
          : entries;
        filtered.unshift(entry);
        saveContextEntries(filtered);
        renderContextList(filtered);

        textElem.value = "";
        const customElem = $("customContextType");
        if (customElem) customElem.value = "";
        editingContextId = null;
        editingContextType = null;
        showMessage("blueprintMessage", "✅ Context saved", "success");
      } catch (err) {
        showMessage("blueprintMessage", `❌ Error: ${err.message || String(err)}`, "error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    async function resetContext() {
      if (!confirm("This will delete all stored context entries and blueprints in Pinecone. Continue?")) {
        return;
      }

      const btn = $("resetContextBtn");
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Resetting...";

      try {
        const res = await apiFetch(`${baseUrl()}/reset-context`, { method: "POST" });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Reset failed");
        }

        saveContextEntries([]);
        renderContextList([]);
        saveBlueprints([]);
        renderBlueprintList([]);
        showMessage("blueprintMessage", "✅ Context store cleared", "success");
      } catch (err) {
        showMessage("blueprintMessage", `❌ Error: ${err.message || String(err)}`, "error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    function clearBlueprintForm() {
      $("blueprintId").value = "";
      $("blueprintDesc").value = "";
      $("blueprintJson").value = "";
      editingBlueprintId = null;
    }

    function parseBlueprintPayload() {
      let blueprintId = $("blueprintId").value.trim();
      let description = $("blueprintDesc").value.trim();
      const rawJson = $("blueprintJson").value.trim();
      let blueprint = null;

      if (rawJson) {
        let parsed = null;
        try {
          parsed = JSON.parse(rawJson);
        } catch (err) {
          showMessage("blueprintMessage", "❌ Blueprint JSON must be valid", "error");
          return null;
        }

        if (
          parsed &&
          typeof parsed === "object" &&
          parsed.id &&
          parsed.description &&
          parsed.blueprint
        ) {
          blueprintId = String(parsed.id || "").trim();
          description = String(parsed.description || "").trim();
          blueprint = parsed.blueprint;
        } else {
          blueprint = parsed;
        }
      }

      if (!blueprintId) {
        showMessage("blueprintMessage", "❌ Enter a blueprint id", "error");
        return null;
      }
      if (!description) {
        showMessage("blueprintMessage", "❌ Enter a blueprint description", "error");
        return null;
      }
      if (!blueprint || typeof blueprint !== "object" || Array.isArray(blueprint)) {
        showMessage("blueprintMessage", "❌ Blueprint JSON must be an object", "error");
        return null;
      }

      return { id: blueprintId, description, blueprint };
    }

    async function uploadBlueprint() {
      const payload = parseBlueprintPayload();
      if (!payload) return;

      const btn = $("saveBlueprintBtn");
      const originalText = btn.innerHTML;
      btn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Saving...";

      try {
        if (editingBlueprintId && editingBlueprintId !== payload.id) {
          try {
            await deleteContextEntry(editingBlueprintId);
          } catch (err) {
            console.warn("Failed to delete old blueprint:", err);
          }
        }

        const res = await apiFetch(`${baseUrl()}/context-blueprint`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Blueprint upload failed");
        }

        const entries = loadBlueprints();
        const entry = {
          blueprint_id: payload.id,
          description: payload.description,
          blueprint: payload.blueprint,
          created_at: new Date().toISOString(),
        };
        const filtered = editingBlueprintId
          ? entries.filter((e) => e.blueprint_id !== editingBlueprintId)
          : entries;
        filtered.unshift(entry);
        saveBlueprints(filtered);
        renderBlueprintList(filtered);

        clearBlueprintForm();
        showMessage("blueprintMessage", "✅ Blueprint saved", "success");
      } catch (err) {
        showMessage("blueprintMessage", `❌ Error: ${err.message || String(err)}`, "error");
      } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    async function ask(questionText) {
      const doc_id = $("docId").textContent;
      if (!doc_id || doc_id === "Not uploaded") {
        showMessage("askMessage", "❌ Please upload a PDF first", "error");
        return;
      }

      const question = questionText || $("question").value.trim();
      if (!question) {
        showMessage("askMessage", "❌ Please enter a question", "error");
        return;
      }

      const btn = $("askBtn");
      const sBtn = $("summaryBtn");
      const originalText = btn.innerHTML;
      
      btn.disabled = true;
      sBtn.disabled = true;
      btn.innerHTML = "<span class='icon'>⏳</span> Analyzing...";
      
      $("answer").innerHTML = "🤔 Analyzing your question...";
      $("citations").style.display = "none";
      $("evidence").innerHTML = "<div class='status-box status-loading'>🔍 Retrieving relevant sections...</div>";
      showMessage("askMessage", "", "");

      try {
        const sectionElem = $("sectionFilter");
        const pageStartElem = $("pageStart");
        const pageEndElem = $("pageEnd");
        const section = sectionElem ? sectionElem.value : "";
        const pageStart = pageStartElem ? parseInt(pageStartElem.value, 10) : NaN;
        const pageEnd = pageEndElem ? parseInt(pageEndElem.value, 10) : NaN;
        const contextTypes = getActiveContextTypes();
        const payload = {
          doc_id,
          question,
          top_k: 6,
          context_types: contextTypes.length ? contextTypes : undefined,
        };
        if (section) payload.section = section;
        if (Number.isFinite(pageStart)) payload.page_start = pageStart;
        if (Number.isFinite(pageEnd)) payload.page_end = pageEnd;

        const res = await apiFetch(`${baseUrl()}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data?.detail || "Chat request failed");
        }

        // Render answer (structured JSON or plain text)
        let rawAnswer = data.answer ?? "—";
        let structured = null;

        if (rawAnswer && typeof rawAnswer === "string") {
          const candidate = extractJsonCandidate(rawAnswer);
          if (candidate) {
            try {
              structured = JSON.parse(candidate);
            } catch (e) {
              structured = null;
            }
            if (!structured) {
              const answer = extractJsonStringValue(candidate, "answer");
              const findings = extractJsonStringList(candidate, "key_findings");
              const conclusions = extractJsonStringList(candidate, "main_conclusions");
              const citations = extractJsonStringList(candidate, "citations");
              if (answer || findings.length || conclusions.length || citations.length) {
                structured = {
                  answer: answer || "",
                  key_findings: findings,
                  main_conclusions: conclusions,
                  citations: citations,
                };
              }
            }
          }
        } else if (rawAnswer && typeof rawAnswer === "object") {
          structured = rawAnswer;
        }

        if (structured) {
          const evidenceList =
            structured.citations ||
            structured.evidence ||
            data.evidence ||
            data.citations ||
            [];
          const summaryText =
            structured.summary ||
            structured.answer ||
            structured.response ||
            "";
          let exportText = String(summaryText || "");
          if (!exportText && (structured.key_findings || structured.main_conclusions)) {
            const findings = Array.isArray(structured.key_findings) ? structured.key_findings : [];
            const conclusions = Array.isArray(structured.main_conclusions) ? structured.main_conclusions : [];
            exportText = [
              findings.length ? "Key findings:\n- " + findings.join("\n- ") : "",
              conclusions.length ? "Main conclusions:\n- " + conclusions.join("\n- ") : "",
            ]
              .filter(Boolean)
              .join("\n\n");
          }

          $("answer").innerHTML = renderAnswerPayload(structured);
          const citationNumbers = extractCitations(String(summaryText));
          const active = renderCitations(citationNumbers, evidenceList);
          renderEvidence(evidenceList, active);
          persistSectionsFromEvidence(doc_id, evidenceList);
          lastAnswerText = exportText;
          lastEvidence = evidenceList || [];
          lastCitations = active || [];
        } else {
          rawAnswer = String(rawAnswer);
          const cleanAnswer = rawAnswer.replace(/\[e\d+\]/g, "").trim(); // Remove citation markers
          const markdownAnswer = renderMarkdown(cleanAnswer);
          $("answer").innerHTML = markdownAnswer;

          const citationNumbers = extractCitations(rawAnswer);
          const evidenceList = data.evidence || data.citations || [];
          const active = renderCitations(citationNumbers, evidenceList);
          renderEvidence(evidenceList, active);
          persistSectionsFromEvidence(doc_id, evidenceList);
          lastAnswerText = cleanAnswer;
          lastEvidence = evidenceList || [];
          lastCitations = active || [];
        }
      } catch (err) {
        $("answer").innerHTML = "—";
        $("citations").style.display = "none";
        showMessage("askMessage", `❌ Error: ${err.message || String(err)}`, "error");
      } finally {
        btn.disabled = false;
        sBtn.disabled = false;
        btn.innerHTML = originalText;
      }
    }

    $("uploadBtn").addEventListener("click", (e) => {
      console.log("Upload button clicked", e);
      uploadPdf();
    });
    $("file").addEventListener("change", (e) => {
      const file = e.target.files[0] || null;
      setSelectedFile(file);
    });
    if ($("dropZone")) {
      const dropZone = $("dropZone");
      dropZone.addEventListener("click", () => {
        const fileInput = $("file");
        if (fileInput) fileInput.click();
      });
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
      dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
      });
      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (!file) return;
        if (!isPdfFile(file)) {
          showMessage("uploadMessage", "❌ Only PDF files are supported", "error");
          return;
        }
        setSelectedFile(file);
        const fileInput = $("file");
        if (fileInput) fileInput.value = "";
      });
    }
    $("docSelect").addEventListener("change", (e) => {
      const value = e.target.value || null;
      setDocId(value);
    });
    if ($("contextExecBtn")) {
      $("contextExecBtn").addEventListener("click", () => uploadContext("executive_brief", $("contextExecBtn")));
    }
    if ($("contextTechBtn")) {
      $("contextTechBtn").addEventListener("click", () => uploadContext("technical_analysis", $("contextTechBtn")));
    }
    if ($("contextLegalBtn")) {
      $("contextLegalBtn").addEventListener("click", () => uploadContext("legal_compliance", $("contextLegalBtn")));
    }
    if ($("contextCustomBtn")) {
      $("contextCustomBtn").addEventListener("click", () => uploadContext("custom", $("contextCustomBtn")));
    }
    $("resetContextBtn").addEventListener("click", () => resetContext());
    $("saveBlueprintBtn").addEventListener("click", () => uploadBlueprint());
    $("clearBlueprintBtn").addEventListener("click", () => {
      clearBlueprintForm();
      showMessage("blueprintMessage", "", "");
    });
    $("resetKnowledgeBtn").addEventListener("click", () => resetKnowledge());
    $("exportMarkdownBtn").addEventListener("click", () => downloadMarkdown());
    $("printAnswerBtn").addEventListener("click", () => window.print());

    if ($("ctxFilterSelect")) {
      $("ctxFilterSelect").addEventListener("change", () => getActiveContextTypes());
    }
    $("askBtn").addEventListener("click", () => ask());
    $("summaryBtn").addEventListener("click", () => ask("Provide a concise summary of the document"));

    $("question").addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !$("askBtn").disabled) ask();
    });

    const baseUrlInput = $("baseUrl");
    if (baseUrlInput) {
      baseUrlInput.addEventListener("change", () => {
        setDocId(null);
        checkAuth();
      });
    }

    // Initialize doc selector from localStorage
    const docs = loadDocs();
    renderDocSelect(docs, null);
    renderDocList(docs, null);
    renderContextList(loadContextEntries());
    renderBlueprintList(loadBlueprints());
    applyContextFilters(loadContextFilters());

    const savedDoc = localStorage.getItem(LAST_DOC_KEY);
    if (savedDoc && docs.find((d) => d.doc_id === savedDoc)) {
      setDocId(savedDoc);
    } else {
      updateDocUI(null);
    }

    if ($("loginBtn")) {
      $("loginBtn").addEventListener("click", () => login());
    }
    if ($("logoutBtn")) {
      $("logoutBtn").addEventListener("click", () => logout());
    }
    if ($("loginPassword")) {
      $("loginPassword").addEventListener("keypress", (e) => {
        if (e.key === "Enter") login();
      });
    }

    setupSectionNav();
    checkAuth();
  
