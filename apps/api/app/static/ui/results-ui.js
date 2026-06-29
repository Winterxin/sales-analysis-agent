const formatBlank = (value) => (value === undefined || value === null || value === "" ? "-" : value);

export function renderResultsPage(results) {
  renderResultActions(results.downloads || {});
  renderResultMeta(results.dataset || {});
  renderKpis(results.kpis || []);
  renderSummary(results.summary || []);
  renderSections(results.sections || []);
  renderModeling(results.modeling);
  renderActionPlan(results.action_plan);
  renderDataNote(results.data_note);
}

export function renderResultsLoading() {
  document.getElementById("results-summary").innerHTML = `
    <div class="results-empty"><strong>正在加载分析结果</strong></div>
  `;
}

export function renderResultsError(message) {
  document.getElementById("results-summary").innerHTML = `
    <div class="results-empty error"><strong>${escapeHtml(message || "分析结果暂不可用")}</strong></div>
  `;
}

function renderResultActions(downloads) {
  const actions = document.getElementById("results-actions");
  actions.innerHTML = "";
  const items = [
    ["打开 HTML 报告", downloads.client_report_html],
    ["下载 JSON", downloads.client_report_json || downloads.report_json],
    ["下载 Notebook", downloads.notebook, "primary"],
  ];
  for (const [label, href, variant] of items) {
    if (!href) continue;
    const link = document.createElement("a");
    link.className = `results-button ${variant || ""}`;
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = label;
    actions.appendChild(link);
  }
}

function renderResultMeta(dataset) {
  document.getElementById("results-meta").innerHTML = `
    ${metaItem("文件", dataset.filename)}
    ${metaItem("记录 / 字段", [dataset.row_count, dataset.column_count].filter(Boolean).join(" / "))}
    ${metaItem("时间跨度", dataset.time_span_label || (dataset.date_span_days ? `${dataset.date_span_days} 天` : "-"))}
    ${metaItem("分析章节", dataset.section_count ? `${dataset.section_count} 个` : "-")}
    <div class="results-tags">
      <span>Notebook 内容整理</span>
      <span>动态章节</span>
    </div>
  `;
}

function renderKpis(kpis) {
  document.getElementById("results-kpis").innerHTML = kpis
    .map((kpi) => `
      <article class="result-kpi">
        <span>${escapeHtml(kpi.label)}</span>
        <b>${escapeHtml(kpi.value)}</b>
      </article>
    `)
    .join("");
}

function renderSummary(summary) {
  const cards = summary.length
    ? summary
    : ["分析完成后，这里展示 Notebook 结论中的核心判断。"];
  document.getElementById("results-summary").innerHTML = `
    <div class="results-section-head">
      <h2>核心结论</h2>
      <div class="results-tags"><span>来自 Notebook 结论</span><span class="risk">客户阅读位</span></div>
    </div>
    <div class="summary-grid">
      ${cards.map((item) => `<article><strong>${escapeHtml(shortTitle(item))}</strong><p>${escapeHtml(item)}</p></article>`).join("")}
    </div>
  `;
}

function renderSections(sections) {
  document.getElementById("results-sections").innerHTML = sections
    .map((section) => renderSection(section))
    .join("");
  for (const button of document.querySelectorAll("[data-toggle-full-text]")) {
    button.addEventListener("click", () => toggleFullText(button));
  }
}

function renderSection(section) {
  const table = renderTable(section.table);
  const chart = renderChart(section.chart);
  const fullText = section.is_long_text
    ? `<button class="text-toggle" type="button" data-toggle-full-text>展开完整分析</button>
       <p class="full-text" hidden>${escapeHtml(section.full_text)}</p>`
    : "";
  return `
    <article class="result-chapter">
      <header>
        <div>
          <h2>${escapeHtml(section.title)}</h2>
          <p>${escapeHtml(section.summary || "展示本章节的图表、结论和证据。")}</p>
        </div>
        <span class="chapter-kind">${escapeHtml(section.kind || "分析")}</span>
      </header>
      <div class="chapter-grid">
        <div>
          <p class="takeaway">${escapeHtml(section.takeaway || section.summary || "")}</p>
          <ul>${(section.findings || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
          ${fullText}
          ${table}
        </div>
        ${chart}
      </div>
    </article>
  `;
}

function renderChart(chart = {}) {
  const payload = chart.payload || {};
  const type = chart.type || "table";
  if (type === "line" && Array.isArray(payload.x) && Array.isArray(payload.y)) {
    return `<div class="chart-card"><strong>${escapeHtml(payload.series_name || "趋势图")}</strong>${svgLine(payload.x, payload.y)}<p>图表按容器自适应，避免裁切。</p></div>`;
  }
  if ((type === "bar" || type === "stacked_bar" || type === "histogram") && Array.isArray(payload.x) && Array.isArray(payload.y)) {
    return `<div class="chart-card"><strong>${escapeHtml(payload.series_name || "对比图")}</strong>${svgBars(payload.x, payload.y)}<p>接入真实产物后按数据点数量自动抽样。</p></div>`;
  }
  if (type === "scatter" && Array.isArray(payload.x) && Array.isArray(payload.y)) {
    return `<div class="chart-card"><strong>${escapeHtml(payload.series_name || "关系图")}</strong>${svgScatter(payload.x, payload.y)}<p>用于展示两个指标之间的关系。</p></div>`;
  }
  return `<div class="chart-card"><strong>图表</strong><div class="chart-empty">暂无可直接渲染的图表数据</div></div>`;
}

function renderModeling(modeling) {
  const node = document.getElementById("results-modeling");
  if (!modeling) {
    node.innerHTML = "";
    node.classList.add("hidden");
    return;
  }
  node.classList.remove("hidden");
  const metrics = modeling.metrics || {};
  node.innerHTML = `
    <div class="results-section-head">
      <h2>${escapeHtml(modeling.title || "建模分析")}</h2>
      <div class="results-tags"><span>可选章节</span><span>人工复核</span></div>
    </div>
    <div class="model-grid">
      <article class="model-card">
        <p class="takeaway">${escapeHtml(modeling.narrative || (modeling.findings || [])[0] || "模型结果用于辅助业务复核。")}</p>
        ${scoreRow("Best Model", modeling.best_model)}
        ${Object.entries(metrics).map(([label, value]) => scoreRow(label, value)).join("")}
        ${modeling.is_long_text ? `<button class="text-toggle" type="button" data-toggle-full-text>展开完整建模说明</button><p class="full-text" hidden>${escapeHtml(modeling.full_text || "")}</p>` : ""}
      </article>
      <article class="chart-card">
        <strong>模型误判结构</strong>
        ${renderConfusionMatrix(modeling.confusion_matrix || [])}
        <p>训练细节和可修改代码仍保留在 Notebook。</p>
      </article>
    </div>
  `;
  for (const button of node.querySelectorAll("[data-toggle-full-text]")) {
    button.addEventListener("click", () => toggleFullText(button));
  }
}

function renderActionPlan(actionPlan) {
  const node = document.getElementById("results-action-plan");
  const rows = actionPlan?.rows || [];
  const columns = actionPlan?.columns || [];
  if (!rows.length || !columns.length) {
    node.innerHTML = "";
    node.classList.add("hidden");
    return;
  }
  node.classList.remove("hidden");
  node.innerHTML = `
    <div class="results-section-head">
      <h2>行动建议</h2>
      <div class="results-tags"><span class="risk">P1/P2/P3</span><span>来自 Notebook Action Plan</span></div>
    </div>
    <div class="action-grid">
      ${rows.map((row) => `
        <article class="action-card">
          <span class="chapter-kind">${escapeHtml(row.priority || row.Priority || "Action")}</span>
          <strong>${escapeHtml(row.issue || row.Issue || "行动项")}</strong>
          <p>${escapeHtml(row.action || row.Action || "")}</p>
          ${row.evidence ? `<small>${escapeHtml(row.evidence)}</small>` : ""}
        </article>
      `).join("")}
    </div>
  `;
}

function renderDataNote(dataNote) {
  const node = document.getElementById("results-data-note");
  const findings = dataNote?.findings || [];
  node.innerHTML = `
    <div class="results-section-head">
      <h2>数据说明</h2>
      <div class="results-tags"><span>轻量展示</span></div>
    </div>
    <div class="summary-grid">
      <article><strong>字段与规模</strong><p>记录数 ${escapeHtml(formatBlank(dataNote?.profile?.row_count))}，字段数 ${escapeHtml(formatBlank(dataNote?.profile?.column_count))}。</p></article>
      <article><strong>清洗摘要</strong><p>${escapeHtml(findings[0] || "数据清洗和字段识别信息默认轻量展示。")}</p></article>
    </div>
  `;
}

function renderTable(table = {}) {
  const rows = table.rows || [];
  const columns = table.columns || [];
  if (!rows.length || !columns.length) return "";
  return `
    <div class="result-table-wrap">
      <table>
        <thead><tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((row) => `<tr>${columns.map((column) => `<td>${escapeHtml(formatBlank(row[column]))}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderConfusionMatrix(rows) {
  const cells = rows.length ? rows.slice(0, 4).map((row) => row.count) : ["TN", "FP", "FN", "TP"];
  return `<div class="matrix">${cells.map((cell) => `<div>${escapeHtml(formatBlank(cell))}</div>`).join("")}</div>`;
}

function svgLine(labels, values) {
  const points = normalizePoints(values).map((point, index) => `${index * (100 / Math.max(values.length - 1, 1))},${point}`).join(" ");
  return `<svg class="chart-canvas" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="趋势图"><polyline points="${points}" /></svg>`;
}

function svgBars(labels, values) {
  const nums = numericValues(values);
  const max = Math.max(...nums, 1);
  const width = 100 / Math.max(nums.length, 1);
  return `<svg class="chart-canvas" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="柱状图">
    ${nums.map((value, index) => {
      const height = Math.max(4, Math.abs(value) / max * 82);
      return `<rect x="${index * width + 1}" y="${96 - height}" width="${Math.max(width - 2, 2)}" height="${height}" />`;
    }).join("")}
  </svg>`;
}

function svgScatter(xs, ys) {
  const xPoints = normalizePoints(xs).map((value) => 100 - value);
  const yPoints = normalizePoints(ys);
  return `<svg class="chart-canvas" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label="散点图">
    ${yPoints.map((y, index) => `<circle cx="${xPoints[index] || index * 10}" cy="${y}" r="2.2" />`).join("")}
  </svg>`;
}

function normalizePoints(values) {
  const nums = numericValues(values);
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  const span = max - min || 1;
  return nums.map((value) => 88 - ((value - min) / span) * 72);
}

function numericValues(values) {
  return values.slice(0, 24).map((value) => Number(value) || 0);
}

function scoreRow(label, value) {
  return `<div class="score-row"><span>${escapeHtml(label)}</span><b>${escapeHtml(formatBlank(value))}</b></div>`;
}

function metaItem(label, value) {
  return `<div class="meta-item"><span>${escapeHtml(label)}</span><b>${escapeHtml(formatBlank(value))}</b></div>`;
}

function toggleFullText(button) {
  const text = button.nextElementSibling;
  const expanded = text.hasAttribute("hidden");
  text.toggleAttribute("hidden", !expanded);
  button.textContent = expanded ? "收起完整分析" : "展开完整分析";
}

function shortTitle(text) {
  return String(text).split(/[，。,.]/)[0].slice(0, 24) || "分析结论";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
