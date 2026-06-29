const appShell = document.getElementById("app-shell");
const fileInput = document.getElementById("csv-file");
const fileName = document.getElementById("file-name");
const submitButton = document.getElementById("submit-button");
const openPreviewButton = document.getElementById("open-preview-button");
const expandPreviewButton = document.getElementById("expand-preview-button");
const statusNode = document.getElementById("status");
const statusPill = document.getElementById("status-pill");
const steps = [...document.querySelectorAll(".step")];
const downloadList = document.getElementById("download-list");
const previewFrame = document.getElementById("report-preview-frame");
const previewEmpty = document.getElementById("preview-empty");
const openReportLink = document.getElementById("open-report-link");
const analysisTab = document.getElementById("analysis-tab");
const detailTab = document.getElementById("detail-tab");
const analysisPage = document.getElementById("analysis-page");
const resultsPage = document.getElementById("results-page");
const detailPage = document.getElementById("detail-page");
const detailStatus = document.getElementById("detail-status");
const detailProfile = document.getElementById("detail-profile");
const detailModuleCount = document.getElementById("detail-module-count");
const fieldTable = document.getElementById("field-table");
const viewDetailLink = document.getElementById("view-detail-link");
const profileInputs = [...document.querySelectorAll("input[name='llm-profile']")];
const languageInputs = [...document.querySelectorAll("input[name='output-language']")];

let currentStepIndex = 0;

export const UI_TEXT = {
  en: {
    analysis: "Analysis",
    taskDetails: "Task Details",
    headline: "Upload a Sales CSV and Generate Analysis Artifacts",
    startAnalysis: "Start Analysis",
    runMode: "Run Mode",
    quick: "Quick",
    full: "Full",
    quickHint: "Fast complete artifacts with core LLM stages and lightweight chart selection.",
    fullHint: "Deeper enhancement with chart intent planning, revision, and extra reflections.",
    chooseCsv: "Choose Sales CSV",
    uploadFile: "Upload File",
    csvHint: "Supports order details, retail transactions, and ecommerce sales data.",
    noFileSelected: "No file selected",
    artifactDownloads: "Artifact Downloads",
    notebook: "Notebook",
    businessReview: "Business Review",
    clientReport: "Client Report",
    reportPreview: "Report Preview",
    previewReport: "Preview Report",
    reset: "Reset",
    waitingForFile: "Waiting for a file.",
    waiting: "Waiting",
    analysisCompleted: "Analysis completed",
    analysisFailed: "Analysis failed",
    language: "Language",
    pages: "Pages",
    open: "Open",
    close: "Close",
    closePreview: "Close Preview",
    collapsePreview: "Collapse Preview",
    expand: "Expand",
    viewTaskDetails: "View Task Details",
    taskStatus: "Task Status",
    moduleCount: "Module Count",
    fieldMapping: "Field Mapping",
    sourceField: "Source Field",
    mappedMeaning: "Mapped Meaning",
    waitingForUpload: "Waiting for upload.",
    noFieldMapping: "No field mapping is available.",
    previewEmpty: "The report preview opens after analysis completes.",
    analyzing: "Analyzing...",
    creatingTask: "Creating task",
    uploadingFields: "Uploading and mapping fields",
    planGenerated: "Analysis plan generated",
    runningAgent: "Running agent analysis",
    buildingReports: "Building reports",
    restoredLatestTask: "Restored the latest task.",
    done: "Done",
    failed: "Failed",
    running: "Running",
    agentFlow: "Agent Flow",
    stepCreateTitle: "Create Task",
    stepCreateNote: "Prepare runtime folder and task record.",
    stepUploadTitle: "Upload and Map Fields",
    stepUploadNote: "Detect date, sales, product, profit, and related fields.",
    stepPlanTitle: "Plan Analysis",
    stepPlanNote: "Select trend, product, order, profit, and modeling modules.",
    stepRunTitle: "Run Analysis",
    stepRunNote: "Run Python analysis, chart selection, Notebook generation, and execution.",
    stepReportsTitle: "Build Reports",
    stepReportsNote: "Prepare Client Report, Notebook, and Business Review.",
    detailDescription: "Detailed task state, field mapping, module count, trace, and errors live here.",
  },
  "zh-CN": {
    analysis: "分析",
    taskDetails: "任务详情",
    headline: "上传销售 CSV 并生成分析产物",
    startAnalysis: "开始分析",
    runMode: "运行模式",
    quick: "快速",
    full: "完整",
    quickHint: "快速生成完整产物，保留核心 LLM 阶段和轻量图表选择。",
    fullHint: "更完整的增强模式，包含图表意图规划、修订和更多反思。",
    chooseCsv: "选择销售 CSV",
    uploadFile: "上传文件",
    csvHint: "支持订单明细、零售流水和电商销售数据。",
    noFileSelected: "未选择文件",
    artifactDownloads: "产物下载",
    notebook: "Notebook",
    businessReview: "经营复盘",
    clientReport: "客户报告",
    reportPreview: "报告预览",
    previewReport: "预览报告",
    reset: "重置",
    waitingForFile: "等待上传文件。",
    waiting: "等待",
    analysisCompleted: "分析完成",
    analysisFailed: "分析失败",
    language: "语言",
    pages: "页面",
    open: "打开",
    close: "关闭",
    closePreview: "关闭预览",
    collapsePreview: "收起预览",
    expand: "展开",
    viewTaskDetails: "查看任务详情",
    taskStatus: "任务状态",
    moduleCount: "模块数量",
    fieldMapping: "字段识别",
    sourceField: "原始字段",
    mappedMeaning: "识别语义",
    waitingForUpload: "等待上传。",
    noFieldMapping: "暂无字段识别结果。",
    previewEmpty: "分析完成后自动打开报告。",
    analyzing: "分析中...",
    creatingTask: "创建任务",
    uploadingFields: "上传并识别字段",
    planGenerated: "分析计划已生成",
    runningAgent: "执行 Agent 分析",
    buildingReports: "生成报告",
    restoredLatestTask: "已恢复最近任务。",
    done: "完成",
    failed: "失败",
    running: "运行中",
    agentFlow: "Agent 流程",
    stepCreateTitle: "创建任务",
    stepCreateNote: "准备运行目录和任务记录。",
    stepUploadTitle: "上传与字段识别",
    stepUploadNote: "识别日期、销售额、商品、利润等字段。",
    stepPlanTitle: "分析计划",
    stepPlanNote: "选择趋势、商品、订单、利润和建模模块。",
    stepRunTitle: "执行分析",
    stepRunNote: "运行 Python 分析、图表选择、Notebook 生成与执行。",
    stepReportsTitle: "生成报告",
    stepReportsNote: "整理客户报告、Notebook 和经营复盘。",
    detailDescription: "这里显示任务状态、字段映射、模块数量、trace 和错误信息。",
  },
};

export const elements = {
  fileInput,
  fileName,
  form: document.getElementById("analysis-form"),
  resetButton: document.getElementById("reset-button"),
  openPreviewButton,
  closePreviewButton: document.getElementById("close-preview-button"),
  drawerTab: document.getElementById("drawer-tab"),
  expandPreviewButton,
  analysisTab,
  detailTab,
  viewDetailLink,
  languageInputs,
};

export function selectedProfile() {
  return document.querySelector("input[name='llm-profile']:checked")?.value || "quick";
}

export function selectedOutputLanguage() {
  return document.querySelector("input[name='output-language']:checked")?.value || "en";
}

function normalizedLanguage(language) {
  return language === "zh-CN" ? "zh-CN" : "en";
}

export function t(key) {
  const language = normalizedLanguage(selectedOutputLanguage());
  return UI_TEXT[language][key] || UI_TEXT.en[key] || key;
}

function refreshFileName() {
  const file = fileInput.files?.[0];
  fileName.textContent = file?.name || t("noFileSelected");
}

function refreshDownloadPlaceholders() {
  const activeLinks = [...downloadList.querySelectorAll("a:not(.disabled)")];
  if (activeLinks.length) {
    const labels = [t("notebook"), t("businessReview"), "HTML", "JSON"];
    for (const [index, link] of activeLinks.entries()) {
      link.textContent = labels[index] || link.textContent;
    }
    return;
  }
  downloadList.innerHTML = `
    <a class="download disabled" href="#">${t("notebook")}</a>
    <a class="download disabled" href="#">${t("businessReview")}</a>
    <a class="download disabled" href="#">HTML</a>
    <a class="download disabled" href="#">JSON</a>
  `;
}

function refreshStepTimes() {
  for (const step of steps) {
    const state = step.dataset.stepState || "";
    const timeNode = step.querySelector(".step-time");
    if (!state) {
      timeNode.textContent = "-";
    } else if (state === "done") {
      timeNode.textContent = t("done");
    } else if (state === "failed") {
      timeNode.textContent = t("failed");
    } else {
      timeNode.textContent = t("running");
    }
  }
}

function refreshPreviewButtonText() {
  const isOpen = appShell.classList.contains("preview-open");
  openPreviewButton.textContent = isOpen ? t("closePreview") : t("previewReport");
}

function refreshStatusText() {
  const key = statusNode.dataset.statusKey;
  if (key) {
    statusNode.textContent = t(key);
    statusPill.textContent = key === "waitingForFile" ? t("waiting") : t(key);
  }
}

export function applyUiLanguage() {
  const language = normalizedLanguage(selectedOutputLanguage());
  document.documentElement.lang = language === "zh-CN" ? "zh-CN" : "en";
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  document.getElementById("language-switch")?.setAttribute("aria-label", t("language"));
  document.querySelector(".nav")?.setAttribute("aria-label", t("pages"));
  refreshFileName();
  refreshDownloadPlaceholders();
  refreshStepTimes();
  refreshPreviewButtonText();
  refreshStatusText();
  expandPreviewButton.textContent = appShell.classList.contains("preview-expanded")
    ? t("collapsePreview")
    : t("expand");
  if (fieldTable.dataset.emptyState === "waiting") {
    fieldTable.innerHTML = `<tr><td colspan="2">${t("waitingForUpload")}</td></tr>`;
  } else if (fieldTable.dataset.emptyState === "no-mapping") {
    fieldTable.innerHTML = `<tr><td colspan="2">${t("noFieldMapping")}</td></tr>`;
  }
}

export function setOutputLanguage(language) {
  const normalized = normalizedLanguage(language);
  for (const input of languageInputs) {
    input.checked = input.value === normalized;
  }
  applyUiLanguage();
}

export function openPreview() {
  appShell.classList.add("preview-open");
  openPreviewButton.textContent = t("closePreview");
  openPreviewButton.setAttribute("aria-expanded", "true");
}

export function closePreview() {
  appShell.classList.remove("preview-open");
  openPreviewButton.textContent = t("previewReport");
  openPreviewButton.setAttribute("aria-expanded", "false");
}

export function togglePreview() {
  if (appShell.classList.contains("preview-open")) {
    closePreview();
    return;
  }
  openPreview();
}

export function togglePreviewSize() {
  const expanded = !appShell.classList.contains("preview-expanded");
  appShell.classList.toggle("preview-expanded", expanded);
  expandPreviewButton.textContent = expanded ? t("collapsePreview") : t("expand");
  expandPreviewButton.setAttribute("aria-pressed", String(expanded));
}

export function setPage(page) {
  const currentPage = page === "detail" ? "detail" : "analysis";
  const analysis = currentPage === "analysis";
  const detail = currentPage === "detail";
  analysisPage.classList.toggle("hidden", !analysis);
  resultsPage.classList.remove("active");
  detailPage.classList.toggle("active", detail);
  appShell.classList.remove("results-mode");
  analysisTab.setAttribute("aria-current", analysis ? "page" : "false");
  detailTab.setAttribute("aria-current", detail ? "page" : "false");
}

export function setRunning(isRunning) {
  appShell.classList.toggle("running", isRunning);
  submitButton.disabled = isRunning;
  fileInput.disabled = isRunning;
  for (const input of profileInputs) {
    input.disabled = isRunning;
  }
  for (const input of languageInputs) {
    input.disabled = isRunning;
  }
  submitButton.textContent = isRunning ? t("analyzing") : t("startAnalysis");
}

export function setStatus(text, isError = false, statusKey = "") {
  if (statusKey) {
    statusNode.dataset.statusKey = statusKey;
  } else {
    delete statusNode.dataset.statusKey;
  }
  statusNode.textContent = statusKey ? t(statusKey) : text;
  statusNode.style.color = isError ? "var(--danger)" : "";
  statusPill.textContent = statusKey === "waitingForFile" ? t("waiting") : statusNode.textContent;
}

export function setStep(index, state) {
  currentStepIndex = index;
  const step = steps[index];
  step.className = `step ${state}`;
  step.dataset.stepState = state;
  step.querySelector(".step-time").textContent =
    state === "done" ? t("done") : state === "failed" ? t("failed") : t("running");
}

export function resetSteps() {
  for (const step of steps) {
    step.className = "step";
    delete step.dataset.stepState;
    step.querySelector(".step-time").textContent = "-";
  }
}

export function renderDownloads(taskId) {
  const items = [
    [t("notebook"), `/api/v1/analysis/tasks/${taskId}/artifacts/notebook`],
    [t("businessReview"), `/api/v1/analysis/tasks/${taskId}/artifacts/business-review`],
    ["HTML", `/api/v1/analysis/tasks/${taskId}/artifacts/client-report-html`],
    ["JSON", `/api/v1/analysis/tasks/${taskId}/artifacts/client-report-json`],
  ];
  downloadList.innerHTML = "";
  for (const [label, href] of items) {
    const link = document.createElement("a");
    link.className = "download";
    link.href = href;
    link.textContent = label;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    downloadList.appendChild(link);
  }
}

export function setPreviewUrl(taskId) {
  const previewUrl = `/api/v1/analysis/tasks/${taskId}/artifacts/client-report-html`;
  previewFrame.src = previewUrl;
  previewFrame.classList.remove("hidden");
  previewEmpty.classList.add("hidden");
  openReportLink.href = previewUrl;
}

export function renderFieldMapping(mapping) {
  const entries = Object.entries(mapping || {});
  fieldTable.innerHTML = "";
  if (!entries.length) {
    fieldTable.dataset.emptyState = "no-mapping";
    fieldTable.innerHTML = `<tr><td colspan="2">${t("noFieldMapping")}</td></tr>`;
    return;
  }
  delete fieldTable.dataset.emptyState;
  for (const [original, canonical] of entries) {
    const row = document.createElement("tr");
    const originalCell = document.createElement("td");
    const canonicalCell = document.createElement("td");
    originalCell.textContent = original;
    canonicalCell.textContent = canonical;
    row.append(originalCell, canonicalCell);
    fieldTable.appendChild(row);
  }
}

export function resetUi({ clearFile = false } = {}) {
  closePreview();
  resetSteps();
  setStatus(t("waitingForFile"), false, "waitingForFile");
  viewDetailLink.classList.add("hidden");
  if (clearFile) {
    fileInput.value = "";
  }
  refreshFileName();
  previewFrame.classList.add("hidden");
  previewFrame.removeAttribute("src");
  previewEmpty.classList.remove("hidden");
  openReportLink.href = "#";
  refreshDownloadPlaceholders();
  detailStatus.textContent = "-";
  detailProfile.textContent = "-";
  detailModuleCount.textContent = "-";
  fieldTable.dataset.emptyState = "waiting";
  fieldTable.innerHTML = `<tr><td colspan="2">${t("waitingForUpload")}</td></tr>`;
}

export function setDetailState({ status = "-", profile = "-", moduleCount = "-", fieldMapping } = {}) {
  detailStatus.textContent = status;
  detailProfile.textContent = profile;
  detailModuleCount.textContent = moduleCount;
  if (fieldMapping !== undefined) {
    renderFieldMapping(fieldMapping);
  }
}

export function restoreTaskView(task) {
  renderDownloads(task.taskId);
  setPreviewUrl(task.taskId);
  setDetailState({
    status: task.status || "completed",
    profile: task.profile || "-",
    moduleCount: task.moduleCount || "-",
    fieldMapping: task.fieldMapping,
  });
  for (let index = 0; index < steps.length; index += 1) {
    setStep(index, "done");
  }
  setStatus(t("restoredLatestTask"), false, "restoredLatestTask");
}

export function showFailure(message) {
  setStep(currentStepIndex, "failed");
  detailStatus.textContent = "failed";
  viewDetailLink.classList.remove("hidden");
  setStatus(message || t("analysisFailed"), true);
}

applyUiLanguage();
