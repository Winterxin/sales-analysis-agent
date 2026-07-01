import { cancelTask, createTask, getTask, runAnalysis, uploadFile } from "./api.js?v=runtimev3";
import { clearLastTask, readLastTask, saveLastTask } from "./storage.js?v=runtimev3";
import {
  closePreview,
  elements,
  openPreview,
  applyUiLanguage,
  renderFieldMapping,
  resetUi,
  restoreTaskView,
  selectedOutputLanguage,
  selectedProfile,
  setDetailState,
  setOutputLanguage,
  setPage,
  setRunning,
  setStatus,
  setStep,
  setStopEnabled,
  showFailure,
  t,
  togglePreview,
  togglePreviewSize,
} from "./ui.js?v=runtimev3";
import { validateSelectedFile } from "./validation.js?v=runtimev3";

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled"]);
const POLL_INTERVAL_MS = 1000;

let currentTask = null;
let pollTimer = null;
let pollInFlight = false;
let previewOpenedForTaskId = "";

function taskIdFromUrl() {
  return new URLSearchParams(window.location.search).get("task_id");
}

function updateUrlTaskId(taskId) {
  const url = new URL(window.location.href);
  if (taskId) {
    url.searchParams.set("task_id", taskId);
  } else {
    url.searchParams.delete("task_id");
  }
  window.history.replaceState({}, "", url);
}

function savedTaskMetadata() {
  const saved = readLastTask();
  return saved && typeof saved === "object" ? saved : {};
}

function mergeTaskMetadata(task) {
  const saved = savedTaskMetadata();
  return {
    ...task,
    taskId: task.task_id || task.taskId,
    profile: saved.profile || task.profile || "-",
    outputLanguage: saved.outputLanguage || selectedOutputLanguage(),
    fieldMapping: saved.fieldMapping,
    moduleCount: saved.moduleCount || task.artifact_manifest?.report?.module_count || "-",
  };
}

function persistTask(task) {
  const merged = mergeTaskMetadata(task);
  saveLastTask({
    taskId: merged.taskId,
    profile: merged.profile,
    outputLanguage: merged.outputLanguage,
    status: merged.status,
    moduleCount: merged.moduleCount,
    fieldMapping: merged.fieldMapping,
  });
}

function stopPolling() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
  }
  pollTimer = null;
  pollInFlight = false;
}

function schedulePoll(taskId) {
  stopPolling();
  pollTimer = window.setTimeout(() => pollTaskStatus(taskId), POLL_INTERVAL_MS);
}

async function pollTaskStatus(taskId) {
  if (pollInFlight) {
    schedulePoll(taskId);
    return;
  }
  pollInFlight = true;
  try {
    const task = await getTask(taskId);
    applyTaskState(task);
    if (!TERMINAL_STATUSES.has(task.status)) {
      schedulePoll(taskId);
    }
  } catch (error) {
    showFailure(error.message || t("analysisFailed"));
    setRunning(false);
  } finally {
    pollInFlight = false;
  }
}

function startPolling(taskId) {
  schedulePoll(taskId);
}

function applyTaskState(task) {
  currentTask = mergeTaskMetadata(task);
  persistTask(currentTask);
  restoreTaskView(currentTask);
  if (TERMINAL_STATUSES.has(currentTask.status)) {
    stopPolling();
    if (currentTask.status === "completed" && previewOpenedForTaskId !== currentTask.taskId) {
      previewOpenedForTaskId = currentTask.taskId;
      openPreview();
    }
  }
}

async function restoreLastTask() {
  const params = new URLSearchParams(window.location.search);
  const requestedPage = params.get("page");
  const fallbackPage = requestedPage === "detail" ? "detail" : "analysis";
  const saved = savedTaskMetadata();
  const taskId = taskIdFromUrl() || saved.taskId;
  if (saved.outputLanguage) {
    setOutputLanguage(saved.outputLanguage);
  }
  if (!taskId) {
    setPage(fallbackPage);
    return;
  }

  try {
    const task = await getTask(taskId);
    updateUrlTaskId(taskId);
    applyTaskState(task);
    if (!TERMINAL_STATUSES.has(task.status)) {
      startPolling(taskId);
    }
  } catch {
    clearLastTask();
    updateUrlTaskId("");
    resetUi();
  }
  setPage(fallbackPage);
}

function updateSelectedFile() {
  const file = elements.fileInput.files?.[0];
  elements.fileName.textContent = file?.name || t("noFileSelected");

  const validationError = validateSelectedFile(file);
  if (validationError && elements.fileInput.files?.length) {
    setStatus(validationError, true);
    return;
  }
  setStatus(t("waitingForFile"), false, "waitingForFile");
}

async function handleAnalysisSubmit(event) {
  event.preventDefault();
  const file = elements.fileInput.files?.[0];
  const validationError = validateSelectedFile(file);
  if (validationError) {
    setStatus(validationError, true);
    return;
  }

  stopPolling();
  resetUi();
  setRunning(true);

  const profile = selectedProfile();
  const outputLanguage = selectedOutputLanguage();
  let taskId = "";
  let fieldMapping = {};
  setDetailState({ profile });

  try {
    setStep(0, "active");
    setStatus(t("creatingTask"), false, "creatingTask");
    const createBody = await createTask();
    taskId = createBody.task_id;
    updateUrlTaskId(taskId);
    setStep(0, "done");

    setStep(1, "active");
    setStatus(t("uploadingFields"), false, "uploadingFields");
    const uploadBody = await uploadFile(taskId, file);
    fieldMapping = uploadBody.schema_mapping?.field_mapping || {};
    renderFieldMapping(fieldMapping);
    const moduleCount = uploadBody.analysis_plan?.analysis_plan?.length || "-";
    saveLastTask({
      taskId,
      profile,
      outputLanguage,
      status: "uploaded",
      moduleCount,
      fieldMapping,
    });
    setStep(1, "done");

    setStep(2, "done");
    setDetailState({
      profile,
      moduleCount,
      fieldMapping,
    });

    const runBody = await runAnalysis(taskId, profile, outputLanguage);
    applyTaskState({
      ...runBody,
      task_id: taskId,
      profile,
      outputLanguage,
      fieldMapping,
      moduleCount,
    });
    startPolling(taskId);
  } catch (error) {
    showFailure(error.message || t("analysisFailed"));
    setRunning(false);
  }
}

async function handleStopClick() {
  const taskId = currentTask?.taskId || taskIdFromUrl();
  if (!taskId) {
    return;
  }
  setStopEnabled(false, { stopping: true });
  try {
    const task = await cancelTask(taskId);
    setStatus(t("stopRequested"), false);
    applyTaskState(task);
    startPolling(taskId);
  } catch (error) {
    showFailure(error.message || t("analysisFailed"));
  }
}

elements.analysisTab.addEventListener("click", () => setPage("analysis"));
elements.detailTab.addEventListener("click", () => setPage("detail"));
elements.viewDetailLink.addEventListener("click", () => setPage("detail"));
elements.fileInput.addEventListener("change", updateSelectedFile);
elements.openPreviewButton.addEventListener("click", togglePreview);
elements.drawerTab.addEventListener("click", openPreview);
elements.closePreviewButton.addEventListener("click", closePreview);
elements.expandPreviewButton.addEventListener("click", togglePreviewSize);
elements.stopButton.addEventListener("click", handleStopClick);
for (const input of elements.languageInputs) {
  input.addEventListener("change", applyUiLanguage);
}
elements.resetButton.addEventListener("click", () => {
  stopPolling();
  clearLastTask();
  updateUrlTaskId("");
  resetUi({ clearFile: true });
});
elements.form.addEventListener("submit", handleAnalysisSubmit);

restoreLastTask();
