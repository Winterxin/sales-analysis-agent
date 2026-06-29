import { createTask, runAnalysis, uploadFile } from "./api.js?v=g75a4";
import { clearLastTask, readLastTask, saveLastTask } from "./storage.js?v=g75a4";
import {
  closePreview,
  elements,
  openPreview,
  applyUiLanguage,
  renderDownloads,
  renderFieldMapping,
  resetUi,
  restoreTaskView,
  selectedOutputLanguage,
  selectedProfile,
  setDetailState,
  setOutputLanguage,
  setPage,
  setPreviewUrl,
  setRunning,
  setStatus,
  setStep,
  showFailure,
  t,
  togglePreview,
  togglePreviewSize,
} from "./ui.js?v=g75a4";
import { validateSelectedFile } from "./validation.js?v=g75a4";

function restoreLastTask() {
  const params = new URLSearchParams(window.location.search);
  const requestedPage = params.get("page");
  const fallbackPage = requestedPage === "detail" ? "detail" : "analysis";
  const taskIdFromUrl = params.get("task_id");
  if (taskIdFromUrl) {
    restoreTaskView({ taskId: taskIdFromUrl, status: "completed" });
    setPage(fallbackPage);
    return;
  }

  const task = readLastTask();
  if (task?.taskId) {
    setOutputLanguage(task.outputLanguage || "en");
    restoreTaskView(task);
    setPage(fallbackPage);
    return;
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
    setStep(0, "done");

    setStep(1, "active");
    setStatus(t("uploadingFields"), false, "uploadingFields");
    const uploadBody = await uploadFile(taskId, file);
    fieldMapping = uploadBody.schema_mapping?.field_mapping || {};
    renderFieldMapping(fieldMapping);
    setStep(1, "done");

    setStep(2, "active");
    setStatus(t("planGenerated"), false, "planGenerated");
    setDetailState({
      profile,
      moduleCount: uploadBody.analysis_plan?.analysis_plan?.length || "-",
    });
    setStep(2, "done");

    setStep(3, "active");
    setStatus(t("runningAgent"), false, "runningAgent");
    const runBody = await runAnalysis(taskId, profile, outputLanguage);
    setStep(3, "done");

    setStep(4, "active");
    setStatus(t("buildingReports"), false, "buildingReports");
    renderDownloads(taskId);
    setPreviewUrl(taskId);
    setDetailState({
      status: "completed",
      profile,
      moduleCount: runBody.report?.module_count || "-",
      fieldMapping,
    });
    saveLastTask({
      taskId,
      profile,
      outputLanguage,
      status: "completed",
      moduleCount: runBody.report?.module_count || "-",
      fieldMapping,
    });
    setStep(4, "done");
    setStatus(t("analysisCompleted"), false, "analysisCompleted");
    openPreview();
  } catch (error) {
    showFailure(error.message || t("analysisFailed"));
  } finally {
    setRunning(false);
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
for (const input of elements.languageInputs) {
  input.addEventListener("change", applyUiLanguage);
}
elements.resetButton.addEventListener("click", () => {
  clearLastTask();
  resetUi({ clearFile: true });
});
elements.form.addEventListener("submit", handleAnalysisSubmit);

restoreLastTask();
