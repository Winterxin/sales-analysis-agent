import { getAnalysisResults } from "./api.js";
import { renderResultsError, renderResultsLoading, renderResultsPage } from "./results-ui.js";

let currentTaskId = "";

export function setResultsTask(taskId) {
  currentTaskId = taskId || currentTaskId;
}

export async function loadAnalysisResults(taskId = currentTaskId) {
  if (!taskId) {
    renderResultsError("还没有可展示的分析任务。");
    return;
  }
  currentTaskId = taskId;
  renderResultsLoading();
  try {
    const results = await getAnalysisResults(taskId);
    renderResultsPage(results);
  } catch (error) {
    renderResultsError(error.message || "加载分析结果失败");
  }
}
