async function parseResponse(response, fallbackMessage) {
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || fallbackMessage);
  }
  return body;
}

export async function createTask() {
  const response = await fetch("/api/v1/analysis/tasks", { method: "POST" });
  return parseResponse(response, "Failed to create task");
}

export async function uploadFile(taskId, file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`/api/v1/analysis/tasks/${taskId}/upload`, {
    method: "POST",
    body: formData,
  });
  return parseResponse(response, "Upload failed");
}

export async function runAnalysis(taskId, profile, outputLanguage = "en") {
  const params = new URLSearchParams({
    llm_profile: profile || "quick",
    output_language: outputLanguage || "en",
  });
  const response = await fetch(`/api/v1/analysis/tasks/${taskId}/run?${params.toString()}`, {
    method: "POST",
  });
  return parseResponse(response, "Analysis failed");
}

export async function getAnalysisResults(taskId) {
  const response = await fetch(`/api/v1/analysis/tasks/${taskId}/results`);
  return parseResponse(response, "Failed to load analysis results");
}
