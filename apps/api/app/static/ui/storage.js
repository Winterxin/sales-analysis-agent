const LAST_TASK_KEY = "sales-analysis:last-task";

export function saveLastTask(task) {
  try {
    localStorage.setItem(LAST_TASK_KEY, JSON.stringify(task));
  } catch {
    // localStorage may be unavailable in private or restricted browser contexts.
  }
}

export function readLastTask() {
  try {
    return JSON.parse(localStorage.getItem(LAST_TASK_KEY) || "null");
  } catch {
    return null;
  }
}

export function clearLastTask() {
  try {
    localStorage.removeItem(LAST_TASK_KEY);
  } catch {
    // Nothing to clear when storage is unavailable.
  }
}
