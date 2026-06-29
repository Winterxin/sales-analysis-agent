export const MAX_CSV_BYTES = 50 * 1024 * 1024;

export function validateSelectedFile(file) {
  if (!file) {
    return "Please choose a CSV file first.";
  }
  if (!file.name.toLowerCase().endsWith(".csv")) {
    return "Please choose a CSV file (.csv).";
  }
  if (file.size > MAX_CSV_BYTES) {
    return "File size must not exceed 50 MB.";
  }
  return "";
}
