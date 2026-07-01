from __future__ import annotations

from fastapi.testclient import TestClient


def test_app_page_renders_upload_form(client: TestClient) -> None:
    response = client.get("/app")

    assert response.status_code == 200
    assert "Sales Analysis Agent" in response.text
    assert 'type="file"' in response.text
    assert "Start Analysis" in response.text
    assert 'id="stop-button"' in response.text
    assert "Run Mode" in response.text
    assert "Report Language" not in response.text
    assert 'id="language-switch"' in response.text
    assert "Language" in response.text
    assert 'value="en"' in response.text
    assert 'value="zh-CN"' in response.text
    assert "Quick" in response.text
    assert "Full" in response.text
    assert "Fast run with core LLM stages." in response.text
    assert "Deeper run with chart planning and revision." in response.text
    assert "Fast complete artifacts with core LLM stages" not in response.text
    assert "Upload File" in response.text
    assert "Artifact Downloads" in response.text
    assert "Business Review" in response.text
    assert "HTML" in response.text
    assert "JSON" in response.text
    assert "Task Details" in response.text
    assert 'id="llm-status-pill"' in response.text
    assert 'class="live-activity' in response.text
    assert "Current Stage" in response.text
    assert "LLM Status" in response.text
    assert 'id="results-tab"' not in response.text
    assert "results-page" in response.text
    assert "The report preview opens after analysis completes." in response.text
    assert "默认保持居中" not in response.text
    assert "/static/ui/app.css" in response.text
    assert "/static/ui/results.css" in response.text
    assert 'type="module"' in response.text
    assert "/static/ui/app.js" in response.text


def test_app_page_includes_client_report_preview_drawer(client: TestClient) -> None:
    response = client.get("/app")

    assert response.status_code == 200
    assert "Report Preview" in response.text
    assert "report-preview-frame" in response.text
    assert "expand-preview-button" in response.text
    assert "Expand" in response.text


def test_app_static_assets_are_served(client: TestClient) -> None:
    css_response = client.get("/static/ui/app.css")
    app_js_response = client.get("/static/ui/app.js")
    ui_js_response = client.get("/static/ui/ui.js")
    storage_js_response = client.get("/static/ui/storage.js")
    validation_js_response = client.get("/static/ui/validation.js")
    api_js_response = client.get("/static/ui/api.js")
    results_js_response = client.get("/static/ui/results.js")
    results_ui_js_response = client.get("/static/ui/results-ui.js")
    results_css_response = client.get("/static/ui/results.css")

    assert css_response.status_code == 200
    assert ".preview-drawer" in css_response.text
    assert ".preview-open.preview-expanded" in css_response.text
    assert ".file-button" in css_response.text
    assert ".secondary.danger" in css_response.text
    assert ".llm-status-pill" in css_response.text
    assert ".live-activity" in css_response.text
    assert ".status-row p" in css_response.text
    assert "display: none" in css_response.text
    assert ".step.cancelled" in css_response.text
    assert "white-space: nowrap" in css_response.text
    assert app_js_response.status_code == 200
    assert "restoreLastTask" in app_js_response.text
    assert "pollTaskStatus" in app_js_response.text
    assert "cancelTask" in app_js_response.text
    assert "validateSelectedFile" in app_js_response.text
    assert "setRunning" in app_js_response.text
    assert 'from "./ui.js?v=runtimev3"' in app_js_response.text
    assert 'from "./api.js?v=runtimev3"' in app_js_response.text
    assert 'from "./storage.js?v=runtimev3"' in app_js_response.text
    assert 'from "./validation.js?v=runtimev3"' in app_js_response.text
    assert "setOutputLanguage" in app_js_response.text
    assert "UI_TEXT" in ui_js_response.text
    assert "Language" in ui_js_response.text
    assert "语言" in ui_js_response.text
    assert "./results.js" not in app_js_response.text
    assert "loadAnalysisResults" not in app_js_response.text
    assert 'setPage("results")' not in app_js_response.text
    assert ui_js_response.status_code == 200
    assert "client-report-html" in ui_js_response.text
    assert "setLlmStatus" in ui_js_response.text
    assert "setStopVisible" in ui_js_response.text
    assert "setLiveActivity" in ui_js_response.text
    assert "localizedActivityMessage" in ui_js_response.text
    assert '"Working on the current analysis stage.": "正在处理当前分析阶段。"' in ui_js_response.text
    assert 'task.status === "cancelled"' in ui_js_response.text
    assert 'applyStepState(activeIndex, "cancelled")' in ui_js_response.text
    assert 'setStatus(statusLabel("cancelled"), false)' in ui_js_response.text
    assert 'setStatus(statusLabel("cancelled"), true)' not in ui_js_response.text
    assert "Current Stage" in ui_js_response.text
    assert "LLM Status" in ui_js_response.text
    assert "togglePreviewSize" in ui_js_response.text
    assert "showFailure" in ui_js_response.text
    assert "resultsTab" not in ui_js_response.text
    assert 'page === "results"' not in ui_js_response.text
    assert "Analyzing..." in ui_js_response.text
    assert "Collapse Preview" in ui_js_response.text
    assert "收起预览" in ui_js_response.text
    assert "No file selected" in ui_js_response.text
    assert "未选择文件" in ui_js_response.text
    assert "/artifacts/notebook" in ui_js_response.text
    assert "/artifacts/business-review" in ui_js_response.text
    assert "/artifacts/client-report-html" in ui_js_response.text
    assert "/artifacts/client-report-json" in ui_js_response.text
    assert storage_js_response.status_code == 200
    assert "sales-analysis:last-task" in storage_js_response.text
    assert validation_js_response.status_code == 200
    assert "File size must not exceed" in validation_js_response.text
    assert "Please choose a CSV file" in validation_js_response.text
    assert api_js_response.status_code == 200
    assert "runAnalysis" in api_js_response.text
    assert "getTask" in api_js_response.text
    assert "cancelTask" in api_js_response.text
    assert "method: \"POST\"" in api_js_response.text
    assert "output_language" in api_js_response.text
    assert "getAnalysisResults" in api_js_response.text
    assert "WebSocket" not in app_js_response.text + ui_js_response.text + api_js_response.text
    assert "EventSource" not in app_js_response.text + ui_js_response.text + api_js_response.text
    assert "/cancel" in api_js_response.text
    assert results_js_response.status_code == 200
    assert "loadAnalysisResults" in results_js_response.text
    assert results_ui_js_response.status_code == 200
    assert "renderResultsPage" in results_ui_js_response.text
    assert "toggle-full-text" in results_ui_js_response.text
    assert results_css_response.status_code == 200
    assert ".results-page" in results_css_response.text
    assert ".chart-canvas" in results_css_response.text
