// THAI-MOD Forensic Console: Insult & Defamation Detection (Slide 5 Workflow)
const commentInput = document.getElementById("commentInput");
const analyzeButton = document.getElementById("analyzeButton");
const historyList = document.getElementById("historyList");
const triageBanner = document.getElementById("triageBanner");
const resultAccent = document.getElementById("resultAccent");
const actionTag = document.getElementById("actionTag");
const processingPanel = document.getElementById("processingPanel");
const processingLabel = document.getElementById("processingLabel");

const predictedLabel = document.getElementById("predictedLabel");
const recommendation = document.getElementById("recommendation");
const insultScore = document.getElementById("insultScore");
const defamationScore = document.getElementById("defamationScore");
const confidence = document.getElementById("confidence");

const targetStatus = document.getElementById("targetStatus");
const targetEntities = document.getElementById("targetEntities");
const allegationStatus = document.getElementById("allegationStatus");
const allegationCues = document.getElementById("allegationCues");
const hedgedBadge = document.getElementById("hedgedBadge");
const explanationText = document.getElementById("explanationText");
const highlightedDisplay = document.getElementById("highlightedDisplay");
const disclaimerText = document.getElementById("disclaimerText");

const modelVersion = document.getElementById("modelVersion");
const healthStatus = document.getElementById("healthStatus");

const recentHistory = [];
let currentPrediction = null;

async function ensureAnalyzerAccess() {
  try {
    const response = await fetch("/api/auth/me");
    if (!response.ok) return;
    const payload = await response.json();
    if (payload.protect_analyzer && !payload.authenticated) {
      window.location.assign("/login?next=/");
    }
  } catch (_error) {
    // Ignore auth check failures
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function toPercent(val) {
  if (val === null || val === undefined || isNaN(val)) return "-";
  return `${Math.round(Number(val) * 100)}%`;
}

function trimText(text, maxLength = 100) {
  if (!text) return "-";
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}...` : text;
}

function renderHighlightedText(rawText, spans) {
  if (!rawText) {
    return '<span class="placeholder-text">ไม่มีข้อความ</span>';
  }
  if (!spans || spans.length === 0) {
    return escapeHtml(rawText);
  }

  // Filter valid spans that fit within rawText
  const validSpans = spans
    .filter((s) => typeof s.start === "number" && typeof s.end === "number" && s.start >= 0 && s.end <= rawText.length && s.start < s.end)
    .sort((a, b) => a.start - b.start);

  if (validSpans.length === 0) {
    return escapeHtml(rawText);
  }

  // Remove overlaps greedily
  const nonOverlapping = [];
  let lastEnd = 0;
  for (const span of validSpans) {
    if (span.start >= lastEnd) {
      nonOverlapping.push(span);
      lastEnd = span.end;
    }
  }

  let html = "";
  let cur = 0;
  for (const span of nonOverlapping) {
    if (span.start > cur) {
      html += escapeHtml(rawText.slice(cur, span.start));
    }
    const cls = span.kind === "target_rule" ? "hl-target" : "hl-allegation";
    const title = span.kind === "target_rule" ? "Target" : "Allegation";
    html += `<mark class="${cls}" title="${title}">${escapeHtml(rawText.slice(span.start, span.end))}</mark>`;
    cur = span.end;
  }
  if (cur < rawText.length) {
    html += escapeHtml(rawText.slice(cur));
  }
  return html;
}

function setPrediction(result, rawText) {
  currentPrediction = result;

  const isAbstained = result.status === "abstained";
  const label = result.predicted_label;

  let bannerClass = "neutral";
  let actionLabel = "Stand by";
  let recText = "ANALYSIS_COMPLETE";

  if (isAbstained) {
    bannerClass = "abstained";
    actionLabel = "Abstained / Review";
    recText = "HUMAN_REVIEW_RECOMMENDED";
    predictedLabel.textContent = `Abstained (${result.reason || "out_of_scope"})`;
  } else if (label === "Defamation") {
    bannerClass = "defamation";
    actionLabel = "Defamation Risk";
    recText = "ESCALATE_LEGAL_REVIEW";
    predictedLabel.textContent = "Defamation (เข้าข่ายหมิ่นประมาท)";
  } else if (label === "Insult") {
    bannerClass = "insult";
    actionLabel = "Insult Detected";
    recText = "FLAG_FOR_MODERATION";
    predictedLabel.textContent = "Insult (ดูหมิ่น/ด่าทอ)";
  } else {
    predictedLabel.textContent = label || "-";
  }

  recommendation.textContent = recText;
  triageBanner.className = `result-banner ${bannerClass}`;
  resultAccent.className = `result-accent ${bannerClass}`;
  actionTag.className = `result-action ${bannerClass}`;
  actionTag.textContent = actionLabel;

  // Probabilities and confidence
  if (result.probabilities) {
    insultScore.textContent = toPercent(result.probabilities.Insult);
    defamationScore.textContent = toPercent(result.probabilities.Defamation);
  } else {
    insultScore.textContent = "-";
    defamationScore.textContent = "-";
  }
  confidence.textContent = toPercent(result.confidence);

  // Target identification
  if (result.target) {
    targetStatus.textContent = result.target.status || "Unknown";
    targetStatus.style.color = result.target.status === "Identified" ? "var(--primary)" : "var(--muted)";
    if (result.target.entities && result.target.entities.length > 0) {
      targetEntities.innerHTML = result.target.entities
        .map((e) => `<span class="tag-pill tag-target">${escapeHtml(e.text)} <small>(${escapeHtml(e.type)})</small></span>`)
        .join("");
    } else {
      targetEntities.innerHTML = '<span class="placeholder-text">ไม่พบเป้าหมายที่ระบุตัวได้</span>';
    }
  }

  // Allegation analysis
  if (result.allegation_analysis) {
    allegationStatus.textContent = result.allegation_analysis.result || "Unclear";
    if (result.allegation_analysis.hedged) {
      hedgedBadge.classList.remove("hidden");
    } else {
      hedgedBadge.classList.add("hidden");
    }

    if (result.allegation_analysis.cues && result.allegation_analysis.cues.length > 0) {
      allegationCues.innerHTML = result.allegation_analysis.cues
        .map((c) => `<span class="tag-pill tag-allegation">${escapeHtml(c.text)} <small>[${escapeHtml(c.category)}]</small></span>`)
        .join("");
    } else {
      allegationCues.innerHTML = '<span class="placeholder-text">ไม่พบคำกล่าวหาที่ชัดเจน</span>';
    }
  }

  // Explanation
  explanationText.textContent = result.explanation || "-";

  // Text Evidence Highlighting
  highlightedDisplay.innerHTML = renderHighlightedText(rawText, result.evidence_spans);

  // Disclaimer
  if (result.disclaimer) {
    disclaimerText.textContent = result.disclaimer;
  }
}

function renderHistory() {
  historyList.innerHTML = "";
  if (recentHistory.length === 0) {
    historyList.innerHTML = `
      <tr class="history-empty">
        <td colspan="5">ยังไม่มีประวัติการตรวจสอบ เริ่มทดสอบโดยเลือกตัวอย่างด้านบน หรือพิมพ์ข้อความแล้วกด Analyze</td>
      </tr>
    `;
    return;
  }

  recentHistory.forEach((item) => {
    const row = document.createElement("tr");
    let badgeClass = "neutral";
    if (item.predicted_label === "Defamation") badgeClass = "defamation";
    else if (item.predicted_label === "Insult") badgeClass = "insult";
    else if (item.status === "abstained") badgeClass = "abstained";

    const displayLabel = item.predicted_label || item.status || "-";
    const confDisplay = item.confidence ? toPercent(item.confidence) : "-";

    row.innerHTML = `
      <td>${new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</td>
      <td><span class="decision-badge ${badgeClass}">${escapeHtml(item.status)}</span></td>
      <td><span class="label-badge ${badgeClass}">${escapeHtml(displayLabel)}</span></td>
      <td>${confDisplay}</td>
      <td class="history-comment">${escapeHtml(trimText(item.text))}</td>
    `;
    historyList.appendChild(row);
  });
}

async function loadSystemInfo() {
  try {
    const [infoRes, healthRes] = await Promise.all([
      fetch("/api/binary/insult-defamation/info"),
      fetch("/api/health"),
    ]);

    if (infoRes.ok) {
      const info = await infoRes.json();
      modelVersion.textContent = info.model_name || "WangchanBERTa V2";
    } else {
      modelVersion.textContent = "WangchanBERTa V2";
    }

    if (healthRes.ok) {
      const health = await healthRes.json();
      healthStatus.textContent = health.status === "ok" ? "Online" : "Degraded";
    } else {
      healthStatus.textContent = "Ready";
    }
  } catch (_error) {
    modelVersion.textContent = "WangchanBERTa V2 (Local)";
    healthStatus.textContent = "Ready";
  }
}

async function analyzeComment() {
  const text = commentInput.value.trim();
  if (!text) {
    commentInput.focus();
    return;
  }

  analyzeButton.disabled = true;
  analyzeButton.textContent = "Screening...";
  processingPanel.classList.remove("hidden");
  processingLabel.textContent = "Running Evasion Normalizer, Target Extractor & WangchanBERTa Classifier...";

  recommendation.textContent = "SCREENING_ACTIVE";
  predictedLabel.textContent = "processing...";
  insultScore.textContent = "-";
  defamationScore.textContent = "-";
  confidence.textContent = "-";
  triageBanner.className = "result-banner neutral";
  resultAccent.className = "result-accent";
  actionTag.className = "result-action neutral";
  actionTag.textContent = "Processing";

  try {
    const response = await fetch("/api/binary/insult-defamation/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Request failed with status ${response.status}`);
    }

    const result = await response.json();
    setPrediction(result, text);

    recentHistory.unshift({
      ...result,
      text,
      timestamp: new Date().toISOString(),
    });
    if (recentHistory.length > 8) {
      recentHistory.pop();
    }
    renderHistory();
  } catch (error) {
    predictedLabel.textContent = "error";
    recommendation.textContent = "API_ERROR";
    explanationText.textContent = `เกิดข้อผิดพลาดในการเรียก API: ${error.message}`;
    triageBanner.className = "result-banner risk";
    resultAccent.className = "result-accent risk";
    actionTag.className = "result-action risk";
    actionTag.textContent = "Check API";
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze Comment (ตรวจวิเคราะห์)";
    processingPanel.classList.add("hidden");
  }
}

analyzeButton.addEventListener("click", analyzeComment);

document.querySelectorAll(".sample-button").forEach((button) => {
  button.addEventListener("click", () => {
    commentInput.value = button.dataset.text;
    analyzeComment();
  });
});

renderHistory();
ensureAnalyzerAccess();
loadSystemInfo();
