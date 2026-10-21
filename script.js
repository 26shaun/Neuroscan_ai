(function () {
  "use strict";

  var MAX_BYTES = 5 * 1024 * 1024;
  var MAX_SCANS = 20;
  var ALLOWED_TYPES = ["image/jpeg", "image/png"];
  var REQUEST_TIMEOUT_MS = 120000;
  var GRADCAM_PREFIX = "data:image/png;base64,";
  var API_BASE = window.location.protocol === "file:" ? "http://localhost:8000" : "";

  var CLASS_LABELS = {
    glioma: "Glioma",
    meningioma: "Meningioma",
    pituitary: "Pituitary Tumor",
    notumor: "No Tumor Detected"
  };
  var CLASS_ORDER = ["glioma", "meningioma", "pituitary", "notumor"];

  var el = {
    dropzone: document.getElementById("dropzone"),
    input: document.getElementById("file-input"),
    count: document.getElementById("scan-count"),
    error: document.getElementById("error"),
    analyze: document.getElementById("analyze-btn"),
    reset: document.getElementById("reset-btn"),
    loading: document.getElementById("loading"),
    resultsEmpty: document.getElementById("results-empty"),
    list: document.getElementById("scan-list"),
    referral: document.getElementById("referral"),
    statTotal: document.getElementById("stat-total"),
    statFlagged: document.getElementById("stat-flagged"),
    statClear: document.getElementById("stat-clear"),
    statLatency: document.getElementById("stat-latency")
  };

  var state = {
    scans: [],
    nextNumber: 1,
    busy: false,
    stats: { total: 0, flagged: 0, clear: 0 }
  };

  function make(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function pad(n) {
    return n < 10 ? "0" + n : String(n);
  }

  function formatBytes(bytes) {
    if (bytes < 1024 * 1024) {
      return (bytes / 1024).toFixed(0) + " KB";
    }
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = false;
  }

  function clearError() {
    el.error.textContent = "";
    el.error.hidden = true;
  }

  function validateFile(file) {
    if (ALLOWED_TYPES.indexOf(file.type) === -1) {
      return "unsupported file type (use JPEG or PNG)";
    }
    if (file.size === 0) {
      return "the file is empty";
    }
    if (file.size > MAX_BYTES) {
      return "file is " + formatBytes(file.size) + ", the limit is 5 MB";
    }
    return null;
  }

  function pendingScans() {
    return state.scans.filter(function (scan) {
      return scan.status === "ready";
    });
  }

  function updateControls() {
    var ready = pendingScans().length;
    var total = state.scans.length;
    el.analyze.disabled = state.busy || ready === 0;
    el.reset.disabled = state.busy || total === 0;
    el.analyze.textContent = state.busy ? "Analyzing\u2026" : "Analyze scans";
    el.dropzone.setAttribute("aria-disabled", String(state.busy));
    el.count.textContent = total === 0
      ? "No scans selected."
      : total + (total === 1 ? " scan" : " scans") + " added, " + ready + " ready to analyze.";
    el.resultsEmpty.hidden = total > 0;
    el.loading.hidden = !state.busy;
    el.referral.hidden = !state.scans.some(function (scan) {
      return scan.status === "done";
    });
  }

  function buildBars(result) {
    var bars = make("ul", "bars");
    CLASS_ORDER.forEach(function (key) {
      var percent = (result.probabilities[key] * 100).toFixed(1);
      var item = make("li", key === result.predicted_class ? "is-top" : "");

      var head = make("div", "bar-head");
      head.appendChild(make("span", "label", CLASS_LABELS[key]));
      head.appendChild(make("span", "value", percent + "%"));

      var track = make("div", "bar-track");
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", CLASS_LABELS[key] + " probability");
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", percent);
      var fill = make("div", "bar-fill");
      fill.style.width = percent + "%";
      track.appendChild(fill);

      item.appendChild(head);
      item.appendChild(track);
      bars.appendChild(item);
    });
    return bars;
  }

  function buildFigure(src, alt, caption) {
    var figure = make("figure");
    var img = make("img");
    img.src = src;
    img.alt = alt;
    figure.appendChild(img);
    figure.appendChild(make("figcaption", "", caption));
    return figure;
  }

  function buildCard(scan) {
    var card = make("li", "scan-card");

    var head = make("div", "scan-head");
    head.appendChild(make("h3", "", scan.label));
    head.appendChild(make("span", "scan-file", scan.file.name + " (" + formatBytes(scan.file.size) + ")"));

    var badgeText = { ready: "Ready", analyzing: "Analyzing\u2026", done: "Complete", error: "Failed" }[scan.status];
    var badgeClass = scan.status === "done" ? "badge is-done" : scan.status === "error" ? "badge is-error" : "badge";
    head.appendChild(make("span", badgeClass, badgeText));

    var remove = make("button", "btn btn-secondary btn-small", "Remove");
    remove.type = "button";
    remove.disabled = state.busy;
    remove.setAttribute("aria-label", "Remove " + scan.label);
    remove.addEventListener("click", function () {
      removeScan(scan.id);
    });
    head.appendChild(remove);
    card.appendChild(head);

    var body = make("div", "scan-body");
    var images = make("div", "scan-images");
    images.appendChild(buildFigure(scan.previewUrl, "MRI scan for " + scan.label, "MRI scan"));

    if (scan.status === "done") {
      var result = scan.result;
      if (result.gradcam && result.gradcam.indexOf(GRADCAM_PREFIX) === 0) {
        images.appendChild(buildFigure(result.gradcam, "Grad-CAM heatmap for " + scan.label, "Grad-CAM"));
      }
      body.appendChild(images);

      var summary = make("div");
      summary.appendChild(make("p", "scan-label", "AI prediction (screening result)"));
      summary.appendChild(make("p", "scan-pred", result.prediction));
      summary.appendChild(make("p", "scan-meta", "Confidence " + (result.confidence * 100).toFixed(1) + "%"));
      body.appendChild(summary);

      var flag = make("div", result.specialist_review ? "flag is-review" : "flag");
      flag.appendChild(make("strong", "", result.specialist_review
        ? "Specialist review recommended"
        : "No flag raised"));
      var reasons = Array.isArray(result.review_reasons) ? result.review_reasons : [];
      if (result.specialist_review && reasons.length > 0) {
        flag.appendChild(make("p", "", reasons.join(". ") + "."));
      } else if (!result.specialist_review) {
        flag.appendChild(make("p", "", "Review by a qualified professional is still required."));
      }
      body.appendChild(flag);

      body.appendChild(buildBars(result));
      if (!result.gradcam && result.gradcam_error) {
        body.appendChild(make("p", "scan-note", result.gradcam_error));
      }
    } else {
      body.appendChild(images);
      if (scan.status === "error") {
        body.appendChild(make("p", "scan-error", scan.error));
      } else if (scan.status === "analyzing") {
        body.appendChild(make("p", "scan-note", "Analysis in progress."));
      } else {
        body.appendChild(make("p", "scan-note", "Not analyzed yet."));
      }
    }

    card.appendChild(body);
    return card;
  }

  function renderScans() {
    var fragment = document.createDocumentFragment();
    state.scans.forEach(function (scan) {
      fragment.appendChild(buildCard(scan));
    });
    el.list.replaceChildren(fragment);
    updateControls();
  }

  function addFiles(fileList) {
    var rejected = [];
    Array.prototype.slice.call(fileList).forEach(function (file) {
      if (state.scans.length >= MAX_SCANS) {
        rejected.push(file.name + ": session limit of " + MAX_SCANS + " scans reached");
        return;
      }
      var problem = validateFile(file);
      if (problem) {
        rejected.push(file.name + ": " + problem);
        return;
      }
      state.scans.push({
        id: state.nextNumber,
        label: "Person " + pad(state.nextNumber),
        file: file,
        previewUrl: URL.createObjectURL(file),
        status: "ready",
        result: null,
        error: null
      });
      state.nextNumber += 1;
    });

    if (rejected.length > 0) {
      showError("Some files were not added. " + rejected.join("; ") + ".");
    } else {
      clearError();
    }
    renderScans();
  }

  function removeScan(id) {
    if (state.busy) {
      return;
    }
    state.scans = state.scans.filter(function (scan) {
      if (scan.id === id) {
        URL.revokeObjectURL(scan.previewUrl);
        return false;
      }
      return true;
    });
    renderScans();
  }

  function clearAll() {
    if (state.busy) {
      return;
    }
    state.scans.forEach(function (scan) {
      URL.revokeObjectURL(scan.previewUrl);
    });
    state.scans = [];
    state.nextNumber = 1;
    el.input.value = "";
    clearError();
    renderScans();
  }

  function extractErrorMessage(response, body) {
    if (body && typeof body.detail === "string") {
      return body.detail;
    }
    if (body && Array.isArray(body.detail) && body.detail.length > 0 && body.detail[0].msg) {
      return body.detail[0].msg;
    }
    return "The server returned an error (HTTP " + response.status + ").";
  }

  function isValidResult(result) {
    if (!result || typeof result !== "object" || !result.probabilities) {
      return false;
    }
    if (typeof result.confidence !== "number" || typeof result.prediction !== "string") {
      return false;
    }
    return CLASS_ORDER.indexOf(result.predicted_class) !== -1 && CLASS_ORDER.every(function (key) {
      return typeof result.probabilities[key] === "number";
    });
  }

  function applyResult(scan, result) {
    if (result && result.status === "error") {
      scan.status = "error";
      scan.error = typeof result.error === "string" ? result.error : "Analysis failed for this scan.";
    } else if (isValidResult(result)) {
      scan.status = "done";
      scan.result = result;
    } else {
      scan.status = "error";
      scan.error = "The server returned an unexpected result for this scan.";
    }
  }

  function updateStats(scans, elapsedMs) {
    scans.forEach(function (scan) {
      if (scan.status !== "done") {
        return;
      }
      state.stats.total += 1;
      if (scan.result.specialist_review) {
        state.stats.flagged += 1;
      }
      if (scan.result.predicted_class === "notumor") {
        state.stats.clear += 1;
      }
    });
    el.statTotal.textContent = String(state.stats.total);
    el.statFlagged.textContent = String(state.stats.flagged);
    el.statClear.textContent = String(state.stats.clear);
    el.statLatency.textContent = (elapsedMs / 1000).toFixed(1) + " s";
  }

  async function analyze() {
    var batch = pendingScans();
    if (batch.length === 0 || state.busy) {
      return;
    }

    clearError();
    state.busy = true;
    batch.forEach(function (scan) {
      scan.status = "analyzing";
    });
    renderScans();

    var controller = new AbortController();
    var timer = window.setTimeout(function () {
      controller.abort();
    }, REQUEST_TIMEOUT_MS);

    try {
      var formData = new FormData();
      batch.forEach(function (scan) {
        formData.append("files", scan.file, scan.file.name);
      });

      var started = performance.now();
      var response = await fetch(API_BASE + "/predict/batch", {
        method: "POST",
        body: formData,
        signal: controller.signal
      });

      var body = null;
      try {
        body = await response.json();
      } catch (parseError) {
        body = null;
      }

      if (!response.ok) {
        throw new Error(extractErrorMessage(response, body));
      }
      if (!body || !Array.isArray(body.results) || body.results.length !== batch.length) {
        throw new Error("The server returned an unexpected response format.");
      }

      batch.forEach(function (scan, index) {
        applyResult(scan, body.results[index]);
      });
      updateStats(batch, performance.now() - started);
    } catch (error) {
      batch.forEach(function (scan) {
        scan.status = "ready";
      });
      if (error.name === "AbortError") {
        showError("The request timed out after 2 minutes. Try fewer scans.");
      } else if (error instanceof TypeError) {
        showError("Cannot reach the screening service. Check that the server is running.");
      } else {
        showError(error.message);
      }
    } finally {
      window.clearTimeout(timer);
      state.busy = false;
      renderScans();
    }
  }

  el.dropzone.addEventListener("click", function () {
    if (!state.busy) {
      el.input.click();
    }
  });

  el.dropzone.addEventListener("keydown", function (event) {
    if ((event.key === "Enter" || event.key === " ") && !state.busy) {
      event.preventDefault();
      el.input.click();
    }
  });

  el.input.addEventListener("change", function () {
    if (el.input.files && el.input.files.length > 0) {
      addFiles(el.input.files);
    }
    el.input.value = "";
  });

  ["dragenter", "dragover"].forEach(function (name) {
    el.dropzone.addEventListener(name, function (event) {
      event.preventDefault();
      if (!state.busy) {
        el.dropzone.classList.add("is-dragover");
      }
    });
  });

  ["dragleave", "dragend"].forEach(function (name) {
    el.dropzone.addEventListener(name, function () {
      el.dropzone.classList.remove("is-dragover");
    });
  });

  el.dropzone.addEventListener("drop", function (event) {
    event.preventDefault();
    el.dropzone.classList.remove("is-dragover");
    var files = event.dataTransfer && event.dataTransfer.files;
    if (!state.busy && files && files.length > 0) {
      addFiles(files);
    }
  });

  window.addEventListener("dragover", function (event) {
    event.preventDefault();
  });
  window.addEventListener("drop", function (event) {
    event.preventDefault();
  });

  el.analyze.addEventListener("click", analyze);
  el.reset.addEventListener("click", clearAll);

  window.addEventListener("beforeunload", function () {
    state.scans.forEach(function (scan) {
      URL.revokeObjectURL(scan.previewUrl);
    });
  });

  renderScans();
})();