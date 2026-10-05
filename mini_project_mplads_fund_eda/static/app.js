// MPLADS Fund Intelligence & Risk Analytics - Minimalist High-Performance Client

let statesCache = [];
let chartStates = null;
let chartRisk = null;
let chartCategories = null;

let mpPage = 1;
const mpLimit = 20;

const PRESETS = {
  high: {
    state: "Madhya Pradesh",
    allocated: 50000000,
    sanc_amt: 48000000,
    sanc_cnt: 60,
    disb_amt: 22500000,
    comp_cnt: 30
  },
  backlog: {
    state: "Uttar Pradesh",
    allocated: 50000000,
    sanc_amt: 45000000,
    sanc_cnt: 55,
    disb_amt: 3500000,
    comp_cnt: 4
  },
  zero: {
    state: "Bihar",
    allocated: 50000000,
    sanc_amt: 0,
    sanc_cnt: 0,
    disb_amt: 0,
    comp_cnt: 0
  },
  avg: {
    state: "Maharashtra",
    allocated: 50000000,
    sanc_amt: 30000000,
    sanc_cnt: 35,
    disb_amt: 14000000,
    comp_cnt: 18
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initListeners();
  loadData();
});

// --- Tab Navigation ---
function initTabs() {
  const btns = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      panels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(`tab-${btn.dataset.tab}`);
      if (target) target.classList.add("active");
    });
  });
}

// --- Listeners ---
function initListeners() {
  document.getElementById("btn-refresh").addEventListener("click", loadData);

  // State search
  document.getElementById("state-search").addEventListener("input", (e) => {
    filterStates(e.target.value);
  });

  // MP Filters
  let searchTimer;
  document.getElementById("mp-search").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      mpPage = 1;
      fetchMps();
    }, 200);
  });

  document.getElementById("mp-state-filter").addEventListener("change", () => {
    mpPage = 1;
    fetchMps();
  });

  document.getElementById("mp-risk-filter").addEventListener("change", () => {
    mpPage = 1;
    fetchMps();
  });

  document.getElementById("mp-sort").addEventListener("change", () => {
    mpPage = 1;
    fetchMps();
  });

  // Pagination
  document.getElementById("btn-prev").addEventListener("click", () => {
    if (mpPage > 1) {
      mpPage--;
      fetchMps();
    }
  });

  document.getElementById("btn-next").addEventListener("click", () => {
    mpPage++;
    fetchMps();
  });

  // Simulator Presets
  document.getElementById("sim-preset").addEventListener("change", (e) => {
    const val = e.target.value;
    if (PRESETS[val]) {
      const p = PRESETS[val];
      document.getElementById("sim-state").value = p.state;
      document.getElementById("sim-allocated").value = p.allocated;
      document.getElementById("sim-sanctioned-amt").value = p.sanc_amt;
      document.getElementById("sim-sanctioned-cnt").value = p.sanc_cnt;
      document.getElementById("sim-disbursed-amt").value = p.disb_amt;
      document.getElementById("sim-completed-cnt").value = p.comp_cnt;
      updateCurrencyHints();
    }
  });

  // Currency Hint Listeners
  ["sim-allocated", "sim-sanctioned-amt", "sim-disbursed-amt"].forEach(id => {
    document.getElementById(id).addEventListener("input", updateCurrencyHints);
  });

  // Sim Form
  document.getElementById("sim-form").addEventListener("submit", handleSimSubmit);

  // Modal Close
  document.getElementById("modal-close").addEventListener("click", closeModal);
  document.getElementById("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") closeModal();
  });
}

function updateCurrencyHints() {
  document.getElementById("hint-alloc").textContent = formatCrores(document.getElementById("sim-allocated").value);
  document.getElementById("hint-sanc").textContent = formatCrores(document.getElementById("sim-sanctioned-amt").value);
  document.getElementById("hint-disb").textContent = formatCrores(document.getElementById("sim-disbursed-amt").value);
}

// --- Data Loading ---
async function loadData() {
  try {
    await Promise.all([
      fetchOverview(),
      fetchStates(),
      fetchMps(),
      fetchCategories()
    ]);
  } catch (err) {
    console.error("Dashboard error:", err);
  }
}

async function fetchOverview() {
  const res = await fetch("/api/overview");
  const d = await res.json();

  document.getElementById("kpi-allocated").textContent = formatCrores(d.total_allocated);
  document.getElementById("kpi-disbursed").textContent = formatCrores(d.total_disbursed);
  document.getElementById("kpi-backlog").textContent = formatCrores(d.total_backlog);
  document.getElementById("kpi-utilization").textContent = `${(d.avg_utilization_rate * 100).toFixed(1)}%`;
  document.getElementById("kpi-median-sub").textContent = `Median: ${(d.median_utilization_rate * 100).toFixed(1)}%`;

  document.getElementById("kpi-risk-count").textContent = `${d.at_risk_count} / ${d.total_mps}`;
  document.getElementById("kpi-risk-pct").textContent = `${d.at_risk_percent}% at-risk`;

  document.getElementById("kpi-anomalies").textContent = `${d.zero_activity_count + d.orphan_count}`;
  document.getElementById("kpi-anomaly-detail").textContent = `${d.zero_activity_count} Inactive | ${d.orphan_count} Orphans`;

  renderRiskDoughnut(d.at_risk_count, d.total_mps - d.at_risk_count);
}

async function fetchStates() {
  const res = await fetch("/api/states");
  statesCache = await res.json();

  // Populate Dropdowns
  const stateFilter = document.getElementById("mp-state-filter");
  const simState = document.getElementById("sim-state");
  stateFilter.innerHTML = '<option value="All">All States</option>';
  simState.innerHTML = '<option value="">Select State</option>';

  const names = statesCache.map(s => s.state).filter(Boolean).sort();
  names.forEach(n => {
    stateFilter.insertAdjacentHTML("beforeend", `<option value="${n}">${n}</option>`);
    simState.insertAdjacentHTML("beforeend", `<option value="${n}">${n}</option>`);
  });

  renderStatesTable(statesCache);
  renderStatesBar(statesCache);
}

function renderStatesTable(states) {
  const tbody = document.getElementById("states-tbody");
  if (!states.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px;">No states found.</td></tr>';
    return;
  }

  tbody.innerHTML = states.map(s => `
    <tr>
      <td><strong>${s.state}</strong></td>
      <td class="num">${s.mp_count}</td>
      <td class="num">${formatCrores(s.total_allocated)}</td>
      <td class="num">${formatCrores(s.total_disbursed)}</td>
      <td class="num"><strong>${(s.avg_utilization * 100).toFixed(1)}%</strong></td>
      <td class="num">${(s.median_utilization * 100).toFixed(1)}%</td>
      <td class="num">${formatCrores(s.avg_backlog)}</td>
      <td>
        <span class="badge ${s.at_risk_pct > 50 ? 'badge-risk' : 'badge-track'}">
          ${s.at_risk_pct.toFixed(0)}% (${s.at_risk_count}/${s.mp_count})
        </span>
      </td>
    </tr>
  `).join("");
}

function filterStates(query) {
  const q = query.toLowerCase().trim();
  renderStatesTable(statesCache.filter(s => s.state.toLowerCase().includes(q)));
}

async function fetchMps() {
  const search = document.getElementById("mp-search").value;
  const state = document.getElementById("mp-state-filter").value;
  const risk = document.getElementById("mp-risk-filter").value;
  const [sortBy, sortOrder] = document.getElementById("mp-sort").value.split(":");

  const params = new URLSearchParams({
    page: mpPage,
    limit: mpLimit,
    sort_by: sortBy,
    sort_order: sortOrder
  });

  if (search) params.append("search", search);
  if (state && state !== "All") params.append("state", state);
  if (risk && risk !== "all") params.append("risk_filter", risk);

  const res = await fetch(`/api/mps?${params.toString()}`);
  const r = await res.json();

  renderMpsTable(r.data);

  // Pagination
  const start = r.total === 0 ? 0 : (r.page - 1) * mpLimit + 1;
  const end = Math.min(r.page * mpLimit, r.total);
  document.getElementById("pagination-label").textContent = `Showing ${start}-${end} of ${r.total} MPs`;
  document.getElementById("page-num").textContent = `${r.page} / ${r.total_pages || 1}`;
  document.getElementById("btn-prev").disabled = (r.page <= 1);
  document.getElementById("btn-next").disabled = (r.page >= r.total_pages || r.total_pages === 0);
}

function renderMpsTable(mps) {
  const tbody = document.getElementById("mps-tbody");
  if (!mps || !mps.length) {
    tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 24px;">No matching records.</td></tr>';
    return;
  }

  tbody.innerHTML = mps.map((mp, idx) => {
    const isAtRisk = mp.at_risk === 1;
    return `
      <tr>
        <td>
          <div style="font-weight: 500;">${mp.honble_members_of_parliament}</div>
          ${mp.has_no_activity ? '<span class="badge badge-neutral" style="font-size:0.65rem;">Inactive</span>' : ''}
          ${mp.has_orphan_completions ? '<span class="badge badge-neutral" style="font-size:0.65rem;">Orphan</span>' : ''}
        </td>
        <td>${mp.state}</td>
        <td>${mp.constituency || "—"}</td>
        <td class="num">${formatCrores(mp.allocated_amount)}</td>
        <td class="num">${formatCrores(mp.total_disbursed_amount)}</td>
        <td class="num"><strong>${(mp.utilization_rate * 100).toFixed(1)}%</strong></td>
        <td class="num">${formatCrores(mp.sanctioned_backlog)}</td>
        <td class="num">${mp.sanctioned_work_count} / ${mp.completed_work_count}</td>
        <td>
          <span class="badge ${isAtRisk ? 'badge-risk' : 'badge-track'}">
            ${isAtRisk ? 'At Risk' : 'On Track'} (${mp.risk_score}%)
          </span>
        </td>
        <td>
          <button class="btn btn-secondary" onclick="showMpModal(${idx})" style="padding: 3px 8px; font-size: 0.72rem;">View</button>
        </td>
      </tr>
    `;
  }).join("");

  window._currentMps = mps;
}

window.showMpModal = function(idx) {
  const mp = window._currentMps[idx];
  if (!mp) return;

  document.getElementById("modal-mp").textContent = mp.honble_members_of_parliament;
  document.getElementById("modal-sub").textContent = `${mp.constituency || 'General'}, ${mp.state}`;

  const isAtRisk = mp.at_risk === 1;
  document.getElementById("modal-body").innerHTML = `
    <div style="margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center;">
      <span class="badge ${isAtRisk ? 'badge-risk' : 'badge-track'}" style="font-size: 0.85rem; padding: 4px 10px;">
        ${isAtRisk ? 'AT RISK' : 'ON TRACK'} (${mp.risk_score}% Probability)
      </span>
      <span style="font-size: 0.78rem; color: var(--text-muted);">Utilization: ${(mp.utilization_rate * 100).toFixed(1)}%</span>
    </div>
    <div class="stat-row">
      <div class="stat-item"><div class="label">Allocated</div><div class="val">${formatCrores(mp.allocated_amount)}</div></div>
      <div class="stat-item"><div class="label">Disbursed</div><div class="val">${formatCrores(mp.total_disbursed_amount)}</div></div>
      <div class="stat-item"><div class="label">Sanction Backlog</div><div class="val">${formatCrores(mp.sanctioned_backlog)}</div></div>
      <div class="stat-item"><div class="label">Works (S / C)</div><div class="val">${mp.sanctioned_work_count} / ${mp.completed_work_count}</div></div>
    </div>
  `;

  document.getElementById("modal").classList.remove("d-none");
};

function closeModal() {
  document.getElementById("modal").classList.add("d-none");
}

async function fetchCategories() {
  const res = await fetch("/api/categories");
  const cats = await res.json();

  const tbody = document.getElementById("cat-tbody");
  if (!cats || !cats.length) {
    tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; padding: 20px;">No categories.</td></tr>';
    return;
  }

  tbody.innerHTML = cats.map(c => `
    <tr>
      <td><strong>${c.category}</strong></td>
      <td class="num">${formatCrores(c.total_sanction_amount)}</td>
      <td class="num">${formatCrores(c.total_disbursed_amount)}</td>
    </tr>
  `).join("");

  renderCategoriesBar(cats);
}

// --- Simulator ---
async function handleSimSubmit(e) {
  e.preventDefault();

  const payload = {
    state: document.getElementById("sim-state").value,
    allocated_amount: parseFloat(document.getElementById("sim-allocated").value) || 0,
    total_sanction_amount: parseFloat(document.getElementById("sim-sanctioned-amt").value) || 0,
    sanctioned_work_count: parseInt(document.getElementById("sim-sanctioned-cnt").value, 10) || 0,
    total_disbursed_amount: parseFloat(document.getElementById("sim-disbursed-amt").value) || 0,
    completed_work_count: parseInt(document.getElementById("sim-completed-cnt").value, 10) || 0
  };

  const btn = document.getElementById("btn-simulate");
  btn.disabled = true;
  btn.textContent = "Computing...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const d = await res.json();

    document.getElementById("sim-placeholder").classList.add("d-none");
    const rBox = document.getElementById("sim-result-box");
    rBox.classList.remove("d-none");

    const isRisk = d.at_risk === 1;
    const tag = document.getElementById("sim-status-tag");
    tag.textContent = d.risk_status;
    tag.className = `status-tag ${isRisk ? 'risk' : 'track'}`;

    const badge = document.getElementById("sim-badge");
    badge.textContent = `${d.risk_probability}% Risk`;
    badge.className = `badge ${isRisk ? 'badge-risk' : 'badge-track'}`;

    document.getElementById("sim-prob-label").textContent = `${d.risk_probability}%`;
    const meter = document.getElementById("sim-meter");
    meter.style.width = `${d.risk_probability}%`;
    meter.className = `meter-fill ${isRisk ? 'risk' : 'track'}`;

    document.getElementById("sim-util-val").textContent = `${d.metrics.utilization_pct}%`;
    document.getElementById("sim-median-val").textContent = `${d.metrics.state_median_pct}%`;
    document.getElementById("sim-backlog-val").textContent = formatCrores(d.metrics.sanctioned_backlog);
    document.getElementById("sim-ratio-val").textContent = `${(d.metrics.completion_ratio * 100).toFixed(1)}%`;

    document.getElementById("sim-factors").innerHTML = d.risk_factors.map(f => `<li>${f}</li>`).join("");
  } catch (err) {
    alert("Inference failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Risk Inference";
  }
}

// --- Chart.js Renderers ---
function renderStatesBar(states) {
  if (!states || !states.length) return;

  const valid = states.filter(s => s.mp_count >= 2);
  const subset = [...valid.slice(0, 5), ...valid.slice(-5).reverse()];

  const labels = subset.map(s => s.state);
  const data = subset.map(s => (s.avg_utilization * 100).toFixed(1));
  const colors = subset.map((s, idx) => idx < 5 ? "#34d399" : "#f87171");

  const ctx = document.getElementById("chart-states-bar").getContext("2d");
  if (chartStates) chartStates.destroy();

  chartStates = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{ data: data, backgroundColor: colors, borderRadius: 4 }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        y: {
          grid: { color: "#27272a" },
          ticks: { color: "#71717a", font: { family: "Inter" }, callback: v => `${v}%` }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#a1a1aa", font: { family: "Inter" }, maxRotation: 40 }
        }
      }
    }
  });
}

function renderRiskDoughnut(riskCount, trackCount) {
  const ctx = document.getElementById("chart-risk-doughnut").getContext("2d");
  if (chartRisk) chartRisk.destroy();

  chartRisk = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["At Risk", "On Track"],
      datasets: [{
        data: [riskCount, trackCount],
        backgroundColor: ["#f87171", "#34d399"],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { color: "#a1a1aa", font: { family: "Inter" } }
        }
      },
      cutout: "75%"
    }
  });
}

function renderCategoriesBar(cats) {
  if (!cats || !cats.length) return;

  const labels = cats.map(c => c.category);
  const sanc = cats.map(c => (c.total_sanction_amount / 1e7).toFixed(2));
  const disb = cats.map(c => (c.total_disbursed_amount / 1e7).toFixed(2));

  const ctx = document.getElementById("chart-categories-bar").getContext("2d");
  if (chartCategories) chartCategories.destroy();

  chartCategories = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: "Sanctioned (₹ Cr)", data: sanc, backgroundColor: "#38bdf8", borderRadius: 4 },
        { label: "Disbursed (₹ Cr)", data: disb, backgroundColor: "#34d399", borderRadius: 4 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: "#a1a1aa", font: { family: "Inter" } } }
      },
      scales: {
        y: {
          grid: { color: "#27272a" },
          ticks: { color: "#71717a", callback: v => `₹${v} Cr` }
        },
        x: {
          grid: { display: false },
          ticks: { color: "#a1a1aa" }
        }
      }
    }
  });
}

// --- Helpers ---
function formatCrores(amount) {
  if (amount === undefined || amount === null || isNaN(amount)) return "₹0.00";
  const num = typeof amount === "string" ? parseFloat(amount) : amount;
  if (isNaN(num)) return "₹0.00";
  const abs = Math.abs(num);
  if (abs >= 1e7) {
    return `${num < 0 ? '-' : ''}₹${(abs / 1e7).toFixed(2)} Cr`;
  } else if (abs >= 1e5) {
    return `${num < 0 ? '-' : ''}₹${(abs / 1e5).toFixed(2)} L`;
  } else {
    return `${num < 0 ? '-' : ''}₹${abs.toLocaleString('en-IN')}`;
  }
}
