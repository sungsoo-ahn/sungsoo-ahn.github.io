---
layout: page
permalink: /gpu-scheduler/
title: GPU Scheduler
description: SPML Lab GPU Cluster Scheduler
nav: false
---

<style>
/* ---- Top bar ---- */
.gpu-topbar { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem; }
.gpu-topbar-left { display: flex; align-items: center; gap: 0.5rem; }

/* ---- Tabs ---- */
.gpu-tabs { display: flex; list-style: none; padding: 0; margin: 0 0 1rem 0; border-bottom: 2px solid var(--global-divider-color); }
.gpu-tab { padding: 0.5rem 1.2rem; cursor: pointer; color: var(--global-text-color); opacity: 0.6; border-bottom: 2px solid transparent; margin-bottom: -2px; transition: opacity 0.15s; font-weight: 500; }
.gpu-tab:hover { opacity: 0.85; }
.gpu-tab.active { opacity: 1; border-bottom-color: var(--global-theme-color, #0d6efd); }

/* ---- Buttons ---- */
.gpu-btn { display: inline-block; padding: 0.35rem 0.9rem; border: none; border-radius: 4px; cursor: pointer; font-size: 0.85rem; font-weight: 500; color: #fff; background: var(--global-theme-color, #0d6efd); transition: filter 0.15s; }
.gpu-btn:hover { filter: brightness(1.1); }
.gpu-btn-sm { padding: 0.2rem 0.55rem; font-size: 0.78rem; }
.gpu-btn-primary { background: var(--global-theme-color, #0d6efd); }
.gpu-btn-secondary { background: #6c757d; }
.gpu-btn-danger { background: #dc3545; }

/* ---- Select & Input ---- */
.gpu-select, .gpu-input { padding: 0.3rem 0.5rem; border: 1px solid var(--global-divider-color); border-radius: 4px; background: var(--global-card-bg-color, #fff); color: var(--global-text-color); font-size: 0.85rem; }
.gpu-select-full { width: 100%; }
.gpu-input { width: 100%; box-sizing: border-box; }

/* ---- Stats ---- */
.gpu-stats-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 1.2rem; }
.gpu-stat-card { background: var(--global-card-bg-color, #fff); border: 1px solid var(--global-divider-color); border-radius: 6px; padding: 0.8rem; text-align: center; }
.gpu-stat-value { font-size: 1.6rem; font-weight: 700; }
.gpu-stat-label { font-size: 0.78rem; color: var(--global-text-color); opacity: 0.65; margin-top: 0.15rem; }

/* ---- Heatmap ---- */
.gpu-heatmap-wrap { overflow-x: auto; margin-bottom: 1rem; }
.gpu-heatmap { width: 100%; border-collapse: collapse; font-size: 0.75rem; table-layout: fixed; }
.gpu-heatmap th, .gpu-heatmap td { border: 1px solid var(--global-divider-color); padding: 0; text-align: center; }
.gpu-heatmap th { padding: 0.3rem 0.2rem; font-weight: 600; background: var(--global-card-bg-color, #fff); }
.gpu-hm-node { width: 150px; text-align: left !important; padding: 0.25rem 0.4rem !important; white-space: nowrap; }
.gpu-hm-gpu { width: calc((100% - 150px) / 8); }
.gpu-cell { height: 32px; cursor: default; position: relative; vertical-align: middle; }
.gpu-cell-free { background: rgba(40, 167, 69, 0.18); cursor: pointer; }
.gpu-cell-free:hover { background: rgba(40, 167, 69, 0.35); }
.gpu-cell-reserved { background: rgba(255, 193, 7, 0.30); }
.gpu-cell-active { background: rgba(220, 53, 69, 0.25); }
.gpu-cell-offline { background: rgba(128, 128, 128, 0.18); }
.gpu-cell-text { font-size: 0.65rem; font-weight: 600; line-height: 32px; }

/* ---- Legend ---- */
.gpu-legend { display: flex; gap: 1rem; font-size: 0.78rem; margin-bottom: 1rem; flex-wrap: wrap; }
.gpu-legend-item { display: flex; align-items: center; gap: 0.3rem; }
.gpu-legend-swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; border: 1px solid var(--global-divider-color); }

/* ---- Badges ---- */
.gpu-tier-badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; color: #fff; font-size: 0.68rem; font-weight: 600; vertical-align: middle; }
.gpu-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; vertical-align: middle; }
.gpu-status-badge { display: inline-block; padding: 0.1rem 0.4rem; border-radius: 3px; font-size: 0.72rem; font-weight: 500; }
.gpu-status-active { background: rgba(220, 53, 69, 0.15); color: #dc3545; }
.gpu-status-assigned { background: rgba(255, 193, 7, 0.20); color: #b58900; }
.gpu-status-waiting { background: rgba(108, 117, 125, 0.15); color: #6c757d; }
.gpu-status-reserved { background: rgba(255, 193, 7, 0.20); color: #b58900; }
.gpu-status-offline { background: rgba(128, 128, 128, 0.20); color: #6c757d; }

/* ---- Panel / Table ---- */
.gpu-panel { background: var(--global-card-bg-color, #fff); border: 1px solid var(--global-divider-color); border-radius: 6px; padding: 1rem; margin-bottom: 1rem; }
.gpu-panel h3 { margin: 0 0 0.8rem 0; font-size: 1.05rem; }
.gpu-panel h4 { margin: 0 0 0.5rem 0; font-size: 0.95rem; }
.gpu-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.gpu-table th, .gpu-table td { padding: 0.4rem 0.5rem; border-bottom: 1px solid var(--global-divider-color); text-align: left; vertical-align: middle; }
.gpu-table th { font-weight: 600; opacity: 0.7; }
.gpu-actions { white-space: nowrap; }
.gpu-muted { font-size: 0.82rem; opacity: 0.6; }

/* ---- Form ---- */
.gpu-form { max-width: 500px; }
.gpu-form-group { margin-bottom: 0.75rem; }
.gpu-form-group > label { display: block; font-size: 0.82rem; font-weight: 600; margin-bottom: 0.2rem; }
.gpu-form-inline { display: flex; flex-wrap: wrap; gap: 0.75rem; max-width: 100%; align-items: flex-end; }
.gpu-form-inline .gpu-form-group { flex: 1 1 120px; margin-bottom: 0; }
.gpu-radio-group { display: flex; gap: 0.8rem; }
.gpu-radio { font-size: 0.85rem; cursor: pointer; display: flex; align-items: center; gap: 0.25rem; }
.gpu-slot-picker { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.gpu-cost-display { padding: 0.5rem 0.7rem; background: rgba(13, 110, 253, 0.08); border-radius: 4px; font-weight: 500; font-size: 0.85rem; }
.gpu-validation { margin-bottom: 0.5rem; min-height: 1.2rem; }
.gpu-error { color: #dc3545; font-size: 0.82rem; }

/* ---- Node grid ---- */
.gpu-node-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.75rem; }
.gpu-node-card { background: var(--global-card-bg-color, #fff); border: 1px solid var(--global-divider-color); border-radius: 6px; padding: 0.8rem; }
.gpu-node-down { opacity: 0.55; }
.gpu-node-header { margin-bottom: 0.3rem; }
.gpu-node-model { font-size: 0.85rem; font-weight: 500; margin-bottom: 0.3rem; }
.gpu-node-details { font-size: 0.75rem; opacity: 0.7; line-height: 1.4; margin-bottom: 0.5rem; }
.gpu-node-usage { font-size: 0.78rem; }
.gpu-usage-bar { height: 6px; background: var(--global-divider-color); border-radius: 3px; margin-bottom: 0.25rem; overflow: hidden; }
.gpu-usage-fill { height: 100%; background: var(--global-theme-color, #0d6efd); border-radius: 3px; transition: width 0.3s; }

/* ---- Responsive ---- */
@media (max-width: 600px) {
  .gpu-stats-row { grid-template-columns: repeat(2, 1fr); }
  .gpu-hm-node { width: 100px; font-size: 0.68rem; }
  .gpu-cell { height: 26px; }
  .gpu-cell-text { font-size: 0.55rem; line-height: 26px; }
}
</style>

<div id="gpu-app">
  <div style="text-align:center;padding:2rem;opacity:0.5">Loading GPU Scheduler...</div>
</div>

<script src="{{ '/assets/js/gpu-scheduler.js' | relative_url }}"></script>
