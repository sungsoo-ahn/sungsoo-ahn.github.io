(function () {
  'use strict';

  // ============================================================
  // Constants
  // ============================================================

  var KEYS = {
    nodes: 'gpu_nodes',
    users: 'gpu_users',
    requests: 'gpu_requests',
    currentUser: 'gpu_current_user',
    initialized: 'gpu_initialized'
  };

  var TIER = {
    1: { name: 'T1', model: 'RTX 3090', mult: 1 },
    2: { name: 'T2', model: 'Max Pro 6000', mult: 2 },
    3: { name: 'T3', model: 'H200', mult: 4 },
    4: { name: 'T4', model: 'B200', mult: 6 }
  };

  var CONTIGUOUS = {
    1: [[0], [1], [2], [3], [4], [5], [6], [7]],
    2: [[0, 1], [2, 3], [4, 5], [6, 7]],
    4: [[0, 1, 2, 3], [4, 5, 6, 7]],
    8: [[0, 1, 2, 3, 4, 5, 6, 7]]
  };

  var MAX_DURATION = {
    1: { normal: 7, queued: 3 },
    2: { normal: 7, queued: 3 },
    4: { normal: 4, queued: 2 },
    8: { normal: 3, queued: 2 }
  };

  var DAY = 86400000;
  var HOUR = 3600000;

  // ============================================================
  // Store — localStorage CRUD
  // ============================================================

  var Store = {
    get: function (key) {
      try { return JSON.parse(localStorage.getItem(key)); } catch (e) { return null; }
    },
    set: function (key, val) {
      localStorage.setItem(key, JSON.stringify(val));
    },
    remove: function (key) {
      localStorage.removeItem(key);
    }
  };

  // ============================================================
  // Helpers
  // ============================================================

  function uuid() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (c) {
      var r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  function esc(str) {
    var d = document.createElement('div');
    d.textContent = str || '';
    return d.innerHTML;
  }

  function getInitials(name) {
    if (!name) return '?';
    if (/[\u3131-\uD79D]/.test(name)) return name.length >= 2 ? name.slice(-2) : name;
    return name.split(' ').map(function (w) { return w[0]; }).join('').toUpperCase().slice(0, 2);
  }

  function fmtDate(iso) {
    if (!iso) return '-';
    var d = new Date(iso);
    return (d.getMonth() + 1) + '/' + d.getDate() + ' ' +
      String(d.getHours()).padStart(2, '0') + ':' + String(d.getMinutes()).padStart(2, '0');
  }

  function fmtDateShort(iso) {
    if (!iso) return '-';
    var d = new Date(iso);
    return (d.getMonth() + 1) + '/' + d.getDate();
  }

  function timeLeft(endIso) {
    var diff = new Date(endIso).getTime() - Date.now();
    if (diff <= 0) return 'Expired';
    var hours = Math.floor(diff / HOUR);
    var days = Math.floor(hours / 24);
    var rem = hours % 24;
    if (days > 0) return days + 'd ' + rem + 'h';
    if (hours > 0) return hours + 'h';
    return '<1h';
  }

  function timeWaiting(createdIso) {
    var diff = Date.now() - new Date(createdIso).getTime();
    if (diff < HOUR) return Math.floor(diff / 60000) + 'm';
    var hours = Math.floor(diff / HOUR);
    if (hours < 24) return hours + 'h';
    return Math.floor(hours / 24) + 'd ' + (hours % 24) + 'h';
  }

  function tierBadge(tier) {
    var colors = { 1: '#6c757d', 2: '#0d6efd', 3: '#6f42c1', 4: '#dc3545' };
    return '<span class="gpu-tier-badge" style="background:' + (colors[tier] || '#6c757d') + '">' + TIER[tier].name + '</span>';
  }

  function trafficDot(color) {
    var hex = { green: '#28a745', yellow: '#ffc107', red: '#dc3545', grey: '#6c757d' };
    return '<span class="gpu-dot" style="background:' + (hex[color] || hex.grey) + '" title="' + color + '"></span>';
  }

  // ============================================================
  // Data accessors
  // ============================================================

  function getNodes() { return Store.get(KEYS.nodes) || []; }
  function getUsers() { return Store.get(KEYS.users) || []; }
  function getRequests() { return Store.get(KEYS.requests) || []; }
  function getCurrentUserId() { return Store.get(KEYS.currentUser); }
  function setCurrentUserId(id) { Store.set(KEYS.currentUser, id); }
  function getUserById(id) { return getUsers().find(function (u) { return u.id === id; }); }
  function getNodeById(id) { return getNodes().find(function (n) { return n.id === id; }); }

  // ============================================================
  // Domain logic
  // ============================================================

  function getTrafficLight(node) {
    if (!node.driverVersion) return 'grey';
    if (node.status === 'maintenance') return 'red';
    var cudaMajor = parseFloat(node.cudaVersion);
    var driverAge = (Date.now() - new Date(node.driverUpdatedAt).getTime()) / DAY;
    if (cudaMajor >= 12 && driverAge < 90) return 'green';
    return 'yellow';
  }

  function getNodeGpuStatus(nodeId) {
    var reqs = getRequests().filter(function (r) {
      return r.nodeId === nodeId && (r.status === 'active' || r.status === 'assigned');
    });
    var gpus = [];
    for (var i = 0; i < 8; i++) gpus.push(null);
    reqs.forEach(function (r) {
      if (r.gpuIds) {
        r.gpuIds.forEach(function (idx) {
          gpus[idx] = {
            status: r.status,
            userId: r.userId,
            purpose: r.purpose,
            expectedEndAt: r.expectedEndAt,
            requestId: r.id
          };
        });
      }
    });
    return gpus;
  }

  function getAvailableSlots(nodeId, gpuCount) {
    var gpuStatus = getNodeGpuStatus(nodeId);
    return CONTIGUOUS[gpuCount].filter(function (slot) {
      return slot.every(function (idx) { return gpuStatus[idx] === null; });
    });
  }

  function calcPointCost(gpuCount, tier, days) {
    return gpuCount * (TIER[tier] ? TIER[tier].mult : 1) * days;
  }

  function getUserActivePoints(userId) {
    return getRequests()
      .filter(function (r) { return r.userId === userId && (r.status === 'active' || r.status === 'assigned'); })
      .reduce(function (sum, r) {
        var tier = r.minTier;
        if (r.nodeId) {
          var node = getNodeById(r.nodeId);
          if (node) tier = node.gpuTier;
        }
        return sum + calcPointCost(r.gpuCount, tier, r.durationDays);
      }, 0);
  }

  function hasQueueForTier(tier) {
    return getRequests().some(function (r) { return r.status === 'waiting' && r.minTier <= tier; });
  }

  function getMaxDuration(gpuCount, tier) {
    var cfg = MAX_DURATION[gpuCount] || MAX_DURATION[1];
    return hasQueueForTier(tier) ? cfg.queued : cfg.normal;
  }

  function calcPriority(userId, waitingCount) {
    var base = Math.max(0, 1.0 - waitingCount * 0.1);
    var activePoints = getUserActivePoints(userId);
    var fairness = Math.max(0, 1.0 - activePoints / 50);
    return Math.round((base + fairness) * 100) / 100;
  }

  // ============================================================
  // Auto-assignment engine
  // ============================================================

  function tryAssignRequests() {
    var requests = getRequests();
    var nodes = getNodes();
    var waiting = requests
      .filter(function (r) { return r.status === 'waiting'; })
      .sort(function (a, b) { return b.priorityScore - a.priorityScore; });

    var changed = false;

    waiting.forEach(function (req) {
      // Build candidate nodes
      var candidates = nodes.filter(function (n) {
        return n.status === 'active' && n.gpuTier >= req.minTier;
      });

      // Sort: preferred node first, then lowest tier that satisfies (resource efficiency)
      candidates.sort(function (a, b) {
        if (req.preferredNodeId) {
          if (a.id === req.preferredNodeId && b.id !== req.preferredNodeId) return -1;
          if (b.id === req.preferredNodeId && a.id !== req.preferredNodeId) return 1;
        }
        return a.gpuTier - b.gpuTier;
      });

      for (var i = 0; i < candidates.length; i++) {
        var node = candidates[i];
        // Re-read gpu status each time since earlier iterations may have assigned
        var slots = getAvailableSlotsFromRequests(requests, node.id, req.gpuCount);
        if (slots.length > 0) {
          var now = new Date();
          req.status = 'assigned';
          req.nodeId = node.id;
          req.gpuIds = slots[0];
          req.assignedAt = now.toISOString();
          req.expectedEndAt = new Date(now.getTime() + req.durationDays * DAY).toISOString();
          changed = true;
          break;
        }
      }
    });

    if (changed) {
      Store.set(KEYS.requests, requests);
    }
    return changed;
  }

  // Like getAvailableSlots but works on an in-memory requests array (for batch assignment)
  function getAvailableSlotsFromRequests(requests, nodeId, gpuCount) {
    var gpus = [];
    for (var i = 0; i < 8; i++) gpus.push(null);
    requests.forEach(function (r) {
      if (r.nodeId === nodeId && (r.status === 'active' || r.status === 'assigned') && r.gpuIds) {
        r.gpuIds.forEach(function (idx) { gpus[idx] = r; });
      }
    });
    return CONTIGUOUS[gpuCount].filter(function (slot) {
      return slot.every(function (idx) { return gpus[idx] === null; });
    });
  }

  // ============================================================
  // Request operations
  // ============================================================

  function submitRequest(userId, gpuCount, minTier, preferredNodeId, purpose, memo, durationDays) {
    var requests = getRequests();
    var waitingCount = requests.filter(function (r) { return r.status === 'waiting'; }).length;
    var now = new Date();
    var req = {
      id: uuid(),
      userId: userId,
      gpuCount: gpuCount,
      minTier: minTier,
      preferredNodeId: preferredNodeId || null,
      purpose: purpose,
      memo: memo || '',
      durationDays: durationDays,
      priorityScore: calcPriority(userId, waitingCount),
      status: 'waiting',
      nodeId: null,
      gpuIds: null,
      assignedAt: null,
      startedAt: null,
      expectedEndAt: null,
      actualEndAt: null,
      createdAt: now.toISOString()
    };
    requests.push(req);
    Store.set(KEYS.requests, requests);
    tryAssignRequests();
    return req;
  }

  function startRequest(id) {
    var requests = getRequests();
    var r = requests.find(function (x) { return x.id === id; });
    if (!r || r.status !== 'assigned') return false;
    var now = new Date();
    r.status = 'active';
    r.startedAt = now.toISOString();
    r.expectedEndAt = new Date(now.getTime() + r.durationDays * DAY).toISOString();
    Store.set(KEYS.requests, requests);
    return true;
  }

  function endRequest(id) {
    var requests = getRequests();
    var r = requests.find(function (x) { return x.id === id; });
    if (!r || r.status !== 'active') return false;
    r.status = 'completed';
    r.actualEndAt = new Date().toISOString();
    Store.set(KEYS.requests, requests);
    tryAssignRequests();
    return true;
  }

  function cancelRequest(id) {
    var requests = getRequests();
    var r = requests.find(function (x) { return x.id === id; });
    if (!r) return false;
    if (r.status === 'waiting' || r.status === 'assigned' || r.status === 'active') {
      var wasOccupying = r.status === 'assigned' || r.status === 'active';
      r.status = 'completed';
      r.actualEndAt = new Date().toISOString();
      Store.set(KEYS.requests, requests);
      if (wasOccupying) tryAssignRequests();
      return true;
    }
    return false;
  }

  function extendRequest(id, extraDays) {
    var requests = getRequests();
    var r = requests.find(function (x) { return x.id === id; });
    if (!r || r.status !== 'active') return false;
    var node = getNodeById(r.nodeId);
    if (node && hasQueueForTier(node.gpuTier)) return false;
    r.expectedEndAt = new Date(new Date(r.expectedEndAt).getTime() + extraDays * DAY).toISOString();
    r.durationDays += extraDays;
    Store.set(KEYS.requests, requests);
    return true;
  }

  function checkExpired() {
    var requests = getRequests();
    var now = Date.now();
    var changed = false;
    requests.forEach(function (r) {
      // Active request past end time (+30min grace)
      if (r.status === 'active' && r.expectedEndAt && new Date(r.expectedEndAt).getTime() + 1800000 < now) {
        r.status = 'expired';
        r.actualEndAt = new Date().toISOString();
        changed = true;
      }
      // Assigned but not started within 2h
      if (r.status === 'assigned' && r.assignedAt && new Date(r.assignedAt).getTime() + 2 * HOUR < now) {
        r.status = 'expired';
        r.actualEndAt = new Date().toISOString();
        changed = true;
      }
    });
    if (changed) {
      Store.set(KEYS.requests, requests);
      tryAssignRequests();
      renderApp();
    }
  }

  // ============================================================
  // Seed mock data
  // ============================================================

  function seedMockData() {
    // Re-seed if migrating from old version (had reservations/queue, not requests)
    if (Store.get(KEYS.initialized) && !Store.get(KEYS.requests)) {
      Object.values(KEYS).forEach(function (k) { Store.remove(k); });
      Store.remove('gpu_reservations');
      Store.remove('gpu_queue');
    }
    if (Store.get(KEYS.initialized)) return;
    var now = Date.now();

    // --- Nodes ---
    var nodes = [];
    function addNodes(start, count, tier, model, driver, cuda, cudnn, pt, type, daysAgo, statusOverride) {
      for (var i = 0; i < count; i++) {
        var idx = start + i;
        var st = (statusOverride && statusOverride[i]) || 'active';
        nodes.push({
          id: uuid(),
          name: 'node-' + String(idx).padStart(2, '0'),
          gpuModel: model, gpuTier: tier, gpuCount: 8,
          driverVersion: driver, cudaVersion: cuda,
          cudnnVersion: cudnn, pytorchVersion: pt,
          osVersion: 'Ubuntu 22.04', nodeType: type,
          cloudExpireAt: type === 'cloud' ? new Date(now + 180 * DAY).toISOString() : null,
          driverUpdatedAt: new Date(now - daysAgo * DAY).toISOString(),
          status: st, createdAt: new Date(now - 90 * DAY).toISOString()
        });
      }
    }
    addNodes(1, 6, 1, 'RTX 3090', '525.89.02', '12.0', '8.7.0', '2.1.0', 'on-premise', 60,
      [null, null, null, null, null, 'maintenance']);
    addNodes(7, 4, 2, 'Max Pro 6000', '535.129.03', '12.2', '8.9.2', '2.2.0', 'on-premise', 45, null);
    addNodes(11, 6, 3, 'H200', '550.54.15', '12.4', '8.9.7', '2.3.0', 'cloud', 20,
      [null, null, null, null, 'offline', null]);
    addNodes(17, 4, 4, 'B200', '560.28.03', '12.5', '9.0.0', '2.4.0', 'cloud', 10, null);

    // --- Users ---
    var users = [
      { name: '김나영', role: 'student', weeklyPoints: 32, maxConcurrentPoints: 16, maxTier: 3 },
      { name: '우동엽', role: 'senior', weeklyPoints: 64, maxConcurrentPoints: 32, maxTier: 4 },
      { name: '박성현', role: 'student', weeklyPoints: 32, maxConcurrentPoints: 16, maxTier: 3 },
      { name: '김민규', role: 'student', weeklyPoints: 32, maxConcurrentPoints: 16, maxTier: 3 },
      { name: '김성수', role: 'admin', weeklyPoints: 999, maxConcurrentPoints: 999, maxTier: 4 },
      { name: '이지은', role: 'student', weeklyPoints: 32, maxConcurrentPoints: 16, maxTier: 3 },
      { name: '정현우', role: 'senior', weeklyPoints: 64, maxConcurrentPoints: 32, maxTier: 4 },
      { name: '최서연', role: 'newbie', weeklyPoints: 16, maxConcurrentPoints: 8, maxTier: 2 }
    ].map(function (u) {
      return Object.assign({ id: uuid(), isActive: true, createdAt: new Date(now - 60 * DAY).toISOString() }, u);
    });

    // --- Requests (migrated from old reservations + queue) ---
    var requests = [];

    function addActiveReq(nodeIdx, gpuIds, userIdx, purpose, daysAgo, durationDays) {
      var startedAt = new Date(now - daysAgo * DAY);
      var assignedAt = new Date(startedAt.getTime() - 0.04 * DAY);
      var endAt = new Date(startedAt.getTime() + durationDays * DAY);
      requests.push({
        id: uuid(), userId: users[userIdx].id,
        gpuCount: gpuIds.length, minTier: nodes[nodeIdx].gpuTier,
        preferredNodeId: null, purpose: purpose, memo: '',
        durationDays: durationDays, priorityScore: 1.0,
        status: 'active',
        nodeId: nodes[nodeIdx].id, gpuIds: gpuIds,
        assignedAt: assignedAt.toISOString(),
        startedAt: startedAt.toISOString(),
        expectedEndAt: endAt.toISOString(),
        actualEndAt: null,
        createdAt: assignedAt.toISOString()
      });
    }

    function addAssignedReq(nodeIdx, gpuIds, userIdx, purpose, durationDays) {
      var assignedAt = new Date(now - 0.04 * DAY);
      var endAt = new Date(assignedAt.getTime() + durationDays * DAY);
      requests.push({
        id: uuid(), userId: users[userIdx].id,
        gpuCount: gpuIds.length, minTier: nodes[nodeIdx].gpuTier,
        preferredNodeId: null, purpose: purpose, memo: '',
        durationDays: durationDays, priorityScore: 1.0,
        status: 'assigned',
        nodeId: nodes[nodeIdx].id, gpuIds: gpuIds,
        assignedAt: assignedAt.toISOString(),
        startedAt: null,
        expectedEndAt: endAt.toISOString(),
        actualEndAt: null,
        createdAt: assignedAt.toISOString()
      });
    }

    function addCompletedReq(nodeIdx, gpuIds, userIdx, purpose, daysAgo, durationDays) {
      var startedAt = new Date(now - daysAgo * DAY);
      var endAt = new Date(startedAt.getTime() + durationDays * DAY);
      requests.push({
        id: uuid(), userId: users[userIdx].id,
        gpuCount: gpuIds.length, minTier: nodes[nodeIdx].gpuTier,
        preferredNodeId: null, purpose: purpose, memo: '',
        durationDays: durationDays, priorityScore: 0.5,
        status: 'completed',
        nodeId: nodes[nodeIdx].id, gpuIds: gpuIds,
        assignedAt: startedAt.toISOString(),
        startedAt: startedAt.toISOString(),
        expectedEndAt: endAt.toISOString(),
        actualEndAt: endAt.toISOString(),
        createdAt: startedAt.toISOString()
      });
    }

    function addWaitingReq(userIdx, gpuCount, minTier, purpose, durationDays, hoursAgo, priority) {
      requests.push({
        id: uuid(), userId: users[userIdx].id,
        gpuCount: gpuCount, minTier: minTier,
        preferredNodeId: null, purpose: purpose, memo: '',
        durationDays: durationDays, priorityScore: priority,
        status: 'waiting',
        nodeId: null, gpuIds: null,
        assignedAt: null, startedAt: null,
        expectedEndAt: null, actualEndAt: null,
        createdAt: new Date(now - hoursAgo * HOUR).toISOString()
      });
    }

    // === Active requests ===
    // T1 nodes (node-01 ~ node-05 active, node-06 maintenance)
    addActiveReq(0, [0, 1], 0, 'Protein diffusion training', 1.5, 3);       // 김나영 on node-01
    addActiveReq(1, [0, 1, 2, 3], 1, 'GNN pretraining', 2, 4);             // 우동엽 on node-02
    addActiveReq(2, [0, 1], 2, 'Baseline evaluation', 0.5, 2);              // 박성현 on node-03
    addActiveReq(3, [4, 5, 6, 7], 3, 'VAE ablation study', 1, 3);           // 김민규 on node-04

    // T2 nodes (node-07 ~ node-10)
    addActiveReq(6, [0, 1, 2, 3], 1, 'Molecule generation', 1.2, 3);        // 우동엽 on node-07
    addActiveReq(7, [0, 1], 5, 'Contrastive learning', 0.8, 2);             // 이지은 on node-08

    // T3 nodes (node-11 ~ node-16, node-15 offline)
    addActiveReq(10, [0, 1, 2, 3, 4, 5, 6, 7], 6, 'LLM fine-tuning', 0.8, 3);  // 정현우 on node-11 (full)
    addActiveReq(11, [0, 1, 2, 3], 0, 'Protein structure prediction', 1.8, 4);  // 김나영 on node-12
    addActiveReq(12, [0, 1], 5, 'Drug binding prediction', 0.6, 2);              // 이지은 on node-13

    // T4 nodes (node-17 ~ node-20)
    addActiveReq(16, [0, 1, 2, 3], 1, 'Foundation model training', 0.3, 3);     // 우동엽 on node-17
    addActiveReq(17, [0, 1], 6, 'Scaling experiments', 0.7, 2);                  // 정현우 on node-18

    // === Assigned (awaiting user to click Start) ===
    addAssignedReq(4, [0, 1], 3, 'Hyperparameter sweep', 2);                // 김민규 on node-05
    addAssignedReq(18, [4, 5], 7, 'Tutorial experiments', 2);               // 최서연 on node-19

    // === Completed (recent history) ===
    addCompletedReq(0, [4, 5, 6, 7], 2, 'Benchmark runs', 5, 2);
    addCompletedReq(9, [0, 1], 3, 'Debugging session', 6, 1);
    addCompletedReq(19, [0, 1, 2, 3], 6, 'Multi-GPU training', 4, 3);

    // === Waiting (in the queue, no GPUs yet) ===
    // These requests can't be auto-assigned because all matching nodes are busy
    addWaitingReq(3, 8, 3, 'Large-scale H200 training', 3, 3, 1.20);        // 김민규 wants 8x T3 — node-11 full
    addWaitingReq(5, 4, 2, 'Ablation study', 2, 1.5, 0.95);                 // 이지은 wants 4x T2
    addWaitingReq(7, 4, 4, 'Full B200 training run', 3, 0.5, 0.60);         // 최서연 wants 4x T4 (tier-locked, can't use)

    Store.set(KEYS.nodes, nodes);
    Store.set(KEYS.users, users);
    Store.set(KEYS.requests, requests);
    Store.set(KEYS.initialized, true);
    // Clean up old keys if they exist
    Store.remove('gpu_reservations');
    Store.remove('gpu_queue');
  }

  // ============================================================
  // State
  // ============================================================

  var currentTab = 'dashboard';
  var prefillNodeId = null; // set when clicking free heatmap cell

  // ============================================================
  // Renderers
  // ============================================================

  function renderApp() {
    var app = document.getElementById('gpu-app');
    if (!app) return;

    var userId = getCurrentUserId();
    var users = getUsers();

    var html = '';

    // --- Top bar ---
    html += '<div class="gpu-topbar">';
    html += '<div class="gpu-topbar-left">';
    html += '<label for="gpu-user-select">User:</label> ';
    html += '<select id="gpu-user-select" class="gpu-select">';
    html += '<option value="">-- Select --</option>';
    users.forEach(function (u) {
      html += '<option value="' + u.id + '"' + (u.id === userId ? ' selected' : '') + '>' + esc(u.name) + ' (' + u.role + ')</option>';
    });
    html += '</select>';
    html += '</div>';
    html += '<button id="gpu-reset-btn" class="gpu-btn gpu-btn-danger">Reset Data</button>';
    html += '</div>';

    // --- Tabs ---
    html += '<ul class="gpu-tabs">';
    ['dashboard', 'request', 'nodes'].forEach(function (tab) {
      var labels = { dashboard: 'Dashboard', request: 'Request', nodes: 'Nodes' };
      html += '<li class="gpu-tab' + (currentTab === tab ? ' active' : '') + '" data-tab="' + tab + '">' + labels[tab] + '</li>';
    });
    html += '</ul>';

    // --- Tab content ---
    html += '<div class="gpu-tab-content">';
    if (currentTab === 'dashboard') html += renderDashboard();
    else if (currentTab === 'request') html += renderRequestTab();
    else if (currentTab === 'nodes') html += renderNodesTab();
    html += '</div>';

    app.innerHTML = html;
  }

  function renderDashboard() {
    var html = '';
    html += renderHowItWorks();
    html += renderStats();
    html += renderHeatmap();
    html += renderMyRequests();
    return html;
  }

  function renderHowItWorks() {
    var html = '<div class="gpu-panel gpu-how-it-works">';
    html += '<h3>How It Works</h3>';
    html += '<ol style="margin:0 0 0 1.2rem;padding:0;font-size:0.85rem;line-height:1.6">';
    html += '<li><strong>Submit a request</strong> \u2014 specify how many GPUs you need, the minimum tier, and duration. Optionally pick a preferred node.</li>';
    html += '<li><strong>Auto-assignment</strong> \u2014 the system matches your request to available GPUs automatically, prioritized by a fairness score. If nothing is available, you wait in the queue.</li>';
    html += '<li><strong>Start within 2 hours</strong> \u2014 once assigned, click <em>Start</em> to begin. If you don\u2019t start within 2h, the assignment expires and GPUs go to the next person.</li>';
    html += '<li><strong>End or extend</strong> \u2014 release GPUs when done. If no one is waiting, you can extend by 1 day when less than 24h remain.</li>';
    html += '</ol>';
    html += '<p style="font-size:0.8rem;opacity:0.6;margin:0.5rem 0 0 0">';
    html += 'Cost = GPUs \u00d7 tier multiplier \u00d7 days. ';
    html += 'Tiers: T1(\u00d71) &lt; T2(\u00d72) &lt; T3(\u00d74) &lt; T4(\u00d76). ';
    html += 'Click any green cell in the heatmap to jump to the request form with that node pre-selected.';
    html += '</p>';
    html += '</div>';
    return html;
  }

  function renderStats() {
    var nodes = getNodes();
    var requests = getRequests();

    var totalGpus = nodes.length * 8;
    var activeNodes = nodes.filter(function (n) { return n.status === 'active'; });
    var occupiedGpus = 0;
    activeNodes.forEach(function (n) {
      var st = getNodeGpuStatus(n.id);
      st.forEach(function (g) { if (g) occupiedGpus++; });
    });
    var offlineGpus = nodes.filter(function (n) { return n.status !== 'active'; }).length * 8;
    var freeGpus = totalGpus - occupiedGpus - offlineGpus;
    var activeCount = requests.filter(function (r) { return r.status === 'active' || r.status === 'assigned'; }).length;
    var queueLen = requests.filter(function (r) { return r.status === 'waiting'; }).length;

    var cards = [
      { label: 'Total GPUs', value: totalGpus, color: '#0d6efd' },
      { label: 'Free', value: freeGpus, color: '#28a745' },
      { label: 'Active', value: activeCount, color: '#fd7e14' },
      { label: 'Queue Length', value: queueLen, color: '#6f42c1' }
    ];

    var html = '<div class="gpu-stats-row">';
    cards.forEach(function (c) {
      html += '<div class="gpu-stat-card">';
      html += '<div class="gpu-stat-value" style="color:' + c.color + '">' + c.value + '</div>';
      html += '<div class="gpu-stat-label">' + c.label + '</div>';
      html += '</div>';
    });
    html += '</div>';
    return html;
  }

  function renderHeatmap() {
    var nodes = getNodes();
    var html = '<div class="gpu-heatmap-wrap"><table class="gpu-heatmap">';

    // Header
    html += '<thead><tr><th class="gpu-hm-node">Node</th>';
    for (var g = 0; g < 8; g++) html += '<th class="gpu-hm-gpu">GPU ' + g + '</th>';
    html += '</tr></thead><tbody>';

    nodes.forEach(function (node) {
      var gpuStatus = getNodeGpuStatus(node.id);
      var tl = getTrafficLight(node);
      var isDown = node.status !== 'active';

      html += '<tr>';
      html += '<td class="gpu-hm-node">';
      html += trafficDot(tl) + ' ';
      html += '<strong>' + esc(node.name) + '</strong> ';
      html += tierBadge(node.gpuTier);
      html += '</td>';

      for (var i = 0; i < 8; i++) {
        var gs = gpuStatus[i];
        var cls = 'gpu-cell';
        var title = '';
        var content = '';

        if (isDown) {
          cls += ' gpu-cell-offline';
          title = node.status;
          content = '<span class="gpu-cell-text">\u2014</span>';
        } else if (gs) {
          cls += gs.status === 'active' ? ' gpu-cell-active' : ' gpu-cell-reserved';
          var user = getUserById(gs.userId);
          var uname = user ? user.name : '?';
          title = uname + ' \u2014 ' + (gs.purpose || '') + ' (' + timeLeft(gs.expectedEndAt) + ')';
          content = '<span class="gpu-cell-text">' + esc(getInitials(uname)) + '</span>';
        } else {
          cls += ' gpu-cell-free';
          title = 'Free \u2014 click to request';
          content = '';
        }

        html += '<td class="' + cls + '" title="' + esc(title) + '" data-node="' + node.id + '" data-gpu="' + i + '">';
        html += content;
        html += '</td>';
      }
      html += '</tr>';
    });

    html += '</tbody></table></div>';

    // Legend
    html += '<div class="gpu-legend">';
    html += '<span class="gpu-legend-item"><span class="gpu-legend-swatch gpu-cell-free"></span> Free</span>';
    html += '<span class="gpu-legend-item"><span class="gpu-legend-swatch gpu-cell-reserved"></span> Assigned</span>';
    html += '<span class="gpu-legend-item"><span class="gpu-legend-swatch gpu-cell-active"></span> Active</span>';
    html += '<span class="gpu-legend-item"><span class="gpu-legend-swatch gpu-cell-offline"></span> Offline</span>';
    html += '</div>';
    return html;
  }

  function renderMyRequests() {
    var userId = getCurrentUserId();
    if (!userId) return '<div class="gpu-panel"><p class="gpu-muted">Select a user to see your requests.</p></div>';

    var user = getUserById(userId);
    var myReqs = getRequests().filter(function (r) {
      return r.userId === userId && (r.status === 'active' || r.status === 'assigned');
    });
    var activePoints = getUserActivePoints(userId);

    var html = '<div class="gpu-panel">';
    html += '<h3>My Requests \u2014 ' + esc(user.name) + '</h3>';
    html += '<p class="gpu-muted">Points in use: <strong>' + activePoints + '</strong> / ' + user.maxConcurrentPoints + ' max concurrent</p>';

    if (myReqs.length === 0) {
      html += '<p class="gpu-muted">No active requests.</p>';
    } else {
      html += '<table class="gpu-table"><thead><tr><th>Node</th><th>GPUs</th><th>Purpose</th><th>Status</th><th>Time Left</th><th>Actions</th></tr></thead><tbody>';
      myReqs.forEach(function (r) {
        var node = r.nodeId ? getNodeById(r.nodeId) : null;
        var nodeName = node ? node.name : '-';
        html += '<tr>';
        html += '<td>' + esc(nodeName) + '</td>';
        html += '<td>' + (r.gpuIds ? r.gpuIds.join(', ') : '-') + '</td>';
        html += '<td>' + esc(r.purpose) + '</td>';
        html += '<td><span class="gpu-status-badge gpu-status-' + r.status + '">' + r.status + '</span></td>';
        html += '<td>' + (r.expectedEndAt ? timeLeft(r.expectedEndAt) : '-') + '</td>';
        html += '<td class="gpu-actions">';

        if (r.status === 'assigned') {
          html += '<button class="gpu-btn gpu-btn-sm gpu-btn-primary" data-action="start" data-id="' + r.id + '">Start</button> ';
          html += '<button class="gpu-btn gpu-btn-sm gpu-btn-danger" data-action="cancel" data-id="' + r.id + '">Cancel</button>';
        } else if (r.status === 'active') {
          html += '<button class="gpu-btn gpu-btn-sm gpu-btn-primary" data-action="end" data-id="' + r.id + '">End</button> ';
          var hoursLeft = (new Date(r.expectedEndAt).getTime() - Date.now()) / HOUR;
          if (hoursLeft < 24) {
            html += '<button class="gpu-btn gpu-btn-sm gpu-btn-secondary" data-action="extend" data-id="' + r.id + '">Extend +1d</button> ';
          }
          html += '<button class="gpu-btn gpu-btn-sm gpu-btn-danger" data-action="cancel" data-id="' + r.id + '">Cancel</button>';
        }

        html += '</td></tr>';
      });
      html += '</tbody></table>';
    }
    html += '</div>';
    return html;
  }

  // ============================================================
  // Request tab — unified form + queue table
  // ============================================================

  function renderRequestTab() {
    var html = '';
    html += renderRequestForm();
    html += renderQueueTable();
    return html;
  }

  function renderRequestForm() {
    var userId = getCurrentUserId();
    var user = userId ? getUserById(userId) : null;
    var nodes = getNodes().filter(function (n) { return n.status === 'active'; });

    var html = '<div class="gpu-panel"><h3>New Request</h3>';

    if (!userId) {
      html += '<p class="gpu-muted">Select a user first.</p></div>';
      return html;
    }

    html += '<form id="gpu-request-form" class="gpu-form">';

    // GPU Count
    html += '<div class="gpu-form-group">';
    html += '<label>GPU Count</label>';
    html += '<div class="gpu-radio-group">';
    [1, 2, 4, 8].forEach(function (c) {
      html += '<label class="gpu-radio"><input type="radio" name="req-gpu-count" value="' + c + '"' + (c === 1 ? ' checked' : '') + '> ' + c + '</label>';
    });
    html += '</div></div>';

    // Min Tier
    html += '<div class="gpu-form-group">';
    html += '<label>Min Tier</label>';
    html += '<select id="req-min-tier" class="gpu-select gpu-select-full">';
    [1, 2, 3, 4].forEach(function (t) {
      var disabled = user && t > user.maxTier ? ' disabled' : '';
      html += '<option value="' + t + '"' + disabled + '>' + TIER[t].name + ' (' + TIER[t].model + ')' + (t > user.maxTier ? ' — locked' : '') + '</option>';
    });
    html += '</select></div>';

    // Preferred Node (optional)
    html += '<div class="gpu-form-group">';
    html += '<label>Preferred Node (optional)</label>';
    html += '<select id="req-preferred-node" class="gpu-select gpu-select-full">';
    html += '<option value="">Any available</option>';
    nodes.forEach(function (n) {
      var disabled = user && n.gpuTier > user.maxTier ? ' disabled' : '';
      var tl = getTrafficLight(n);
      var selected = (prefillNodeId === n.id) ? ' selected' : '';
      html += '<option value="' + n.id + '"' + disabled + selected + '>' + n.name + ' [' + TIER[n.gpuTier].name + ' ' + n.gpuModel + '] ' + tl + '</option>';
    });
    html += '</select></div>';

    // Duration
    html += '<div class="gpu-form-group">';
    html += '<label>Duration (days)</label>';
    html += '<input type="number" id="req-duration" class="gpu-input" min="1" max="7" value="1">';
    html += '</div>';

    // Purpose
    html += '<div class="gpu-form-group">';
    html += '<label>Purpose</label>';
    html += '<input type="text" id="req-purpose" class="gpu-input" placeholder="e.g. Protein diffusion training">';
    html += '</div>';

    // Memo
    html += '<div class="gpu-form-group">';
    html += '<label>Memo (optional)</label>';
    html += '<input type="text" id="req-memo" class="gpu-input" placeholder="Optional notes">';
    html += '</div>';

    // Cost estimate
    html += '<div class="gpu-form-group">';
    html += '<div id="req-cost" class="gpu-cost-display">Estimated cost: 1 point</div>';
    html += '</div>';

    // Validation
    html += '<div id="req-validation" class="gpu-validation"></div>';

    html += '<button type="submit" class="gpu-btn gpu-btn-primary">Submit Request</button>';
    html += '</form></div>';

    // Clear prefill after rendering
    prefillNodeId = null;

    return html;
  }

  function renderQueueTable() {
    var requests = getRequests()
      .filter(function (r) { return r.status === 'waiting' || r.status === 'assigned'; })
      .sort(function (a, b) {
        // Assigned first, then waiting; within each group sort by priority
        if (a.status !== b.status) return a.status === 'assigned' ? -1 : 1;
        return b.priorityScore - a.priorityScore;
      });
    var userId = getCurrentUserId();

    var html = '<div class="gpu-panel"><h3>Queue</h3>';

    if (requests.length === 0) {
      html += '<p class="gpu-muted">Queue is empty.</p>';
    } else {
      html += '<table class="gpu-table"><thead><tr><th>#</th><th>User</th><th>GPUs</th><th>Min Tier</th><th>Purpose</th><th>Priority</th><th>Status</th><th>Time</th><th></th></tr></thead><tbody>';
      requests.forEach(function (r, i) {
        var user = getUserById(r.userId);
        html += '<tr>';
        html += '<td>' + (i + 1) + '</td>';
        html += '<td>' + esc(user ? user.name : '?') + '</td>';
        html += '<td>' + r.gpuCount + '</td>';
        html += '<td>' + tierBadge(r.minTier) + '</td>';
        html += '<td>' + esc(r.purpose) + '</td>';
        html += '<td>' + r.priorityScore.toFixed(2) + '</td>';
        html += '<td><span class="gpu-status-badge gpu-status-' + r.status + '">' + r.status + '</span></td>';
        html += '<td>';
        if (r.status === 'waiting') {
          html += timeWaiting(r.createdAt);
        } else if (r.status === 'assigned') {
          var node = getNodeById(r.nodeId);
          html += (node ? node.name : '?') + ' GPU ' + (r.gpuIds ? r.gpuIds.join(',') : '?');
        }
        html += '</td>';
        html += '<td class="gpu-actions">';
        if (r.status === 'assigned' && userId === r.userId) {
          html += '<button class="gpu-btn gpu-btn-sm gpu-btn-primary" data-action="start" data-id="' + r.id + '">Start</button> ';
        }
        if (userId === r.userId) {
          html += '<button class="gpu-btn gpu-btn-sm gpu-btn-danger" data-action="cancel" data-id="' + r.id + '">Cancel</button>';
        }
        html += '</td>';
        html += '</tr>';
      });
      html += '</tbody></table>';
    }

    html += '</div>';
    return html;
  }

  // ============================================================
  // Nodes tab
  // ============================================================

  function renderNodesTab() {
    var nodes = getNodes();
    var html = '<div class="gpu-node-grid">';

    nodes.forEach(function (node) {
      var tl = getTrafficLight(node);
      var gpuStatus = getNodeGpuStatus(node.id);
      var usedCount = gpuStatus.filter(function (g) { return g !== null; }).length;
      var isDown = node.status !== 'active';

      html += '<div class="gpu-node-card' + (isDown ? ' gpu-node-down' : '') + '">';
      html += '<div class="gpu-node-header">';
      html += trafficDot(tl) + ' <strong>' + esc(node.name) + '</strong> ' + tierBadge(node.gpuTier);
      if (isDown) html += ' <span class="gpu-status-badge gpu-status-offline">' + node.status + '</span>';
      html += '</div>';
      html += '<div class="gpu-node-model">' + esc(node.gpuModel) + ' &times; 8</div>';
      html += '<div class="gpu-node-details">';
      html += 'CUDA ' + esc(node.cudaVersion) + ' &middot; Driver ' + esc(node.driverVersion) + '<br>';
      html += 'cuDNN ' + esc(node.cudnnVersion) + ' &middot; PyTorch ' + esc(node.pytorchVersion) + '<br>';
      html += esc(node.osVersion) + ' &middot; ' + node.nodeType;
      if (node.cloudExpireAt) html += '<br>Cloud expires: ' + fmtDateShort(node.cloudExpireAt);
      html += '</div>';
      html += '<div class="gpu-node-usage">';
      if (isDown) {
        html += '<span class="gpu-muted">Unavailable</span>';
      } else {
        html += '<div class="gpu-usage-bar"><div class="gpu-usage-fill" style="width:' + (usedCount / 8 * 100) + '%"></div></div>';
        html += '<span class="gpu-muted">' + usedCount + '/8 GPUs in use</span>';
      }
      html += '</div>';
      html += '</div>';
    });

    html += '</div>';
    return html;
  }

  // ============================================================
  // Request form — live cost update
  // ============================================================

  function updateRequestCost() {
    var costDiv = document.getElementById('req-cost');
    if (!costDiv) return;

    var gpuCountEl = document.querySelector('input[name="req-gpu-count"]:checked');
    var tierEl = document.getElementById('req-min-tier');
    var durEl = document.getElementById('req-duration');
    if (!gpuCountEl || !tierEl || !durEl) return;

    var gpuCount = parseInt(gpuCountEl.value);
    var tier = parseInt(tierEl.value);
    var dur = parseInt(durEl.value) || 1;
    var cost = calcPointCost(gpuCount, tier, dur);

    costDiv.textContent = 'Estimated cost: ' + cost + ' points (' + gpuCount + ' GPU \u00d7 ' + TIER[tier].name + ' ' + TIER[tier].mult + 'x \u00d7 ' + dur + 'd)';
  }

  // ============================================================
  // Event handling (delegation)
  // ============================================================

  function bindEvents() {
    var app = document.getElementById('gpu-app');
    if (!app) return;

    app.addEventListener('click', function (e) {
      var target = e.target;

      // Tab click
      var tab = target.closest('.gpu-tab');
      if (tab) {
        currentTab = tab.dataset.tab;
        renderApp();
        return;
      }

      // Reset button
      if (target.id === 'gpu-reset-btn') {
        if (confirm('Reset all data? This cannot be undone.')) {
          Object.values(KEYS).forEach(function (k) { Store.remove(k); });
          Store.remove('gpu_reservations');
          Store.remove('gpu_queue');
          seedMockData();
          currentTab = 'dashboard';
          renderApp();
        }
        return;
      }

      // GPU cell click (free cells → switch to Request tab with preferred node)
      var cell = target.closest('.gpu-cell-free');
      if (cell) {
        prefillNodeId = cell.dataset.node;
        currentTab = 'request';
        renderApp();
        return;
      }

      // Action buttons
      var action = target.dataset.action;
      var id = target.dataset.id;

      if (action === 'start' && id) { startRequest(id); renderApp(); }
      if (action === 'end' && id) { endRequest(id); renderApp(); }
      if (action === 'cancel' && id) { cancelRequest(id); renderApp(); }
      if (action === 'extend' && id) { extendRequest(id, 1); renderApp(); }
    });

    app.addEventListener('change', function (e) {
      var target = e.target;

      // User selector
      if (target.id === 'gpu-user-select') {
        setCurrentUserId(target.value || null);
        renderApp();
        return;
      }

      // Request form field changes → update cost
      if (target.name === 'req-gpu-count' || target.id === 'req-min-tier' || target.id === 'req-duration') {
        updateRequestCost();
      }
    });

    app.addEventListener('input', function (e) {
      if (e.target.id === 'req-duration') updateRequestCost();
    });

    app.addEventListener('submit', function (e) {
      e.preventDefault();

      if (e.target.id === 'gpu-request-form') {
        handleRequestSubmit();
        return;
      }
    });
  }

  function handleRequestSubmit() {
    var userId = getCurrentUserId();
    var user = getUserById(userId);
    var gpuCount = parseInt(document.querySelector('input[name="req-gpu-count"]:checked').value);
    var minTier = parseInt(document.getElementById('req-min-tier').value);
    var preferredNode = document.getElementById('req-preferred-node').value || null;
    var duration = parseInt(document.getElementById('req-duration').value);
    var purpose = document.getElementById('req-purpose').value.trim();
    var memo = document.getElementById('req-memo').value.trim();
    var validationDiv = document.getElementById('req-validation');

    function showError(msg) {
      if (validationDiv) validationDiv.innerHTML = '<span class="gpu-error">' + esc(msg) + '</span>';
    }

    if (!userId || !user) { showError('Select a user.'); return; }
    if (!purpose) { showError('Enter a purpose.'); return; }
    if (minTier > user.maxTier) { showError('Min tier exceeds your max tier (' + TIER[user.maxTier].name + ').'); return; }
    if (duration < 1 || duration > 7) { showError('Duration must be 1-7 days.'); return; }

    var cost = calcPointCost(gpuCount, minTier, duration);
    var currentPoints = getUserActivePoints(userId);
    if (currentPoints + cost > user.maxConcurrentPoints) {
      showError('Point budget exceeded. Current: ' + currentPoints + ', cost: ' + cost + ', max: ' + user.maxConcurrentPoints);
      return;
    }

    submitRequest(userId, gpuCount, minTier, preferredNode, purpose, memo, duration);
    currentTab = 'dashboard';
    renderApp();
  }

  // ============================================================
  // Init
  // ============================================================

  function init() {
    seedMockData();
    bindEvents();
    renderApp();
    setInterval(checkExpired, 60000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
