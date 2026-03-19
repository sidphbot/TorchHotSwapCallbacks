/* research.js — Cytoscape.js graph tab for the Research & Learning module */
/* global S, cytoscape, showLoader, hideLoader, api */

(function () {
  'use strict';

  var cy = null;
  var currentLayout = 'hierarchical';
  var currentStreamId = null;
  var initialized = false;
  var _graphLoader = null;
  var _firstLoadDone = false;  // track whether first load has completed
  var _lastElementHash = '';   // track element changes for differential updates

  // ---------------------------------------------------------------------------
  // Node style mappings
  // ---------------------------------------------------------------------------

  var STATUS_COLORS = {
    proposed: '#6c7a89',
    discovered: '#9b59b6',
    shadow: '#ff6b6b',
    testing: '#f39c12',
    confirmed: '#2ecc71',
    refuted: '#e74c3c',
    inconclusive: '#95a5a6',
    exported: '#00d4aa',
    archived: '#4a4a4a',
  };

  var OUTCOME_COLORS = {
    improved: '#2ecc71',
    degraded: '#e74c3c',
    neutral: '#95a5a6',
    timeout: '#f39c12',
  };

  // Distinct run colors for the tree branches
  var RUN_COLORS = [
    '#3d9eff', '#ff9833', '#33dd77', '#ff4d5e',
    '#9966ff', '#33ccdd', '#ffd233', '#ff66b2',
  ];

  function nodeShape(d) {
    if (d.node_type === 'root') return 'round-rectangle';
    if (d.node_type === 'run') return 'round-rectangle';
    if (d.node_type === 'observation') return 'ellipse';
    if (d.node_type === 'hypothesis') return 'round-rectangle';
    if (d.node_type === 'evidence') return 'diamond';
    return 'ellipse';
  }

  function nodeColor(d) {
    if (d.node_type === 'root') return '#00d4aa';
    if (d.node_type === 'run') return d._run_color || '#3d9eff';
    if (d.node_type === 'observation') return '#00bcd4';
    if (d.node_type === 'hypothesis') return STATUS_COLORS[d.status] || '#6c7a89';
    if (d.node_type === 'evidence') return OUTCOME_COLORS[d.outcome] || '#95a5a6';
    return '#888';
  }

  function nodeSize(d) {
    if (d.node_type === 'root') return 40;
    if (d.node_type === 'run') return 34;
    var base = 22;
    var count = d.evidence_count || 0;
    return Math.min(base + count * 4, 55);
  }

  function nodeOpacity(d) {
    if (d.node_type === 'root' || d.node_type === 'run') return 1.0;
    var map = {
      proposed: 0.6,
      discovered: 0.5,
      shadow: 0.4,
      testing: 0.8,
      confirmed: 1.0,
      refuted: 0.5,
      inconclusive: 0.5,
      exported: 1.0,
      archived: 0.3,
    };
    if (d.node_type === 'hypothesis') return map[d.status] || 0.6;
    if (d.diluted) return 0.45;
    return 0.85;
  }

  function nodeLabel(d) {
    if (d.node_type === 'root') return d.label || 'Experiment';
    if (d.node_type === 'run') return d.label || d.run_id || 'run';
    if (d.node_type === 'observation') {
      var obsText = d.text ? d.text.slice(0, 40) : 'obs';
      return obsText;
    }
    if (d.node_type === 'hypothesis') {
      var lbl = d.condition ? d.condition.slice(0, 40) : 'hyp';
      if (d.status) lbl += '\n[' + d.status + ']';
      return lbl;
    }
    if (d.node_type === 'evidence') {
      var evLbl = d.outcome || 'ev';
      if (d.delta) {
        var keys = Object.keys(d.delta);
        if (keys.length > 0) evLbl += '\n' + keys[0] + ': ' + (typeof d.delta[keys[0]] === 'number' ? d.delta[keys[0]].toFixed(4) : d.delta[keys[0]]);
      }
      return evLbl;
    }
    return d.id;
  }

  function borderWidth(d) {
    if (d.node_type === 'root') return 2;
    if (d.node_type === 'run') return 2;
    var conf = d.nn_confidence || 0;
    return Math.max(conf * 4, 0.5);
  }

  // ---------------------------------------------------------------------------
  // Cytoscape initialization
  // ---------------------------------------------------------------------------

  function initCytoscape(containerId) {
    if (typeof cytoscape === 'undefined') {
      console.warn('research.js: Cytoscape.js not loaded');
      return null;
    }
    var container = document.getElementById(containerId);
    if (!container) return null;

    var rect = container.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) {
      console.warn('research.js: container too small, deferring');
      return null;
    }

    cy = cytoscape({
      container: container,
      style: [
        {
          selector: 'node',
          style: {
            label: function (el) { return nodeLabel(el.data()); },
            shape: function (el) { return nodeShape(el.data()); },
            'background-color': function (el) { return nodeColor(el.data()); },
            width: function (el) { return nodeSize(el.data()); },
            height: function (el) { return nodeSize(el.data()); },
            opacity: function (el) { return nodeOpacity(el.data()); },
            'border-width': function (el) { return borderWidth(el.data()); },
            'border-color': function (el) {
              var d = el.data();
              if (d.node_type === 'root') return '#00d4aa';
              if (d.node_type === 'run') return d._run_color || '#3d9eff';
              var conf = d.nn_confidence || 0;
              return conf > 0.5 ? '#2ecc71' : '#e74c3c';
            },
            color: '#e0e0e0',
            'font-size': '10px',
            'font-family': 'JetBrains Mono, monospace',
            'text-valign': 'bottom',
            'text-halign': 'center',
            'text-margin-y': 8,
            'text-wrap': 'wrap',
            'text-max-width': '160px',
            'text-background-color': 'rgba(8,12,18,0.75)',
            'text-background-opacity': 0.75,
            'text-background-padding': '3px',
            'text-background-shape': 'roundrectangle',
          },
        },
        // Root node — larger, accent colored, label inside
        {
          selector: 'node[node_type = "root"]',
          style: {
            'font-size': '11px',
            'font-weight': 'bold',
            'text-valign': 'center',
            'text-halign': 'center',
            color: '#080c12',
            'text-margin-y': 0,
            'text-wrap': 'none',
            'border-color': '#00d4aa',
          },
        },
        // Run node — clickable, label inside
        {
          selector: 'node[node_type = "run"]',
          style: {
            'font-size': '10px',
            'font-weight': 'bold',
            'text-valign': 'center',
            'text-halign': 'center',
            color: '#080c12',
            'text-margin-y': 0,
            'text-wrap': 'none',
            'cursor': 'pointer',
          },
        },
        {
          selector: 'edge',
          style: {
            width: function (el) {
              var t = el.data().edge_type;
              if (t === 'run_branch' || t === 'root_branch') return 2.5;
              return 1.5;
            },
            'line-color': function (el) {
              var d = el.data();
              if (d.edge_type === 'root_branch') return '#00d4aa';
              if (d.edge_type === 'run_branch') return d._run_color || '#3d9eff';
              if (d.edge_type === 'supports') return '#2ecc71';
              if (d.edge_type === 'contradicts') return '#e74c3c';
              if (d.edge_type === 'refines_to') return '#00bcd4';
              return '#444';
            },
            'target-arrow-color': function (el) {
              var d = el.data();
              if (d.edge_type === 'root_branch') return '#00d4aa';
              if (d.edge_type === 'run_branch') return d._run_color || '#3d9eff';
              if (d.edge_type === 'supports') return '#2ecc71';
              if (d.edge_type === 'contradicts') return '#e74c3c';
              return '#00bcd4';
            },
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            opacity: function (el) {
              var t = el.data().edge_type;
              return (t === 'root_branch' || t === 'run_branch') ? 0.85 : 0.6;
            },
          },
        },
        {
          selector: 'node[status = "testing"]',
          style: {
            'border-style': 'double',
            'border-width': 3,
            'border-color': '#f39c12',
          },
        },
        {
          selector: 'node[status = "shadow"]',
          style: {
            'border-style': 'dashed',
            'border-width': 2,
            'border-color': '#ff6b6b',
            'background-opacity': 0.3,
          },
        },
        {
          selector: 'node[?diluted]',
          style: {
            'overlay-color': '#ff6b6b',
            'overlay-padding': 3,
            'overlay-opacity': 0.15,
          },
        },
      ],
      layout: { name: 'grid' },
      minZoom: 0.2,
      maxZoom: 3,
      selectionType: 'additive',
    });

    // Semantic zoom
    cy.on('zoom', function () {
      var z = cy.zoom();
      cy.nodes().forEach(function (n) {
        var t = n.data().node_type;
        if (t === 'root' || t === 'run') return; // always show labels
        if (z < 0.4) {
          n.style('font-size', '0px');
        } else if (z < 0.7) {
          n.style('font-size', '8px');
          n.style('text-max-width', '100px');
        } else {
          n.style('font-size', '10px');
          n.style('text-max-width', '160px');
        }
      });
    });

    // Click handler — run nodes trigger focus, others show detail
    cy.on('tap', 'node', function (evt) {
      var d = evt.target.data();
      if (d.node_type === 'run' && d.run_dir) {
        // Focus this run in Metrics tab
        if (typeof window._focusRunFromResearch === 'function') {
          window._focusRunFromResearch(d.run_dir, d.label || d.run_id);
        }
      } else {
        showNodeDetail(d);
      }
    });

    // Hover tooltip for run nodes
    cy.on('mouseover', 'node', function (evt) {
      var d = evt.target.data();
      var container = document.getElementById(containerId);
      if (!container) return;
      var tip = container.querySelector('.cy-tooltip');
      if (!tip) {
        tip = document.createElement('div');
        tip.className = 'cy-tooltip';
        tip.style.cssText = 'position:absolute;z-index:20;pointer-events:none;' +
          'background:rgba(8,12,18,0.92);border:1px solid var(--border-bright);' +
          'border-radius:4px;padding:6px 8px;font-size:9px;font-family:var(--font-mono);' +
          'color:var(--text-primary);max-width:220px;line-height:1.4;display:none;';
        container.appendChild(tip);
      }
      var html = '';
      if (d.node_type === 'run') {
        html = '<b style="color:' + (d._run_color || '#3d9eff') + '">' + esc(d.label || d.run_id) + '</b>';
        if (d.step_count) html += '<br>' + d.step_count + ' steps';
        if (d.run_dir) html += '<br><span style="color:var(--cyan)">Click to focus</span>';
      } else if (d.node_type === 'hypothesis') {
        html = '<b style="color:' + (STATUS_COLORS[d.status] || '#888') + '">' + esc(d.condition || '') + '</b>';
        html += '<br>Status: ' + d.status;
        html += '<br>Confidence: ' + ((d.confidence || 0) * 100).toFixed(0) + '%';
        if (d.evidence_count) html += '<br>Evidence: ' + d.evidence_count;
        if (d.nn_confidence > 0) html += '<br>NN: ' + (d.nn_confidence * 100).toFixed(0) + '%';
      } else if (d.node_type === 'evidence') {
        html = '<b style="color:' + (OUTCOME_COLORS[d.outcome] || '#888') + '">' + (d.outcome || 'evidence') + '</b>';
        if (d.delta) html += '<br>' + esc(JSON.stringify(d.delta));
      } else if (d.node_type === 'observation') {
        html = esc(d.text || 'observation');
      } else if (d.node_type === 'root') {
        html = '<b>Experiment Root</b>';
      }
      if (html) {
        tip.innerHTML = html;
        tip.style.display = 'block';
        var pos = evt.target.renderedPosition();
        tip.style.left = (pos.x + 15) + 'px';
        tip.style.top = (pos.y - 10) + 'px';
      }
    });

    cy.on('mouseout', 'node', function () {
      var container = document.getElementById(containerId);
      if (!container) return;
      var tip = container.querySelector('.cy-tooltip');
      if (tip) tip.style.display = 'none';
    });

    return cy;
  }

  // ---------------------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------------------

  async function fetchGraph(streamId) {
    var url = '/api/research/graph';
    if (streamId) url += '?stream_id=' + encodeURIComponent(streamId);
    try {
      var resp = await fetch(url);
      var data = await resp.json();
      return data.elements || [];
    } catch (e) {
      console.warn('Failed to fetch research graph:', e);
      return [];
    }
  }

  async function fetchRuns() {
    try {
      var data = await api('GET', '/api/runs/discover');
      return (data && data.runs) ? data.runs : [];
    } catch (e) {
      return [];
    }
  }

  async function fetchStreams() {
    try {
      var resp = await fetch('/api/research/streams');
      var data = await resp.json();
      return data.streams || [];
    } catch (e) {
      return [];
    }
  }

  async function fetchStats() {
    try {
      var resp = await fetch('/api/research/stats');
      return await resp.json();
    } catch (e) {
      return {};
    }
  }

  // ---------------------------------------------------------------------------
  // Tree builder — merges research graph + discovered runs into a mindmap
  // ---------------------------------------------------------------------------

  function buildTreeElements(graphElements, runs) {
    var elements = [];
    var existingIds = new Set();

    // Root node
    elements.push({ data: { id: '_root', node_type: 'root', label: 'Experiment' } });

    // Run nodes — one per discovered run
    var runMap = {};  // run_id → run color
    runs.forEach(function (run, idx) {
      var runId = '_run_' + (run.run_id || idx);
      var color = RUN_COLORS[idx % RUN_COLORS.length];
      runMap[run.run_id || ''] = { nodeId: runId, color: color };

      elements.push({
        data: {
          id: runId,
          node_type: 'run',
          run_id: run.run_id,
          label: run.label || run.run_id || 'run-' + idx,
          run_dir: run.dir || '',
          step_count: run.step_count || 0,
          _run_color: color,
        },
      });
      // Edge: root → run
      elements.push({
        data: {
          id: '_edge_root_' + runId,
          source: '_root',
          target: runId,
          edge_type: 'root_branch',
        },
      });
    });

    // If no runs discovered, create a default "current" run node
    if (runs.length === 0) {
      var defaultRunId = '_run_current';
      runMap[''] = { nodeId: defaultRunId, color: RUN_COLORS[0] };
      elements.push({
        data: {
          id: defaultRunId,
          node_type: 'run',
          run_id: 'current',
          label: 'Current Run',
          run_dir: '',
          _run_color: RUN_COLORS[0],
        },
      });
      elements.push({
        data: {
          id: '_edge_root_current',
          source: '_root',
          target: defaultRunId,
          edge_type: 'root_branch',
        },
      });
    }

    // Add all original graph elements (hypotheses, observations, evidence)
    graphElements.forEach(function (el) {
      if (el.data && el.data.id) {
        existingIds.add(el.data.id);
        elements.push(el);
      }
    });

    // Connect hypotheses to their run nodes
    // Try to match via stream_id, condition text, or just attach to first run
    graphElements.forEach(function (el) {
      var d = el.data;
      if (!d || d.source || d.target) return;  // skip edges
      if (d.node_type !== 'hypothesis') return;

      // Try to find a matching run
      var attached = false;
      Object.keys(runMap).forEach(function (runId) {
        if (attached) return;
        // Match by condition containing run_id or label
        var runInfo = runMap[runId];
        var condLower = (d.condition || '').toLowerCase();
        var runIdLower = runId.toLowerCase();
        if (runIdLower && condLower.indexOf(runIdLower) !== -1) {
          elements.push({
            data: {
              id: '_branch_' + runInfo.nodeId + '_' + d.id,
              source: runInfo.nodeId,
              target: d.id,
              edge_type: 'run_branch',
              _run_color: runInfo.color,
            },
          });
          attached = true;
        }
      });

      // If no match, attach to first run (or default)
      if (!attached) {
        var firstKey = Object.keys(runMap)[0] || '';
        var first = runMap[firstKey];
        if (first) {
          elements.push({
            data: {
              id: '_branch_' + first.nodeId + '_' + d.id,
              source: first.nodeId,
              target: d.id,
              edge_type: 'run_branch',
              _run_color: first.color,
            },
          });
        }
      }
    });

    return elements;
  }

  // ---------------------------------------------------------------------------
  // Element hash — detect whether graph data actually changed
  // ---------------------------------------------------------------------------

  function computeElementHash(elements) {
    // Quick hash: sorted element IDs + key data fields
    var parts = [];
    elements.forEach(function (el) {
      var d = el.data;
      if (!d) return;
      if (d.source) {
        parts.push('e:' + d.id);
      } else {
        parts.push('n:' + d.id + ':' + (d.status || '') + ':' + (d.outcome || '') + ':' + (d.evidence_count || 0));
      }
    });
    parts.sort();
    return parts.join('|');
  }

  // ---------------------------------------------------------------------------
  // Graph update — seamless after first load
  // ---------------------------------------------------------------------------

  var _refreshInFlight = false;

  async function refreshGraph() {
    if (_refreshInFlight) return;  // prevent overlapping refreshes
    _refreshInFlight = true;

    try {
      if (!cy) {
        cy = initCytoscape('researchCyContainer');
        if (!cy) return;
      }

      cy.resize();

      // Only show loader on first load — subsequent refreshes are seamless
      if (!_firstLoadDone) {
        var container = document.getElementById('researchCyContainer');
        if (!_graphLoader && container) {
          _graphLoader = showLoader(container, 'Loading research graph\u2026');
        }
      }

      // Fetch graph data and run list in parallel
      var results = await Promise.all([
        fetchGraph(currentStreamId),
        fetchRuns(),
      ]);
      var graphElements = results[0];
      var runs = results[1];

      // Build tree structure
      var treeElements = buildTreeElements(graphElements, runs);

      // Check if data actually changed
      var newHash = computeElementHash(treeElements);
      if (_firstLoadDone && newHash === _lastElementHash) {
        // Data unchanged — skip DOM update, just update stats
        updateStats();
        return;
      }
      _lastElementHash = newHash;

      if (!_firstLoadDone) {
        // First load — full replace + layout + hide loader
        cy.elements().remove();
        if (treeElements.length > 0) {
          cy.add(treeElements);
          applyLayout(currentLayout);
          setTimeout(function () {
            cy.resize();
            cy.fit(undefined, 30);
            hideLoader(_graphLoader);
            _graphLoader = null;
            _firstLoadDone = true;
          }, 400);
        } else {
          hideLoader(_graphLoader);
          _graphLoader = null;
          _firstLoadDone = true;
        }
      } else {
        // Subsequent refresh — differential update preserving viewport
        var zoom = cy.zoom();
        var pan = cy.pan();

        // Build ID sets for new and existing elements
        var newIds = new Set();
        var newEdgeIds = new Set();
        treeElements.forEach(function (el) {
          if (el.data.source) newEdgeIds.add(el.data.id);
          else newIds.add(el.data.id);
        });

        // Remove elements that no longer exist
        cy.elements().forEach(function (el) {
          var id = el.data('id');
          if (el.isEdge()) {
            if (!newEdgeIds.has(id)) el.remove();
          } else {
            if (!newIds.has(id)) el.remove();
          }
        });

        // Add or update elements
        var existingNodeIds = new Set();
        var existingEdgeIds = new Set();
        cy.nodes().forEach(function (n) { existingNodeIds.add(n.data('id')); });
        cy.edges().forEach(function (e) { existingEdgeIds.add(e.data('id')); });

        var toAdd = [];
        treeElements.forEach(function (el) {
          var id = el.data.id;
          if (el.data.source) {
            // Edge
            if (!existingEdgeIds.has(id)) toAdd.push(el);
          } else {
            // Node
            if (existingNodeIds.has(id)) {
              // Update existing node data (status, evidence_count, etc.)
              var existingNode = cy.getElementById(id);
              if (existingNode.length) {
                var oldData = existingNode.data();
                var changed = false;
                Object.keys(el.data).forEach(function (k) {
                  if (el.data[k] !== oldData[k]) changed = true;
                });
                if (changed) existingNode.data(el.data);
              }
            } else {
              toAdd.push(el);
            }
          }
        });

        if (toAdd.length > 0) {
          cy.add(toAdd);
          // Only re-layout if new nodes were added
          applyLayout(currentLayout);
        }

        // Restore viewport
        cy.viewport({ zoom: zoom, pan: pan });
      }

      updateStats();
    } finally {
      _refreshInFlight = false;
    }
  }

  async function updateStats() {
    var stats = await fetchStats();
    var el = document.getElementById('researchStats');
    if (el && stats) {
      // Build rich info panel HTML
      var hyps = stats.hypotheses || 0;
      var obs = stats.observations || 0;
      var ev = stats.evidence || 0;
      var edges = stats.edges || 0;

      var html = '<div class="research-info-grid">';

      // Row 1: Summary counts
      html += '<div class="research-info-row">';
      html += '<span class="research-info-stat"><b>' + hyps + '</b> hypotheses</span>';
      html += '<span class="research-info-stat"><b>' + obs + '</b> observations</span>';
      html += '<span class="research-info-stat"><b>' + ev + '</b> evidence</span>';
      html += '<span class="research-info-stat"><b>' + edges + '</b> edges</span>';
      html += '</div>';

      // Row 2: Status breakdown (if we have hypotheses)
      if (hyps > 0 && stats.status_counts) {
        html += '<div class="research-info-row research-info-status">';
        var statusOrder = ['confirmed', 'testing', 'proposed', 'discovered', 'refuted', 'inconclusive', 'shadow', 'exported', 'archived'];
        statusOrder.forEach(function (s) {
          var count = stats.status_counts[s] || 0;
          if (count > 0) {
            var c = STATUS_COLORS[s] || '#888';
            html += '<span class="research-status-badge" style="border-color:' + c + ';color:' + c + '">' + count + ' ' + s + '</span>';
          }
        });
        html += '</div>';
      }

      // Row 3: Warnings
      if (stats.shadow_count > 0 || stats.diluted_count > 0) {
        html += '<div class="research-info-row research-info-warnings">';
        if (stats.shadow_count > 0) {
          html += '<span style="color:#ff6b6b">\u26A0 ' + stats.shadow_count + ' shadow mutations</span>';
        }
        if (stats.diluted_count > 0) {
          html += '<span style="color:#f39c12">' + stats.diluted_count + ' diluted results</span>';
        }
        html += '</div>';
      }

      html += '</div>';
      el.innerHTML = html;
    }
    // Contamination banner
    var banner = document.getElementById('researchContamBanner');
    if (banner) {
      if (stats && stats.unacknowledged_shadows > 0) {
        banner.style.display = 'block';
        banner.textContent = '\u26A0 ' + stats.unacknowledged_shadows + ' unacknowledged mutation(s) \u2014 downstream results may be diluted';
      } else {
        banner.style.display = 'none';
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Layouts
  // ---------------------------------------------------------------------------

  function applyLayout(mode) {
    if (!cy || cy.elements().length === 0) return;
    currentLayout = mode;

    var opts;
    if (mode === 'hierarchical') {
      opts = {
        name: 'breadthfirst',
        directed: true,
        roots: cy.nodes('[node_type = "root"]'),
        spacingFactor: 1.4,
        animate: true,
        animationDuration: 300,
      };
    } else if (mode === 'force') {
      opts = {
        name: 'cose',
        animate: true,
        animationDuration: 400,
        nodeRepulsion: function () { return 10000; },
        idealEdgeLength: function () { return 90; },
        gravity: 0.3,
      };
    } else if (mode === 'timeline') {
      var typeY = { root: 0, run: 1, observation: 2, hypothesis: 2, evidence: 3 };
      cy.nodes().forEach(function (n) {
        var d = n.data();
        var step = d.step || d.step_created || 0;
        if (d.node_type === 'root') step = -50;
        if (d.node_type === 'run') step = -20;
        var y = (typeY[d.node_type] || 0) * 120;
        n.position({ x: step * 3, y: y });
      });
      cy.fit(undefined, 40);
      return;
    } else {
      opts = { name: 'grid', animate: true };
    }

    cy.layout(opts).run();
  }

  // ---------------------------------------------------------------------------
  // Node detail panel
  // ---------------------------------------------------------------------------

  function showNodeDetail(d) {
    var panel = document.getElementById('researchDetail');
    if (!panel) return;

    var html = '<div style="font-size:10px;font-family:var(--font-mono);line-height:1.6">';

    // Close button
    html += '<div style="text-align:right"><button class="btn btn-sm" onclick="document.getElementById(\'researchDetail\').style.display=\'none\'" style="font-size:8px;padding:1px 4px">\u2715</button></div>';

    // Type badge
    var typeColor = nodeColor(d);
    html += '<div style="font-weight:700;color:' + typeColor + ';margin-bottom:6px;font-size:11px;text-transform:uppercase;letter-spacing:0.5px">' + (d.node_type || '').toUpperCase() + '</div>';

    if (d.node_type === 'run') {
      html += '<div style="font-weight:600;color:var(--text-primary)">' + esc(d.label || d.run_id) + '</div>';
      if (d.step_count) html += '<div style="color:var(--text-muted)">' + d.step_count + ' steps</div>';
      if (d.run_dir) {
        html += '<div style="margin-top:6px"><button class="btn btn-sm btn-accent" onclick="window._focusRunFromResearch(\'' + esc(d.run_dir) + '\',\'' + esc(d.label || d.run_id) + '\')" style="font-size:9px">Focus in Metrics</button></div>';
      }
    } else if (d.node_type === 'observation') {
      html += '<div>' + esc(d.text) + '</div>';
      html += '<div style="color:var(--text-muted)">step: ' + d.step + ' \u00B7 source: ' + d.source + '</div>';
      if (d.tags && d.tags.length) html += '<div style="color:var(--cyan)">' + d.tags.join(', ') + '</div>';
    } else if (d.node_type === 'hypothesis') {
      html += '<div><b>Condition:</b> ' + esc(d.condition) + '</div>';
      html += '<div><b>Expected:</b> ' + esc(d.expected_outcome) + '</div>';
      html += '<div><b>Status:</b> <span style="color:' + (STATUS_COLORS[d.status] || '#888') + '">' + d.status + '</span></div>';
      html += '<div><b>Confidence:</b> ' + ((d.confidence || 0) * 100).toFixed(0) + '%';
      if (d.nn_confidence > 0) html += ' \u00B7 <b>NN:</b> ' + (d.nn_confidence * 100).toFixed(0) + '%';
      html += '</div>';
      html += '<div><b>Evidence:</b> ' + (d.evidence_count || 0) + ' \u00B7 <b>Tests:</b> ' + (d.test_count || 0) + '</div>';
      if (d.intervention && Object.keys(d.intervention).length) {
        html += '<div style="margin-top:4px"><b>Intervention:</b></div>';
        html += '<pre style="font-size:9px;background:rgba(0,0,0,0.2);padding:4px;border-radius:3px;overflow-x:auto;white-space:pre-wrap">' + esc(JSON.stringify(d.intervention, null, 1)) + '</pre>';
      }

      if (d.diluted) {
        html += '<div style="color:#ff6b6b;margin-top:4px;font-weight:700">\u26A0 DILUTED</div>';
      }

      // Action buttons
      html += '<div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:4px">';
      if (d.status === 'shadow') {
        html += '<button class="btn btn-sm" onclick="window._researchAckShadow(\'' + d.id + '\')">Ack</button>';
        html += '<button class="btn btn-sm btn-accent" onclick="window._researchPromoteShadow(\'' + d.id + '\')">Promote</button>';
      } else if (d.status === 'proposed') {
        html += '<button class="btn btn-sm btn-accent" onclick="window._researchTestHyp(\'' + d.id + '\')">Test</button>';
      }
      if (['testing', 'proposed'].indexOf(d.status) !== -1) {
        html += '<button class="btn btn-sm" onclick="window._researchConcludeHyp(\'' + d.id + '\',\'confirmed\')">Confirm</button>';
        html += '<button class="btn btn-sm" onclick="window._researchConcludeHyp(\'' + d.id + '\',\'refuted\')">Refute</button>';
      }
      html += '</div>';
    } else if (d.node_type === 'evidence') {
      html += '<div><b>Outcome:</b> <span style="color:' + (OUTCOME_COLORS[d.outcome] || '#888') + '">' + d.outcome + '</span></div>';
      if (d.step !== undefined) html += '<div><b>Step:</b> ' + d.step + '</div>';
      if (d.delta && Object.keys(d.delta).length) {
        html += '<div style="margin-top:4px"><b>Deltas:</b></div>';
        html += '<pre style="font-size:9px;background:rgba(0,0,0,0.2);padding:4px;border-radius:3px;white-space:pre-wrap">' + esc(JSON.stringify(d.delta, null, 1)) + '</pre>';
      }
    }

    html += '</div>';
    panel.innerHTML = html;
    panel.style.display = 'block';
  }

  function esc(s) {
    if (!s) return '';
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/'/g, '&#39;');
  }

  // ---------------------------------------------------------------------------
  // Stream filter
  // ---------------------------------------------------------------------------

  async function populateStreamFilter() {
    var sel = document.getElementById('researchStreamFilter');
    if (!sel) return;
    var streams = await fetchStreams();
    sel.innerHTML = '<option value="">All streams</option>';
    streams.forEach(function (s) {
      var opt = document.createElement('option');
      opt.value = s.stream_id;
      opt.textContent = s.name + (s.status === 'concluded' ? ' (done)' : '');
      sel.appendChild(opt);
    });
  }

  // ---------------------------------------------------------------------------
  // Actions (exposed globally for inline onclick handlers)
  // ---------------------------------------------------------------------------

  window._researchTestHyp = async function (hypId) {
    try {
      await fetch('/api/research/hypotheses/' + hypId + '/test', { method: 'POST' });
      refreshGraph();
    } catch (e) {
      console.warn('Test hypothesis failed:', e);
    }
  };

  window._researchAckShadow = async function (hypId) {
    try {
      await fetch('/api/research/shadows/' + hypId + '/acknowledge', { method: 'POST' });
      refreshGraph();
    } catch (e) {
      console.warn('Acknowledge shadow failed:', e);
    }
  };

  window._researchPromoteShadow = async function (hypId) {
    var condition = prompt('Condition (what was the reason for this mutation?):', '');
    var expected = prompt('Expected outcome:', '');
    try {
      await fetch('/api/research/shadows/' + hypId + '/promote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ condition: condition || '', expected_outcome: expected || '' }),
      });
      refreshGraph();
    } catch (e) {
      console.warn('Promote shadow failed:', e);
    }
  };

  window._researchConcludeHyp = async function (hypId, status) {
    try {
      await fetch('/api/research/hypotheses/' + hypId + '/conclude', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: status, reason: '' }),
      });
      refreshGraph();
    } catch (e) {
      console.warn('Conclude hypothesis failed:', e);
    }
  };

  // ---------------------------------------------------------------------------
  // Tab initialization
  // ---------------------------------------------------------------------------

  function initResearchTab() {
    if (initialized) return;

    var container = document.getElementById('researchCyContainer');
    if (!container) return;

    var rect = container.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) {
      setTimeout(initResearchTab, 300);
      return;
    }

    initialized = true;

    // Show loading immediately (first load only)
    _graphLoader = showLoader(container, 'Initializing research graph\u2026');

    cy = initCytoscape('researchCyContainer');

    // Layout selector
    var layoutSel = document.getElementById('researchLayout');
    if (layoutSel) {
      layoutSel.addEventListener('change', function () {
        applyLayout(this.value);
      });
    }

    // Stream filter
    var streamSel = document.getElementById('researchStreamFilter');
    if (streamSel) {
      streamSel.addEventListener('change', function () {
        currentStreamId = this.value || null;
        // Force full refresh on stream change
        _firstLoadDone = false;
        _lastElementHash = '';
        refreshGraph();
      });
    }

    // Refresh button
    var btnRefresh = document.getElementById('btnResearchRefresh');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', function () {
        populateStreamFilter();
        // Force full refresh on manual click
        _lastElementHash = '';
        refreshGraph();
      });
    }

    // Export recipe button
    var btnExport = document.getElementById('btnResearchExportRecipe');
    if (btnExport) {
      btnExport.addEventListener('click', async function () {
        try {
          var resp = await fetch('/api/research/export-recipe', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ out: null }),
          });
          var data = await resp.json();
          alert('Recipe exported to: ' + (data.path || 'unknown'));
        } catch (e) {
          alert('Export failed: ' + e.message);
        }
      });
    }

    // Load conditions button
    var btnLoadCond = document.getElementById('btnResearchLoadConditions');
    if (btnLoadCond) {
      btnLoadCond.addEventListener('click', async function () {
        var task = prompt('Task (mnist or cifar10):', 'mnist');
        if (!task) return;
        var baseDir = prompt('Base run directory (with checkpoints/):', '');
        try {
          var resp = await fetch('/api/research/load-conditions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task: task, base_run_dir: baseDir || '' }),
          });
          var data = await resp.json();
          alert('Loaded ' + (data.loaded || 0) + ' continuation conditions as hypotheses');
          refreshGraph();
        } catch (e) {
          alert('Load failed: ' + e.message);
        }
      });
    }

    // Launch selected tests button
    var btnLaunchTests = document.getElementById('btnResearchLaunchTests');
    if (btnLaunchTests) {
      btnLaunchTests.addEventListener('click', async function () {
        if (!cy) return;
        var selected = cy.nodes(':selected').filter('[node_type = "hypothesis"][status = "proposed"]');
        if (selected.length === 0) {
          alert('Select proposed hypothesis nodes first (shift+click to multi-select)');
          return;
        }
        var baseDir = prompt('Base run directory (with checkpoints/):', '');
        if (!baseDir) return;
        var ids = [];
        selected.forEach(function (n) { ids.push(n.data().id); });
        btnLaunchTests.disabled = true;
        btnLaunchTests.textContent = 'Running...';
        try {
          var resp = await fetch('/api/research/launch-tests', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hypothesis_ids: ids, base_run_dir: baseDir }),
          });
          var data = await resp.json();
          var results = data.results || [];
          var msg = results.map(function (r) {
            return r.hypothesis_id + ': ' + (r.outcome || r.error || 'unknown');
          }).join('\n');
          alert('Results:\n' + msg);
          refreshGraph();
        } catch (e) {
          alert('Launch failed: ' + e.message);
        } finally {
          btnLaunchTests.disabled = false;
          btnLaunchTests.textContent = 'Launch Selected';
        }
      });
    }

    // Initial load
    populateStreamFilter();
    refreshGraph();
  }

  // Auto-refresh (only when tab visible, 10s interval)
  var refreshTimer = null;

  function startAutoRefresh() {
    if (refreshTimer) return;
    refreshTimer = setInterval(function () {
      var tab = document.querySelector('.tab-content[data-tab="research"]');
      if (tab && tab.classList.contains('active')) {
        refreshGraph();
      }
    }, 10000);
  }

  // ---------------------------------------------------------------------------
  // Eval status polling
  // ---------------------------------------------------------------------------

  var _lastFocusRun = null;
  var _evalDone = false;
  var _evalStatusTimer = null;

  async function pollEvalStatus() {
    try {
      var resp = await fetch('/api/eval/status');
      var data = await resp.json();
      if (!data || !data.active) {
        _hideEvalBanner();
        return;
      }

      var banner = document.getElementById('evalProgressBanner');
      var textEl = document.getElementById('evalProgressText');
      var barEl = document.getElementById('evalProgressBar');
      var runLabel = document.getElementById('evalRunningLabel');
      if (!banner) return;

      banner.style.display = 'flex';
      var pct = data.total > 0 ? Math.round(data.completed / data.total * 100) : 0;
      textEl.textContent = 'Eval: ' + data.completed + '/' + data.total;
      barEl.style.width = pct + '%';

      if (data.running) {
        runLabel.textContent = 'running: ' + data.running;
      } else if (data.done) {
        runLabel.textContent = 'complete';
        barEl.style.background = '#2ecc71';
      } else {
        runLabel.textContent = '';
      }

      if (data.focus_run && data.focus_run !== _lastFocusRun) {
        _lastFocusRun = data.focus_run;
        _autoFocusRun(data.focus_run);
      }

      if (data.done && !_evalDone) {
        _evalDone = true;
        refreshGraph();
      }
    } catch (e) {
      // Eval status not available — normal for non-eval mode
    }
  }

  async function _autoFocusRun(condName) {
    if (!window._initialLoadDone) {
      setTimeout(function () { _autoFocusRun(condName); }, 500);
      return;
    }
    try {
      var resp = await fetch('/api/runs/discover');
      var data = await resp.json();
      if (!data || !data.runs) return;
      for (var i = 0; i < data.runs.length; i++) {
        var run = data.runs[i];
        if (run.run_id === condName || run.label === condName) {
          if (run.dir && typeof window._focusRunFromResearch === 'function') {
            window._focusRunFromResearch(run.dir, run.label || condName);
          }
          return;
        }
      }
    } catch (e) {
      // ignore
    }
  }

  function _hideEvalBanner() {
    var banner = document.getElementById('evalProgressBanner');
    if (banner) banner.style.display = 'none';
  }

  // ---------------------------------------------------------------------------
  // Tab switch listener
  // ---------------------------------------------------------------------------

  document.addEventListener('DOMContentLoaded', function () {
    var tabs = document.querySelector('.tabs');
    if (tabs) {
      tabs.addEventListener('click', function (e) {
        var t = e.target.closest('.tab');
        if (t && t.dataset.tab === 'research') {
          setTimeout(function () {
            if (!initialized) {
              initResearchTab();
            } else if (cy) {
              cy.resize();
              refreshGraph();
            }
            startAutoRefresh();
          }, 100);
        }
      });
    }

    _evalStatusTimer = setInterval(pollEvalStatus, 3000);
    setTimeout(pollEvalStatus, 500);
  });
})();
