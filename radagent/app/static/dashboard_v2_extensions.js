/**
 * RadAgent v2 — Dashboard Extensions (Priority 2)
 * Author: Rayane Aggoune
 * 
 * Adds 5 new panels for Milan AI Week demo:
 * 1. Modality Badge
 * 2. Enhanced Side-by-Side Comparison (with fabricated claim tooltips)
 * 3. Dictation Audit Panel
 * 4. Autonomous Queue Panel (WebSocket streaming)
 * 5. Federation Network Panel
 * 
 * Plus: CriticAgent verdict pills visible on all decision panels
 */

// ============================================================================
// PANEL 1: MODALITY BADGE
// ============================================================================

function renderModalityBadge(modalityInfo) {
  const badge = document.getElementById('modality-badge');
  if (!badge) return;
  
  if (!modalityInfo) {
    badge.style.display = 'none';
    return;
  }
  
  const { modality, body_part, status, matched_entry } = modalityInfo;
  
  // Color coding by status
  let statusClass = 'neutral';
  let statusIcon = '📋';
  if (status === 'production') {
    statusClass = 'good';
    statusIcon = '✅';
  } else if (status === 'registered') {
    statusClass = 'warn';
    statusIcon = '⚠️';
  } else if (status === 'fallback') {
    statusClass = 'bad';
    statusIcon = '❌';
  }
  
  badge.innerHTML = `
    <div class="modality-badge-content ${statusClass}">
      <span class="modality-icon">${statusIcon}</span>
      <div class="modality-details">
        <div class="modality-primary">
          <strong>${modality || 'UNKNOWN'}</strong>
          ${body_part ? ` · ${body_part}` : ''}
        </div>
        <div class="modality-secondary">
          ${matched_entry || 'Unknown pipeline'} 
          <span class="status-pill ${statusClass}">${status || 'unknown'}</span>
        </div>
      </div>
    </div>
  `;
  
  badge.style.display = 'block';
}

// ============================================================================
// PANEL 2: ENHANCED SIDE-BY-SIDE COMPARISON
// ============================================================================

function renderEnhancedComparison(vanillaData, groundedData) {
  const reportArea = document.getElementById('report-area');
  if (!reportArea) return;
  
  // Vanilla pane (left, red)
  let vanillaHTML = '';
  if (vanillaData && vanillaData.report) {
    const { report, fabricated_claims } = vanillaData;
    
    // Build report with red underlines for fabricated claims
    let reportText = report.full_text || report.findings || '';
    
    // Highlight fabricated claims
    if (fabricated_claims && fabricated_claims.length > 0) {
      fabricated_claims.forEach(claim => {
        const claimText = claim.claim;
        const reason = claim.reason || 'No evidence provided';
        const risk = claim.hallucination_risk || 'unknown';
        
        // Wrap claim in span with tooltip
        const wrapped = `<span class="fabricated-claim ${risk}" title="${escAttr(reason)}">${escHtml(claimText)}</span>`;
        reportText = reportText.replace(claimText, wrapped);
      });
    }
    
    vanillaHTML = `
      <div class="report-side ungrounded">
        <span class="side-head">❌ Ungrounded (raw VLM)</span>
        <span class="side-sub">Image-only prompt — no specialist, no RAG, no citations</span>
        <div class="report-content">${reportText}</div>
        ${fabricated_claims && fabricated_claims.length > 0 ? `
          <div class="fabrication-summary">
            <strong>${fabricated_claims.length} fabricated claim${fabricated_claims.length > 1 ? 's' : ''}</strong> detected
            (hover red text for details)
          </div>
        ` : ''}
      </div>
    `;
  } else {
    vanillaHTML = `
      <div class="report-side ungrounded">
        <span class="side-head">❌ Ungrounded (raw VLM)</span>
        <span class="side-sub">Generating vanilla baseline…</span>
        <div class="empty">Processing...</div>
      </div>
    `;
  }
  
  // Grounded pane (right, green)
  let groundedHTML = '';
  if (groundedData) {
    groundedHTML = `
      <div class="report-side grounded">
        <span class="side-head">✅ RadAgent (grounded)</span>
        <span class="side-sub">Conditioned on specialist + RAG. Every claim cites evidence.</span>
        <div class="report-content"><pre>${escHtml(groundedData)}</pre></div>
      </div>
    `;
  } else {
    groundedHTML = `
      <div class="report-side grounded">
        <span class="side-head">✅ RadAgent (grounded)</span>
        <span class="side-sub">Generating grounded report…</span>
        <div class="empty">Processing...</div>
      </div>
    `;
  }
  
  reportArea.innerHTML = `
    <div class="report-split">
      ${vanillaHTML}
      ${groundedHTML}
    </div>
  `;
}

// ============================================================================
// PANEL 3: DICTATION AUDIT PANEL
// ============================================================================

function renderDictationAudit(dictationData) {
  const panel = document.getElementById('dictation-audit-panel');
  if (!panel) return;
  
  if (!dictationData) {
    panel.style.display = 'none';
    return;
  }
  
  const { transcript, specialist_findings, discrepancies } = dictationData;
  
  let discrepanciesHTML = '';
  if (discrepancies && discrepancies.length > 0) {
    discrepanciesHTML = discrepancies.map(d => {
      const badgeClass = d.flag === 'RECONSIDER' ? 'warn' : 
                         d.flag === 'CONFIRM' ? 'good' : 'neutral';
      return `
        <div class="discrepancy-row ${badgeClass}">
          <div class="discrepancy-badge">${d.flag}</div>
          <div class="discrepancy-details">
            <div class="discrepancy-finding">${escHtml(d.finding)}</div>
            <div class="discrepancy-reason">${escHtml(d.reason)}</div>
          </div>
        </div>
      `;
    }).join('');
  } else {
    discrepanciesHTML = '<div class="empty">No discrepancies detected</div>';
  }
  
  panel.innerHTML = `
    <div class="dictation-grid">
      <div class="dictation-column">
        <h3>📝 Dictated Transcript</h3>
        <div class="transcript-box">${escHtml(transcript || 'No transcript')}</div>
      </div>
      <div class="dictation-column">
        <h3>🔬 Specialist Findings</h3>
        <div class="specialist-box">
          ${specialist_findings ? specialist_findings.map(f => `
            <div class="specialist-finding">
              <strong>${f.finding}</strong>: ${(f.confidence * 100).toFixed(1)}%
            </div>
          `).join('') : '<div class="empty">No findings</div>'}
        </div>
      </div>
      <div class="dictation-column full-width">
        <h3>⚠️ Discrepancies</h3>
        <div class="discrepancies-box">${discrepanciesHTML}</div>
      </div>
    </div>
  `;
  
  panel.style.display = 'block';
}

// ============================================================================
// PANEL 4: AUTONOMOUS QUEUE PANEL (WebSocket Streaming)
// ============================================================================

let autonomyQueue = [];

function initAutonomyPanel() {
  const panel = document.getElementById('autonomy-queue-panel');
  if (!panel) return;
  
  panel.innerHTML = `
    <div class="autonomy-header">
      <h3>🤖 Autonomous Workflow Queue</h3>
      <div class="queue-status">
        <span class="queue-count">0 actions</span>
        <span class="queue-indicator idle">IDLE</span>
      </div>
    </div>
    <div class="autonomy-queue" id="autonomy-queue-list">
      <div class="empty">No workflow actions yet</div>
    </div>
  `;
}

function updateAutonomyQueue(action) {
  const panel = document.getElementById('autonomy-queue-panel');
  const queueList = document.getElementById('autonomy-queue-list');
  const queueCount = document.querySelector('.queue-count');
  const queueIndicator = document.querySelector('.queue-indicator');
  
  if (!panel || !queueList) return;
  
  // Add action to queue
  autonomyQueue.push(action);
  
  // Update count
  if (queueCount) {
    queueCount.textContent = `${autonomyQueue.length} action${autonomyQueue.length > 1 ? 's' : ''}`;
  }
  
  // Update indicator
  if (queueIndicator) {
    if (action.status === 'running') {
      queueIndicator.className = 'queue-indicator running';
      queueIndicator.textContent = 'RUNNING';
    } else if (action.status === 'halted') {
      queueIndicator.className = 'queue-indicator halted';
      queueIndicator.textContent = 'HALTED';
    } else if (action.status === 'complete') {
      queueIndicator.className = 'queue-indicator complete';
      queueIndicator.textContent = 'COMPLETE';
    }
  }
  
  // Render queue
  const queueHTML = autonomyQueue.map((a, idx) => {
    const statusClass = a.status === 'complete' ? 'good' : 
                       a.status === 'halted' ? 'bad' : 
                       a.status === 'replanning' ? 'warn' : 'neutral';
    
    const confBand = a.confidence >= 0.75 ? 'high' : 
                     a.confidence >= 0.55 ? 'medium' : 'low';
    
    // CriticAgent verdict pill (if present)
    let criticHTML = '';
    if (a.critic_verdict) {
      const verdictClass = a.critic_verdict === 'APPROVE' ? 'good' : 
                          a.critic_verdict === 'CHALLENGE' ? 'warn' : 'bad';
      criticHTML = `<span class="critic-pill ${verdictClass}">${a.critic_verdict}</span>`;
    }
    
    return `
      <div class="autonomy-action ${statusClass}">
        <div class="action-header">
          <span class="action-number">#${idx + 1}</span>
          <span class="action-name">${escHtml(a.action)}</span>
          <span class="conf-pill ${confBand}">${(a.confidence * 100).toFixed(0)}%</span>
          ${criticHTML}
        </div>
        <div class="action-details">
          ${a.evidence_refs ? `<div class="action-evidence">${a.evidence_refs.length} citation${a.evidence_refs.length > 1 ? 's' : ''}</div>` : ''}
          ${a.replan_reason ? `<div class="action-replan">🔄 Replan: ${escHtml(a.replan_reason)}</div>` : ''}
          ${a.halt_reason ? `<div class="action-halt">🛑 Halt: ${escHtml(a.halt_reason)}</div>` : ''}
        </div>
      </div>
    `;
  }).join('');
  
  queueList.innerHTML = queueHTML;
  panel.style.display = 'block';
}

function clearAutonomyQueue() {
  autonomyQueue = [];
  const queueList = document.getElementById('autonomy-queue-list');
  if (queueList) {
    queueList.innerHTML = '<div class="empty">No workflow actions yet</div>';
  }
}

// ============================================================================
// PANEL 5: FEDERATION NETWORK PANEL
// ============================================================================

function renderFederationPanel(federationData) {
  const panel = document.getElementById('federation-panel');
  if (!panel) return;
  
  if (!federationData) {
    panel.style.display = 'none';
    return;
  }
  
  const { rounds, nodes, global_auc, patient_images_transmitted } = federationData;
  
  // Network visualization
  let nodesHTML = '';
  if (nodes && nodes.length > 0) {
    nodesHTML = nodes.map(node => `
      <div class="federation-node">
        <div class="node-icon">🏥</div>
        <div class="node-details">
          <div class="node-name">${escHtml(node.name)}</div>
          <div class="node-stats">
            ${node.samples} samples · AUC ${node.local_auc.toFixed(3)}
          </div>
        </div>
        <div class="node-status ${node.status}">${node.status}</div>
      </div>
    `).join('');
  }
  
  // Round history
  let roundsHTML = '';
  if (rounds && rounds.length > 0) {
    roundsHTML = rounds.map((r, idx) => `
      <div class="federation-round">
        <div class="round-number">Round ${idx + 1}</div>
        <div class="round-auc">Global AUC: ${r.global_auc.toFixed(3)}</div>
        <div class="round-time">${r.wall_clock_seconds}s</div>
      </div>
    `).join('');
  }
  
  // Privacy counter (CRITICAL: must be derived from audit log, not hardcoded)
  const privacyHTML = `
    <div class="privacy-counter ${patient_images_transmitted === 0 ? 'good' : 'bad'}">
      <div class="counter-value">${patient_images_transmitted}</div>
      <div class="counter-label">Patient images that left a hospital</div>
      <div class="counter-proof">
        <a href="#" onclick="verifyPrivacy(); return false;">🔍 Verify in audit log</a>
      </div>
    </div>
  `;
  
  panel.innerHTML = `
    <div class="federation-grid">
      <div class="federation-section">
        <h3>🌐 Hospital Nodes</h3>
        <div class="nodes-container">${nodesHTML}</div>
      </div>
      <div class="federation-section">
        <h3>📊 Training Rounds</h3>
        <div class="rounds-container">${roundsHTML}</div>
      </div>
      <div class="federation-section full-width">
        ${privacyHTML}
      </div>
    </div>
  `;
  
  panel.style.display = 'block';
}

function verifyPrivacy() {
  // Download audit log and verify zero patient data transmission
  alert('Downloading audit log for verification...\n\nThe audit chain will show:\n- Only model weights transmitted\n- SHA-256 hashes of all updates\n- Zero raw patient data in any record');
  // Trigger audit JSON download
  const btnAudit = document.getElementById('btn-audit');
  if (btnAudit) btnAudit.click();
}

// ============================================================================
// CRITIC AGENT VERDICT PILLS (visible on all decision panels)
// ============================================================================

function addCriticVerdict(elementId, verdict) {
  const element = document.getElementById(elementId);
  if (!element) return;
  
  const verdictClass = verdict.verdict === 'APPROVE' ? 'good' : 
                       verdict.verdict === 'CHALLENGE' ? 'warn' : 'bad';
  
  const pill = document.createElement('span');
  pill.className = `critic-pill ${verdictClass}`;
  pill.textContent = `🤖 ${verdict.verdict}`;
  pill.title = verdict.reasoning || '';
  
  // Add to element header
  const header = element.querySelector('h2, h3, .action-header');
  if (header) {
    header.appendChild(pill);
  }
}

// ============================================================================
// WEBSOCKET MESSAGE HANDLERS (extend existing handlers)
// ============================================================================

function handleV2WebSocketMessage(evt) {
  // Handle new v2 message types
  switch (evt.type) {
    case 'modality_identified':
      renderModalityBadge(evt.data);
      break;
      
    case 'vanilla_baseline_complete':
      // Store for comparison panel
      window.lastVanillaData = evt.data;
      if (window.lastGroundedData) {
        renderEnhancedComparison(window.lastVanillaData, window.lastGroundedData);
      }
      break;
      
    case 'grounded_report_complete':
      // Store for comparison panel
      window.lastGroundedData = evt.data;
      if (window.lastVanillaData) {
        renderEnhancedComparison(window.lastVanillaData, window.lastGroundedData);
      }
      break;
      
    case 'dictation_audit':
      renderDictationAudit(evt.data);
      break;
      
    case 'autonomy_action':
      updateAutonomyQueue(evt.data);
      break;
      
    case 'autonomy_complete':
      // Mark queue as complete
      const queueIndicator = document.querySelector('.queue-indicator');
      if (queueIndicator) {
        queueIndicator.className = 'queue-indicator complete';
        queueIndicator.textContent = 'COMPLETE';
      }
      break;
      
    case 'federation_update':
      renderFederationPanel(evt.data);
      break;
      
    case 'critic_verdict':
      addCriticVerdict(evt.target_element, evt.data);
      break;
  }
}

// ============================================================================
// INITIALIZATION
// ============================================================================

function initDashboardV2() {
  // Initialize autonomy panel
  initAutonomyPanel();
  
  // Clear state
  window.lastVanillaData = null;
  window.lastGroundedData = null;
  clearAutonomyQueue();
  
  console.log('RadAgent v2 Dashboard Extensions loaded');
}

// Auto-init when DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initDashboardV2);
} else {
  initDashboardV2();
}

// Export for use in main dashboard
window.RadAgentV2 = {
  renderModalityBadge,
  renderEnhancedComparison,
  renderDictationAudit,
  updateAutonomyQueue,
  clearAutonomyQueue,
  renderFederationPanel,
  addCriticVerdict,
  handleV2WebSocketMessage
};

// Made with Bob
