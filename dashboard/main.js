// State & Interactivity Variables
let latestState = null;
let selectedHazardId = null;
let roverHistory = [];

const statusIndicator = document.getElementById('connection-status');
const roverDataContainer = document.getElementById('rover-data');
const hazardList = document.getElementById('hazard-list');
const inspectorModal = document.getElementById('hazard-inspector');

// Canvas Setup
const canvas = document.getElementById('mapCanvas');
const ctx = canvas.getContext('2d');
const SCALE = 10; // Spatial Map Zoom (10 pixels = 1 meter)
const CENTER_X = canvas.width / 2;
const CENTER_Y = canvas.height / 2;

// Chart.js Telemetry Setup
Chart.defaults.color = '#888';
Chart.defaults.font.family = "'Courier New', monospace";

const createLineChart = (ctxId, label, color) => {
    return new Chart(document.getElementById(ctxId), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: label,
                data: [],
                borderColor: color,
                borderWidth: 2,
                tension: 0.3,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                x: { display: false },
                y: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.1)' } }
            },
            plugins: { legend: { display: false } }
        }
    });
};

const batteryChart = createLineChart('batteryChart', 'Battery %', '#00ff88');
const lossChart = createLineChart('lossChart', 'Packet Loss %', '#ff2a2a');

function updateChart(chart, value) {
    if (chart.data.labels.length > 30) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    chart.data.labels.push('');
    chart.data.datasets[0].data.push(value);
    chart.update();
}

// ---------------------------------------------------------
// 60 FPS RADAR RENDER LOOP
// ---------------------------------------------------------
function renderRadar() {
    requestAnimationFrame(renderRadar);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const time = Date.now();

    // 1. Radar Distance Rings & Crosshairs
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';
    ctx.lineWidth = 1;
    for (let r = 50; r < canvas.width; r += 100) {
        ctx.beginPath(); ctx.arc(CENTER_X, CENTER_Y, r, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.beginPath(); ctx.moveTo(0, CENTER_Y); ctx.lineTo(canvas.width, CENTER_Y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CENTER_X, 0); ctx.lineTo(CENTER_X, canvas.height); ctx.stroke();

    // 2. Rotating Radar Beam
    const sweepAngle = (time / 1000) % (Math.PI * 2);
    ctx.save();
    ctx.translate(CENTER_X, CENTER_Y);
    ctx.rotate(sweepAngle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, canvas.width, 0, 0.25);
    ctx.lineTo(0, 0);
    const grad = ctx.createLinearGradient(0, 0, 0, canvas.width / 2);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.5)');
    grad.addColorStop(1, 'rgba(0, 240, 255, 0.0)');
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.beginPath(); ctx.moveTo(0, 0); ctx.lineTo(canvas.width, 0);
    ctx.strokeStyle = '#00f0ff'; ctx.lineWidth = 1.5; ctx.stroke();
    ctx.restore();

    // 3. Draw Landing Site / Origin (0,0) Marker
    const originX = CENTER_X;
    const originY = CENTER_Y;
    ctx.save();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(originX, originY, 12, 0, Math.PI * 2);
    ctx.stroke();
    // Crosshairs
    ctx.beginPath();
    ctx.moveTo(originX - 18, originY); ctx.lineTo(originX + 18, originY);
    ctx.moveTo(originX, originY - 18); ctx.lineTo(originX, originY + 18);
    ctx.stroke();
    ctx.fillStyle = '#00ff88';
    ctx.font = 'bold 10px Courier New';
    ctx.fillText('LANDING SITE (0,0)', originX + 15, originY + 15);
    ctx.restore();

    if (!latestState) return;

    // 4. Draw Trajectory Path (Planned Path & Historical Trail)
    if (latestState.path && latestState.path.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
        ctx.setLineDash([6, 6]);
        ctx.lineWidth = 2;
        latestState.path.forEach((pt, idx) => {
            const px = CENTER_X + (pt[0] * SCALE);
            const py = CENTER_Y - (pt[1] * SCALE);
            if (idx === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Rover Movement Trail
    if (roverHistory.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = '#38bdf8';
        ctx.lineWidth = 1.5;
        roverHistory.forEach((pt, idx) => {
            const hx = CENTER_X + (pt.x * SCALE);
            const hy = CENTER_Y - (pt.y * SCALE);
            if (idx === 0) ctx.moveTo(hx, hy);
            else ctx.lineTo(hx, hy);
        });
        ctx.stroke();
    }

    // 5. Draw Rover (RV-1)
    const r1 = latestState.rovers["1"];
    let rx = CENTER_X, ry = CENTER_Y;
    if (r1) {
        rx = CENTER_X + (r1.x * SCALE);
        ry = CENTER_Y - (r1.y * SCALE);

        ctx.save();
        ctx.translate(rx, ry);
        ctx.rotate(-r1.heading * Math.PI / 180);

        ctx.beginPath();
        ctx.moveTo(10, 0);
        ctx.lineTo(-7, -6);
        ctx.lineTo(-7, 6);
        ctx.closePath();
        ctx.fillStyle = '#38bdf8';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.restore();

        ctx.fillStyle = '#38bdf8';
        ctx.font = 'bold 11px Courier New';
        ctx.fillText(`RV-1`, rx + 14, ry - 5);
    }

    // 6. Draw Hazards (Rocks)
    for (const [id, h] of Object.entries(latestState.hazards)) {
        const cx = CENTER_X + (h.x_m * SCALE);
        const cy = CENTER_Y - (h.y_m * SCALE);

        const isSelected = (id === selectedHazardId);
        const isFresh = h.state === 'FRESH';
        const strokeColor = isFresh ? '#ff2a2a' : '#ffb800';

        // Target lock pulsing ring if selected
        if (isSelected) {
            const pulse = Math.sin(time / 150) * 4 + 16;
            ctx.beginPath();
            ctx.arc(cx, cy, pulse, 0, Math.PI * 2);
            ctx.strokeStyle = '#00f0ff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Tactical Vector Line from Rover to Selected Rock
            if (r1) {
                ctx.beginPath();
                ctx.moveTo(rx, ry);
                ctx.lineTo(cx, cy);
                ctx.strokeStyle = '#00f0ff';
                ctx.setLineDash([4, 4]);
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.setLineDash([]);

                // Vector Distance Tag on canvas
                const midX = (rx + cx) / 2;
                const midY = (ry + cy) / 2;
                const dist = Math.hypot(h.x_m - r1.x, h.y_m - r1.y).toFixed(1);
                ctx.fillStyle = '#00f0ff';
                ctx.font = '10px Courier New';
                ctx.fillText(`${dist}m`, midX + 5, midY - 5);
            }
        }

        // Diamond Icon
        ctx.save();
        ctx.translate(cx, cy);

        ctx.beginPath();
        ctx.moveTo(0, -7);
        ctx.lineTo(7, 0);
        ctx.lineTo(0, 7);
        ctx.lineTo(-7, 0);
        ctx.closePath();

        ctx.fillStyle = isFresh ? 'rgba(255, 42, 42, 0.4)' : 'rgba(255, 184, 0, 0.4)';
        ctx.fill();
        ctx.strokeStyle = isSelected ? '#00f0ff' : strokeColor;
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();

        ctx.restore();

        // Canvas Text Label
        ctx.fillStyle = '#ffffff';
        ctx.font = '11px Courier New';
        ctx.fillText(`${id}`, cx + 10, cy - 2);
        ctx.fillStyle = strokeColor;
        ctx.font = '9px Courier New';
        ctx.fillText(`${(h.display_confidence * 100).toFixed(0)}%`, cx + 10, cy + 8);
    }
}

// Start Render Engine
requestAnimationFrame(renderRadar);

// ---------------------------------------------------------
// CANVAS CLICK INTERACTION & INSPECTOR
// ---------------------------------------------------------
canvas.addEventListener('click', (e) => {
    if (!latestState) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleX;
    const clickY = (e.clientY - rect.top) * scaleY;

    let clickedHazard = null;
    let minDist = 25; // 25px radius click tolerance

    for (const [id, h] of Object.entries(latestState.hazards)) {
        const cx = CENTER_X + (h.x_m * SCALE);
        const cy = CENTER_Y - (h.y_m * SCALE);
        const dist = Math.hypot(clickX - cx, clickY - cy);
        if (dist < minDist) {
            minDist = dist;
            clickedHazard = id;
        }
    }

    if (clickedHazard) {
        selectHazard(clickedHazard);
    } else {
        selectedHazardId = null;
        inspectorModal.classList.add('hidden');
    }
});

document.getElementById('close-inspector').addEventListener('click', () => {
    selectedHazardId = null;
    inspectorModal.classList.add('hidden');
});

function selectHazard(id) {
    if (!latestState || !latestState.hazards[id]) return;
    selectedHazardId = id;
    const h = latestState.hazards[id];
    const r1 = latestState.rovers["1"] || { x: 0, y: 0, heading: 0 };

    // Relative Vector Math Calculation
    const dx = h.x_m - r1.x;
    const dy = h.y_m - r1.y;
    const range = Math.hypot(dx, dy).toFixed(2);

    let angleRad = Math.atan2(dy, dx);
    let bearingDeg = (angleRad * 180 / Math.PI);
    let relBearing = (bearingDeg - r1.heading + 360) % 360;

    // Update Inspector Modal Text Fields
    document.getElementById('insp-id').textContent = h.id;
    document.getElementById('insp-class').textContent = `CLASS ID: ${h.class} (ROCK)`;
    document.getElementById('insp-conf').textContent = `${(h.display_confidence * 100).toFixed(1)}%`;
    document.getElementById('insp-global').textContent = `[${h.x_m.toFixed(2)}m, ${h.y_m.toFixed(2)}m]`;
    document.getElementById('insp-rel').textContent = `${range}m @ ${relBearing.toFixed(1)}°`;
    document.getElementById('insp-offset').textContent = `ΔX: ${dx >= 0 ? '+' : ''}${dx.toFixed(2)}m, ΔY: ${dy >= 0 ? '+' : ''}${dy.toFixed(2)}m`;
    document.getElementById('insp-age').textContent = `${h.age.toFixed(1)}s (${h.state})`;

    inspectorModal.classList.remove('hidden');
}

// ---------------------------------------------------------
// TELEMETRY & WEBSOCKET ENGINE
// ---------------------------------------------------------
function updateUI(state) {
    const r1 = state.rovers["1"];
    if (r1) {
        // Record trail history
        if (roverHistory.length === 0 || Math.hypot(r1.x - roverHistory[roverHistory.length - 1].x, r1.y - roverHistory[roverHistory.length - 1].y) > 0.5) {
            roverHistory.push({ x: r1.x, y: r1.y });
            if (roverHistory.length > 100) roverHistory.shift();
        }

        const distFromOrigin = Math.hypot(r1.x, r1.y).toFixed(2);
        roverDataContainer.innerHTML = `
            <div style="margin-bottom: 4px;">STATUS: <span class="data-value" style="color: ${r1.state === 'BLACKOUT' ? 'var(--danger-red)' : 'var(--success-green)'}">${r1.state}</span></div>
            <div style="margin-bottom: 4px;">POS (GLOBAL): <span class="data-value">[${r1.x.toFixed(2)}m, ${r1.y.toFixed(2)}m]</span></div>
            <div style="margin-bottom: 4px;">DIST FROM BASE: <span class="data-value">${distFromOrigin}m</span></div>
            <div style="margin-bottom: 4px;">HEADING: <span class="data-value">${r1.heading.toFixed(1)}°</span></div>
        `;
        updateChart(batteryChart, r1.battery);
        updateChart(lossChart, (state.network.loss * 100).toFixed(1));
    }

    // Refresh Inspector if an item is selected
    if (selectedHazardId && state.hazards[selectedHazardId]) {
        selectHazard(selectedHazardId);
    }

    // Update Sidebar Table
    hazardList.innerHTML = '';
    const sortedHazards = Object.values(state.hazards).sort((a, b) => b.display_confidence - a.display_confidence);
    sortedHazards.forEach(h => {
        const isSelected = h.id === selectedHazardId;
        const color = h.state === 'FRESH' ? 'var(--danger-red)' : 'var(--warning-orange)';
        const li = document.createElement('li');
        li.className = isSelected ? 'selected-row' : '';
        li.style.color = color;
        li.innerHTML = `
            <span><strong>${h.id}</strong></span>
            <span>ROCK</span>
            <span>${(h.display_confidence * 100).toFixed(0)}%</span>
            <span>${h.age.toFixed(1)}s</span>
        `;
        li.onclick = () => selectHazard(h.id);
        hazardList.appendChild(li);
    });

    if (document.activeElement.tagName !== "INPUT") {
        document.getElementById('loss').value = state.network.loss;
        document.getElementById('latency').value = state.network.latency_ms;
    }
}

// ---------------------------------------------------------
// SERVER ENDPOINT RESOLUTION (Handles Port 8000 & Port 55152)
// ---------------------------------------------------------
const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const isDevPort = window.location.port !== '8000';
const wsHost = isDevPort ? 'localhost:8000' : window.location.host;
const apiHost = isDevPort ? 'http://localhost:8000' : '';

// WebSocket Connection
const ws = new WebSocket(`${protocol}//${wsHost}/ws`);

ws.onopen = () => {
    statusIndicator.textContent = 'LINK ESTABLISHED';
    statusIndicator.className = 'status online';
};

ws.onmessage = (event) => {
    latestState = JSON.parse(event.data);
    updateUI(latestState);
};

ws.onclose = () => {
    statusIndicator.textContent = 'SIGNAL LOST';
    statusIndicator.className = 'status offline';
};

// Network Control API Handlers (Points directly to Backend)
document.getElementById('network-form').addEventListener('submit', (e) => {
    e.preventDefault();
    fetch(`${apiHost}/api/network`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            loss: parseFloat(document.getElementById('loss').value),
            latency_ms: parseInt(document.getElementById('latency').value)
        })
    }).then(res => res.json()).then(data => {
        console.log("Network parameters updated:", data);
    }).catch(err => console.error("Network update error:", err));
});

document.getElementById('btn-blackout').addEventListener('click', () => {
    fetch(`${apiHost}/api/blackout`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({seconds: 10})
    }).then(res => res.json()).then(data => {
        console.log("Blackout triggered:", data);
    }).catch(err => console.error("Blackout trigger error:", err));
});