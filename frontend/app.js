// ==================================================
// CONFIG
// ==================================================

const API_BASE = "http://127.0.0.1:8000";

// ==================================================
// STATE
// ==================================================

let currentCollection = null;
let currentSessionId = null;

// ==================================================
// DOM
// ==================================================

const collectionList = document.getElementById("collection-list");
const sessionList = document.getElementById("session-list");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const sourcePanel = document.getElementById("source-panel");
const metricsPanel = document.getElementById("metrics-panel");
const newChatBtn = document.getElementById("new-chat-btn");
const uploadBtn = document.getElementById("upload-btn");
const newCollectionBtn = document.getElementById("new-collection-btn");
const uploadModal = document.getElementById("upload-modal");
const collectionModal = document.getElementById("collection-modal");
const uploadConfirmBtn = document.getElementById("upload-confirm");
const createCollectionBtn = document.getElementById("create-collection-confirm");
const closeUploadBtn = document.getElementById("close-upload-modal");
const closeCollectionBtn = document.getElementById("close-collection-modal");
const pdfUploadInput = document.getElementById("pdf-upload");
const collectionNameInput = document.getElementById("collection-name");
const statDocuments = document.getElementById("stat-documents");
const statSessions = document.getElementById("stat-sessions");
const statMessages = document.getElementById("stat-messages");
const chatHeaderTitle = document.querySelector(".chat-header h2");

// ==================================================
// STARTUP
// ==================================================

document.addEventListener("DOMContentLoaded", async () => {
    attachEvents();
    await loadCollections();
});

// ==================================================
// EVENTS
// ==================================================

function attachEvents() {
    sendBtn.addEventListener("click", sendMessage);

    newChatBtn.addEventListener("click", async () => {
        if (!currentCollection) {
            alert("Select a collection first.");
            return;
        }
        await createSession();
    });

    uploadBtn.addEventListener("click", () => {
        if (!currentCollection) {
            alert("Select a collection first.");
            return;
        }
        uploadModal.classList.remove("hidden");
    });

    newCollectionBtn.addEventListener("click", () => {
        collectionModal.classList.remove("hidden");
    });

    closeUploadBtn.addEventListener("click", () => {
        uploadModal.classList.add("hidden");
    });

    closeCollectionBtn.addEventListener("click", () => {
        collectionModal.classList.add("hidden");
    });

    // Close session dropdowns when clicking anywhere else
    document.addEventListener("click", () => {
        document.querySelectorAll(".menu-dropdown").forEach(d => d.classList.add("hidden"));
    });

    uploadConfirmBtn.addEventListener("click", uploadDocuments);

    createCollectionBtn.addEventListener("click", createCollection);

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

// ==================================================
// COLLECTIONS
// ==================================================

async function loadCollections() {
    try {
        const response = await fetch(`${API_BASE}/collections`);
        const data = await response.json();
        await renderCollections(data.collections || []);
    } catch (error) {
        console.error("COLLECTION ERROR:", error);
        alert("Failed to load collections.");
    }
}

async function loadCollectionStats(collectionName) {
    try {
        const response = await fetch(`${API_BASE}/collections/${collectionName}/stats`);
        const data = await response.json();

        statDocuments.textContent = data.documents ?? 0;
        statSessions.textContent = data.sessions ?? 0;
        statMessages.textContent = data.messages ?? 0;
    } catch (error) {
        console.error("STATS ERROR:", error);
    }
}

// ==================================================
// COLLECTIONS RENDERING & ACTIONS
// ==================================================

async function renderCollections(collections) {
    collectionList.innerHTML = "";
    let collectionExists = false;

    collections.forEach((collection) => {
        const item = document.createElement("div");
        item.className = "collection-item";
        
        if (currentCollection === collection.name) {
            item.classList.add("active");
            collectionExists = true;
        }

        // Title Element
        const titleSpan = document.createElement("span");
        titleSpan.className = "session-title-text"; 
        titleSpan.textContent = collection.name;

        // Dropdown Menu HTML
        const actions = document.createElement("div");
        actions.className = "session-actions"; 
        actions.innerHTML = `
            <button class="menu-btn">⋮</button>
            <div class="menu-dropdown hidden">
                <div class="menu-action delete-btn delete-text">Delete</div>
            </div>
        `;

        // Load Collection on click
        titleSpan.addEventListener("click", async () => {
            document.querySelectorAll(".collection-item").forEach((el) => el.classList.remove("active"));
            item.classList.add("active");

            currentCollection = collection.name;
            currentSessionId = null;
            chatMessages.innerHTML = "";
            sourcePanel.innerHTML = "";
            metricsPanel.innerHTML = "";

            await loadSessions();
            await loadCollectionStats(collection.name);
            chatHeaderTitle.textContent = collection.name;
        });

        // Toggle Dropdown Menu
        const menuBtn = actions.querySelector(".menu-btn");
        const dropdown = actions.querySelector(".menu-dropdown");
        
        menuBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            document.querySelectorAll(".menu-dropdown").forEach(d => {
                if (d !== dropdown) d.classList.add("hidden");
            });
            dropdown.classList.toggle("hidden");
        });

        // Handle Delete Collection
        actions.querySelector(".delete-btn").addEventListener("click", async (e) => {
            e.stopPropagation();
            dropdown.classList.add("hidden");
            
            if (confirm(`Delete collection "${collection.name}"? This will permanently erase the PDFs, Vector DB, and all chat sessions.`)) {
                await deleteCollectionAPI(collection.name);
            }
        });

        item.appendChild(titleSpan);
        item.appendChild(actions);
        collectionList.appendChild(item);
    });

    if (collections.length > 0) {
        if (!currentCollection || !collectionExists) {
            currentCollection = collections[0].name;
            const first = document.querySelector(".collection-item");
            if (first) first.classList.add("active");
        }
        chatHeaderTitle.textContent = currentCollection;
        await loadCollectionStats(currentCollection);
        await loadSessions();
    } else {
        currentCollection = null;
        chatHeaderTitle.textContent = "No Collections";
        statDocuments.textContent = 0;
        statSessions.textContent = 0;
        statMessages.textContent = 0;
        sessionList.innerHTML = "";
    }
}
// ==================================================
// COLLECTION API HELPERS
// ==================================================

async function deleteCollectionAPI(collectionName) {
    try {
        const response = await fetch(`${API_BASE}/collections/${collectionName}`, { 
            method: "DELETE" 
        });
        
        if (!response.ok) throw new Error("Server rejected deletion");
        
        // If we deleted the collection we are currently looking at, clear the main screen
        if (currentCollection === collectionName) {
            currentCollection = null;
            currentSessionId = null;
            chatMessages.innerHTML = "";
            sourcePanel.innerHTML = "";
            metricsPanel.innerHTML = "";
        }
        
        // Refresh the sidebar
        await loadCollections();
        
    } catch (e) {
        console.error("Delete failed:", e);
        alert("Failed to delete collection from database.");
    }
}

// ==================================================
// SESSIONS & GROUPING
// ==================================================

async function loadSessions() {
    try {
        const response = await fetch(`${API_BASE}/sessions`);
        const data = await response.json();

        const filteredSessions = (data.sessions || []).filter(
            (session) => !currentCollection || session.collection_name === currentCollection
        );
        
        renderSessions(filteredSessions);
    } catch (error) {
        console.error("SESSION ERROR:", error);
    }
}

function groupSessionsByDate(sessions) {
    const groups = { "Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": [] };
    const now = new Date();
    
    // Create zeroed-out dates for strict midnight boundaries
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    const lastWeek = new Date(today);
    lastWeek.setDate(lastWeek.getDate() - 7);

    sessions.forEach(session => {
        // Force UTC "Z" to ensure exact timezone translation
        const sessionDate = new Date(session.updated_at + "Z"); 
        const dateOnly = new Date(sessionDate.getFullYear(), sessionDate.getMonth(), sessionDate.getDate());

        if (dateOnly.getTime() === today.getTime()) {
            groups["Today"].push(session);
        } else if (dateOnly.getTime() === yesterday.getTime()) {
            groups["Yesterday"].push(session);
        } else if (dateOnly >= lastWeek) {
            groups["Previous 7 Days"].push(session);
        } else {
            groups["Older"].push(session);
        }
    });
    return groups;
}

function renderSessions(sessions) {
    sessionList.innerHTML = "";
    
    const groupedSessions = groupSessionsByDate(sessions);

    for (const [groupName, groupArray] of Object.entries(groupedSessions)) {
        if (groupArray.length === 0) continue; // Skip empty groups

        // 1. Create Group Header
        const header = document.createElement("div");
        header.className = "session-group-header";
        header.textContent = groupName;
        sessionList.appendChild(header);

        // 2. Render Sessions for this Group
        groupArray.forEach(session => {
            const item = document.createElement("div");
            item.className = "session-item";
            if (session.id === currentSessionId) item.classList.add("active");

            // Title element
            const titleSpan = document.createElement("span");
            titleSpan.className = "session-title-text";
            titleSpan.textContent = session.title || `Chat ${session.id}`;

            // Actions dropdown
            const actions = document.createElement("div");
            actions.className = "session-actions";
            actions.innerHTML = `
                <button class="menu-btn">⋮</button>
                <div class="menu-dropdown hidden">
                    <div class="menu-action rename-btn">Rename</div>
                    <div class="menu-action delete-btn delete-text">Delete</div>
                </div>
            `;

            // Setup Click to Load Chat (on the title text, not the container)
            titleSpan.addEventListener("click", async () => {
                document.querySelectorAll(".session-item").forEach(el => el.classList.remove("active"));
                item.classList.add("active");
                currentSessionId = session.id;
                await loadHistory(session.id);
            });

            // Setup Dropdown Toggle
            const menuBtn = actions.querySelector(".menu-btn");
            const dropdown = actions.querySelector(".menu-dropdown");
            
            menuBtn.addEventListener("click", (e) => {
                e.stopPropagation(); // Stop click from bubbling
                // Close all other dropdowns first
                document.querySelectorAll(".menu-dropdown").forEach(d => {
                    if (d !== dropdown) d.classList.add("hidden");
                });
                dropdown.classList.toggle("hidden");
            });

            // Handle Rename Action
            actions.querySelector(".rename-btn").addEventListener("click", async (e) => {
                e.stopPropagation();
                dropdown.classList.add("hidden");
                const newTitle = prompt("Enter new chat name:", session.title);
                if (newTitle && newTitle.trim() !== "") {
                    await renameSessionAPI(session.id, newTitle.trim());
                }
            });

            // Handle Delete Action
            actions.querySelector(".delete-btn").addEventListener("click", async (e) => {
                e.stopPropagation();
                dropdown.classList.add("hidden");
                if (confirm(`Delete "${session.title}"? This cannot be undone.`)) {
                    await deleteSessionAPI(session.id);
                }
            });

            item.appendChild(titleSpan);
            item.appendChild(actions);
            sessionList.appendChild(item);
        });
    }
}

// ==================================================
// SESSION API HELPERS
// ==================================================

async function renameSessionAPI(sessionId, newTitle) {
    try {
        await fetch(`${API_BASE}/sessions/${sessionId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: newTitle })
        });
        await loadSessions(); // Refresh list to show new name
    } catch (e) {
        console.error("Rename failed:", e);
    }
}

async function deleteSessionAPI(sessionId) {
    try {
        await fetch(`${API_BASE}/sessions/${sessionId}`, { method: "DELETE" });
        if (currentSessionId === sessionId) {
            currentSessionId = null; // Clear active view if you deleted it
            chatMessages.innerHTML = "";
            chatHeaderTitle.textContent = currentCollection;
        }
        await loadSessions();
        await loadCollectionStats(currentCollection); // Update message count stats!
    } catch (e) {
        console.error("Delete failed:", e);
    }
}

async function createCollection() {
    const collectionName = collectionNameInput.value.trim();

    if (!collectionName) {
        alert("Enter collection name.");
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/collections`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                name: collectionName,
                description: ""
            })
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();

        collectionModal.classList.add("hidden");
        collectionNameInput.value = "";

        currentCollection = data.name;
        
        await loadCollections();
        await createSession();

        alert("Collection created.");
    } catch (error) {
        console.error(error);
        alert("Collection creation failed.");
    }
}

async function createSession() {
    try {
        const response = await fetch(`${API_BASE}/sessions`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                collection_name: currentCollection,
                title: "New Chat"
            })
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();

        currentSessionId = data.session_id;
        chatMessages.innerHTML = "";
        
        await loadSessions();

        return data.session_id;
    } catch (error) {
        console.error("CREATE SESSION ERROR:", error);
        alert("Failed to create session.");
    }
}

// ==================================================
// HISTORY
// ==================================================

async function loadHistory(sessionId) {
    try {
        const response = await fetch(`${API_BASE}/history/${sessionId}`);
        const data = await response.json();

        chatMessages.innerHTML = "";

        data.messages.forEach((message) => {
            appendMessage(message.role, message.content);
        });
    } catch (error) {
        console.error("HISTORY ERROR:", error);
    }
}

// ==================================================
// CHAT
// ==================================================

async function sendMessage() {
    const query = chatInput.value.trim();

    if (!query) return;

    if (!currentSessionId) {
        alert("Create a session first.");
        return;
    }

    appendMessage("user", query);
    chatInput.value = "";

    const loading = appendMessage("assistant", "Thinking...");

    try {
        const payload = {
            session_id: currentSessionId,
            query: query,
            dense_k: Number(document.getElementById("dense-k")?.value || 20),
            bm25_k: Number(document.getElementById("bm25-k")?.value || 20),
            rerank_candidates: Number(document.getElementById("rerank-k")?.value || 20),
            final_k: Number(document.getElementById("final-k")?.value || 5)
        };

        const response = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        const data = await response.json();

        loading.remove();
        appendMessage("assistant", data.answer || "No response.");
        renderSources(data.sources || []);
        renderMetrics(data.retrieval_metrics || {});

        setTimeout(async () => {
            await loadSessions();
            await loadCollectionStats(currentCollection);
        }, 300);

    } catch (error) {
        loading.remove();
        appendMessage("assistant", "Backend Error");
        console.error("CHAT ERROR:", error);
    }
}

// ==================================================
// MESSAGES
// ==================================================

function appendMessage(role, content) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${role}`;

    const body = document.createElement("div");
    body.className = "message-content";
    body.textContent = content;

    wrapper.appendChild(body);
    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    return wrapper;
}

// ==================================================
// SOURCES (ADVANCED)
// ==================================================

function renderSources(sources) {
    sourcePanel.innerHTML = "";

    if (!sources || sources.length === 0) {
        sourcePanel.innerHTML = "<p style='color: var(--muted); font-size: 0.9rem;'>No sources retrieved for this query.</p>";
        return;
    }

    sources.forEach((source) => {
        // Fallbacks in case your backend uses slightly different keys
        const paper = source.paper_name || source.filename || "Unknown Document";
        const page = source.page || source.page_number || "-";
        const score = source.score ?? 0;
        
        // Ensure your backend generate.py returns the actual text chunk as "content" or "text"
        const previewText = source.content || source.text || "No text preview available.";

        const card = document.createElement("div");
        card.className = "source-card";

        card.innerHTML = `
            <div class="source-header">
                <div class="source-title-group">
                    <div class="source-title">${paper}</div>
                    <div class="source-meta">
                        <span>Page ${page}</span>
                    </div>
                </div>
                <div class="source-score">${Number(score).toFixed(4)}</div>
            </div>
            <div class="source-preview">
                "${previewText.substring(0, 200)}${previewText.length > 200 ? '...' : ''}"
            </div>
        `;

        sourcePanel.appendChild(card);
    });
}

// ==================================================
// METRICS & PIPELINE VISUALIZATION
// ==================================================

function renderMetrics(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) {
        metricsPanel.innerHTML = "<p style='color: var(--muted); font-size: 0.9rem;'>Awaiting next query...</p>";
        return;
    }

    // Parse the metrics (adding fallbacks so it doesn't break)
    const denseMs = metrics.dense_ms || 0;
    const bm25Ms = metrics.bm25_ms || 0;
    const rerankMs = metrics.rerank_ms || 0;
    const totalMs = metrics.total_ms || (denseMs + bm25Ms + rerankMs);

    // Build the interactive pipeline visualization
    metricsPanel.innerHTML = `
        <div class="pipeline-step">
            <span class="step-label">Dense Retrieval</span>
            <span class="step-value">${denseMs} ms</span>
        </div>
        
        <div class="pipeline-step">
            <span class="step-label">BM25 Retrieval</span>
            <span class="step-value">${bm25Ms} ms</span>
        </div>
        
        <div class="pipeline-step">
            <span class="step-label">RRF Fusion & Reranking</span>
            <span class="step-value">${rerankMs} ms</span>
        </div>

        <div class="pipeline-step">
            <span class="step-label" style="color: var(--accent); font-weight: 600;">Total Pipeline Latency</span>
            <span class="step-value" style="font-size: 0.95rem; font-weight: 700;">${totalMs} ms</span>
        </div>
    `;
}

// ==================================================
// UPLOADS
// ==================================================

async function uploadDocuments() {
    const files = pdfUploadInput.files;

    if (!files.length) {
        alert("Choose PDFs first.");
        return;
    }

    const formData = new FormData();
    formData.append("collection_name", currentCollection);

    for (let file of files) {
        formData.append("files", file);
    }

    try {
        const response = await fetch(`${API_BASE}/collections/${currentCollection}/add`, {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error(await response.text());
        }

        uploadModal.classList.add("hidden");
        pdfUploadInput.value = "";

        alert("Documents uploaded.");

        await loadCollectionStats(currentCollection);
    } catch (error) {
        console.error(error);
        alert("Upload failed.");
    }
}