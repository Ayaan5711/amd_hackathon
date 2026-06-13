/**
 * PulseIQ Frontend Application
 * Vanilla JS - no frameworks required
 */

// State
let currentSessionId = null;
let isProcessing = false;

// DOM Elements
const uploadView = document.getElementById('upload-view');
const chatView = document.getElementById('chat-view');
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadError = document.getElementById('upload-error');
const errorText = document.getElementById('error-text');
const newSessionBtn = document.getElementById('new-session-btn');
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('message-input');
const sendBtn = document.getElementById('send-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// Initialize
function init() {
    setupEventListeners();
    setupDragAndDrop();
}

// Event Listeners
function setupEventListeners() {
    // Upload zone click
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // New session
    newSessionBtn.addEventListener('click', resetSession);
    
    // Chat
    sendBtn.addEventListener('click', sendMessage);
    messageInput.addEventListener('keydown', handleInputKeydown);
    messageInput.addEventListener('input', autoResizeTextarea);
    
    // Suggestion chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            messageInput.value = query;
            sendMessage();
        });
    });
}

// Drag and Drop
function setupDragAndDrop() {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, () => {
            uploadZone.classList.remove('dragover');
        });
    });
    
    uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

// File Handling
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

async function handleFile(file) {
    // Validate
    if (!file.name.endsWith('.csv')) {
        showUploadError('Please upload a CSV file');
        return;
    }
    
    if (file.size > 50 * 1024 * 1024) {
        showUploadError('File too large. Maximum size is 50MB');
        return;
    }
    
    hideUploadError();
    showLoading('Uploading...');
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        
        const data = await response.json();
        currentSessionId = data.session_id;
        
        // Show chat section
        showChatSection(data);
        
    } catch (error) {
        showUploadError(error.message);
    } finally {
        hideLoading();
    }
}

// Chat Section
function showChatSection(data) {
    uploadView.style.display = 'none';
    chatView.style.display = 'flex';
    newSessionBtn.style.display = 'flex';
    
    // Update data info
    document.getElementById('data-filename').textContent = data.filename;
    document.getElementById('data-stats').textContent = 
        `${data.row_count.toLocaleString()} rows · ${data.column_count} cols`;
    
    // Enable input
    sendBtn.disabled = false;
    messageInput.focus();
}

function resetSession() {
    currentSessionId = null;
    uploadView.style.display = 'flex';
    chatView.style.display = 'none';
    newSessionBtn.style.display = 'none';
    messagesContainer.innerHTML = `
        <div class="welcome-card">
            <div class="welcome-icon">
                <svg viewBox="0 0 48 48" fill="none">
                    <defs>
                        <linearGradient id="welcomeGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" style="stop-color:#667eea"/>
                            <stop offset="100%" style="stop-color:#764ba2"/>
                        </linearGradient>
                    </defs>
                    <circle cx="24" cy="24" r="22" fill="url(#welcomeGrad)" fill-opacity="0.1"/>
                    <path d="M24 14v20M14 24h20" stroke="url(#welcomeGrad)" stroke-width="2.5" stroke-linecap="round"/>
                </svg>
            </div>
            <h2 class="welcome-title">Ready to analyze</h2>
            <p class="welcome-text">Your survey data has been processed. Ask me anything about it.</p>
        </div>
        <div class="suggestion-chips">
            <button class="chip" data-query="Which department has the highest satisfaction?">
                <span class="chip-icon"></span>
                Compare departments
            </button>
            <button class="chip" data-query="Show me NPS trends by quarter">
                <span class="chip-icon"></span>
                View trends
            </button>
            <button class="chip" data-query="What are the main themes in comments?">
                <span class="chip-icon"></span>
                Extract themes
            </button>
            <button class="chip" data-query="Are there any outliers in the data?">
                <span class="chip-icon"></span>
                Find anomalies
            </button>
            <button class="chip" data-query="What actions should we take?">
                <span class="chip-icon"></span>
                Get recommendations
            </button>
        </div>
    `;
    messageInput.value = '';
    fileInput.value = '';
    
    // Re-attach chip listeners
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            messageInput.value = query;
            sendMessage();
        });
    });
}

// Messaging
async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isProcessing || !currentSessionId) return;
    
    // Add user message
    addMessage('user', message);
    messageInput.value = '';
    autoResizeTextarea();
    
    // Show typing indicator
    showTypingIndicator();
    isProcessing = true;
    sendBtn.disabled = true;
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Request failed');
        }
        
        const data = await response.json();
        
        // Remove typing indicator and add response
        removeTypingIndicator();
        addMessage('assistant', data.response, {
            followUps: data.follow_up_suggestions,
            toolCalls: data.tool_calls,
            evidence: data.evidence
        });
        
    } catch (error) {
        removeTypingIndicator();
        addMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

function addMessage(role, content, options = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'You' : 'AI';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Tool badges
    if (options.toolCalls && options.toolCalls.length > 0) {
        const toolsDiv = document.createElement('div');
        toolsDiv.className = 'tool-badges';
        options.toolCalls.forEach(tool => {
            const badge = document.createElement('span');
            badge.className = 'tool-badge';
            badge.textContent = formatToolName(tool.tool_name);
            toolsDiv.appendChild(badge);
        });
        contentDiv.appendChild(toolsDiv);
    }
    
    // Message text
    const textDiv = document.createElement('div');
    textDiv.innerHTML = formatMessage(content);
    contentDiv.appendChild(textDiv);
    
    // Follow-up suggestions
    if (options.followUps && options.followUps.length > 0) {
        const followUpsDiv = document.createElement('div');
        followUpsDiv.className = 'follow-ups';
        
        options.followUps.forEach(suggestion => {
            const btn = document.createElement('button');
            btn.className = 'follow-up-btn';
            btn.textContent = suggestion;
            btn.addEventListener('click', () => {
                messageInput.value = suggestion;
                sendMessage();
            });
            followUpsDiv.appendChild(btn);
        });
        
        contentDiv.appendChild(followUpsDiv);
    }
    
    // Evidence panel
    if (options.evidence && Object.keys(options.evidence).length > 0) {
        const evidenceDiv = document.createElement('details');
        evidenceDiv.className = 'evidence-panel';
        evidenceDiv.innerHTML = `
            <summary>View source data</summary>
            <pre>${JSON.stringify(options.evidence, null, 2)}</pre>
        `;
        contentDiv.appendChild(evidenceDiv);
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    messagesContainer.appendChild(messageDiv);
    scrollToBottom();
}

function formatMessage(text) {
    // Convert newlines to <br>
    return text.replace(/\n/g, '<br>');
}

function formatToolName(name) {
    // Convert snake_case to Title Case
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message assistant typing';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    messagesContainer.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Input Handling
function handleInputKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// UI Helpers
function showLoading(text = 'Processing...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showUploadError(message) {
    errorText.textContent = message;
    uploadError.style.display = 'flex';
}

function hideUploadError() {
    uploadError.style.display = 'none';
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Initialize app
document.addEventListener('DOMContentLoaded', init);
