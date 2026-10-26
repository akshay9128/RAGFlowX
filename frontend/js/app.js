function appState() {
    return {
        activeTab: 'chat',
        sessions: [],
        currentSessionId: null,
        messages: [],
        inputQuery: '',
        settings: { top_k: 5, use_chain_of_thought: false },
        citationModal: { open: false, filename: '', page_number: '', text: '' },
        previewModal: { open: false, filename: '', file_size: 0, pages: [] },
        documents: [],
        docSearchQuery: '',
        uploading: false,
        isDragging: false,
        uploadedChips: [],
        toast: { show: false, message: '', type: 'success' },
        toastShownKeys: new Set(),
        sessionToken: 0,
        previewToken: 0,
        historyLoading: false,
        historyError: null,
        sending: false,

        get filteredDocuments() {
            if (!this.docSearchQuery || !this.docSearchQuery.trim()) return this.documents;
            const q = this.docSearchQuery.toLowerCase();
            return this.documents.filter(d => d && d.filename && d.filename.toLowerCase().includes(q));
        },

        async init() {
            await this.loadSessions();
            if (!this.currentSessionId && this.sessions.length > 0) {
                await this.selectSession(this.sessions[0].session_id);
            }
        },

        showToast(message, type) {
            const key = String(type || 'success') + '|' + String(message);
            if (this.toastShownKeys.has(key)) return;
            this.toastShownKeys.add(key);
            this.toast = { show: true, message: String(message), type: type || 'success' };
            setTimeout(() => { this.toast.show = false; }, 4000);
        },

        escapeHtml(str) {
            if (str == null) return '';
            return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
        },

        sanitizeMarkdown(markdownContent) {
            if (!markdownContent) return '';
            let rawHtml = null;
            try {
                if (typeof marked !== 'undefined' && typeof marked.parse === 'function') {
                    rawHtml = marked.parse(markdownContent);
                }
            } catch (e) { rawHtml = null; }
            if (rawHtml == null) {
                return this.escapeHtml(markdownContent);
            }
            if (typeof DOMPurify !== 'undefined' && typeof DOMPurify.sanitize === 'function') {
                try {
                    return DOMPurify.sanitize(rawHtml);
                } catch (e) {
                    return this.escapeHtml(markdownContent);
                }
            }
            return this.escapeHtml(markdownContent);
        },

        async loadSessions() {
            try {
                const res = await fetch('/api/v1/sessions');
                if (!res.ok) throw new Error('Failed to load sessions (' + res.status + ')');
                const data = await res.json();
                this.sessions = Array.isArray(data) ? data : [];
            } catch (e) {
                this.showToast('Failed to load sessions', 'error');
            }
        },

        async selectSession(sessionId) {
            if (!sessionId) return;
            const token = ++this.sessionToken;
            this.currentSessionId = sessionId;
            this.messages = [];
            this.uploadedChips = [];
            this.historyLoading = true;
            this.historyError = null;
            this.sending = false;
            try {
                const res = await fetch('/api/v1/sessions/' + sessionId + '/history');
                if (!res.ok) throw new Error('Failed to load history (' + res.status + ')');
                const data = await res.json();
                const turns = Array.isArray(data && data.turns) ? data.turns : [];
                if (token === this.sessionToken) {
                    this.messages = turns;
                    this.historyLoading = false;
                }
            } catch (e) {
                if (token === this.sessionToken) {
                    this.historyLoading = false;
                    this.historyError = e && e.message ? e.message : String(e);
                    this.showToast('Failed to load conversation history', 'error');
                }
            }
        },

        async retrySelectCurrent() {
            if (this.currentSessionId) {
                await this.selectSession(this.currentSessionId);
            }
        },

        async createSession() {
            try {
                const res = await fetch('/api/v1/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: 'New Conversation' })
                });
                if (!res.ok) throw new Error('Failed to create session (' + res.status + ')');
                const data = await res.json();
                if (!data || !data.session_id) throw new Error('Session created but no session ID returned');
                this.currentSessionId = data.session_id;
                await this.selectSession(data.session_id);
                await this.loadSessions();
            } catch (e) {
                this.showToast('Failed to create new session: ' + (e && e.message ? e.message : 'error'), 'error');
            }
        },

        async deleteSession(sessionId) {
            if (!sessionId) return;
            const wasCurrent = (this.currentSessionId === sessionId);
            const token = ++this.sessionToken;
            try {
                const res = await fetch('/api/v1/sessions/' + sessionId, { method: 'DELETE' });
                if (res.status !== 200 && res.status !== 404) {
                    throw new Error('Failed to delete session (' + res.status + ')');
                }
                await this.loadSessions();
                if (wasCurrent) {
                    if (token === this.sessionToken) {
                        const remaining = this.sessions.length > 0 ? this.sessions[0].session_id : null;
                        if (remaining) {
                            await this.selectSession(remaining);
                        } else {
                            this.currentSessionId = null;
                            this.messages = [];
                            this.historyLoading = false;
                            this.historyError = null;
                        }
                    }
                }
            } catch (e) {
                if (token === this.sessionToken) {
                    this.showToast('Failed to delete session', 'error');
                }
            }
        },

        async sendMessage() {
            if (this.sending) return;
            if (!this.inputQuery || !this.inputQuery.trim()) return;
            if (!this.currentSessionId) {
                await this.createSession();
                if (!this.currentSessionId) return;
            }

            const intendedSessionId = this.currentSessionId;
            const query = this.inputQuery;
            this.inputQuery = '';

            this.messages.push({ role: 'user', content: query });
            this.sending = true;
            const token = this.sessionToken;

            try {
                const res = await fetch('/api/v1/sessions/' + intendedSessionId + '/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        top_k: this.settings && this.settings.top_k ? this.settings.top_k : 5,
                        use_chain_of_thought: this.settings && this.settings.use_chain_of_thought === true
                    })
                });

                if (!res.ok) {
                    let errDetail = 'Server error (' + res.status + ')';
                    try {
                        const errData = await res.json();
                        if (errData && errData.detail) errDetail = errData.detail;
                    } catch (_) {}
                    throw new Error(errDetail);
                }

                const data = await res.json();
                if (!data || data.answer == null) {
                    throw new Error('Invalid response: missing answer');
                }

                if (token === this.sessionToken && this.currentSessionId === intendedSessionId) {
                    this.messages.push({
                        role: 'assistant',
                        content: data.answer,
                        citations: Array.isArray(data.citations) ? data.citations : [],
                        groundedness_score: data.groundedness_score !== undefined ? data.groundedness_score : null
                    });
                }

                await this.loadSessions();
            } catch (e) {
                this.inputQuery = query;
                if (token === this.sessionToken && this.currentSessionId === intendedSessionId) {
                    this.messages.push({
                        role: 'assistant',
                        content: 'Error: ' + (e && e.message ? e.message : 'unknown error')
                    });
                }
                this.showToast('Chat request failed: ' + (e && e.message ? e.message : 'error'), 'error');
            } finally {
                this.sending = false;
            }
        },

        async loadDocuments() {
            try {
                const res = await fetch('/api/v1/documents/');
                if (!res.ok) throw new Error('Failed to load documents (' + res.status + ')');
                const data = await res.json();
                this.documents = Array.isArray(data && data.documents) ? data.documents : [];
            } catch (e) {
                this.showToast('Failed to load documents', 'error');
            }
        },

        async previewDocument(documentId) {
            if (!documentId) return;
            const token = ++this.previewToken;
            try {
                const res = await fetch('/api/v1/documents/' + documentId);
                if (!res.ok) throw new Error('Failed to fetch document details (' + res.status + ')');
                const data = await res.json();
                if (token === this.previewToken) {
                    this.previewModal = {
                        open: true,
                        filename: (data && data.metadata && data.metadata.filename) ? data.metadata.filename : 'Unknown',
                        file_size: (data && data.metadata && data.metadata.file_size) ? data.metadata.file_size : 0,
                        pages: Array.isArray(data && data.pages) ? data.pages : []
                    };
                }
            } catch (e) {
                if (token === this.previewToken) {
                    this.showToast('Error previewing document: ' + (e && e.message ? e.message : 'unknown error'), 'error');
                }
            }
        },

        async deleteDocumentSafely(documentId, filename) {
            if (!documentId) return;
            if (!confirm('Delete "' + (filename || 'this document') + '" and all associated chunks, vector indexes, and BM25 index?')) return;

            const derivedStores = [
                { name: 'Chunks', url: '/api/v1/chunks/' + documentId, method: 'DELETE' },
                { name: 'Vector Store Index', url: '/api/v1/vector-store/document/' + documentId, method: 'DELETE' },
                { name: 'BM25 Index', url: '/api/v1/bm25/document/' + documentId, method: 'DELETE' }
            ];
            const sourceStore = { name: 'Document Record', url: '/api/v1/documents/' + documentId, method: 'DELETE' };

            let failures = [];
            let skipped = [];

            for (let i = 0; i < derivedStores.length; i++) {
                const store = derivedStores[i];
                try {
                    const res = await fetch(store.url, { method: store.method });
                    if (res.status === 200) continue;
                    if (res.status === 404) { skipped.push(store.name); continue; }
                    failures.push(store.name);
                } catch (err) {
                    failures.push(store.name);
                }
            }

            if (failures.length > 0) {
                this.showToast('Cleanup aborted. Failed stores: ' + failures.join(', '), 'error');
                await this.loadDocuments();
                return;
            }

            try {
                const res = await fetch(sourceStore.url, { method: sourceStore.method });
                if (res.status === 200) {
                    if (skipped.length > 0) {
                        this.showToast('Deleted. Some stores already absent: ' + skipped.join(', '));
                    } else {
                        this.showToast('"' + (filename || 'document') + '" deleted successfully.');
                    }
                } else if (res.status === 404) {
                    this.showToast('Document record already absent. Derived stores cleaned.');
                } else {
                    failures.push(sourceStore.name);
                    this.showToast('Derived stores cleaned, but source record deletion failed (' + res.status + ')', 'error');
                }
            } catch (err) {
                failures.push(sourceStore.name);
                this.showToast('Derived stores cleaned, but source record deletion failed.', 'error');
            }

            await this.loadDocuments();
        },

        async handleFileUpload(e) {
            const files = e && e.target && e.target.files ? e.target.files : null;
            if (!files || files.length === 0) return;
            try {
                await this.processFiles(files);
            } finally {
                if (e && e.target) e.target.value = '';
            }
        },

        async handleFileDrop(e) {
            const files = e && e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files : null;
            if (!files || files.length === 0) return;
            await this.processFiles(files);
        },

        async processFiles(files) {
            const allowedExtensions = ['.pdf', '.doc', '.docx', '.txt', '.md', '.csv', '.xlsx', '.pptx', '.html', '.json', '.rtf', '.zip'];
            const maxFileSizeBytes = 20 * 1024 * 1024;
            this.uploading = true;
            try {
                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    if (!file) continue;
                    const filename = (file.name || '').toLowerCase();
                    const ext = filename.includes('.') ? '.' + filename.split('.').pop() : '';
                    if (!allowedExtensions.includes(ext)) {
                        this.showToast('Unsupported file type \'' + ext + '\'.', 'error');
                        continue;
                    }
                    if (file.size > maxFileSizeBytes) {
                        this.showToast('File "' + (file.name || 'unnamed') + '" exceeds 20MB limit.', 'error');
                        continue;
                    }
                    const formData = new FormData();
                    formData.append('file', file);
                    try {
                        const res = await fetch('/api/v1/pipeline/ingest', { method: 'POST', body: formData });
                        if (res.ok) {
                            this.showToast((file.name || 'File') + ' ingested successfully!', 'success');
                            this.uploadedChips.push({ name: file.name || 'unnamed' });
                            await this.loadDocuments();
                        } else {
                            let errDetail = '';
                            try { const d = await res.json(); if (d && d.detail) errDetail = d.detail; } catch (_) {}
                            this.showToast('Upload failed for ' + (file.name || 'unnamed') + ': ' + errDetail, 'error');
                        }
                    } catch (err) {
                        this.showToast('Network error uploading ' + (file.name || 'unnamed'), 'error');
                    }
                }
            } finally {
                this.uploading = false;
            }
        },

        openCitationModal(cit) {
            if (!cit) return;
            this.citationModal = {
                open: true,
                filename: cit.filename || 'Unknown',
                page_number: cit.page_number != null ? cit.page_number : '?',
                text: cit.quoted_text || 'Source text passage.'
            };
        }
    }
}
