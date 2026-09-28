/**
 * XRefactor Frontend Application (Vue 3)
 * Provides interactive UI for pipeline orchestration and configuration
 */

const { createApp } = Vue;

const app = createApp({
    data() {
        return {
            currentView: 'home',
            status: {
                connected: false,
                message: 'Connecting...'
            },
            pipelineRunning: false,
            pipelineProgress: 0,
            pipelineElapsedSeconds: 0,
            pipelineOutput: null,
            pipelineError: null,
            lastResults: null,
            charts: {
                nodeType: null,
                edgeType: null,
                confidence: null
            },
            network: null,
            
            pipelineInput: {
                dataDirectory: '../Data/1273091433/jeesite',
                device: 'cpu',
                outputDirectory: './outputs',
                configPath: './configs/config.yaml'
            },
            
            config: {
                cpg: {
                    language: 'java',
                    include_data_flow: true,
                    include_control_flow: true,
                    include_call_graph: true
                },
                gnn: {
                    model_type: 'gat',
                    hidden_dims: '256,256',
                    output_dim: 128,
                    num_heads: 8,
                    num_layers: 3,
                    dropout: 0.1
                },
                transformer: {
                    model_type: 'microsoft/codebert-base',
                    hidden_size: 768,
                    num_layers: 6,
                    num_attention_heads: 12
                }
            }
        };
    },
    
    mounted() {
        this.checkHealth();
        this.checkPipelineStatus();
        // Check health every 30 seconds
        setInterval(() => this.checkHealth(), 30000);
        // Poll pipeline status so an in-progress run survives a page reload
        setInterval(() => this.checkPipelineStatus(), 5000);
    },
    
    computed: {
        /**
         * Flatten recorded pipeline metrics into rows for the performance table
         */
        metricsRows() {
            const metrics = this.lastResults?.raw?.metrics || {};
            return Object.keys(metrics).map(name => ({
                name,
                count: metrics[name].count ?? 0,
                mean: metrics[name].mean ?? 0,
                median: metrics[name].median ?? 0,
                min: metrics[name].min ?? 0,
                max: metrics[name].max ?? 0,
                stdev: metrics[name].stdev ?? 0
            }));
        },
        
        /**
         * Size-capped CPG graph payload for network visualization
         */
        graphData() {
            return this.lastResults?.raw?.stages?.cpg?.graph || this.lastResults?.raw?.graph || { nodes: [], edges: [], truncated: false, total_nodes: 0, shown_nodes: 0 };
        },
        
        /**
         * Confidence scores per refactoring suggestion for chart rendering
         */
        confidenceChartData() {
            const suggestions = this.lastResults?.raw?.stages?.transformer?.refactoring_suggestions || [];
            return {
                labels: suggestions.map((s, i) => `#${i + 1} ${s.refactoring_type || ''}`),
                data: suggestions.map(s => s.confidence ?? 0)
            };
        }
    },
    
    watch: {
        lastResults() {
            this.$nextTick(() => this.renderVisualizations());
        },
        currentView(view) {
            if (view === 'results') {
                this.$nextTick(() => this.renderVisualizations());
            }
        }
    },
    methods: {
        /**
         * Check if backend is running
         */
        async checkHealth() {
            try {
                const response = await axios.get('/api/health', { timeout: 5000 });
                this.status.connected = true;
                this.status.message = 'Connected to backend';
                console.log('✓ Backend health check passed');
            } catch (error) {
                this.status.connected = false;
                this.status.message = 'Backend unreachable';
                console.error('✗ Backend health check failed:', error.message);
            }
        },
        
        /**
         * Poll backend for an in-progress pipeline run (covers page reloads/new tabs)
         */
        async checkPipelineStatus() {
            try {
                const response = await axios.get('/api/pipeline/status', { timeout: 5000 });
                if (response.data.running) {
                    this.pipelineRunning = true;
                    this.pipelineElapsedSeconds = response.data.elapsed_seconds || 0;
                } else if (this.pipelineRunning && this.pipelineElapsedSeconds > 0) {
                    // A run we were tracking has finished since our last poll
                    this.pipelineRunning = false;
                    this.pipelineElapsedSeconds = 0;
                }
            } catch (error) {
                // Non-fatal: status polling failures shouldn't disrupt the UI
                console.debug('Pipeline status check failed:', error.message);
            }
        },
        
        /**
         * Run the complete XRefactor pipeline
         */
        async runFullPipeline() {
            if (!this.pipelineInput.dataDirectory) {
                alert('Please specify a data directory');
                return;
            }
            
            this.pipelineRunning = true;
            this.pipelineError = null;
            this.pipelineOutput = null;
            this.pipelineProgress = 10;
            
            try {
                console.log('Starting full pipeline...');
                
                const response = await axios.post('/api/pipeline/run', {
                    data_directory: this.pipelineInput.dataDirectory,
                    config_path: this.pipelineInput.configPath,
                    device: this.pipelineInput.device,
                    output_directory: this.pipelineInput.outputDirectory
                }, {
                    timeout: 600000  // 10 minute timeout
                });
                
                this.pipelineProgress = 100;
                this.pipelineOutput = response.data;
                this.lastResults = this.normalizePipelineResults(response.data);
                
                console.log('✓ Pipeline completed successfully');
                this.showNotification('Pipeline completed successfully!', 'success');
                
            } catch (error) {
                console.error('✗ Pipeline error:', error);
                this.pipelineError = error.response?.data?.message || error.message;
                this.showNotification(`Pipeline failed: ${this.pipelineError}`, 'error');
                
            } finally {
                this.pipelineRunning = false;
                this.pipelineElapsedSeconds = 0;
            }
        },
        
        /**
         * Run a specific pipeline stage
         */
        async runStage(stageName) {
            if (stageName === 'cpg_construction' && !this.pipelineInput.dataDirectory) {
                alert('Please specify a data directory');
                return;
            }
            
            this.pipelineRunning = true;
            this.pipelineError = null;
            this.pipelineOutput = null;
            
            try {
                console.log(`Starting stage: ${stageName}`);
                
                const response = await axios.post(`/api/pipeline/stage/${stageName}`, {
                    data_directory: this.pipelineInput.dataDirectory,
                    config_path: this.pipelineInput.configPath,
                    device: this.pipelineInput.device
                });
                
                this.pipelineOutput = response.data;
                if (stageName === 'cpg_construction') {
                    this.lastResults = this.normalizePipelineResults({ results: response.data, timestamp: new Date().toISOString() });
                }
                console.log(`✓ Stage ${stageName} completed`);
                this.showNotification(`Stage ${stageName} completed!`, 'success');
                
            } catch (error) {
                console.error(`✗ Stage ${stageName} error:`, error);
                this.pipelineError = error.response?.data?.message || error.message;
                this.showNotification(`Stage failed: ${this.pipelineError}`, 'error');
                
            } finally {
                this.pipelineRunning = false;
            }
        },
        
        /**
         * Save current configuration to localStorage
         */
        saveConfiguration() {
            try {
                localStorage.setItem('xrefactor_config', JSON.stringify(this.config));
                this.showNotification('Configuration saved!', 'success');
                console.log('✓ Configuration saved to localStorage');
            } catch (error) {
                console.error('✗ Save configuration error:', error);
                this.showNotification('Failed to save configuration', 'error');
            }
        },
        
        /**
         * Load configuration from localStorage
         */
        loadConfiguration() {
            try {
                const saved = localStorage.getItem('xrefactor_config');
                if (saved) {
                    this.config = JSON.parse(saved);
                    this.showNotification('Configuration loaded!', 'success');
                    console.log('✓ Configuration loaded from localStorage');
                } else {
                    this.showNotification('No saved configuration found', 'info');
                }
            } catch (error) {
                console.error('✗ Load configuration error:', error);
                this.showNotification('Failed to load configuration', 'error');
            }
        },
        
        /**
         * Show notification message
         */
        showNotification(message, type = 'info') {
            // Simple notification (can be enhanced with a toast library)
            console.log(`[${type.toUpperCase()}] ${message}`);
            alert(message);
        },
        
        /**
         * Normalize pipeline result payload for UI consumption
         */
        normalizePipelineResults(payload) {
            if (!payload) return null;
            const results = payload.results || payload;
            const cpgStats = results?.stages?.cpg?.statistics || results?.statistics || null;
            const edgeTypes = cpgStats?.edge_types || cpgStats?.edgeTypes || {};
            const normalized = {
                status: results.status || payload.status || 'unknown',
                timestamp: results.timestamp || payload.timestamp || new Date().toISOString(),
                stats: {
                    cpg: {
                        total_nodes: cpgStats?.total_nodes ?? cpgStats?.totalNodes ?? 0,
                        total_edges: cpgStats?.total_edges ?? cpgStats?.totalEdges ?? 0,
                        include_data_flow: cpgStats?.include_data_flow ?? false,
                        include_control_flow: cpgStats?.include_control_flow ?? false,
                        include_call_graph: cpgStats?.include_call_graph ?? false,
                        data_flow_edges: edgeTypes?.data_flow ?? 0,
                        control_flow_edges: edgeTypes?.control_flow ?? 0,
                        call_graph_edges: edgeTypes?.calls ?? 0,
                        edge_types: edgeTypes,
                        node_types: cpgStats?.node_types ?? {}
                    }
                },
                transformer: {
                    suggestion_count: results?.stages?.transformer?.refactoring_suggestions?.length || 0,
                    model: results?.stages?.transformer?.model || null
                },
                raw: results
            };
            return normalized;
        },

        /**
         * Format JSON for display
         */
        formatJSON(obj) {
            return JSON.stringify(obj, null, 2);
        },
        
        /**
         * Render all Results-tab visualizations (charts + network graph)
         */
        renderVisualizations() {
            if (this.currentView !== 'results' || !this.lastResults || typeof Chart === 'undefined') return;
            
            const nodeTypes = this.lastResults.stats?.cpg?.node_types || {};
            this.renderBarChart('nodeType', 'nodeTypeChart', Object.keys(nodeTypes), Object.values(nodeTypes), 'Nodes', '#667eea');
            
            const edgeTypes = this.lastResults.stats?.cpg?.edge_types || {};
            this.renderBarChart('edgeType', 'edgeTypeChart', Object.keys(edgeTypes), Object.values(edgeTypes), 'Edges', '#4caf50');
            
            if (this.confidenceChartData.labels.length) {
                this.renderBarChart('confidence', 'confidenceChart', this.confidenceChartData.labels, this.confidenceChartData.data, 'Confidence', '#ff9800');
            }
            
            this.renderGraphNetwork();
        },
        
        /**
         * Generic bar chart renderer; destroys any previous instance on the same canvas
         */
        renderBarChart(chartKey, canvasRef, labels, data, label, color) {
            const canvas = this.$refs[canvasRef];
            if (!canvas) return;
            
            if (this.charts[chartKey]) {
                this.charts[chartKey].destroy();
            }
            
            this.charts[chartKey] = new Chart(canvas, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [{
                        label,
                        data,
                        backgroundColor: color
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true } }
                }
            });
        },
        
        /**
         * Render the Code Property Graph using vis-network
         */
        renderGraphNetwork() {
            const container = this.$refs.cpgNetwork;
            if (!container || typeof vis === 'undefined') return;
            
            const typeColors = {
                class: '#667eea',
                method: '#4caf50',
                field: '#ff9800',
                function: '#9c27b0',
                variable: '#03a9f4',
                statement: '#9e9e9e'
            };
            
            const nodes = new vis.DataSet((this.graphData.nodes || []).map(n => ({
                id: n.id,
                label: n.label,
                title: `${n.type} · ${n.file || ''}:${n.line || ''}`,
                color: typeColors[n.type] || '#607d8b'
            })));
            
            const edges = new vis.DataSet((this.graphData.edges || []).map(e => ({
                from: e.source,
                to: e.target,
                label: e.type,
                arrows: 'to',
                color: { color: '#c5cae9' },
                font: { size: 8, align: 'middle' }
            })));
            
            if (this.network) {
                this.network.destroy();
            }
            
            this.network = new vis.Network(container, { nodes, edges }, {
                layout: { improvedLayout: true },
                physics: { stabilization: true, barnesHut: { gravitationalConstant: -3000 } },
                nodes: { shape: 'dot', size: 10, font: { size: 10 } },
                edges: { smooth: { type: 'continuous' } }
            });
        }
    }
});

app.mount('#app');

console.log('🚀 XRefactor Frontend loaded successfully');
