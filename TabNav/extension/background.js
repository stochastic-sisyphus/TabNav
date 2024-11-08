import { TabNavAPI } from './src/utils/api';

const api = new TabNavAPI();
let processingTabs = false;

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.type === 'PROCESS_TABS') {
        if (!processingTabs) {
            processTabs().catch(console.error);
        }
        return true;
    }
});

async function processTabs() {
    try {
        processingTabs = true;
        
        // Get current tabs
        const tabs = await chrome.tabs.query({});
        
        // Process tabs
        const result = await api.processTabs(tabs);
        
        // Store results
        await chrome.storage.local.set({ 
            groups: result.groups,
            lastUpdate: new Date().toISOString()
        });
        
        // Notify popup
        chrome.runtime.sendMessage({ 
            type: 'PROCESSING_COMPLETE',
            result 
        });
        
    } catch (error) {
        console.error('Processing failed:', error);
        chrome.runtime.sendMessage({ 
            type: 'PROCESSING_ERROR',
            error: error.message 
        });
    } finally {
        processingTabs = false;
    }
}