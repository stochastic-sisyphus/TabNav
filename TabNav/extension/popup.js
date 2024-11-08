let store;
let unsubscribe;

document.addEventListener('DOMContentLoaded', async () => {
    const processButton = document.getElementById('processButton');
    const loadingIndicator = document.getElementById('loading');
    const resultsContainer = document.getElementById('results');
    const errorContainer = document.getElementById('error');

    store = new TabNavStore();
    await store.loadState();

    unsubscribe = store.subscribe(renderState);

    processButton.addEventListener('click', () => {
        chrome.runtime.sendMessage({ type: 'PROCESS_TABS' });
    });

    renderState(store.state);
});

function renderState(state) {
    const loadingIndicator = document.getElementById('loading');
    const resultsContainer = document.getElementById('results');
    const errorContainer = document.getElementById('error');
    const processButton = document.getElementById('processButton');

    // Update loading state
    loadingIndicator.style.display = state.isProcessing ? 'block' : 'none';
    processButton.disabled = state.isProcessing;

    // Show error if any
    if (state.error) {
        errorContainer.textContent = state.error;
        errorContainer.style.display = 'block';
    } else {
        errorContainer.style.display = 'none';
    }

    // Render results
    if (state.groups.length > 0) {
        resultsContainer.innerHTML = state.groups
            .map(group => renderGroup(group))
            .join('');
    }
}

function renderGroup(group) {
    return `
        <div class="group">
            <div class="group-header">
                <h3>${group.name}</h3>
                <p class="group-summary">${group.summary}</p>
            </div>
            <div class="articles">
                ${group.articles
                    .map(article => renderArticle(article))
                    .join('')}
            </div>
        </div>
    `;
}

function renderArticle(article) {
    return `
        <div class="article">
            <div class="article-header">
                <a href="${article.url}" target="_blank" class="article-title">
                    ${article.title}
                </a>
                <span class="relevance">
                    ${Math.round(article.overlapScore * 100)}% relevant
                </span>
            </div>
            <p class="summary">${article.summary}</p>
            <ul class="key-points">
                ${article.keyPoints
                    .map(point => `<li>${point}</li>`)
                    .join('')}
            </ul>
        </div>
    `;
}