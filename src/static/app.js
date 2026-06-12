document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const statusBadge = document.getElementById('status-badge');
    
    // Stats elements
    const statUsers = document.getElementById('stat-users');
    const statMovies = document.getElementById('stat-movies');
    const statSparsity = document.getElementById('stat-sparsity');
    const statSvdMetrics = document.getElementById('stat-svd-metrics');
    const statCfMetrics = document.getElementById('stat-cf-metrics');
    const statUserCfMetrics = document.getElementById('stat-user-cf-metrics');
    const statNcfMetrics = document.getElementById('stat-ncf-metrics');
    const statContentMetrics = document.getElementById('stat-content-metrics');
    
    // Input panels
    const userIdInput = document.getElementById('user-id-input');
    const loadUserBtn = document.getElementById('load-user-btn');
    const randomUserBtn = document.getElementById('random-user-btn');
    const movieIdInput = document.getElementById('movie-id-input');
    const findSimilarBtn = document.getElementById('find-similar-btn');
    const coldStartBtn = document.getElementById('cold-start-btn');
    
    // Quick select chips
    const chipBtns = document.querySelectorAll('.chip-btn[data-user]');
    
    // Content sections
    const emptyStateSection = document.getElementById('empty-state-section');
    const userHistorySection = document.getElementById('user-history-section');
    const recommendationsSection = document.getElementById('recommendations-section');
    const coldstartSection = document.getElementById('coldstart-section');
    const similarMoviesSection = document.getElementById('similar-movies-section');
    
    // Lists and grids
    const userHistoryGrid = document.getElementById('user-history-grid');
    const historyCount = document.getElementById('history-count');
    const currentUserTitle = document.getElementById('current-user-title');
    const svdRecsList = document.getElementById('svd-recs-list');
    const cfRecsList = document.getElementById('cf-recs-list');
    const userCfRecsList = document.getElementById('user-cf-recs-list');
    const ncfRecsList = document.getElementById('ncf-recs-list');
    const contentRecsList = document.getElementById('content-recs-list');
    const coldstartList = document.getElementById('coldstart-list');
    const coldstartGenres = document.getElementById('coldstart-genres');
    const similarMoviesList = document.getElementById('similar-movies-list');
    const similarTargetTitle = document.getElementById('similar-target-title');

    // 1. Initial Load: Check System Status
    checkSystemStatus();

    async function checkSystemStatus() {
        try {
            const response = await fetch('/api/system-status');
            const data = await response.json();
            
            if (data.ready) {
                statusBadge.textContent = 'Engine Ready';
                statusBadge.parentElement.querySelector('.pulse-dot').style.backgroundColor = '#00F2FE';
                statusBadge.parentElement.querySelector('.pulse-dot').style.boxShadow = '0 0 10px #00F2FE';
                
                // Populate stats
                const meta = data.metadata;
                statUsers.textContent = Number(meta.num_users).toLocaleString();
                statMovies.textContent = Number(meta.num_movies).toLocaleString();
                statSparsity.textContent = (meta.sparsity * 100).toFixed(2) + '%';
                
                const svdRes = data.results['Funk SVD'];
                const cfRes = data.results['Item-CF'];
                const userCfRes = data.results['User-CF'];
                const ncfRes = data.results['Neural-CF'];
                
                statSvdMetrics.textContent = `${svdRes.RMSE.toFixed(4)} / ${svdRes['MAP@10'].toFixed(4)}`;
                statCfMetrics.textContent = `${cfRes.RMSE.toFixed(4)} / ${cfRes['MAP@10'].toFixed(4)}`;
                statUserCfMetrics.textContent = userCfRes ? `${userCfRes.RMSE.toFixed(4)} / ${userCfRes['MAP@10'].toFixed(4)}` : '-- / --';
                statNcfMetrics.textContent = ncfRes ? `${ncfRes.RMSE.toFixed(4)} / ${ncfRes['MAP@10'].toFixed(4)}` : '-- / --';
                const contentRes = data.results['Content-Based'];
                statContentMetrics.textContent = contentRes ? `${contentRes.RMSE.toFixed(4)} / ${contentRes['MAP@10'].toFixed(4)}` : '-- / --';
            } else {
                statusBadge.textContent = 'Training Required';
                statusBadge.parentElement.querySelector('.pulse-dot').style.backgroundColor = '#F43F5E';
                statusBadge.parentElement.querySelector('.pulse-dot').style.boxShadow = '0 0 10px #F43F5E';
                
                statUsers.textContent = 'N/A';
                statMovies.textContent = 'N/A';
                statSparsity.textContent = 'N/A';
                statSvdMetrics.textContent = 'Run data pipeline';
                statCfMetrics.textContent = 'Run evaluation.py';
                statUserCfMetrics.textContent = 'Run evaluation.py';
                statNcfMetrics.textContent = 'Run evaluation.py';
                statContentMetrics.textContent = 'Run evaluation.py';
            }
        } catch (error) {
            console.error('Error fetching system status:', error);
            statusBadge.textContent = 'API Connection Error';
        }
    }

    // 2. Load User Profile & Recommendations
    async function loadUserProfile(userId) {
        if (!userId || userId < 1) return;
        
        // Update input field
        userIdInput.value = userId;
        
        // Set active chip states
        chipBtns.forEach(btn => {
            if (btn.getAttribute('data-user') === userId.toString()) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        // Show loading state
        emptyStateSection.classList.add('hide');
        coldstartSection.classList.add('hide');
        similarMoviesSection.classList.add('hide');
        
        userHistorySection.classList.remove('hide');
        recommendationsSection.classList.remove('hide');
        
        currentUserTitle.textContent = `#${userId}`;
        userHistoryGrid.innerHTML = '<div class="shimmer" style="width: 100%; height: 120px; border-radius: 12px;"></div>';
        svdRecsList.innerHTML = generateShimmerItems(5);
        cfRecsList.innerHTML = generateShimmerItems(5);
        userCfRecsList.innerHTML = generateShimmerItems(5);
        ncfRecsList.innerHTML = generateShimmerItems(5);
        contentRecsList.innerHTML = generateShimmerItems(5);
        
        try {
            // Fetch history
            const historyRes = await fetch(`/api/user-history/${userId}`);
            if (!historyRes.ok) throw new Error('User history not found');
            const historyData = await historyRes.json();
            
            renderHistory(historyData.history);
            
            // Fetch recommendations
            const recsRes = await fetch(`/api/recommendations/${userId}`);
            if (!recsRes.ok) throw new Error('Recommendations not found');
            const recsData = await recsRes.json();
            
            renderRecommendations(recsData.svd, svdRecsList);
            renderRecommendations(recsData.cf, cfRecsList);
            renderRecommendations(recsData.user_cf, userCfRecsList);
            renderRecommendations(recsData.ncf, ncfRecsList);
            renderRecommendations(recsData.content, contentRecsList);
            
        } catch (error) {
            console.error('Error loading user profile:', error);
            userHistoryGrid.innerHTML = `<p class="error-msg">Error: ${error.message}. If the system is brand new, please use the Cold-Start Simulator.</p>`;
            svdRecsList.innerHTML = '';
            cfRecsList.innerHTML = '';
            userCfRecsList.innerHTML = '';
            ncfRecsList.innerHTML = '';
            contentRecsList.innerHTML = '';
        }
    }

    function renderHistory(history) {
        historyCount.textContent = `${history.length} movie${history.length !== 1 ? 's' : ''}`;
        
        if (history.length === 0) {
            userHistoryGrid.innerHTML = '<p class="text-secondary">No historical ratings found.</p>';
            return;
        }
        
        userHistoryGrid.innerHTML = history.map(item => `
            <div class="history-item-card">
                <div>
                    <h4 class="item-title" title="${item.title}">${item.title}</h4>
                    <span class="item-meta">${item.year ? item.year : 'N/A'} // M-ID: ${item.movie_id}</span>
                </div>
                <div class="item-rating-row">
                    <div class="stars">${generateStars(item.rating)}</div>
                    <span class="rating-badge">${item.rating.toFixed(1)}</span>
                </div>
            </div>
        `).join('');
    }

    function renderRecommendations(recs, container) {
        if (recs.length === 0) {
            container.innerHTML = '<p class="text-secondary" style="padding: 20px 0;">No recommendations generated.</p>';
            return;
        }
        
        container.innerHTML = recs.map((item, index) => `
            <div class="rec-item">
                <span class="rec-rank">${index + 1}</span>
                <div class="rec-info">
                    <h4 class="rec-title" title="${item.title}">${item.title}</h4>
                    <span class="rec-year">${item.year ? item.year : 'N/A'} // ID: ${item.movie_id}</span>
                    <p class="rec-explanation" title="${item.explanation}">${item.explanation}</p>
                </div>
                <div class="rec-score-box">
                    <span class="rec-score">${item.score.toFixed(1)}</span>
                    <span class="rec-score-label">Predicted</span>
                </div>
            </div>
        `).join('');
    }

    // 3. Similar Movies Search
    async function loadSimilarMovies(movieId) {
        if (!movieId || movieId < 1) return;
        
        // Hide/Show sections
        emptyStateSection.classList.add('hide');
        userHistorySection.classList.add('hide');
        recommendationsSection.classList.add('hide');
        coldstartSection.classList.add('hide');
        
        similarMoviesSection.classList.remove('hide');
        similarTargetTitle.textContent = `Movie ID #${movieId}`;
        similarMoviesList.innerHTML = generateShimmerCards(4);
        
        try {
            const response = await fetch(`/api/similar-movies/${movieId}`);
            if (!response.ok) throw new Error('Movie ID not found in mapping database');
            const data = await response.json();
            
            similarTargetTitle.textContent = data.similar_movies.length > 0 ? `Movie ID #${movieId}` : `Movie ID #${movieId} (No matches)`;
            
            if (data.similar_movies.length === 0) {
                similarMoviesList.innerHTML = '<p class="text-secondary">No similar movies found. Ensure the movie exists in the training set.</p>';
                return;
            }
            
            similarMoviesList.innerHTML = data.similar_movies.map(item => `
                <div class="horizontal-rec-card">
                    <div>
                        <h4 class="horiz-title" title="${item.title}">${item.title}</h4>
                        <span class="item-meta">${item.year ? item.year : 'N/A'} // M-ID: ${item.movie_id}</span>
                    </div>
                    <div class="horiz-footer">
                        <span class="item-meta">Cosine Similarity</span>
                        <span class="horiz-score horiz-score-cf">${(item.similarity * 100).toFixed(0)}%</span>
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Error fetching similar movies:', error);
            similarMoviesList.innerHTML = `<p class="error-msg" style="grid-column: 1/-1;">Error: ${error.message}.</p>`;
        }
    }

    // 4. Cold Start Simulator
    async function loadColdStart() {
        // Find selected genres
        const genres = [];
        if (document.getElementById('genre-action').checked) genres.push('Action');
        if (document.getElementById('genre-scifi').checked) genres.push('Sci-Fi');
        if (document.getElementById('genre-comedy').checked) genres.push('Comedy');
        if (document.getElementById('genre-drama').checked) genres.push('Drama');
        if (document.getElementById('genre-romance').checked) genres.push('Romance');
        
        // Hide/Show sections
        emptyStateSection.classList.add('hide');
        userHistorySection.classList.add('hide');
        recommendationsSection.classList.add('hide');
        similarMoviesSection.classList.add('hide');
        
        coldstartSection.classList.remove('hide');
        coldstartGenres.textContent = genres.length > 0 ? genres.join(' + ') : 'Global Popularity Charts';
        coldstartList.innerHTML = generateShimmerCards(4);
        
        try {
            const queryParam = genres.length > 0 ? `?genres=${encodeURIComponent(genres.join(','))}` : '';
            const response = await fetch(`/api/coldstart${queryParam}`);
            const data = await response.json();
            
            if (data.recommendations.length === 0) {
                coldstartList.innerHTML = '<p class="text-secondary">No movies found matching these preferences.</p>';
                return;
            }
            
            coldstartList.innerHTML = data.recommendations.map(item => `
                <div class="horizontal-rec-card">
                    <div>
                        <h4 class="horiz-title" title="${item.title}">${item.title}</h4>
                        <span class="item-meta">${item.year ? item.year : 'N/A'} // M-ID: ${item.movie_id}</span>
                        <p class="horiz-desc" title="${item.explanation}">${item.explanation}</p>
                    </div>
                    <div class="horiz-footer">
                        <span class="item-meta">Avg Rating</span>
                        <span class="horiz-score">${item.score.toFixed(1)}★</span>
                    </div>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Error fetching cold-start recommendations:', error);
            coldstartList.innerHTML = '<p class="error-msg" style="grid-column: 1/-1;">Error fetching recommendations.</p>';
        }
    }

    // Event Listeners
    loadUserBtn.addEventListener('click', () => {
        const userId = parseInt(userIdInput.value);
        loadUserProfile(userId);
    });
    
    userIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const userId = parseInt(userIdInput.value);
            loadUserProfile(userId);
        }
    });

    randomUserBtn.addEventListener('click', () => {
        // User IDs are 1 to 943 in MovieLens-100k
        const randomId = Math.floor(Math.random() * 943) + 1;
        loadUserProfile(randomId);
    });

    chipBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const userId = parseInt(btn.getAttribute('data-user'));
            loadUserProfile(userId);
        });
    });

    findSimilarBtn.addEventListener('click', () => {
        const movieId = parseInt(movieIdInput.value);
        loadSimilarMovies(movieId);
    });
    
    movieIdInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const movieId = parseInt(movieIdInput.value);
            loadSimilarMovies(movieId);
        }
    });

    coldStartBtn.addEventListener('click', () => {
        loadColdStart();
    });

    // Helper functions
    function generateStars(rating) {
        const fullStars = Math.floor(rating);
        const halfStar = rating % 1 >= 0.5 ? 1 : 0;
        const emptyStars = 5 - fullStars - halfStar;
        
        let starsHtml = '';
        for (let i = 0; i < fullStars; i++) starsHtml += '<i class="fa-solid fa-star"></i>';
        if (halfStar) starsHtml += '<i class="fa-solid fa-star-half-stroke"></i>';
        for (let i = 0; i < emptyStars; i++) starsHtml += '<i class="fa-regular fa-star"></i>';
        
        return starsHtml;
    }

    function generateShimmerItems(count) {
        let html = '';
        for (let i = 0; i < count; i++) {
            html += `
                <div class="rec-item shimmer" style="height: 72px;">
                    <div style="width: 24px; height: 24px; background: rgba(255,255,255,0.03); border-radius: 4px;"></div>
                    <div style="flex: 1; display: flex; flex-direction: column; gap: 8px;">
                        <div style="width: 60%; height: 14px; background: rgba(255,255,255,0.03); border-radius: 4px;"></div>
                        <div style="width: 40%; height: 10px; background: rgba(255,255,255,0.02); border-radius: 4px;"></div>
                    </div>
                    <div style="width: 40px; height: 24px; background: rgba(255,255,255,0.03); border-radius: 4px;"></div>
                </div>
            `;
        }
        return html;
    }

    function generateShimmerCards(count) {
        let html = '';
        for (let i = 0; i < count; i++) {
            html += `
                <div class="horizontal-rec-card shimmer" style="height: 160px; justify-content: space-between;">
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <div style="width: 80%; height: 16px; background: rgba(255,255,255,0.03); border-radius: 4px;"></div>
                        <div style="width: 40%; height: 10px; background: rgba(255,255,255,0.02); border-radius: 4px;"></div>
                        <div style="width: 90%; height: 12px; background: rgba(255,255,255,0.02); border-radius: 4px; margin-top: 10px;"></div>
                    </div>
                    <div style="width: 100%; height: 16px; background: rgba(255,255,255,0.03); border-radius: 4px; margin-top: 16px;"></div>
                </div>
            `;
        }
        return html;
    }
});
