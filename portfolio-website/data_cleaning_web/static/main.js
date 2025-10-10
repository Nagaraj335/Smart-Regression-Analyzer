document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('url-form');
    const urlInput = document.getElementById('csv-url');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        resultDiv.innerHTML = '';
        loadingDiv.classList.remove('hidden');
        const url = urlInput.value.trim();
        if (!url) {
            resultDiv.innerHTML = '<div class="bg-red-500 text-white p-2 rounded">Please enter a CSV URL.</div>';
            loadingDiv.classList.add('hidden');
            return;
        }
        try {
            const response = await fetch('/api/clean_url', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ url })
            });
            const data = await response.json();
            loadingDiv.classList.add('hidden');
            if (data.success) {
                resultDiv.innerHTML = `<div class="bg-green-600 text-white p-2 rounded">Cleaned file ready: <a href="${data.download_link}" class="underline text-yellow-300" download>Download CSV</a></div>`;
            } else {
                resultDiv.innerHTML = `<div class="bg-red-500 text-white p-2 rounded">${data.error}</div>`;
            }
        } catch (err) {
            loadingDiv.classList.add('hidden');
            resultDiv.innerHTML = '<div class="bg-red-500 text-white p-2 rounded">Server error. Please try again.</div>';
        }
    });
});
