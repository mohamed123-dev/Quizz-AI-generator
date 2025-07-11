const pdfSelect = document.getElementById('pdf-select');
const refreshPdfListBtn = document.getElementById('refresh-pdf-list-btn');
const startQuizBtn = document.getElementById('start-quiz-btn');
const refreshQuizBtn = document.getElementById('refresh-quiz-btn');
const quizContainer = document.getElementById('quiz-container');
const quizResult = document.getElementById('quiz-result');

let quizData = [];
let userAnswers = [];

function setButtonsDisabled(disabled) {
    [refreshPdfListBtn, startQuizBtn, refreshQuizBtn].forEach(btn => {
        if (btn) btn.disabled = disabled;
    });
}

async function populatePdfDropdown(restoreValue = null) {
    pdfSelect.innerHTML = '';
    setButtonsDisabled(true);
    try {
        const response = await fetch('http://localhost:8000/list_pdfs');
        const data = await response.json();
        if (Array.isArray(data.pdfs) && data.pdfs.length > 0) {
            data.pdfs.forEach(pdf => {
                const option = document.createElement('option');
                option.value = pdf;
                option.textContent = pdf;
                pdfSelect.appendChild(option);
            });
            if (restoreValue && data.pdfs.includes(restoreValue)) {
                pdfSelect.value = restoreValue;
            }
        } else {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'Aucun document disponible';
            pdfSelect.appendChild(option);
        }
    } catch (err) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Erreur de chargement';
        pdfSelect.appendChild(option);
        quizContainer.innerHTML = '<div class="quiz-error">Erreur lors du chargement de la liste des PDF.</div>';
    }
    setButtonsDisabled(false);
}

async function fetchQuizForSelectedPdf() {
    const selectedPdf = pdfSelect.value;
    if (!selectedPdf) {
        quizContainer.innerHTML = '<div class="quiz-error">Veuillez choisir un document PDF.</div>';
        return;
    }
    quizContainer.innerHTML = 'Chargement du quiz...';
    setButtonsDisabled(true);
    try {
        const response = await fetch(`http://localhost:8000/get_quiz?filename=${encodeURIComponent(selectedPdf)}`);
        const data = await response.json();
        if (Array.isArray(data) && data.length > 0) {
            quizData = data;
            userAnswers = [];
            renderAllQuestions();
        } else {
            quizContainer.innerHTML = '<div class="quiz-error">Aucun quiz trouvé pour ce document. Veuillez générer un quiz.</div>';
        }
    } catch (err) {
        quizContainer.innerHTML = '<div class="quiz-error">Erreur lors du chargement du quiz.</div>';
    }
    setButtonsDisabled(false);
}

function renderAllQuestions() {
    quizContainer.style.display = 'block';
    quizResult.style.display = 'none';
    let html = '';
    quizData.forEach((q, idx) => {
        html += `<div class="quiz-block">
            <div class="quiz-progress">Question ${idx + 1} / ${quizData.length}</div>
            <div class="quiz-question">${q.question}</div>`;
        q.options.forEach((opt, oidx) => {
            const letter = String.fromCharCode(65 + oidx);
            html += `<label class="quiz-option"><input type="checkbox" name="q${idx}" value="${letter}"> ${letter}. ${opt}</label><br>`;
        });
        html += '</div>';
    });
    html += '<button id="submit-quiz-btn" type="button">Valider mes réponses</button>';
    quizContainer.innerHTML = html;
    document.getElementById('submit-quiz-btn').onclick = handleAllQuizSubmit;
}

function handleAllQuizSubmit() {
    userAnswers = [];
    let allAnswered = true;
    for (let i = 0; i < quizData.length; i++) {
        const checked = Array.from(document.querySelectorAll(`input[name="q${i}"]:checked`)).map(cb => cb.value);
        if (checked.length < 1 || checked.length > 2) {
            allAnswered = false;
            break;
        }
        userAnswers.push(checked);
    }
    if (!allAnswered) {
        alert('Pour chaque question, sélectionnez 1 ou 2 réponses.');
        return;
    }
    showQuizResult();
}

function showQuizResult() {
    let score = 0;
    let resultHtml = '<h2>Résultats du Quiz</h2>';
    quizData.forEach((q, i) => {
        const user = userAnswers[i] || [];
        const correct = q.correct_answers;
        const isCorrect = user.length === correct.length && user.every(ans => correct.includes(ans));
        resultHtml += `<div class="quiz-result-q">
            <div><strong>Q${i+1}:</strong> ${q.question}</div>
            <div>Votre réponse: ${user.join(', ') || 'Aucune'} | Réponse correcte: ${correct.join(', ')}</div>
            <div>${isCorrect ? '✅ Bonne réponse' : '❌ Mauvaise réponse'}</div>
        </div>`;
        if (isCorrect) score++;
    });
    resultHtml += `<div class="quiz-final-score">Score: ${score} / ${quizData.length}</div>`;
    resultHtml += `<div class="quiz-next-set"><button id="next-quiz-btn">Nouveau quiz</button></div>`;
    quizResult.innerHTML = resultHtml;
    quizResult.style.display = 'block';
    quizContainer.style.display = 'none';
    document.getElementById('next-quiz-btn').onclick = async () => {
        await generateQuizForSelectedPdf();
    };
}

async function generateQuizForSelectedPdf() {
    const selectedPdf = pdfSelect.value;
    if (!selectedPdf) {
        quizContainer.innerHTML = '<div class="quiz-error">Veuillez choisir un document PDF.</div>';
        return;
    }
    quizContainer.innerHTML = 'Génération du quiz...';
    setButtonsDisabled(true);
    try {
        const response = await fetch(`http://localhost:8000/generate_quiz?filename=${encodeURIComponent(selectedPdf)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        if (data.quiz && Array.isArray(data.quiz) && data.quiz.length > 0) {
            quizData = data.quiz;
            userAnswers = [];
            renderAllQuestions();
            quizResult.style.display = 'none';
        } else {
            quizContainer.innerHTML = `<div class="quiz-error">Erreur: ${data.error || 'Impossible de générer le quiz.'}</div>`;
        }
    } catch (err) {
        quizContainer.innerHTML = '<div class="quiz-error">Erreur lors de la génération du quiz.</div>';
    }
    setButtonsDisabled(false);
}

startQuizBtn.addEventListener('click', generateQuizForSelectedPdf);
refreshQuizBtn.addEventListener('click', generateQuizForSelectedPdf);

refreshPdfListBtn.addEventListener('click', async () => {
    refreshPdfListBtn.disabled = true;
    refreshPdfListBtn.textContent = 'Rafraîchissement...';
    const previousValue = pdfSelect.value;
    await fetch('http://localhost:8000/refresh_pdf_list', { method: 'POST' });
    await populatePdfDropdown(previousValue);
    refreshPdfListBtn.disabled = false;
    refreshPdfListBtn.textContent = 'Rafraîchir la liste';
    if (!pdfSelect.value) {
        quizContainer.innerHTML = '<div class="quiz-error">Aucun document PDF trouvé dans le dossier data/. Ajoutez des fichiers PDF puis rafraîchissez la page.</div>';
    } else {
        quizContainer.innerHTML = '';
    }
});

pdfSelect.addEventListener('change', fetchQuizForSelectedPdf);

window.addEventListener('DOMContentLoaded', async () => {
    await populatePdfDropdown();
    quizContainer.innerHTML = '';
    quizResult.innerHTML = '';
    resetQuizUI();
    if (!pdfSelect.value) {
        quizContainer.innerHTML = '<div class="quiz-error">Aucun document PDF trouvé dans le dossier data/. Ajoutez des fichiers PDF puis rafraîchissez la page.</div>';
    }
});

function resetQuizUI() {
    quizContainer.innerHTML = '';
    quizResult.innerHTML = '';
    quizContainer.style.display = 'block';
    quizResult.style.display = 'none';
}