import { useState } from 'react';
import { askQuestion, type AskResponse } from '../api/ragApi';
import { AnswerPanel } from '../components/AnswerPanel';
import { QuestionForm } from '../components/QuestionForm';
import { SourcesPanel } from '../components/SourcesPanel';

export function HomePage() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState<AskResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isLoading) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await askQuestion({
        question: trimmedQuestion,
        top_k: 5,
        llm_model: 'gemma4:e2b',
        num_predict: 1024,
      });
      setResult(response);
    } catch {
      setError('Помилка отримання відповіді.');
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="hero">
        <p className="eyebrow">Local RAG Cartography</p>
        <h1>Поставте питання до локального корпусу документів</h1>
        <p>Інтерфейс використовує FastAPI backend і показує відповідь разом із джерелами.</p>
      </section>

      <QuestionForm
        question={question}
        isLoading={isLoading}
        onQuestionChange={setQuestion}
        onSubmit={handleSubmit}
      />

      {error ? <div className="error-message">{error}</div> : null}

      <AnswerPanel answer={result?.answer ?? null} />
      <SourcesPanel sources={result?.sources ?? []} />
    </main>
  );
}
