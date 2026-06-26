import type { Source } from '../api/ragApi';
import { CitedAnswer } from './CitedAnswer';

interface AnswerPanelProps {
  answer: string | null;
  sources: Source[];
}

export function AnswerPanel({ answer, sources }: AnswerPanelProps) {
  return (
    <section className="panel">
      <h2>Відповідь</h2>
      {answer ? (
        <CitedAnswer answer={answer} sources={sources} />
      ) : (
        <p className="muted">Поставте питання, щоб отримати відповідь.</p>
      )}
    </section>
  );
}
