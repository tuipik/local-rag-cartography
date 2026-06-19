interface AnswerPanelProps {
  answer: string | null;
}

export function AnswerPanel({ answer }: AnswerPanelProps) {
  return (
    <section className="panel">
      <h2>Відповідь</h2>
      {answer ? <p className="answer-text">{answer}</p> : <p className="muted">Поставте питання, щоб отримати відповідь.</p>}
    </section>
  );
}
