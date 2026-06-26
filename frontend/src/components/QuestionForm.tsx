interface QuestionFormProps {
  question: string;
  isLoading: boolean;
  onQuestionChange: (question: string) => void;
  onSubmit: () => void;
}

export function QuestionForm({
  question,
  isLoading,
  onQuestionChange,
  onSubmit,
}: QuestionFormProps) {
  return (
    <form
      className="question-form"
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
    >
      <label htmlFor="question">Питання</label>
      <textarea
        id="question"
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="Введіть питання..."
        rows={4}
        disabled={isLoading}
      />
      <button type="submit" disabled={isLoading || !question.trim()}>
        {isLoading ? 'Виконується...' : 'Запитати'}
      </button>
    </form>
  );
}
