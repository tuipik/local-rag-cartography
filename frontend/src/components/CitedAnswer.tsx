import type { Source } from '../api/ragApi';
import { sourceViewHref } from '../api/ragApi';

interface CitedAnswerProps {
  answer: string;
  sources: Source[];
}

const CITATION_PATTERN = /(\[\d+\])/g;

export function CitedAnswer({ answer, sources }: CitedAnswerProps) {
  const sourcesByCitationId = new Map(
    sources.map((source) => [source.citation_id, source]),
  );
  const parts = answer.split(CITATION_PATTERN);

  return (
    <p className="answer-text">
      {parts.map((part, index) => {
        const citationMatch = part.match(/^\[(\d+)\]$/);
        if (!citationMatch) {
          return <span key={`${part}-${index}`}>{part}</span>;
        }

        const citationId = Number(citationMatch[1]);
        const source = sourcesByCitationId.get(citationId);
        if (!source) {
          return <span key={`${part}-${index}`}>{part}</span>;
        }

        return (
          <a
            className="citation-link"
            href={sourceViewHref(source)}
            key={`${part}-${index}`}
            rel="noreferrer"
            target="_blank"
            title={source.relative_path}
          >
            {part}
          </a>
        );
      })}
    </p>
  );
}
