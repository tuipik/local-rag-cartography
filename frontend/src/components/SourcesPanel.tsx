import type { Source } from '../api/ragApi';
import { SourceCard } from './SourceCard';

interface SourcesPanelProps {
  sources: Source[];
}

export function SourcesPanel({ sources }: SourcesPanelProps) {
  return (
    <section className="panel">
      <h2>Джерела</h2>
      {sources.length === 0 ? (
        <p className="muted">Джерела зʼявляться після відповіді.</p>
      ) : (
        <div className="sources-list">
          {sources.map((source) => (
            <SourceCard key={`${source.document_id}-${source.location}`} source={source} />
          ))}
        </div>
      )}
    </section>
  );
}
