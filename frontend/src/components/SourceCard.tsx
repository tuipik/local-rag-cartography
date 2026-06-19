import type { Source } from '../api/ragApi';

interface SourceCardProps {
  source: Source;
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <article className="source-card">
      <div className="source-header">
        <span className="source-type">{source.source_type.toUpperCase()}</span>
        <span className="source-location">{source.location}</span>
      </div>
      <h3>{source.relative_path}</h3>
      <p className="source-preview">{source.preview}</p>
      <p className="source-meta">document_id: {source.document_id}</p>
    </article>
  );
}
