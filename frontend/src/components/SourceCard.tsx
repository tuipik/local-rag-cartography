import type { Source } from '../api/ragApi';
import { sourceDownloadHref, sourceViewHref } from '../api/ragApi';

interface SourceCardProps {
  source: Source;
}

export function SourceCard({ source }: SourceCardProps) {
  return (
    <article className="source-card">
      <div className="source-header">
        <span className="source-type">{source.source_type.toUpperCase()}</span>
        <span className="source-citation">[{source.citation_id}]</span>
        <span className="source-location">{source.location}</span>
      </div>
      <h3>{source.relative_path}</h3>
      <p className="source-preview">{source.preview}</p>
      <div className="source-actions">
        <a href={sourceViewHref(source)} rel="noreferrer" target="_blank">
          Відкрити
        </a>
        <a href={sourceDownloadHref(source)} rel="noreferrer">
          Завантажити
        </a>
      </div>
      <p className="source-meta">document_id: {source.document_id}</p>
    </article>
  );
}
