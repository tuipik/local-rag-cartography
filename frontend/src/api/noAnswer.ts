const NO_ANSWER_MARKERS = [
  'інформацію не знайдено',
  'інформація не знайдена',
  'у наданих документах немає',
  'у наданих документах не знайдено',
  'немає інформації',
];

export function isNoAnswer(answer: string): boolean {
  const normalized = answer.toLowerCase();
  return NO_ANSWER_MARKERS.some((marker) => normalized.includes(marker));
}
