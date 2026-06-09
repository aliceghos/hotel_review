'use client'

function StarSVG({ filled }: { filled: boolean }) {
  return (
    <svg
      className={filled ? 'star-filled' : 'star-empty'}
      width="1em"
      height="1em"
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M10 1.5l2.39 4.84 5.34.78-3.87 3.77.91 5.32L10 13.97l-4.77 2.51.91-5.32L2.27 7.39l5.34-.78L10 1.5z" />
    </svg>
  )
}

export default function StarRating({ star, size = 'md' }: { star: number; size?: 'sm' | 'md' | 'lg' }) {
  const sizeClass = size === 'sm' ? 'text-sm' : size === 'lg' ? 'text-2xl' : 'text-lg'
  return (
    <span className={`inline-flex gap-0.5 ${sizeClass}`} aria-label={`${star}星评分`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <StarSVG key={i} filled={i <= star} />
      ))}
    </span>
  )
}
