'use client'

import { useState } from 'react'

function CloseIcon() {
  return (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

export default function ImageLightbox({ src, onClose }: { src: string; onClose: () => void }) {
  return (
    <div
      className="lightbox-overlay fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div className="relative max-w-5xl max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
        <button
          onClick={onClose}
          className="absolute -top-12 right-0 text-white/80 hover:text-white transition-colors cursor-pointer"
          aria-label="关闭"
        >
          <CloseIcon />
        </button>
        <img
          src={src}
          alt="评论图片"
          className="max-w-full max-h-[85vh] rounded-2xl object-contain shadow-2xl"
        />
      </div>
    </div>
  )
}

export function ImageThumbnail({ src, onClick }: { src: string; onClick: () => void }) {
  const [error, setError] = useState(false)

  const displaySrc = src.replace('_R_150_150', '_R_300_300')

  if (error) return null

  return (
    <button onClick={onClick} className="flex-shrink-0 cursor-pointer group">
      <img
        src={displaySrc}
        alt="评论图片"
        className="w-20 h-20 object-cover rounded-xl border border-stone-200 group-hover:border-amber-300 group-hover:shadow-md transition-all duration-200"
        onError={() => setError(true)}
        loading="lazy"
      />
    </button>
  )
}
