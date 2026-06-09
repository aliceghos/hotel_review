'use client'

import { useState } from 'react'
import type { Comment } from '@/lib/types'
import StarRating from './StarRating'
import ImageLightbox, { ImageThumbnail } from './ImageLightbox'

const CATEGORY_PARENT: Record<string, string> = {
  '房间设施': '设施类', '公共设施': '设施类', '餐饮设施': '设施类',
  '前台服务': '服务类', '客房服务': '服务类', '退房/入住效率': '服务类',
  '交通便利性': '位置类', '周边配套': '位置类', '景观/朝向': '位置类',
  '性价比': '价格类', '价格合理性': '价格类',
  '整体满意度': '体验类', '安静程度': '体验类', '卫生状况': '体验类',
}

const PARENT_COLORS: Record<string, string> = {
  '设施类': 'bg-amber-50 text-amber-700',
  '服务类': 'bg-emerald-50 text-emerald-700',
  '位置类': 'bg-indigo-50 text-indigo-700',
  '价格类': 'bg-orange-50 text-orange-700',
  '体验类': 'bg-rose-50 text-rose-700',
}

function CategoryTag({ name }: { name: string }) {
  const parent = CATEGORY_PARENT[name] || '其他'
  const color = PARENT_COLORS[parent] || 'bg-stone-100 text-stone-600'
  return (
    <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium ${color}`}>
      {name}
    </span>
  )
}

const TRAVEL_ICONS: Record<string, string> = {
  '家庭亲子': 'M17 20h-2v-3a1 1 0 00-1-1h-4a1 1 0 00-1 1v3H7v-4a5 5 0 0110 0v4zM12 3a3 3 0 100 6 3 3 0 000-6z',
  '商务出差': 'M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z',
  '情侣出游': 'M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z',
  '朋友出游': 'M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0z',
  '独自出行': 'M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16zM12 8v4m0 4h.01',
}

function TravelTypeBadge({ type }: { type: string }) {
  const iconPath = TRAVEL_ICONS[type]
  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-stone-100 text-stone-600 rounded-full text-xs font-medium">
      {iconPath && (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d={iconPath} />
        </svg>
      )}
      {type}
    </span>
  )
}

function UsefulIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
    </svg>
  )
}

function CommentIcon() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
    </svg>
  )
}

export default function CommentCard({ comment, keyword }: { comment: Comment; keyword?: string }) {
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  const images = Array.isArray(comment.images) ? comment.images : []
  const categories = [comment.category1, comment.category2, comment.category3].filter(Boolean)
  const isLong = comment.comment.length > 200

  const getFullSrc = (src: string) =>
    src.replace(/_R_\d+_\d+/, '_R_800_600').replace('_R5_Q70_D', '_R5_Q90_D')

  function highlightText(text: string) {
    if (!keyword) return text
    const parts = text.split(new RegExp(`(${keyword.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
    return parts.map((part, i) =>
      part.toLowerCase() === keyword.toLowerCase()
        ? <mark key={i} className="bg-amber-100 text-amber-900 px-0.5 rounded">{part}</mark>
        : part
    )
  }

  const displayText = isLong && !expanded
    ? comment.comment.substring(0, 200) + '...'
    : comment.comment

  return (
    <>
      <div className="card-hover bg-white rounded-2xl shadow-sm border border-stone-200/60 p-5">
        {/* Header */}
        <div className="flex items-start justify-between gap-3 mb-3">
          <div className="flex items-center gap-3 flex-wrap">
            <StarRating star={comment.star} />
            <span className="text-sm text-stone-400">
              {comment.publish_date}
            </span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {comment.travel_type && <TravelTypeBadge type={comment.travel_type} />}
          </div>
        </div>

        {/* Room type */}
        <div className="mb-2">
          <span className="text-sm text-stone-400">房型：</span>
          <span className="text-sm font-medium text-stone-700">{comment.room_type}</span>
        </div>

        {/* Comment text */}
        <div className="mb-3 text-stone-700 leading-relaxed whitespace-pre-line">
          {highlightText(displayText)}
          {isLong && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="ml-1 text-amber-600 hover:text-amber-800 text-sm cursor-pointer font-medium"
            >
              {expanded ? '收起' : '展开全部'}
            </button>
          )}
        </div>

        {/* Images */}
        {images.length > 0 && (
          <div className="flex gap-2 mb-3 overflow-x-auto pb-1">
            {images.map((img, i) => (
              <ImageThumbnail
                key={i}
                src={img}
                onClick={() => setLightboxSrc(getFullSrc(img))}
              />
            ))}
          </div>
        )}

        {/* Categories & stats */}
        <div className="flex items-center justify-between flex-wrap gap-2 pt-3 border-t border-stone-100">
          <div className="flex gap-1.5 flex-wrap">
            {categories.map((cat) => (
              <CategoryTag key={cat} name={cat} />
            ))}
          </div>
          <div className="flex items-center gap-3 text-xs text-stone-400">
            {comment.useful_count > 0 && (
              <span className="inline-flex items-center gap-1">
                <UsefulIcon />{comment.useful_count}
              </span>
            )}
            {comment.review_count > 0 && (
              <span className="inline-flex items-center gap-1">
                <CommentIcon />{comment.review_count}
              </span>
            )}
            <span>质量分 {comment.quality_score}</span>
          </div>
        </div>
      </div>

      {lightboxSrc && (
        <ImageLightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
    </>
  )
}
