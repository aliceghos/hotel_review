'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import type { Comment, Filters } from '@/lib/types'
import CommentCard from '@/components/CommentCard'
import FilterBar from '@/components/FilterBar'
import ChatWidget from '@/components/ChatWidget'

const PAGE_SIZE = 20

function StatsIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
  )
}

function StarIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
    </svg>
  )
}

function ImageIcon() {
  return (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  )
}

function LoadingSpinner() {
  return (
    <svg className="animate-spin w-8 h-8 text-amber-500" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3.5" />
      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

export default function HomePage() {
  const [comments, setComments] = useState<Comment[]>([])
  const [loading, setLoading] = useState(true)
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [filters, setFilters] = useState<Filters>({
    keyword: '',
    star: null,
    fuzzyRoomType: '',
    travelType: '',
    category: '',
    sortBy: 'publish_date',
    sortOrder: 'desc',
  })

  const [stats, setStats] = useState({ total: 0, avgStar: 0, withImages: 0 })

  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [debouncedKeyword, setDebouncedKeyword] = useState('')

  useEffect(() => {
    if (searchTimer.current) clearTimeout(searchTimer.current)
    searchTimer.current = setTimeout(() => {
      setDebouncedKeyword(filters.keyword)
    }, 400)
    return () => { if (searchTimer.current) clearTimeout(searchTimer.current) }
  }, [filters.keyword])

  // Fetch stats
  useEffect(() => {
    async function fetchStats() {
      try {
        const url = new URL('/api/comments', window.location.origin)
        url.searchParams.set('type', 'stats')
        const res = await fetch(url.toString())
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const result = await res.json()
        setStats(result.stats)
      } catch (err) {
        console.error('Failed to fetch stats:', err)
      }
    }
    fetchStats()
  }, [])

  // Fetch comments
  const fetchComments = useCallback(async (pageNum: number, append: boolean) => {
    setLoading(true)
    try {
      const from = (pageNum - 1) * PAGE_SIZE
      const to = from + PAGE_SIZE - 1

      const url = new URL('/api/comments', window.location.origin)
      url.searchParams.set('from', String(from))
      url.searchParams.set('to', String(to))
      url.searchParams.set('count', 'exact')
      url.searchParams.set('sortBy', filters.sortBy)
      url.searchParams.set('sortOrder', filters.sortOrder)

      if (filters.star !== null) {
        url.searchParams.set('star', String(filters.star))
      }
      if (filters.fuzzyRoomType) {
        url.searchParams.set('fuzzyRoomType', filters.fuzzyRoomType)
      }
      if (filters.travelType) {
        url.searchParams.set('travelType', filters.travelType)
      }
      if (filters.category) {
        url.searchParams.set('category', filters.category)
      }
      if (debouncedKeyword) {
        url.searchParams.set('keyword', debouncedKeyword)
      }

      const res = await fetch(url.toString())
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const { data, count, error } = await res.json()

      if (error) {
        console.error('Query error:', error)
        return
      }

      const results = (data as Comment[]) || []
      if (append) {
        setComments(prev => [...prev, ...results])
      } else {
        setComments(results)
      }
      setTotal(count || 0)
      setHasMore(results.length === PAGE_SIZE)
    } catch (err) {
      console.error('Fetch error:', err)
    } finally {
      setLoading(false)
    }
  }, [filters.star, filters.fuzzyRoomType, filters.travelType, filters.category, filters.sortBy, filters.sortOrder, debouncedKeyword])

  // Reset page when filters change
  useEffect(() => {
    setPage(1)
    fetchComments(1, false)
  }, [fetchComments])

  const loadMore = () => {
    const nextPage = page + 1
    setPage(nextPage)
    fetchComments(nextPage, true)
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: '#faf8f5' }}>
      {/* Header */}
      <header style={{ background: 'linear-gradient(135deg, #2c2416 0%, #3d3220 50%, #4d4236 100%)' }}>
        <div className="max-w-6xl mx-auto px-4 py-10">
          <div className="flex items-center gap-3 mb-2">
            {/* Hotel icon */}
            <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'linear-gradient(135deg, #c8a45c 0%, #b8942e 100%)' }}>
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-white tracking-tight">花园酒店 住客评论</h1>
              <p className="text-amber-200/70 text-sm mt-0.5">广州花园酒店住客真实评价浏览系统</p>
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-4 mt-6">
            <div className="relative overflow-hidden rounded-2xl p-4 text-center"
              style={{ background: 'rgba(255,255,255,0.08)', backdropFilter: 'blur(12px)' }}>
              <div className="absolute top-0 left-0 w-full h-0.5 bg-amber-400/60" />
              <div className="flex items-center justify-center gap-2 mb-1 text-amber-300/80">
                <StatsIcon />
              </div>
              <div className="text-2xl font-bold text-white">{stats.total.toLocaleString()}</div>
              <div className="text-xs text-amber-200/60 mt-0.5">总评论数</div>
            </div>
            <div className="relative overflow-hidden rounded-2xl p-4 text-center"
              style={{ background: 'rgba(255,255,255,0.08)', backdropFilter: 'blur(12px)' }}>
              <div className="absolute top-0 left-0 w-full h-0.5 bg-amber-400/60" />
              <div className="flex items-center justify-center gap-2 mb-1 text-amber-300/80">
                <StarIcon />
              </div>
              <div className="text-2xl font-bold text-white">{stats.avgStar.toFixed(1)}</div>
              <div className="text-xs text-amber-200/60 mt-0.5">平均评分</div>
            </div>
            <div className="relative overflow-hidden rounded-2xl p-4 text-center"
              style={{ background: 'rgba(255,255,255,0.08)', backdropFilter: 'blur(12px)' }}>
              <div className="absolute top-0 left-0 w-full h-0.5 bg-amber-400/60" />
              <div className="flex items-center justify-center gap-2 mb-1 text-amber-300/80">
                <ImageIcon />
              </div>
              <div className="text-2xl font-bold text-white">{stats.withImages.toLocaleString()}</div>
              <div className="text-xs text-amber-200/60 mt-0.5">含图评论</div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-6 space-y-5 flex-1">
        <FilterBar filters={filters} onChange={setFilters} total={total} />

        {loading && comments.length === 0 ? (
          <div className="flex items-center justify-center py-24">
            <div className="text-center">
              <div className="mb-4 flex justify-center">
                <LoadingSpinner />
              </div>
              <p className="text-stone-500">加载评论中...</p>
            </div>
          </div>
        ) : !loading && comments.length === 0 && total === 0 ? (
          <div className="text-center py-24">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-stone-100 text-stone-400 mb-4">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
              </svg>
            </div>
            <p className="text-stone-500 text-lg">评论数据暂时不可用</p>
            <p className="text-stone-400 text-sm mt-2">请使用右下角的 AI 问答助手查询酒店信息</p>
          </div>
        ) : comments.length === 0 ? (
          <div className="text-center py-24">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-stone-100 text-stone-400 mb-4">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
              </svg>
            </div>
            <p className="text-stone-500 text-lg">没有找到匹配的评论</p>
            <p className="text-stone-400 text-sm mt-2">试试调整筛选条件</p>
          </div>
        ) : (
          <div className="space-y-4">
            {comments.map((comment) => (
              <CommentCard
                key={comment.id}
                comment={comment}
                keyword={debouncedKeyword || undefined}
              />
            ))}

            {hasMore && (
              <div className="text-center py-4">
                <button
                  onClick={loadMore}
                  disabled={loading}
                  className="px-8 py-2.5 bg-amber-500 text-white rounded-xl hover:bg-amber-600 disabled:opacity-50 cursor-pointer transition-all text-sm font-medium shadow-sm hover:shadow-md"
                >
                  {loading ? '加载中...' : '加载更多'}
                </button>
              </div>
            )}

            {!hasMore && comments.length > 0 && (
              <p className="text-center text-stone-400 text-sm py-4">
                已显示全部 {total} 条评论
              </p>
            )}
          </div>
        )}
      </main>

      <footer className="mt-auto py-6 text-center text-sm text-stone-400 border-t border-stone-200/60">
        花园酒店住客评论浏览系统  |  DSBA 大模型应用
      </footer>

      <ChatWidget />
    </div>
  )
}
