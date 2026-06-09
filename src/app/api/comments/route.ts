import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// ── Types ──────────────────────────────────────────────
interface Comment {
  id: string
  _id: string
  comment: string
  images: string[]
  score: number
  publish_date: string
  room_type: string
  fuzzy_room_type: string
  travel_type: string
  comment_len: number
  useful_count: number
  review_count: number
  quality_score: number
  categories: string[]
  category1: string
  category2: string
  category3: string
  star: number
}

// ── CSV Parser ─────────────────────────────────────────
function parseCSV(text: string): Record<string, string>[] {
  const rows: Record<string, string>[] = []
  const lines = text.split('\n')
  if (lines.length === 0) return rows

  const headers = parseCSVLine(lines[0])
  let current: string[] = []
  let field = ''
  let inQuotes = false

  // Re-join with \n and parse as a stream
  const fullText = text
  let i = 0
  // Skip header
  while (i < fullText.length && fullText[i] !== '\n') i++
  i++ // past newline

  while (i < fullText.length) {
    const ch = fullText[i]

    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < fullText.length && fullText[i + 1] === '"') {
          field += '"'
          i += 2
          continue
        }
        inQuotes = false
        i++
        continue
      }
      field += ch
      i++
      continue
    }

    if (ch === '"') {
      inQuotes = true
      i++
      continue
    }

    if (ch === ',') {
      current.push(field)
      field = ''
      i++
      continue
    }

    if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && i + 1 < fullText.length && fullText[i + 1] === '\n') {
        i++
      }
      current.push(field)
      field = ''
      // Build row
      if (current.length >= headers.length) {
        const row: Record<string, string> = {}
        for (let h = 0; h < headers.length; h++) {
          row[headers[h]] = current[h] || ''
        }
        rows.push(row)
      }
      current = []
      i++
      continue
    }

    field += ch
    i++
  }

  // Last field if no trailing newline
  if (field || current.length > 0) {
    current.push(field)
    if (current.length >= headers.length) {
      const row: Record<string, string> = {}
      for (let h = 0; h < headers.length; h++) {
        row[headers[h]] = current[h] || ''
      }
      rows.push(row)
    }
  }

  return rows
}

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let field = ''
  let inQuotes = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          field += '"'
          i++
          continue
        }
        inQuotes = false
        continue
      }
      field += ch
      continue
    }
    if (ch === '"') { inQuotes = true; continue }
    if (ch === ',') { result.push(field); field = ''; continue }
    field += ch
  }
  result.push(field)
  return result
}

// ── Field parsing ──────────────────────────────────────
function parsePythonList(s: string): string[] {
  if (!s || s === '[]') return []
  // Handle Python-style list: ['item1', 'item2']
  const cleaned = s
    .replace(/^\[/, '')
    .replace(/\]$/, '')
  if (!cleaned.trim()) return []
  // Split by ', ' but handle quoted items
  const items: string[] = []
  let item = ''
  let inQ = false
  for (let i = 0; i < cleaned.length; i++) {
    const ch = cleaned[i]
    if (ch === "'" || ch === '"') {
      inQ = !inQ
      continue
    }
    if (ch === ',' && !inQ) {
      items.push(item.trim())
      item = ''
      continue
    }
    item += ch
  }
  if (item.trim()) items.push(item.trim())
  return items.filter(Boolean)
}

function parseImages(s: string): string[] {
  if (!s || s === '[]' || s === '[""]') return []
  try {
    // Fix double-double-quotes from CSV escaping
    const fixed = s.replace(/""/g, '"')
    const parsed = JSON.parse(fixed)
    return Array.isArray(parsed) ? parsed.filter((u: unknown) => typeof u === 'string' && u.length > 0) : []
  } catch {
    return []
  }
}

function safeFloat(s: string): number {
  const v = parseFloat(s)
  return isNaN(v) ? 0 : v
}

function safeInt(s: string): number {
  const v = parseInt(s, 10)
  return isNaN(v) ? 0 : v
}

// ── Data loading (cached) ──────────────────────────────
let _comments: Comment[] | null = null

function loadComments(): Comment[] {
  if (_comments) return _comments

  const csvPath = path.join(process.cwd(), 'public', 'enriched_comments.csv')
  const text = fs.readFileSync(csvPath, 'utf-8')
  const rows = parseCSV(text)

  _comments = rows.map((row, idx) => {
    const categories = parsePythonList(row.categories || '')
    const images = parseImages(row.images || '')
    const score = safeFloat(row.score)

    return {
      id: row._id || String(idx),
      _id: row._id || '',
      comment: row.comment || '',
      images,
      score,
      publish_date: row.publish_date || '',
      room_type: row.room_type || '',
      fuzzy_room_type: row.fuzzy_room_type || '',
      travel_type: row.travel_type || '',
      comment_len: safeInt(row.comment_len),
      useful_count: safeInt(row.useful_count),
      review_count: safeInt(row.review_count),
      quality_score: safeFloat(row.quality_score),
      categories,
      category1: categories[0] || '',
      category2: categories[1] || '',
      category3: categories[2] || '',
      star: Math.round(score),
    }
  })

  return _comments
}

// ── API Handler ────────────────────────────────────────
export async function GET(request: NextRequest) {
  const { searchParams } = request.nextUrl

  // ── Stats endpoint ──
  if (searchParams.get('type') === 'stats') {
    const all = loadComments()
    const uniqueById = new Map<string, Comment>()
    for (const c of all) uniqueById.set(c._id, c)
    const unique = [...uniqueById.values()]
    const stars = unique.map(c => c.score).filter(s => s > 0)
    const avgStar = stars.length > 0
      ? stars.reduce((sum, s) => sum + s, 0) / stars.length
      : 0
    const withImages = unique.filter(c => c.images.length > 0).length
    return NextResponse.json({
      stats: { total: unique.length, avgStar, withImages },
    })
  }

  const select = searchParams.get('select') || '*'
  const countParam = searchParams.get('count')
  const starFilter = searchParams.get('star')
  const fuzzyRoomType = searchParams.get('fuzzyRoomType') || searchParams.get('fuzzy_room_type')
  const travelType = searchParams.get('travelType') || searchParams.get('travel_type')
  const category = searchParams.get('category')
  const keyword = searchParams.get('keyword')
  const sortBy = searchParams.get('sortBy') || searchParams.get('sort_by') || 'publish_date'
  const sortOrder = searchParams.get('sortOrder') || searchParams.get('sort_order') || 'desc'
  const from = parseInt(searchParams.get('from') || '0', 10)
  const to = parseInt(searchParams.get('to') || '19', 10)

  let comments = loadComments()

  // ── Filters ──
  if (starFilter !== null && starFilter !== '') {
    const star = parseInt(starFilter, 10)
    if (!isNaN(star)) {
      comments = comments.filter(c => c.star === star)
    }
  }

  if (fuzzyRoomType) {
    comments = comments.filter(c => c.fuzzy_room_type === fuzzyRoomType)
  }

  if (travelType) {
    comments = comments.filter(c => c.travel_type === travelType)
  }

  if (category) {
    comments = comments.filter(c =>
      c.category1 === category || c.category2 === category || c.category3 === category
    )
  }

  if (keyword) {
    const kw = keyword.replace(/%/g, '').toLowerCase()
    comments = comments.filter(c => c.comment.toLowerCase().includes(kw))
  }

  // Deduplicate by _id after filtering
  const seen = new Set<string>()
  comments = comments.filter(c => {
    if (seen.has(c._id)) return false
    seen.add(c._id)
    return true
  })

  const total = comments.length

  // ── Sort ──
  const sortField = sortBy === 'star' ? 'score' : sortBy
  comments.sort((a, b) => {
    const aVal = (a as any)[sortField] ?? ''
    const bVal = (b as any)[sortField] ?? ''
    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal
    }
    const cmp = String(aVal).localeCompare(String(bVal))
    return sortOrder === 'asc' ? cmp : -cmp
  })

  // ── Paginate ──
  const page = comments.slice(from, to + 1)

  // ── Field selection ──
  let data: any[]
  if (select === '*' || !select) {
    data = page
  } else {
    const fields = select.split(',').map(f => f.trim())
    data = page.map(c => {
      const obj: any = {}
      for (const f of fields) {
        obj[f] = (c as any)[f]
      }
      return obj
    })
  }

  return NextResponse.json({
    data,
    count: countParam === 'exact' ? total : undefined,
    error: null,
  })
}
