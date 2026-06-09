'use client'

import type { Filters } from '@/lib/types'

const ROOM_TYPES = ['全部', '套房', '大床房', '双床房', '主题房']
const TRAVEL_TYPES = ['全部', '家庭亲子', '商务出差', '情侣出游', '朋友出游', '独自旅行', '代人预订', '其他']
const CATEGORIES = [
  '全部',
  '房间设施', '公共设施', '餐饮设施',
  '前台服务', '客房服务', '退房/入住效率',
  '交通便利性', '周边配套', '景观/朝向',
  '性价比', '价格合理性',
  '整体满意度', '安静程度', '卫生状况',
]
const SORT_OPTIONS = [
  { value: 'publish_date', label: '发布日期' },
  { value: 'star', label: '评分' },
  { value: 'quality_score', label: '质量分' },
  { value: 'useful_count', label: '有用数' },
]

interface FilterBarProps {
  filters: Filters
  onChange: (filters: Filters) => void
  total: number
}

function SearchIcon() {
  return (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
    </svg>
  )
}

function ChevronDown() {
  return (
    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  )
}

export default function FilterBar({ filters, onChange, total }: FilterBarProps) {
  const update = (patch: Partial<Filters>) => onChange({ ...filters, ...patch })

  const isActive = (value: string | null, target: string) =>
    (target === '全部' && !value) || value === target

  const pillClass = (active: boolean) =>
    active
      ? 'bg-amber-500 text-white shadow-sm'
      : 'bg-stone-100 text-stone-600 hover:bg-stone-200'

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-stone-200/60 p-5 space-y-4">
      {/* Search */}
      <div className="flex gap-3">
        <div className="relative flex-1">
          <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-stone-400">
            <SearchIcon />
          </span>
          <input
            type="text"
            value={filters.keyword}
            onChange={(e) => update({ keyword: e.target.value })}
            placeholder="搜索评论关键词..."
            className="w-full pl-10 pr-4 py-2.5 border border-stone-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-amber-500/40 focus:border-amber-400 text-sm bg-stone-50/50 placeholder:text-stone-400"
          />
        </div>
        <div className="text-sm text-stone-500 flex items-center whitespace-nowrap">
          共 <span className="font-semibold text-stone-800 mx-1">{total}</span> 条
        </div>
      </div>

      {/* Star filter */}
      <FilterRow label="评分">
        <PillButton active={filters.star === null} onClick={() => update({ star: null })}>
          全部
        </PillButton>
        {[5, 4, 3, 2, 1].map((s) => (
          <PillButton
            key={s}
            active={filters.star === s}
            onClick={() => update({ star: filters.star === s ? null : s })}
          >
            {s}星
          </PillButton>
        ))}
      </FilterRow>

      {/* Room type filter */}
      <FilterRow label="房型">
        {ROOM_TYPES.map((type) => (
          <PillButton
            key={type}
            active={isActive(filters.fuzzyRoomType, type)}
            onClick={() => update({ fuzzyRoomType: type === '全部' ? '' : type })}
          >
            {type}
          </PillButton>
        ))}
      </FilterRow>

      {/* Travel type filter */}
      <FilterRow label="出行">
        {TRAVEL_TYPES.map((type) => (
          <PillButton
            key={type}
            active={isActive(filters.travelType, type)}
            onClick={() => update({ travelType: type === '全部' ? '' : type })}
          >
            {type}
          </PillButton>
        ))}
      </FilterRow>

      {/* Category filter */}
      <FilterRow label="分类">
        {CATEGORIES.map((cat) => (
          <PillButton
            key={cat}
            active={isActive(filters.category, cat)}
            onClick={() => update({ category: cat === '全部' ? '' : cat })}
          >
            {cat}
          </PillButton>
        ))}
      </FilterRow>

      {/* Sort */}
      <div className="flex items-center gap-3 pt-2 border-t border-stone-100">
        <span className="text-sm text-stone-500 font-medium min-w-[3rem]">排序:</span>
        <div className="relative">
          <select
            value={filters.sortBy}
            onChange={(e) => update({ sortBy: e.target.value as Filters['sortBy'] })}
            className="appearance-none pl-3.5 pr-8 py-1.5 border border-stone-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40 focus:border-amber-400 bg-stone-50/50 cursor-pointer"
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <span className="absolute right-2.5 top-1/2 -translate-y-1/2 text-stone-400 pointer-events-none">
            <ChevronDown />
          </span>
        </div>
        <button
          onClick={() => update({ sortOrder: filters.sortOrder === 'desc' ? 'asc' : 'desc' })}
          className="inline-flex items-center gap-1 px-3.5 py-1.5 border border-stone-200 rounded-xl text-sm hover:bg-stone-50 cursor-pointer transition-colors text-stone-600"
        >
          {filters.sortOrder === 'desc' ? (
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 15l7-7 7 7" />
            </svg>
          )}
          {filters.sortOrder === 'desc' ? '降序' : '升序'}
        </button>
      </div>
    </div>
  )
}

function FilterRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-sm text-stone-500 font-medium min-w-[3rem]">{label}:</span>
      <div className="flex gap-1.5 flex-wrap">
        {children}
      </div>
    </div>
  )
}

function PillButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 rounded-full text-sm cursor-pointer transition-all duration-200 ${
        active
          ? 'bg-amber-500 text-white shadow-sm'
          : 'bg-stone-100 text-stone-600 hover:bg-stone-200'
      }`}
    >
      {children}
    </button>
  )
}
