export type StandardCategory =
  | '房间设施' | '公共设施' | '餐饮设施'
  | '前台服务' | '客房服务' | '退房/入住效率'
  | '交通便利性' | '周边配套' | '景观/朝向'
  | '性价比' | '价格合理性'
  | '整体满意度' | '安静程度' | '卫生状况';

export interface Comment {
  id?: string
  _id: string
  comment: string
  images: string[]
  score: number
  publish_date: string
  room_type: string
  fuzzy_room_type: string
  travel_type: string
  comment_len?: number
  useful_count: number
  review_count: number
  quality_score: number
  categories?: StandardCategory[]
  category1: StandardCategory | null
  category2: StandardCategory | null
  category3: StandardCategory | null
  star: number
}

export interface Filters {
  keyword: string
  star: number | null
  fuzzyRoomType: string
  travelType: string
  category: string
  sortBy: 'publish_date' | 'star' | 'quality_score' | 'useful_count'
  sortOrder: 'asc' | 'desc'
}
