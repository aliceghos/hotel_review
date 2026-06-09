"""Insforge 数据库连接工具（支持本地 CSV 回退）"""

import os
from pathlib import Path
import pandas as pd
import requests


def _load_from_local_csv() -> pd.DataFrame:
    """从本地 CSV 加载评论数据（Insforge 不可用时的回退方案）"""
    data_dir = Path(__file__).parent.parent / "data"
    csv_path = data_dir / "filtered_comments.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
        print(f"✅ 从本地 CSV 加载 {len(df)} 条评论数据")
        return df
    raise FileNotFoundError(f"本地数据文件不存在: {csv_path}")


def get_all_comments_from_insforge() -> pd.DataFrame:
    """从 Insforge 数据库获取所有评论数据

    如果 Insforge 配置不可用或连接失败，自动回退到本地 CSV。

    返回:
        pandas.DataFrame: 评论数据，以 _id 为索引
    """
    base_url = os.getenv("NEXT_PUBLIC_INSFORGE_BASE_URL")
    anon_key = os.getenv("NEXT_PUBLIC_INSFORGE_ANON_KEY")

    if not base_url or not anon_key:
        print("⚠️ 缺少 Insforge 配置，回退到本地 CSV")
        return _load_from_local_csv()

    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Content-Type": "application/json"
    }

    all_data = []
    batch_size = 1000
    offset = 0

    print("正在从 Insforge 数据库获取评论数据...")

    try:
        while True:
            url = f"{base_url}/api/database/records/comments?select=*"
            range_headers = {
                **headers,
                "Range-Unit": "items",
                "Range": f"{offset}-{offset + batch_size - 1}",
                "Prefer": "count=exact"
            }
            response = requests.get(url, headers=range_headers, timeout=30)

            if response.status_code not in (200, 206):
                raise RuntimeError(f"Insforge API 调用失败: {response.status_code}")

            data = response.json()
            if not data:
                break

            all_data.extend(data)
            print(f"  已获取 {len(all_data)} 条评论...")

            if len(data) < batch_size:
                break
            offset += batch_size
    except Exception as e:
        print(f"⚠️ Insforge 连接失败 ({e})，回退到本地 CSV")
        return _load_from_local_csv()

    df = pd.DataFrame(all_data)

    if '_id' in df.columns:
        df.set_index('_id', inplace=True)

    print(f"✅ 成功从 Insforge 加载 {len(df)} 条评论数据")
    return df
