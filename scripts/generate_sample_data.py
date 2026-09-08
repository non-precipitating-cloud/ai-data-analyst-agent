"""生成示例数据集（sales / ecommerce / financial）。

用法：
    python scripts/generate_sample_data.py

生成确定性数据（固定随机种子），方便复现 Demo。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"

RNG = np.random.RandomState(42)

REGIONS = ["华东", "华南", "华北", "西南", "华中", "东北"]

CATEGORIES = {
    "电子产品": {"products": ["智能手机", "笔记本电脑", "蓝牙耳机"], "margin": 0.15},
    "家居用品": {"products": ["智能台灯", "空气净化器", "电饭煲"], "margin": 0.25},
    "服装": {"products": ["男士夹克", "女士连衣裙", "运动鞋"], "margin": 0.35},
    "食品饮料": {"products": ["坚果礼盒", "咖啡豆", "气泡水"], "margin": 0.20},
    "美妆个护": {"products": ["护肤套装", "口红", "洗发水"], "margin": 0.40},
}

PRODUCT_BASE = {
    "智能手机": 900_000, "笔记本电脑": 1_100_000, "蓝牙耳机": 320_000,
    "智能台灯": 150_000, "空气净化器": 420_000, "电饭煲": 180_000,
    "男士夹克": 260_000, "女士连衣裙": 220_000, "运动鞋": 300_000,
    "坚果礼盒": 170_000, "咖啡豆": 130_000, "气泡水": 110_000,
    "护肤套装": 350_000, "口红": 140_000, "洗发水": 90_000,
}

PRODUCT_PRICE = {
    "智能手机": 4500, "笔记本电脑": 6800, "蓝牙耳机": 800,
    "智能台灯": 300, "空气净化器": 1200, "电饭煲": 500,
    "男士夹克": 700, "女士连衣裙": 550, "运动鞋": 620,
    "坚果礼盒": 200, "咖啡豆": 150, "气泡水": 90,
    "护肤套装": 900, "口红": 260, "洗发水": 120,
}

REGION_FACTOR = {"华东": 1.35, "华南": 1.15, "华北": 1.00, "西南": 0.85, "华中": 0.80, "东北": 0.60}

# 季节相位：不同品类在不同月份有高峰
SEASON_PHASE = {
    "电子产品": 2.0, "家居用品": 4.0, "服装": 5.0, "食品饮料": 6.0, "美妆个护": 1.0,
}


def _segment(category: str) -> str:
    mapping = {
        "电子产品": "企业客户", "家居用品": "家庭", "服装": "个人",
        "食品饮料": "家庭", "美妆个护": "个人",
    }
    return mapping[category]


def generate_sales() -> pd.DataFrame:
    """月度 × 地区 × 产品 的销售数据，含 2025 年下滑故事、异常值、销售-利润相关。"""
    rows = []
    for month_idx in range(24):
        year = 2024 if month_idx < 12 else 2025
        month = month_idx % 12 + 1
        date = f"{year}-{month:02d}-01"

        for region in REGIONS:
            for category, meta in CATEGORIES.items():
                for product in meta["products"]:
                    base = PRODUCT_BASE[product] * REGION_FACTOR[region]

                    # 季节性
                    season = 1 + 0.25 * math.sin(
                        2 * math.pi * (month - 1) / 12 + SEASON_PHASE[category]
                    )

                    # 年度趋势：2025 整体下滑 ~12%，华东再降 ~20%，服装再降 ~25%
                    trend = 1.0
                    if year == 2025:
                        trend = 0.88
                        if region == "华东":
                            trend *= 0.80
                        if category == "服装":
                            trend *= 0.75

                    noise = 1 + RNG.normal(0, 0.08)
                    sales = base * season * trend * noise

                    # 折扣：2025 服装加大促销
                    discount = float(RNG.uniform(0.0, 0.30))
                    if year == 2025 and category == "服装":
                        discount *= 1.5
                    discount = min(discount, 0.60)

                    margin = meta["margin"]
                    profit = sales * margin * (1 - 0.8 * discount) * (1 + RNG.normal(0, 0.05))

                    unit_price = PRODUCT_PRICE[product] * (1 + RNG.normal(0, 0.03))
                    quantity = max(1, int(round(sales / unit_price)))

                    rows.append({
                        "date": date,
                        "year": year,
                        "month": month,
                        "region": region,
                        "category": category,
                        "product": product,
                        "sales": round(sales, 2),
                        "quantity": quantity,
                        "profit": round(profit, 2),
                        "discount": round(discount, 4),
                        "unit_price": round(unit_price, 2),
                        "customer_segment": _segment(category),
                    })

    df = pd.DataFrame(rows)

    # 注入异常值：促销爆量（销售极高）+ 高折扣导致负利润
    n_outliers = 12
    outlier_idx = RNG.choice(len(df), size=n_outliers, replace=False)
    df.loc[outlier_idx, "sales"] *= RNG.uniform(3.0, 6.0, size=n_outliers)
    df.loc[outlier_idx, "quantity"] = (
        df.loc[outlier_idx, "sales"] / df.loc[outlier_idx, "unit_price"]
    ).round().astype(int)

    n_neg = 6
    neg_idx = RNG.choice(len(df), size=n_neg, replace=False)
    df.loc[neg_idx, "discount"] = 0.85
    df.loc[neg_idx, "profit"] = df.loc[neg_idx, "sales"] * 0.10 * (1 - 0.8 * 0.85)

    return df


def generate_ecommerce() -> pd.DataFrame:
    """电商订单数据：含退换货、评分、支付方式。"""
    n = 4000
    start = pd.Timestamp("2024-01-01")
    days = RNG.randint(0, 730, size=n)
    dates = start + pd.to_timedelta(days, unit="D")

    regions = RNG.choice(REGIONS, size=n, p=[0.30, 0.20, 0.18, 0.14, 0.10, 0.08])
    categories = list(CATEGORIES.keys())
    cat = RNG.choice(categories, size=n, p=[0.28, 0.20, 0.22, 0.18, 0.12])
    products = [RNG.choice(CATEGORIES[c]["products"]) for c in cat]
    quantity = RNG.randint(1, 5, size=n)
    unit_price = np.array([PRODUCT_PRICE[p] for p in products]) * RNG.uniform(0.9, 1.15, size=n)
    total_amount = quantity * unit_price
    payment = RNG.choice(["支付宝", "微信支付", "银行卡", "货到付款"], size=n, p=[0.42, 0.38, 0.15, 0.05])
    is_returned = RNG.rand(n) < 0.08
    review_score = np.clip(RNG.normal(4.3, 0.8, size=n), 1, 5).round(1)

    return pd.DataFrame({
        "order_id": [f"ORD{i:06d}" for i in range(1, n + 1)],
        "order_date": dates.date,
        "customer_id": [f"C{RNG.randint(1, 600):05d}" for _ in range(n)],
        "region": regions,
        "category": cat,
        "product": products,
        "quantity": quantity,
        "unit_price": unit_price.round(2),
        "total_amount": total_amount.round(2),
        "payment_method": payment,
        "is_returned": is_returned,
        "review_score": review_score,
    })


def generate_financial() -> pd.DataFrame:
    """公司财务数据：按月 × 部门 × 科目 的收入/支出。"""
    departments = ["销售部", "市场部", "研发部", "运营部", "人事部"]
    accounts = ["主营业务收入", "销售成本", "市场费用", "研发支出", "运营支出", "管理费用"]

    rows = []
    for month_idx in range(24):
        year = 2024 if month_idx < 12 else 2025
        month = month_idx % 12 + 1
        date = f"{year}-{month:02d}-01"

        for dept in departments:
            for account in accounts:
                if account == "主营业务收入":
                    amount = 5_000_000 * (0.95 if year == 2025 else 1.0) * (1 + RNG.normal(0, 0.06))
                elif account == "销售成本":
                    amount = -2_200_000 * (1 + RNG.normal(0, 0.05))
                elif account == "市场费用":
                    amount = -800_000 * (1.15 if year == 2025 else 1.0) * (1 + RNG.normal(0, 0.08))
                elif account == "研发支出":
                    amount = -1_400_000 * (1 + RNG.normal(0, 0.04))
                elif account == "运营支出":
                    amount = -600_000 * (1 + RNG.normal(0, 0.07))
                else:
                    amount = -500_000 * (1 + RNG.normal(0, 0.05))
                rows.append({
                    "date": date,
                    "year": year,
                    "month": month,
                    "department": dept,
                    "account": account,
                    "amount": round(amount, 2),
                })

    return pd.DataFrame(rows)


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    sales = generate_sales()
    ecommerce = generate_ecommerce()
    financial = generate_financial()

    sales.to_csv(DATASETS_DIR / "sales.csv", index=False, encoding="utf-8-sig")
    ecommerce.to_csv(DATASETS_DIR / "ecommerce.csv", index=False, encoding="utf-8-sig")
    financial.to_csv(DATASETS_DIR / "financial.csv", index=False, encoding="utf-8-sig")

    print(f"[sales]       {len(sales):>6} rows × {sales.shape[1]} cols")
    print(f"[ecommerce]   {len(ecommerce):>6} rows × {ecommerce.shape[1]} cols")
    print(f"[financial]   {len(financial):>6} rows × {financial.shape[1]} cols")
    print(f"已保存到 {DATASETS_DIR}")


if __name__ == "__main__":
    main()
