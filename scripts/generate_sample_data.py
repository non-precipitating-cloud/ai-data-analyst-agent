"""生成示例数据集（sales / ecommerce / financial），供 Agent 演示与测试使用。

用法：
    python scripts/generate_sample_data.py

脚本用途：
- 在 datasets/ 目录下生成三份带“业务故事”的 CSV：
  sales.csv（月度×地区×品类销售，内置 2025 下滑、季节性、异常值、销售-利润相关）、
  ecommerce.csv（4000 条电商订单，含退换货/评分/支付方式）、
  financial.csv（按月×部门×科目的收入/支出）。
- 生成确定性数据（固定随机种子 42）：每次运行内容一致，方便复现 Demo 与回归测试。
- 输出统一使用 utf-8-sig 编码，Excel 直接打开中文不乱码。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# 项目根目录（scripts 的上一级），保证从任意工作目录运行都能定位 datasets/
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"

# 固定种子的随机数生成器：所有“随机”数据由此产出，保证结果可复现
RNG = np.random.RandomState(42)

REGIONS = ["华东", "华南", "华北", "西南", "华中", "东北"]

# 品类定义：每个品类含代表性产品列表与基准毛利率
CATEGORIES = {
    "电子产品": {"products": ["智能手机", "笔记本电脑", "蓝牙耳机"], "margin": 0.15},
    "家居用品": {"products": ["智能台灯", "空气净化器", "电饭煲"], "margin": 0.25},
    "服装": {"products": ["男士夹克", "女士连衣裙", "运动鞋"], "margin": 0.35},
    "食品饮料": {"products": ["坚果礼盒", "咖啡豆", "气泡水"], "margin": 0.20},
    "美妆个护": {"products": ["护肤套装", "口红", "洗发水"], "margin": 0.40},
}

# 各产品的基准月销售额（未乘地区系数/季节/趋势前），单位：元
PRODUCT_BASE = {
    "智能手机": 900_000, "笔记本电脑": 1_100_000, "蓝牙耳机": 320_000,
    "智能台灯": 150_000, "空气净化器": 420_000, "电饭煲": 180_000,
    "男士夹克": 260_000, "女士连衣裙": 220_000, "运动鞋": 300_000,
    "坚果礼盒": 170_000, "咖啡豆": 130_000, "气泡水": 110_000,
    "护肤套装": 350_000, "口红": 140_000, "洗发水": 90_000,
}

# 各产品的基准单价，用于反推销量，单位：元
PRODUCT_PRICE = {
    "智能手机": 4500, "笔记本电脑": 6800, "蓝牙耳机": 800,
    "智能台灯": 300, "空气净化器": 1200, "电饭煲": 500,
    "男士夹克": 700, "女士连衣裙": 550, "运动鞋": 620,
    "坚果礼盒": 200, "咖啡豆": 150, "气泡水": 90,
    "护肤套装": 900, "口红": 260, "洗发水": 120,
}

# 地区系数：模拟各区域市场体量差异（华东最大、东北最小）
REGION_FACTOR = {"华东": 1.35, "华南": 1.15, "华北": 1.00, "西南": 0.85, "华中": 0.80, "东北": 0.60}

# 季节相位：不同品类在不同月份有高峰
SEASON_PHASE = {
    "电子产品": 2.0, "家居用品": 4.0, "服装": 5.0, "食品饮料": 6.0, "美妆个护": 1.0,
}


def _segment(category: str) -> str:
    """按品类映射客户分群（企业客户/家庭/个人）。

    参数：
        category: 品类名称。
    返回值：
        str: 该品类对应的客户分群名称。
    """
    mapping = {
        "电子产品": "企业客户", "家居用品": "家庭", "服装": "个人",
        "食品饮料": "家庭", "美妆个护": "个人",
    }
    return mapping[category]


def generate_sales() -> pd.DataFrame:
    """生成月度 × 地区 × 产品的销售数据。

    返回值：
        pd.DataFrame: 含 date/year/month/region/category/product/sales/quantity/
        profit/discount/unit_price/customer_segment 列；
        数据内置 2025 年下滑故事（整体 -12%、华东额外 -20%、服装额外 -25%）、
        正弦季节性、促销高折扣、以及注入的异常高销量与负利润样本，
        且利润与销售额正相关，供 Agent 做归因/异常/相关性分析。
    """
    rows = []
    # 24 个月：2024 全年 12 个月 + 2025 全年 12 个月，用于做同比分析
    for month_idx in range(24):
        year = 2024 if month_idx < 12 else 2025
        month = month_idx % 12 + 1
        date = f"{year}-{month:02d}-01"

        for region in REGIONS:
            for category, meta in CATEGORIES.items():
                for product in meta["products"]:
                    # 基准销售额 = 产品基准 × 地区系数
                    base = PRODUCT_BASE[product] * REGION_FACTOR[region]

                    # 季节性：年内正弦波动 ±25%，相位因品类而异（错峰旺季）
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

                    # 随机噪声：乘性正态扰动（标准差 8%），模拟真实经营波动
                    noise = 1 + RNG.normal(0, 0.08)
                    sales = base * season * trend * noise

                    # 折扣：2025 服装加大促销
                    discount = float(RNG.uniform(0.0, 0.30))
                    if year == 2025 and category == "服装":
                        discount *= 1.5
                    discount = min(discount, 0.60)

                    # 利润 = 销售额 × 毛利率 × 折扣侵蚀（折扣越高利润越低）+ 小幅噪声，
                    # 因此利润与销售额天然正相关、与折扣负相关
                    margin = meta["margin"]
                    profit = sales * margin * (1 - 0.8 * discount) * (1 + RNG.normal(0, 0.05))

                    # 单价围绕基准价小幅波动；销量由销售额反推（至少 1 件）
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

    # 注入异常值：随机选 12 行做“促销爆量”，销售额放大 3~6 倍，供异常检测工具发现
    n_outliers = 12
    # replace=False 保证不重复抽中同一行
    outlier_idx = RNG.choice(len(df), size=n_outliers, replace=False)
    df.loc[outlier_idx, "sales"] *= RNG.uniform(3.0, 6.0, size=n_outliers)
    # 销量随异常销售额同步重算，保持两列逻辑自洽
    df.loc[outlier_idx, "quantity"] = (
        df.loc[outlier_idx, "sales"] / df.loc[outlier_idx, "unit_price"]
    ).round().astype(int)

    # 再造 6 行“高折扣低利润”样本：85% 折扣把利润压到极低（接近亏损）
    n_neg = 6
    neg_idx = RNG.choice(len(df), size=n_neg, replace=False)
    df.loc[neg_idx, "discount"] = 0.85
    df.loc[neg_idx, "profit"] = df.loc[neg_idx, "sales"] * 0.10 * (1 - 0.8 * 0.85)

    return df


def generate_ecommerce() -> pd.DataFrame:
    """生成电商订单明细数据（4000 单，跨约 2 年）。

    返回值：
        pd.DataFrame: 含订单号/下单日期/客户号/地区/品类/产品/数量/单价/总额/
        支付方式/是否退货/评分等列；地区、品类、支付方式按设定概率分布抽样，
        退货率约 8%，评分服从均值 4.3 的正态分布并截断到 1~5。
    """
    n = 4000
    start = pd.Timestamp("2024-01-01")
    # 在 730 天窗口内均匀随机偏移，得到随机下单日期
    days = RNG.randint(0, 730, size=n)
    dates = start + pd.to_timedelta(days, unit="D")

    # 各维度按经验概率加权抽样，模拟真实业务分布而非均匀分布
    regions = RNG.choice(REGIONS, size=n, p=[0.30, 0.20, 0.18, 0.14, 0.10, 0.08])
    categories = list(CATEGORIES.keys())
    cat = RNG.choice(categories, size=n, p=[0.28, 0.20, 0.22, 0.18, 0.12])
    # 品类先抽出，再从该品类下等概率选具体产品
    products = [RNG.choice(CATEGORIES[c]["products"]) for c in cat]
    quantity = RNG.randint(1, 5, size=n)
    # 单价在基准价 0.9~1.15 倍间浮动，模拟日常促销波动
    unit_price = np.array([PRODUCT_PRICE[p] for p in products]) * RNG.uniform(0.9, 1.15, size=n)
    total_amount = quantity * unit_price
    payment = RNG.choice(["支付宝", "微信支付", "银行卡", "货到付款"], size=n, p=[0.42, 0.38, 0.15, 0.05])
    # 伯努利试验：约 8% 的订单标记为退货
    is_returned = RNG.rand(n) < 0.08
    # 评分截断在合法区间 [1, 5]
    review_score = np.clip(RNG.normal(4.3, 0.8, size=n), 1, 5).round(1)

    return pd.DataFrame({
        "order_id": [f"ORD{i:06d}" for i in range(1, n + 1)],
        "order_date": dates.date,
        # 客户 ID 在约 600 个客户内随机，天然形成复购行为
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
    """生成公司财务损益类数据（按月 × 部门 × 科目）。

    返回值：
        pd.DataFrame: 含 date/year/month/department/account/amount 列；
        收入科目为正数、支出科目为负数；2025 年收入 -5%、市场费用 +15%，
        与 sales 的下滑故事呼应，供财务结构与同比分析。
    """
    departments = ["销售部", "市场部", "研发部", "运营部", "人事部"]
    accounts = ["主营业务收入", "销售成本", "市场费用", "研发支出", "运营支出", "管理费用"]

    rows = []
    for month_idx in range(24):
        year = 2024 if month_idx < 12 else 2025
        month = month_idx % 12 + 1
        date = f"{year}-{month:02d}-01"

        for dept in departments:
            for account in accounts:
                # 每个科目有独立的基准金额、年度系数与波动幅度；支出用负数表示
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
    """脚本入口：生成三份数据集并写入 datasets/ 目录，打印行数统计。

    返回值：
        None；副作用是创建目录并输出 sales.csv / ecommerce.csv / financial.csv。
    """
    # 目录不存在时自动创建，已存在也不报错
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    sales = generate_sales()
    ecommerce = generate_ecommerce()
    financial = generate_financial()

    # utf-8-sig：带 BOM 的 UTF-8，Excel 双击打开中文不乱码；index=False 不写行号列
    sales.to_csv(DATASETS_DIR / "sales.csv", index=False, encoding="utf-8-sig")
    ecommerce.to_csv(DATASETS_DIR / "ecommerce.csv", index=False, encoding="utf-8-sig")
    financial.to_csv(DATASETS_DIR / "financial.csv", index=False, encoding="utf-8-sig")

    print(f"[sales]       {len(sales):>6} rows × {sales.shape[1]} cols")
    print(f"[ecommerce]   {len(ecommerce):>6} rows × {ecommerce.shape[1]} cols")
    print(f"[financial]   {len(financial):>6} rows × {financial.shape[1]} cols")
    print(f"已保存到 {DATASETS_DIR}")


if __name__ == "__main__":
    main()
