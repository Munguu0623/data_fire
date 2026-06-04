"""
АЛХАМ 4 — XGBoost загвар сургах (шалтгаан бүрд)
================================================
- Зорилт: хороо × долоо хоногт гал гарах магадлал (binary classification)
- Цаг хугацааны хуваалт: 2015-2023 сургалт, 2024-2025 тест (leakage үгүй)
- Class imbalance: scale_pos_weight ашиглана
- Үнэлгээ: PR-AUC, ROC-AUC, recall@top-k (ОБ-ын сургалтын төлөвлөлтөд тохирсон)

Гаралт: model_{cause}.json, predictions.parquet, metrics
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
import json

CAUSE_KEYS = {'ilgal': 'Ил гал', 'tsakhilgaan': 'Цахилгаан', 'yandan': 'Яндан/цонолт'}
TRAIN_END = 2023   # 2015-2023 сургалт
TEST_START = 2024  # 2024-2025 тест

# Бүх шалтгаанд нийтлэг features
BASE_FEATURES = ['week_sin', 'week_cos', 'month_sin', 'month_cos', 'year_idx',
                 'log_pop', 'log_district_pop', 'iso_week', 'month']

def cause_features(key):
    """Тухайн шалтгааны lag/rolling/seasonal features."""
    return [f'{key}_lag1', f'{key}_lag2', f'{key}_lag4',
            f'{key}_roll4', f'{key}_roll8', f'{key}_roll12',
            f'{key}_roll26', f'{key}_roll52', f'{key}_lag52',
            f'{key}_cummean', f'{key}_seasweek']

def recall_at_topk(y_true, y_score, groups, k=10):
    """
    Долоо хоног бүрд эрсдэлийн оноогоор эрэмбэлж дээд k хороог сонгоход
    бодит гарсан галын хэдэн хувийг барьж авч байна вэ.
    ОБ "хамгийн эрсдэлтэй 10 хороонд сургалт хий" гэдэгт яг тохирно.
    """
    dfm = pd.DataFrame({'y': y_true, 's': y_score, 'g': groups})
    total_hit, total_fire = 0, 0
    for _, grp in dfm.groupby('g'):
        topk = grp.nlargest(k, 's')
        total_hit += topk['y'].sum()
        total_fire += grp['y'].sum()
    return total_hit / total_fire if total_fire > 0 else np.nan

def train_one(df, key):
    label = f'fire_{key}'
    feats = BASE_FEATURES + cause_features(key)

    train = df[df['iso_year'] <= TRAIN_END]
    test = df[df['iso_year'] >= TEST_START]

    Xtr, ytr = train[feats], train[label]
    Xte, yte = test[feats], test[label]

    # Class imbalance тэнцвэр
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric='aucpr',
        min_child_weight=5, reg_lambda=1.0,
        n_jobs=-1, random_state=42,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)

    proba = model.predict_proba(Xte)[:, 1]
    # Долоо хоног бүрийн бүлэг (recall@k-д)
    grp = test['iso_year'].astype(str) + '-' + test['iso_week'].astype(str)

    metrics = {
        'cause': CAUSE_KEYS[key],
        'train_rows': len(train), 'test_rows': len(test),
        'pos_rate_train': float(ytr.mean()),
        'pr_auc': float(average_precision_score(yte, proba)),
        'roc_auc': float(roc_auc_score(yte, proba)),
        'recall_top5': float(recall_at_topk(yte.values, proba, grp.values, 5)),
        'recall_top10': float(recall_at_topk(yte.values, proba, grp.values, 10)),
        'recall_top20': float(recall_at_topk(yte.values, proba, grp.values, 20)),
        'baseline_pr': float(yte.mean()),  # санамсаргүй таамаглалын суурь
    }

    model.save_model(f'./data/model_{key}.json')

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    metrics['top_features'] = imp.head(6).to_dict()

    # Тестийн таамаглалыг хадгалах
    pred_df = test[['district', 'khoroo', 'iso_year', 'iso_week', 'month', label]].copy()
    pred_df[f'risk_{key}'] = proba
    return model, metrics, pred_df

def main():
    df = pd.read_parquet('./data/features.parquet')
    all_metrics, preds = [], None

    for key in CAUSE_KEYS:
        model, m, pred_df = train_one(df, key)
        all_metrics.append(m)
        if preds is None:
            preds = pred_df
        else:
            preds = preds.merge(
                pred_df[['district', 'khoroo', 'iso_year', 'iso_week', f'fire_{key}', f'risk_{key}']],
                on=['district', 'khoroo', 'iso_year', 'iso_week'], how='outer')

    preds.to_parquet('./data/predictions.parquet', index=False)
    with open('./data/metrics.json', 'w') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    # Тайлан
    print('=' * 64)
    print('ЗАГВАРЫН ҮР ДҮН (тест: 2024-2025)')
    print('=' * 64)
    for m in all_metrics:
        print(f"\n▶ {m['cause']}")
        print(f"  Гал гарах суурь магадлал: {m['baseline_pr']*100:.2f}%")
        print(f"  PR-AUC:  {m['pr_auc']:.3f}  (санамсаргүй={m['baseline_pr']:.3f})")
        print(f"  ROC-AUC: {m['roc_auc']:.3f}")
        print(f"  Recall@top5  (7 хоног бүр эрсдэлт 5 хороо):  {m['recall_top5']*100:.1f}%")
        print(f"  Recall@top10 (7 хоног бүр эрсдэлт 10 хороо): {m['recall_top10']*100:.1f}%")
        print(f"  Recall@top20 (7 хоног бүр эрсдэлт 20 хороо): {m['recall_top20']*100:.1f}%")
        print(f"  Гол features: {', '.join(list(m['top_features'])[:4])}")

if __name__ == '__main__':
    main()
