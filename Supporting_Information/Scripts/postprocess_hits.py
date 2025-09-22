from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    from postproc.ta_ranker import add_ta_flags, hierarchical_sort
    from postproc.tanimoto_cluster import (
        calc_ecfp_bits,
        cluster_ids,
        cluster_diversity,
        cluster_medoids,
        plot_tanimoto_dendrogram,
        build_prob_matrix,
        cosine_similarity_matrix,
        auto_cut as auto_cut_struct,
        auto_cut_cosine,
        _similarity_matrix,
        _labels_from_threshold,
        knn_communities,
    )
    from postproc.concordance import report_concordance
    try:
        from postproc.safety_filter import add_safety_tags  # type: ignore
    except Exception:
        add_safety_tags = None  # type: ignore
except Exception:  # fallback if executed as module within package
    from .ta_ranker import add_ta_flags, hierarchical_sort  # type: ignore
    from .tanimoto_cluster import (  # type: ignore
        calc_ecfp_bits,
        cluster_ids,
        cluster_diversity,
        cluster_medoids,
        plot_tanimoto_dendrogram,
        build_prob_matrix,
        cosine_similarity_matrix,
        auto_cut as auto_cut_struct,
        auto_cut_cosine,
        _similarity_matrix,
        _labels_from_threshold,
        knn_communities,
    )
    from .concordance import report_concordance  # type: ignore
    try:
        from .safety_filter import add_safety_tags  # type: ignore
    except Exception:
        add_safety_tags = None  # type: ignore


def get_logger() -> logging.Logger:
    logger = logging.getLogger("postproc")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


DEFAULT_WINDOW_GROUPS = {
    "morning": ["CHEMBL239", "CHEMBL3351", "CHEMBL2083"],
    "evening": ["CHEMBL2047", "CHEMBL1947"],
    "postpr": ["CHEMBL5080"],
}


def assign_window_row(row: pd.Series, groups: dict, thr: float, delta: float) -> str:
    def p(tid: str) -> float:
        return float(row.get(f"{tid}_probability", 0.0))
    def group_score(keys: List[str]) -> float:
        return max((p(k) for k in keys), default=0.0)
    score_morning = group_score(groups.get("morning", []))
    score_evening = group_score(groups.get("evening", []))
    score_postpr = group_score(groups.get("postpr", []))
    vals = {
        "morning": score_morning,
        "evening": score_evening,
        "postpr": score_postpr,
    }
    first = max(vals, key=vals.get)
    second = max([k for k in vals if k != first], key=lambda k: vals[k])
    if vals[first] < thr:
        return "unspecified"
    if vals[first] - vals[second] < delta:
        return "dual_" + "_".join(sorted([first, second]))
    mapping = {
        "morning": "morning_powder",
        "evening": "evening_shot",
        "postpr": "postprandial_chew",
    }
    return mapping.get(first, "unspecified")


def suggested_form_from_window(win: str) -> str:
    mapping = {
        "morning_powder": "powder",
        "evening_shot": "shot",
        "postprandial_chew": "chew",
        "dual_evening_postpr": "shot+chew",
        "dual_morning_postpr": "powder+chew",
    }
    return mapping.get(win, "")


def parse_args() -> argparse.Namespace:
    """Запуск без атрибутов — все настройки берутся из config.yaml."""
    p = argparse.ArgumentParser("Post-process ranked hits (config-driven)")
    return p.parse_args([])


def main() -> None:
    _ = parse_args()
    logger = get_logger()

    # Загрузка конфигурации (yaml)
    try:
        try:
            from postproc.config import resolve_config  # type: ignore
        except Exception:
            from .config import resolve_config  # type: ignore
        cfg = resolve_config() or {}
    except Exception:
        cfg = {}

    # Пути ввода/вывода
    io_cfg = cfg.get("io", {}) if isinstance(cfg.get("io"), dict) else {}
    pred_path = Path(io_cfg.get("predictions", "data/advanced_predictions.csv"))
    oof_dir = Path(io_cfg.get("oof_dir", "data/oof"))
    out_dir = Path(io_cfg.get("out_dir", "data/postproc"))
    heat_dir = Path(io_cfg.get("reports_dir", "reports"))
    for p in [out_dir, heat_dir, oof_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if not cfg or not isinstance(cfg, dict) or not cfg.get("_config_path"):
        logger.warning(
            "Config not found or not parsed; using defaults. Install 'pyyaml' and ensure config.yaml is present."
        )
    logger.info("Loading predictions: %s", pred_path)
    df = pd.read_csv(pred_path)

    logger.info("Detecting probability columns")
    prob_cols = [c for c in df.columns if c.endswith("_probability")]
    if not prob_cols:
        raise RuntimeError("No probability columns (*_probability) found in predictions")

    # Чтение весов
    weights = {}
    try:
        import json
        weights_path = cfg.get("weights_path") if isinstance(cfg.get("weights_path"), str) else None
        if weights_path and Path(weights_path).exists():
            with open(weights_path, "r", encoding="utf-8") as f:
                weights = json.load(f)
        else:
            weights = {}
    except Exception:
        weights = {}

    # Пер-таргет пороги
    per_thr = {}
    if isinstance(cfg.get("thresholds", {}).get("per_target"), dict):
        per_thr = dict(cfg.get("thresholds", {}).get("per_target", {}))
        logger.info("Loaded per-target thresholds from config: %d", len(per_thr))

    # TA: базовый порог из конфигурации, либо per-target
    base_ta_thr = float(cfg.get("ta", {}).get("base_thr", 0.5)) if isinstance(cfg.get("ta"), dict) else 0.5
    logger.info("Computing TA/priority with base thr=%.2f (per-target=%s)", base_ta_thr, bool(per_thr))
    # Если есть пер-таргет пороги — используем их для TA
    if per_thr:
        tids = [c.replace("_probability", "") for c in prob_cols]
        thr_vec = np.array([float(per_thr.get(t, base_ta_thr)) for t in tids], dtype=float).reshape(1, -1)
        probs = df[prob_cols].to_numpy(dtype=float)
        ta_counts = (probs >= thr_vec).sum(axis=1).astype(int)
        df["TA"] = ta_counts
        # priority как раньше
        priority = np.full(df.shape[0], "Low", dtype=object)
        priority[(ta_counts >= 1) & (ta_counts <= 2)] = "Med"
        priority[ta_counts > 2] = "High"
        df["priority"] = priority
    else:
        df = add_ta_flags(df, prob_cols, thr=base_ta_thr)

    # Мягкая метрика TA_weighted = sum_t w_t * p_t
    try:
        # Преобразуем вектор вероятностей и весов
        prob_mat_arr = df[prob_cols].to_numpy(dtype=float)
        # Веса в порядке столбцов prob_cols (которые в формате <TID>_probability)
        w_vec = np.array([float(weights.get(c.replace("_probability", ""), 1.0)) for c in prob_cols], dtype=float)
        df["TA_weighted"] = (prob_mat_arr * w_vec.reshape(1, -1)).sum(axis=1)
    except Exception:
        df["TA_weighted"] = df[prob_cols].mean(axis=1)

    # Определяем основной скор (единая точка входа)
    if "main_score" in df.columns:
        main_score = "main_score"
    elif "weighted_sum" in df.columns:
        main_score = "weighted_sum"
    elif "bayes_score" in df.columns:
        main_score = "bayes_score"
    else:
        raise RuntimeError("Не найден ни main_score, ни weighted_sum, ни bayes_score")

    # Рассчёт отпечатков и кластеров
    if "SMILES" not in df.columns:
        raise RuntimeError("Column 'SMILES' is required in predictions to compute ECFP")
    # Гарантируем наличие Name для сортировки/витрины
    if "Name" not in df.columns and "SMILES" in df.columns:
        df["Name"] = df["SMILES"]

    cluster_cfg = cfg.get("cluster", {}) if isinstance(cfg.get("cluster"), dict) else {}
    fp = str(cluster_cfg.get("fp", "morgan"))
    sim = str(cluster_cfg.get("sim", "tanimoto"))
    alpha = float(cluster_cfg.get("alpha", 0.5))
    auto_cut_flag = bool(cluster_cfg.get("auto_cut", False))
    # Кандидаты порогов для auto-cut (если заданы в YAML)
    cand = cluster_cfg.get("candidates")
    if isinstance(cand, (list, tuple)) and len(cand) >= 2:
        try:
            cand = [float(x) for x in cand]
        except Exception:
            cand = None
    else:
        cand = None
    logger.info("Calculating fingerprints (%s) and clustering (sim=%s)", fp, sim)
    # Импорт обновлённого API
    try:
        from postproc.tanimoto_cluster import calc_fingerprint  # type: ignore
    except Exception:
        from .tanimoto_cluster import calc_fingerprint  # type: ignore
    # Параметры отпечатка из конфига (если заданы)
    fp_n_bits = int(cluster_cfg.get("n_bits", 2048))
    fp_radius = int(cluster_cfg.get("radius", 2))
    bits = calc_fingerprint(df["SMILES"].astype(str).tolist(), fp=fp, n_bits=fp_n_bits, radius=fp_radius)
    thr_eff = float(cluster_cfg.get("tan_thr", 0.58))
    if auto_cut_flag:
        try:
            from postproc.tanimoto_cluster import auto_cut, _similarity_matrix  # type: ignore
        except Exception:
            from .tanimoto_cluster import auto_cut, _similarity_matrix  # type: ignore
        thr_eff = auto_cut(bits, sim=sim, alpha=alpha, candidates=cand)
        logger.info("Auto-selected cluster threshold: %.3f", thr_eff)
    logger.info(
        "Config loaded from: %s | Cluster params: fp=%s n_bits=%d radius=%d sim=%s alpha=%.2f thr=%.2f auto_cut=%s",
        str(cfg.get("_config_path", "<none>")), fp, fp_n_bits, fp_radius, sim, alpha, thr_eff, str(auto_cut_flag),
    )
    cids = cluster_ids(bits, thr=thr_eff, sim=sim, alpha=alpha)

    # Фоллбек: если все синглтоны — подвинуть порог к максимуму сходства
    try:
        if len(np.unique(cids)) == bits.shape[0]:
            try:
                from postproc.tanimoto_cluster import _similarity_matrix  # type: ignore
            except Exception:
                from .tanimoto_cluster import _similarity_matrix  # type: ignore
            S = _similarity_matrix(bits, sim=sim, alpha=alpha)
            tri = np.triu(S, 1)
            smax = float(np.max(tri)) if tri.size else 0.0
            thr_low = max(0.20, 0.95 * smax)
            if thr_low < thr_eff:
                logger.warning(
                    "All singletons at thr=%.2f; lowering cut to %.2f (smax=%.2f)",
                    thr_eff,
                    thr_low,
                    smax,
                )
                thr_eff = thr_low
                cids = cluster_ids(bits, thr=thr_eff, sim=sim, alpha=alpha)
            # Дополнительно: kNN-фоллбек, если всё ещё одиночки
            if len(np.unique(cids)) == bits.shape[0]:
                from postproc.tanimoto_cluster import knn_communities  # type: ignore
                knn_k = int(cluster_cfg.get("knn_k", 3)) if isinstance(cluster_cfg, dict) else 3
                cids_knn = knn_communities(S, k=knn_k)
                if len(np.unique(cids_knn)) < len(cids_knn):
                    logger.warning(
                        "Structural clustering: all singletons at thr=%.2f -> switched to kNN communities (k=%d).",
                        thr_eff,
                        knn_k,
                    )
                    cids = cids_knn
            # Сохраним топ-пары сходства для диагностики
            try:
                idx = np.argsort(tri, axis=None)
                if idx.size:
                    tail = idx[-25:]
                    i, j = np.unravel_index(tail, tri.shape)
                    (pd.DataFrame({"i": i, "j": j, "sim": S[i, j]})
                     .sort_values("sim", ascending=False)
                     .to_csv((heat_dir / "similarity_top_pairs.csv"), index=False))
            except Exception:
                pass
        # Если наоборот — один гигантский кластер, применим MST-cut для получения >=3 групп
        if len(np.unique(cids)) <= 1:
            try:
                from scipy.sparse.csgraph import minimum_spanning_tree  # type: ignore
                # матрица расстояний
                if 'S' not in locals():
                    try:
                        from postproc.tanimoto_cluster import _similarity_matrix  # type: ignore
                    except Exception:
                        from .tanimoto_cluster import _similarity_matrix  # type: ignore
                    S = _similarity_matrix(bits, sim=sim, alpha=alpha)
                D = 1.0 - S
                T = minimum_spanning_tree(D).toarray()
                edges = []
                n_nodes = S.shape[0]
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if T[i, j] > 0:
                            edges.append((i, j, float(T[i, j])))
                edges.sort(key=lambda x: x[2], reverse=True)
                # целевое число кластеров
                target_k = int(cluster_cfg.get("target_k", 0)) if isinstance(cluster_cfg, dict) else 0
                if target_k <= 0:
                    import math
                    target_k = max(3, int(math.sqrt(max(1, n_nodes))))
                cut_edges = set((i, j) for i, j, _ in edges[: max(0, target_k - 1)])
                # DSU для компонентов после разреза
                parent = list(range(n_nodes))

                def find(x: int) -> int:
                    while x != parent[x]:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def union(a: int, b: int) -> None:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

                for i, j, _w in edges[max(0, target_k - 1) :]:
                    if (i, j) not in cut_edges:
                        union(i, j)
                roots = [find(i) for i in range(n_nodes)]
                uniq = {r: k for k, r in enumerate(dict.fromkeys(roots))}
                cids_mst = np.array([uniq[r] for r in roots], dtype=int)
                if len(np.unique(cids_mst)) > 1:
                    logger.warning(
                        "Structural clustering: single giant component -> MST cut fallback applied (k=%d).",
                        target_k,
                    )
                    cids = cids_mst
            except Exception:
                pass
    except Exception:
        pass

    div = cluster_diversity(bits, cids, sim=sim, alpha=alpha)

    df["cluster_id"] = cids
    df["div_score"] = div.values

    # Медоиды и дендрограмма
    is_medoid_mask = cluster_medoids(bits, cids, sim=sim, alpha=alpha)
    # Иерархическая сортировка и ранги
    logger.info("Hierarchical sort and ranking")
    df_with_medoids = df.copy()
    df_with_medoids["is_medoid"] = is_medoid_mask
    df_sorted = hierarchical_sort(
        df_with_medoids,
        main_score=main_score,
        diversity_col="div_score",
        tie_breaker="TA_weighted",
    ).copy()
    # упорядочим биты согласно порядку строк после сортировки и сохраним исходный индекс
    _orig_idx = df_sorted.index.values
    bits_sorted = bits[_orig_idx, :]
    df_out = df_sorted.reset_index().rename(columns={"index": "orig_idx"})
    df_out["rank"] = np.arange(1, len(df_out) + 1)

    # Safety/регуляторные теги (если модуль доступен)
    if add_safety_tags is not None:
        try:
            logger.info("Annotating safety/regulatory risk tags")
            df_out = add_safety_tags(df_out, smiles_col="SMILES", name_col="Name")  # type: ignore
        except Exception as e:
            logger.warning("Safety annotation failed: %s", e)

    # Регуляторные аннотации (не перетирают базовые поля)
    try:
        from regdb import annotate_regulatory  # type: ignore
        before_cols = set(df_out.columns)
        df_out = annotate_regulatory(df_out)  # type: ignore
        required = {"InChIKey", "SMILES", "Name"}
        assert required.issubset(set(df_out.columns)), "Регуляторная аннотация потеряла базовые столбцы."
        new_cols = sorted(set(df_out.columns) - before_cols)
        logger.info("Regulatory columns added: %s", ", ".join(new_cols))
    except Exception as e:
        logger.warning("Regulatory annotation skipped: %s", e)

    # Chrono-окна и форма выпуска до сохранения CSV
    groups = cfg.get("window_targets", {}) if isinstance(cfg.get("window_targets"), dict) else {}
    if not groups:
        groups = DEFAULT_WINDOW_GROUPS
    chrono_cfg = cfg.get("chrono", {}) if isinstance(cfg.get("chrono"), dict) else {}
    chrono_delta = float(chrono_cfg.get("delta", 0.03))
    df_out["chrono_window"] = df_out.apply(lambda r: assign_window_row(r, groups, thr=base_ta_thr, delta=chrono_delta), axis=1)
    df_out["suggested_form"] = df_out["chrono_window"].map(suggested_form_from_window).fillna("")

    # --- Murcko scaffolds (optional) ---
    try:
        scfg = cfg.get("scaffold", {}) if isinstance(cfg.get("scaffold"), dict) else {}
        if scfg.get("enabled", True):
            from rdkit import Chem  # type: ignore
            from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
            smiles_col = "SMILES"
            scaff_smis: List[str] = []
            for smi in df_out[smiles_col].fillna("").astype(str).tolist():
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    scaff_smis.append("")
                    continue
                core = MurckoScaffold.GetScaffoldForMol(m)
                s = Chem.MolToSmiles(core) if core is not None else ""
                scaff_smis.append(s)
            df_out["murcko_scaffold"] = scaff_smis
            uniq = {s: i for i, s in enumerate(dict.fromkeys(df_out["murcko_scaffold"].tolist()))}
            df_out["scaffold_id"] = [uniq.get(s, -1) for s in df_out["murcko_scaffold"]]
    except Exception as e:
        logger.warning("Murcko scaffolds failed: %s", e)

    # === MOA кластеризация (единожды) ===
    try:
        mocfg = cfg.get("moa_cluster", {}) if isinstance(cfg.get("moa_cluster"), dict) else {}
        if mocfg.get("enabled", True):
            P, _tnames = build_prob_matrix(df_out)
            S_moa = cosine_similarity_matrix(P, normalize=str(mocfg.get("normalize", "l2")))
            if mocfg.get("auto_cut", True):
                moa_thr = float(auto_cut_cosine(S_moa, candidates=mocfg.get("candidates")))
            else:
                moa_thr = float(mocfg.get("sim_thr", 0.30))
            moa_ids = _labels_from_threshold(S_moa, moa_thr)
            if len(np.unique(moa_ids)) == len(moa_ids):
                moa_knn_k = int(mocfg.get("knn_k", 3))
                moa_ids = knn_communities(S_moa, k=moa_knn_k)
                logger.warning(
                    "MOA threshold %.2f produced singletons; switched to kNN communities (k=%d).",
                    moa_thr,
                    moa_knn_k,
                )
            df_out["moa_cluster_id"] = moa_ids
            # MOA медоиды по косинусной дистанции
            Dm = 1.0 - S_moa
            moa_is_medoid = np.zeros(len(df_out), dtype=bool)
            for cl in np.unique(moa_ids):
                idx = np.where(moa_ids == cl)[0]
                if idx.size == 0:
                    continue
                if idx.size == 1:
                    moa_is_medoid[idx[0]] = True
                else:
                    sums = Dm[np.ix_(idx, idx)].sum(axis=1)
                    moa_is_medoid[idx[int(np.argmin(sums))]] = True
            df_out["moa_is_medoid"] = moa_is_medoid
            # Сохраняем использованные параметры для отчёта
            try:
                import json as _json
                used = {
                    "moa_params_used": {
                        "normalize": str(mocfg.get("normalize", "l2")),
                        "auto_cut": bool(mocfg.get("auto_cut", True)),
                        "sim": "cosine",
                        "candidates": mocfg.get("candidates"),
                        "sim_thr": float(mocfg.get("sim_thr", 0.30)),
                    },
                    "moa_n_clusters": int(len(np.unique(moa_ids))),
                }
                with open((out_dir / "moa_params.json"), "w", encoding="utf-8") as f:
                    _json.dump(used, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
    except Exception as e:
        logger.error("MOA clustering failed (won't write CSV without moa_*): %s", e)
        # Добавим плейсхолдеры, чтобы отчёт продолжился
        df_out["moa_cluster_id"] = df_out.get("moa_cluster_id", pd.Series([-1] * len(df_out)))
        df_out["moa_is_medoid"] = df_out.get("moa_is_medoid", pd.Series([False] * len(df_out)))

    # Гарантируем присутствие колонок в CSV
    cols_to_keep = [
        "murcko_scaffold",
        "scaffold_id",
        "moa_cluster_id",
        "moa_is_medoid",
    ]
    for c in cols_to_keep:
        if c not in df_out.columns:
            df_out[c] = (-1 if c.endswith("_id") else ("" if c.endswith("_scaffold") else False))

    # Сохраним обновлённые предсказания с метками MOA и рангами обратно в predictions_path
    try:
        df_out.to_csv(pred_path, index=False, encoding="utf-8")
        logger.info("Updated predictions saved with MOA labels and ranks: %s", pred_path)
    except Exception as e:
        logger.warning("Failed to update predictions CSV: %s", e)

    # Вывод финальной таблицы
    out_csv = (out_dir / "ranked_hits.csv").resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing ranked hits to %s", out_csv)
    # Добавим calibrated_prob в портфель, если был в исходном файле
    if "calibrated_prob" in df_out.columns:
        pass
    df_out.to_csv(out_csv, index=False)

    # Построение матрицы targets x molecules
    logger.info("Building prob matrix (targets x molecules)")
    targets = [c.replace("_probability", "") for c in prob_cols]
    prob_mat = df_out[[*prob_cols]].T
    prob_mat.index = targets
    # Подписи молекул для heatmap: используем имена, если доступны
    try:
        if "Name" in df_out.columns:
            prob_mat.columns = df_out["Name"].astype(str).tolist()
        elif "SMILES" in df_out.columns:
            prob_mat.columns = df_out["SMILES"].astype(str).tolist()
    except Exception:
        pass

    # Повторную MOA-кластеризацию в отчёте не выполняем; используем уже готовые метки

    # Дендрограмма Танимото
    try:
        labels = None
        if "Name" in df_out.columns:
            labels = df_out["Name"].astype(str).tolist()
        elif "SMILES" in df_out.columns:
            labels = df_out["SMILES"].astype(str).tolist()
        plot_tanimoto_dendrogram(
            bits_sorted,
            (heat_dir / "tanimoto_dendrogram.png"),
            labels=labels,
            thr=thr_eff,
            sim=sim,
            alpha=alpha,
        )
    except Exception as e:
        logger.warning("Dendrogram plotting failed: %s", e)

    # Отчёты по согласованности
    logger.info("Building concordance reports in %s", heat_dir)
    # Прокидываем скан разрывов из config.yaml в окружение для concordance
    try:
        import os
        scan_spec = str(cfg.get("break_gap", {}).get("scan", "")) if isinstance(cfg.get("break_gap"), dict) else ""
        if scan_spec:
            os.environ["CONCORDANCE_SCAN"] = scan_spec
    except Exception:
        pass
    # Передадим calibrated_prob из исходного файла при наличии
    cal_flag = None
    try:
        if "calibrated_prob" in df_out.columns:
            vals = pd.to_numeric(df_out["calibrated_prob"], errors="coerce").dropna().astype(int)
            if not vals.empty:
                cal_flag = int(int(vals.max()) > 0)
    except Exception:
        cal_flag = None

    report_concordance(
        prob_mat,
        bits_sorted,
        oof_dir,
        heat_dir,
        gap_tan_thr=float(cfg.get("break_gap", {}).get("gap_tan", 0.8)) if isinstance(cfg.get("break_gap"), dict) else 0.8,
        gap_delta_thr=float(cfg.get("break_gap", {}).get("gap_thr", 0.4)) if isinstance(cfg.get("break_gap"), dict) else 0.4,
        cluster_thr=thr_eff,
        cluster_sim=sim,
        cluster_alpha=alpha,
        moa_labels=df_out.get("moa_cluster_id").tolist() if "moa_cluster_id" in df_out.columns else None,
        calibrated_prob=cal_flag,
    )

    # Портфель: медоиды с TA >= ta_min (целое число таргетов)
    pf_cfg = cfg.get("portfolio", {}) if isinstance(cfg.get("portfolio"), dict) else {}
    ta_min = int(pf_cfg.get("ta_min", 3))
    K = int(pf_cfg.get("size", 50))
    per_struct = int(pf_cfg.get("per_struct_cluster", pf_cfg.get("per_cluster", 3)))
    per_moa = int(pf_cfg.get("per_moa_cluster", 1))
    cover_win = bool(pf_cfg.get("cover_windows", False))

    cand = df_out[
        (df_out.get("is_medoid", False)) &
        (df_out["TA"] >= ta_min) &
        (df_out.get("risk_level", "green") != "red")
    ].sort_values(["TA", main_score, "TA_weighted"], ascending=[False, False, False]).copy()

    picked_rows = []
    count_struct: dict[int, int] = {}
    count_moa: dict[int, int] = {}
    used_orig_idx: set[int] = set()

    def _can_take(row: pd.Series) -> bool:
        cs = int(row.get("cluster_id", -1))
        cm = int(row.get("moa_cluster_id", -1))
        if per_struct > 0 and count_struct.get(cs, 0) >= per_struct:
            return False
        if per_moa > 0 and count_moa.get(cm, 0) >= per_moa:
            return False
        return True

    for _, r in cand.iterrows():
        if len(picked_rows) >= K:
            break
        if _can_take(r) and int(r.get("orig_idx", -1)) not in used_orig_idx:
            picked_rows.append(r)
            cs = int(r.get("cluster_id", -1)); count_struct[cs] = count_struct.get(cs, 0) + 1
            cm = int(r.get("moa_cluster_id", -1)); count_moa[cm] = count_moa.get(cm, 0) + 1
            used_orig_idx.add(int(r.get("orig_idx", -1)))

    portfolio = pd.DataFrame(picked_rows)

    # Покрытие окон (опционально)
    if cover_win and not portfolio.empty:
        need = {"morning_powder", "evening_shot", "postprandial_chew"}
        have = set(portfolio.get("chrono_window", pd.Series([], dtype=str)).unique().tolist())
        # мягкий пул для добора
        cand_soft = df_out[
            (df_out.get("is_medoid", False)) &
            (df_out["TA"] >= int(pf_cfg.get("ta_min_window", max(1, ta_min - 1)))) &
            (df_out.get("risk_level", "green") != "red")
        ].sort_values(["TA", main_score, "TA_weighted"], ascending=[False, False, False]).copy()
        for w in sorted(list(need - have)):
            if len(portfolio) >= K:
                break
            extra = cand_soft[~cand_soft.index.isin(portfolio.index) & (cand_soft.get("chrono_window") == w)]
            for _, r in extra.iterrows():
                if len(portfolio) >= K:
                    break
                if _can_take(r) and int(r.get("orig_idx", -1)) not in used_orig_idx:
                    portfolio = pd.concat([portfolio, r.to_frame().T], ignore_index=True)
                    cs = int(r.get("cluster_id", -1)); count_struct[cs] = count_struct.get(cs, 0) + 1
                    cm = int(r.get("moa_cluster_id", -1)); count_moa[cm] = count_moa.get(cm, 0) + 1
                    used_orig_idx.add(int(r.get("orig_idx", -1)))
                break

    # Предупреждение и финальный добор до K из мягкого пула
    if portfolio.empty:
        logger.warning("Portfolio is empty (filters too strict). Consider lowering ta_min or relaxing cluster limits.")
    if len(portfolio) < K:
        ta_win = int(pf_cfg.get("ta_min_window", max(1, ta_min - 1)))
        logger.info("Portfolio has %d < %d items; relaxing TA to ta_min_window=%d for fill.", len(portfolio), K, ta_win)
        cand_soft = df_out[
            (df_out.get("is_medoid", False)) &
            (df_out["TA"] >= ta_win) &
            (df_out.get("risk_level", "green") != "red")
        ].sort_values(["TA", main_score, "TA_weighted"], ascending=[False, False, False]).copy()
        for _, r in cand_soft.iterrows():
            if len(portfolio) >= K:
                break
            if int(r.get("orig_idx", -1)) in used_orig_idx:
                continue
            if _can_take(r):
                portfolio = pd.concat([portfolio, r.to_frame().T], ignore_index=True)
                cs = int(r.get("cluster_id", -1)); count_struct[cs] = count_struct.get(cs, 0) + 1
                cm = int(r.get("moa_cluster_id", -1)); count_moa[cm] = count_moa.get(cm, 0) + 1
                used_orig_idx.add(int(r.get("orig_idx", -1)))
    port_csv = (out_dir / "portfolio.csv").resolve()
    port_csv.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing portfolio medoids to %s", port_csv)
    portfolio.to_csv(port_csv, index=False)

    # Сохраним фактически использованные параметры кластера рядом с отчётами
    try:
        import json
        cluster_meta = {
            "config_path": str(cfg.get("_config_path", "")),
            "fp": fp,
            "n_bits": int(fp_n_bits),
            "radius": int(fp_radius),
            "sim": sim,
            "alpha": float(alpha),
            "thr": float(thr_eff),
            "auto_cut": bool(auto_cut_flag),
        }
        with open((heat_dir / "cluster_config_used.json"), "w", encoding="utf-8") as f:
            json.dump(cluster_meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Статистика кластеров для QC
    try:
        from collections import Counter

        def _cluster_stats(labels: np.ndarray) -> dict:
            cnt = Counter(labels.tolist())
            sizes = sorted(cnt.values(), reverse=True)
            return {"n_clusters": int(len(cnt)), "sizes": [int(x) for x in sizes[:20]]}

        stats_struct = _cluster_stats(df_out["cluster_id"].to_numpy()) if "cluster_id" in df_out.columns else None
        stats_moa = _cluster_stats(df_out["moa_cluster_id"].to_numpy()) if "moa_cluster_id" in df_out.columns else None
        with open((heat_dir / "cluster_stats.json"), "w", encoding="utf-8") as f:
            json.dump({"struct": stats_struct, "moa": stats_moa}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    logger.info("Done")


if __name__ == "__main__":
    main()
