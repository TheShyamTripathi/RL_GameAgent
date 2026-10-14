"""
dashboard.py
------------
Streamlit UI for the RL GameAgent project.

    streamlit run dashboard.py

Tabs:  🏠 Overview | 🏋️ Train (background job, live view, Stop) | 🎮 Play | 📊 Analytics
Heavy imports (torch, agent, trainer) are done lazily, only when an action needs them.
"""
from __future__ import annotations

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from configs import GAMES, VARIANTS, make_config
from utils import available_models, list_runs, load_run_log, run_label

st.set_page_config(page_title="RL GameAgent", page_icon="🎮", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.6rem;}
    div[data-testid="stMetric"] {
        background: rgba(120,120,160,0.10); border: 1px solid rgba(120,120,160,0.25);
        border-radius: 12px; padding: 10px 14px;
    }
    .hero {font-size: 2.0rem; font-weight: 800; margin: 0;}
    .sub  {opacity: .7; margin-top: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

METRICS = {
    "Moving average (20 ep)": ("avg20", "Reward"),
    "Moving average (100 ep)": ("avg100", "Reward"),
    "Greedy evaluation": ("eval_mean", "Eval reward"),
    "Episode reward": ("reward", "Reward"),
    "Loss": ("loss", "Huber loss"),
    "Mean Q-value": ("mean_q", "Q"),
    "Epsilon": ("epsilon", "ε"),
}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def device_name() -> str:
    try:
        import torch

        return "CUDA GPU" if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "PyTorch not installed"


def run_options(runs: list[dict]) -> dict[str, dict]:
    """label -> meta, with unique labels."""
    out: dict[str, dict] = {}
    for m in runs:
        label = base = run_label(m)
        i = 2
        while label in out:
            label, i = f"{base} #{i}", i + 1
        out[label] = m
    return out


def fmt(v, nd: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def training_chart(df: pd.DataFrame, height: int = 320):
    base = alt.Chart(df).encode(x=alt.X("episode:Q", title="Episode"))
    raw = base.mark_line(opacity=0.28, color="#6b7cff").encode(
        y=alt.Y("reward:Q", title="Reward"), tooltip=["episode", "reward"])
    avg = base.mark_line(strokeWidth=3, color="#ff7a45").encode(y="avg20:Q", tooltip=["episode", "avg20"])
    layers = [raw, avg]
    ev = df.dropna(subset=["eval_mean"])
    if len(ev):
        layers.append(alt.Chart(ev).mark_point(filled=True, size=70, color="#22c55e").encode(
            x="episode:Q", y="eval_mean:Q", tooltip=["episode", "eval_mean"]))
    return alt.layer(*layers).properties(height=height, width="container").interactive()


def simple_line(df: pd.DataFrame, col: str, title: str, color: str = "#6b7cff"):
    d = df[["episode", col]].dropna()
    return (alt.Chart(d, title=title).mark_line(color=color)
            .encode(x=alt.X("episode:Q", title="Episode"), y=alt.Y(f"{col}:Q", title=None),
                    tooltip=["episode", col])
            .properties(height=200, width="container").interactive())


def multi_run_chart(frames: dict[str, pd.DataFrame], col: str, ylabel: str):
    parts = []
    for label, df in frames.items():
        d = df[["episode", col]].dropna().copy()
        d["run"] = label
        parts.append(d)
    if not parts:
        return None
    long = pd.concat(parts)
    return (alt.Chart(long).mark_line(strokeWidth=2)
            .encode(x=alt.X("episode:Q", title="Episode"), y=alt.Y(f"{col}:Q", title=ylabel),
                    color=alt.Color("run:N", legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["run", "episode", col])
            .properties(height=340, width="container").interactive())


def runs_table(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for m in runs:
        s = m.get("summary") or {}
        rows.append({
            "run": run_label(m), "variant": VARIANTS.get(m.get("variant"), {}).get("label", m.get("variant")),
            "seed": m.get("seed"), "status": m.get("status"), "episodes": s.get("episodes_run"),
            "solved at ep": s.get("first_solved_episode"), "final eval": s.get("final_eval_mean"),
            "± std": s.get("final_eval_std"), "time (s)": s.get("seconds"),
        })
    df = pd.DataFrame(rows)
    for c in ("episodes", "solved at ep", "final eval", "± std", "time (s)"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def draw_job(job) -> None:
    """Pure display of a (possibly running) training job."""
    cfg, hist = job.cfg, list(job.history)
    st.caption(f"Run: `{cfg.run_name}`")
    if not hist:
        st.info("Collecting warm-up experience…")
        return
    last = hist[-1]
    status = "⏹ stopping…" if job.stopping else ("🟢 training" if job.running else "✔ done")
    st.progress(min(1.0, last["episode"] / cfg.episodes),
                text=f"{status} — episode {last['episode']}/{cfg.episodes}")
    df = pd.DataFrame(hist)
    ev = df["eval_mean"].dropna()
    c = st.columns(5)
    c[0].metric("Last reward", fmt(last["reward"]))
    c[1].metric("Avg (20 ep)", fmt(last["avg20"]))
    c[2].metric("Epsilon", f"{last['epsilon']:.3f}")
    c[3].metric("Greedy eval", fmt(ev.iloc[-1]) if len(ev) else "–")
    c[4].metric("Elapsed", f"{last['time']:.0f}s")
    st.altair_chart(training_chart(df))
    st.caption("🟦 episode reward · 🟧 20-episode average · 🟢 greedy evaluation (used to pick the best model)")


@st.fragment(run_every=1.0)
def live_view(job) -> None:
    """Auto-refreshing fragment; triggers one full rerun when training ends."""
    draw_job(job)
    if not job.running:
        st.rerun()


# ----------------------------------------------------------------------------
# sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎮 RL GameAgent")
    game = st.selectbox("Choose a game", list(GAMES), format_func=lambda g: f"{GAMES[g].emoji}  {g}")
    spec = GAMES[game]
    st.markdown(f"**Goal:** {spec.goal}")
    with st.expander("Game details"):
        st.markdown(f"- **State:** {spec.state}\n- **Actions:** {spec.actions}\n"
                    f"- **Reward:** {spec.reward}\n- **Solved:** {spec.solved_text}")
    st.divider()
    st.caption(f"Device: **{device_name()}**")

runs = list_runs(game)
st.markdown('<p class="hero">🎮 Reinforcement Learning Game Agent</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub">Deep Q-Network family · currently on <b>{game}</b></p>', unsafe_allow_html=True)

tab_over, tab_train, tab_play, tab_stats = st.tabs(["🏠 Overview", "🏋️ Train", "🎮 Play", "📊 Analytics"])

# ============================================================================
# OVERVIEW
# ============================================================================
with tab_over:
    finished = [m for m in runs if m.get("status") in ("finished", "stopped") and m.get("summary")]
    best_eval = max((m["summary"]["final_eval_mean"] for m in finished), default=None)
    k = st.columns(3)
    k[0].metric("Runs on this game", len(runs))
    k[1].metric("Solved runs", sum(1 for m in finished if m["summary"].get("solved")))
    k[2].metric("Best final evaluation", fmt(best_eval))

    st.subheader("How the agent learns")
    st.graphviz_chart(
        """
        digraph G {
          rankdir=LR; bgcolor="transparent";
          node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontcolor="#111", fontname="Helvetica", color="#6b7cff"];
          edge [fontname="Helvetica", fontsize=10, color="#888"];
          Env    [label="Gymnasium\\nenvironment"];
          Agent  [label="DQN agent\\nε-greedy policy net"];
          NStep  [label="n-step\\naccumulator"];
          Buffer [label="Replay buffer\\nuniform / prioritized"];
          Learner[label="Learner\\nDouble-DQN target\\nHuber loss"];
          Target [label="Target net\\n(Polyak τ)"];
          Env -> Agent [label="state"]; Agent -> Env [label="action"];
          Agent -> NStep [label="(s,a,r,s')"]; NStep -> Buffer [label="n-step transitions"];
          Buffer -> Learner [label="batch + IS weights"]; Learner -> Buffer [label="new priorities"];
          Target -> Learner [label="bootstrap"]; Learner -> Target [label="soft update"];
          Learner -> Agent [label="gradient step"];
        }
        """
    )

    st.subheader("Algorithm components")
    st.markdown(
        "| Component | Idea | File |\n|---|---|---|\n"
        "| DQN | Neural Q-function + replay + target network | `agent.py`, `networks.py` |\n"
        "| Double DQN | Online net selects, target net evaluates → less over-estimation | `agent.py` |\n"
        "| Dueling head | Separate V(s) and A(s,a) streams | `networks.py` |\n"
        "| Prioritized replay | Sum-tree sampling ∝ TD-error, importance-sampling correction | `replay_buffer.py` |\n"
        "| N-step returns | Faster reward propagation, per-transition discount γᵏ | `replay_buffer.py` |\n"
        "| Greedy evaluation | Fixed-seed evaluation picks the best checkpoint | `train.py` |\n"
        "| Run tracking | Config + metrics + models saved per run for comparison | `utils.py`, `train.py` |"
    )
    st.subheader("Algorithm variants (ablation presets)")
    st.dataframe(pd.DataFrame([{"variant": v["label"], "description": v["description"]} for v in VARIANTS.values()]),
                 hide_index=True)

# ============================================================================
# TRAIN
# ============================================================================
with tab_train:
    job = st.session_state.get("job")
    running = job is not None and job.running

    t1, t2 = st.columns([2, 3])
    with t1:
        variant = st.selectbox("Algorithm variant", list(VARIANTS), index=len(VARIANTS) - 1,
                               format_func=lambda v: VARIANTS[v]["label"])
    with t2:
        st.markdown("&nbsp;")
        st.caption(VARIANTS[variant]["description"])

    with st.expander("⚙️ Hyper-parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        episodes = c1.slider("Episodes", 20, 1000, int(spec.episodes), 10, key=f"ep_{game}")
        lr = c1.select_slider("Learning rate", [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=spec.lr,
                              format_func=lambda v: f"{v:g}", key=f"lr_{game}")
        gamma = c2.slider("Discount γ", 0.90, 0.999, 0.99, 0.001, format="%.3f", key=f"g_{game}")
        eps_decay = c2.slider("ε decay / episode", 0.90, 0.999, float(spec.eps_decay), 0.001,
                              format="%.3f", key=f"ed_{game}")
        batch_size = c3.select_slider("Batch size", [32, 64, 128, 256], value=64, key=f"bs_{game}")
        hidden = c3.select_slider("Hidden units (×2 layers)", [32, 64, 128, 256], value=128, key=f"h_{game}")
        seed = c3.number_input("Seed", 0, 10_000, 42, key=f"seed_{game}")
        early_stop = st.checkbox("Stop early once solved", value=True, key=f"es_{game}")

    overrides = dict(episodes=episodes, lr=lr, gamma=gamma, eps_decay=eps_decay, batch_size=batch_size,
                     hidden=(hidden, hidden), seed=int(seed), early_stop=early_stop)
    with st.expander("📄 Resolved configuration (saved with the run)"):
        st.json(make_config(game, variant, **overrides).to_dict(), expanded=False)

    b1, b2, _ = st.columns([1, 1, 3])
    start = b1.button("🚀 Start training", type="primary", disabled=running)
    stop = b2.button("⏹ Stop", disabled=not running)

    if start:
        from train import TrainingJob          # lazy: pulls in torch

        job = TrainingJob(make_config(game, variant, **overrides)).start()
        st.session_state["job"] = job
    if stop and job is not None:
        job.stop()

    if job is None:
        st.info("Pick a variant and press **Start training**. Training runs in the background: "
                "you can switch tabs, and the best model is saved as it improves.")
    elif job.running:
        live_view(job)
    else:
        draw_job(job)
        if job.error:
            st.error(f"Training failed: {job.error}")
        elif job.result:
            r = job.result
            msg = (f"✅ Solved at episode {r['first_solved_episode']}" if r["solved"]
                   else f"Training {r['status']} after {r['episodes_run']} episodes (not solved).")
            (st.success if r["solved"] else st.warning)(msg)
            m = st.columns(4)
            m[0].metric("Final eval (best model)", f"{r['final_eval_mean']:.1f}", f"± {r['final_eval_std']:.1f}")
            m[1].metric("Episodes", r["episodes_run"])
            m[2].metric("Env steps", f"{r['total_steps']:,}")
            m[3].metric("Time", f"{r['seconds']:.0f}s")
            st.caption(f"Saved: `{r['best_model']}`")

# ============================================================================
# PLAY
# ============================================================================
with tab_play:
    models = available_models(game)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Settings")
        if not models:
            st.warning("No trained model yet for this game. Train one first — the random agent still works.")
        labels = ["🎲 Random agent"] + [m["label"] for m in models]
        choice = st.selectbox("Agent", labels, index=1 if models else 0)
        model_file = None if choice == labels[0] else models[labels.index(choice) - 1]["path"]
        n_eps = st.slider("Episodes to watch", 1, 10, 1)
        fps = st.slider("Playback speed (frames/sec)", 5, 120, 40)
        seed_play = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)
        run_play = st.button("▶️ Play", type="primary")
    with right:
        st.subheader("Live view")
        frame_ph = st.empty()
        s1, s2, s3 = st.columns(3)
        step_ph, score_ph, ep_ph = s1.empty(), s2.empty(), s3.empty()
        result_ph = st.empty()
        if run_play:
            from play import episode_stream        # lazy: pulls in torch

            scores = []
            for ep in range(1, n_eps + 1):
                for out in episode_stream(game, model_file, seed=int(seed_play) + ep if seed_play else None):
                    frame_ph.image(out["frame"])
                    step_ph.metric("Step", out["step"])
                    score_ph.metric("Score", f"{out['score']:.1f}")
                    ep_ph.metric("Episode", f"{ep}/{n_eps}")
                    time.sleep(1.0 / fps)
                scores.append(out["score"])
            result_ph.success(f"Scores: {', '.join(f'{s:.0f}' for s in scores)}  |  mean = {np.mean(scores):.1f}")
        else:
            frame_ph.info("Press **Play** to watch the agent.")

# ============================================================================
# ANALYTICS
# ============================================================================
with tab_stats:
    if not runs:
        st.info("No runs for this game yet. Train an agent first.")
    else:
        opts = run_options(runs)
        tbl = runs_table(runs)

        st.subheader("🏆 Leaderboard")
        st.dataframe(tbl.sort_values("final eval", ascending=False, na_position="last"),
                     hide_index=True,
                     column_config={"final eval": st.column_config.NumberColumn(format="%.1f"),
                                    "± std": st.column_config.NumberColumn(format="%.1f"),
                                    "time (s)": st.column_config.NumberColumn(format="%.0f")})

        done_tbl = tbl.dropna(subset=["final eval"])
        if len(done_tbl):
            st.subheader("🧪 Ablation summary (aggregated over seeds)")
            agg = done_tbl.groupby("variant", sort=False).agg(
                runs=("run", "count"),
                solved_rate=("solved at ep", lambda x: float(x.notna().mean())),
                episodes_to_solve=("solved at ep", "mean"),
                final_eval_mean=("final eval", "mean"),
                final_eval_std=("final eval", "std"),
            ).reset_index()
            st.dataframe(agg, hide_index=True, column_config={
                "solved_rate": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.0f%%"),
                "episodes_to_solve": st.column_config.NumberColumn(format="%.0f"),
                "final_eval_mean": st.column_config.NumberColumn(format="%.1f"),
                "final_eval_std": st.column_config.NumberColumn(format="%.1f")})
            st.caption("Run `python compare.py` to train several variants × seeds automatically.")

        st.subheader("📈 Compare runs")
        default = list(opts)[:3]
        picked = st.multiselect("Runs", list(opts), default=default)
        metric_name = st.radio("Metric", list(METRICS), horizontal=True)
        col, ylabel = METRICS[metric_name]
        frames = {lab: df for lab in picked if (df := load_run_log(opts[lab]["run_name"])) is not None}
        chart = multi_run_chart(frames, col, ylabel) if frames else None
        if chart is not None:
            st.altair_chart(chart)
        else:
            st.caption("Select at least one run with logged data.")

        st.subheader("🔍 Run details")
        one = st.selectbox("Run", list(opts), key="detail_run")
        meta = opts[one]
        df = load_run_log(meta["run_name"])
        s = meta.get("summary") or {}
        if s:
            d = st.columns(5)
            d[0].metric("Final eval", fmt(s.get("final_eval_mean")), f"± {fmt(s.get('final_eval_std'))}")
            d[1].metric("Solved at ep", s.get("first_solved_episode") or "–")
            d[2].metric("Episodes", s.get("episodes_run", "–"))
            d[3].metric("Env steps", f"{s['total_steps']:,}" if s.get("total_steps") else "–")
            d[4].metric("Status", meta.get("status", "?"))
        if df is not None:
            st.altair_chart(training_chart(df))
            g1, g2 = st.columns(2)
            g1.altair_chart(simple_line(df, "loss", "Loss", "#ef4444"))
            g2.altair_chart(simple_line(df, "mean_q", "Mean Q-value", "#8b5cf6"))
            g3, g4 = st.columns(2)
            g3.altair_chart(simple_line(df, "epsilon", "Epsilon (exploration)", "#0ea5e9"))
            g4.altair_chart(simple_line(df, "beta", "PER beta (IS correction)", "#f59e0b"))
            st.download_button("⬇️ Download metrics CSV", df.to_csv(index=False),
                               f"{meta['run_name']}.csv", "text/csv")
        with st.expander("Configuration of this run"):
            st.json(meta.get("config", {}), expanded=False)

    st.divider()
    st.subheader("📏 Evaluate: trained agent vs random")
    models = available_models(game)
    if not models:
        st.caption("Train a model to enable evaluation.")
    else:
        e1, e2 = st.columns([3, 1])
        eval_label = e1.selectbox("Model", [m["label"] for m in models], key="eval_model")
        n_eval = e2.slider("Episodes", 5, 50, 20, key="eval_eps")
        if st.button("Run evaluation"):
            from play import evaluate               # lazy: pulls in torch

            mf = next(m["path"] for m in models if m["label"] == eval_label)
            with st.spinner("Evaluating…"):
                trained, rand = evaluate(game, mf, n_eval), evaluate(game, None, n_eval)
            r1, r2 = st.columns(2)
            r1.metric("Trained agent (mean)", f"{np.mean(trained):.1f}", f"± {np.std(trained):.1f}")
            r2.metric("Random agent (mean)", f"{np.mean(rand):.1f}", f"± {np.std(rand):.1f}")
            long = pd.DataFrame({"agent": ["Trained"] * n_eval + ["Random"] * n_eval, "score": trained + rand})
            st.altair_chart(alt.Chart(long).mark_boxplot(size=60).encode(
                x=alt.X("agent:N", title=None), y=alt.Y("score:Q", title="Episode score"),
                color=alt.Color("agent:N", legend=None)).properties(height=300, width="container"))
