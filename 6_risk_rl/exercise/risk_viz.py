"""Presentation, environment & quiz helpers for the WE4 notebook 06:
"Risk-Aware RL — the scenic route that was profitable right up until it wasn't".

Same idea as `intro_viz`, `pg_viz` and `ac_viz`: every HTML/CSS illustration,
interactive widget, quiz *answer key* and matplotlib visual lives here, out of
the notebook, so the teaching cells stay about the *idea*.

It also hides two numbers on purpose: **the gorge rim's fall probability and
the undermined ground's fall probability**. The notebook only ever calls
`Trail().step(action)` and looks at what comes back — that is what makes the
exercise honest.

    import risk_viz as rv
    rv.the_briefing()
    env = rv.Trail()

Students are told not to read this file. (You are, presumably, not a student.)
"""
import json as _json

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

# ===========================================================================
#  The mountain vocabulary (kept here so every widget labels things the same)
# ===========================================================================
N_ROWS, N_COLS = 6, 8
N_STATES = N_ROWS * N_COLS              # the 48 map cells (the solver adds one bookkeeping state)
ACTIONS = ["Up", "Right", "Down", "Left"]
ACTION_ARROW = ["↑", "→", "↓", "←"]
N_ACTIONS = 4

START = (5, 0)                          # the village depot, bottom-left
GOAL = (5, 7)                           # the mountain lodge, bottom-right

# ---------------------------------------------------------------------------
#  ONE RULE GOVERNS DANGER ON THIS MAP:
#      a cell is risky  <=>  it shares an edge with a hazard.
#  Hazards are the gorge (a chasm south of the map, under GORGE_COLS only) and
#  the open pits. Stepping into a pit is not a risk, it is a certainty.
#  Everything else follows, including why the depot and the lodge are safe:
#  there is simply no gorge beneath them.
# ---------------------------------------------------------------------------
GORGE_COLS = (2, 3, 4, 5)               # the chasm runs under these columns of the bottom row
PIT_CELLS = frozenset({(3, 2), (3, 5)}) # open sinkholes
SERVICE_ROW = 4                         # the old service road, +1 a stretch

_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # Up, Right, Down, Left


def _touches_gorge(rc):
    return rc[0] == N_ROWS - 1 and rc[1] in GORGE_COLS


def _touches_pit(rc):
    return any((rc[0] + dr, rc[1] + dc) in PIT_CELLS for dr, dc in _DELTAS)


# the panorama path: the gorge rim, where the view (and the tips) are
RIM_CELLS = frozenset((N_ROWS - 1, c) for c in GORGE_COLS)
# undermined ground: everything touching a pit, the pits themselves excluded
UNDERMINED_CELLS = frozenset(
    (r, c) for r in range(N_ROWS) for c in range(N_COLS)
    if (r, c) not in PIT_CELLS and _touches_pit((r, c)))
# every cell the walker can fall from or into: exactly what a shield must mask
DANGER_CELLS = RIM_CELLS | UNDERMINED_CELLS | PIT_CELLS

MOVE = -1.0                             # every move costs effort
ROAD_BONUS = +1.0                       # the old service road: villagers wave, small tips
SCENIC_BONUS = +5.0                     # tourists tip on the panorama (walking east, toward the lodge)
FINISH = +50.0                          # the delivery bonus at the lodge
FALL_DAMAGE = 100.0                     # one fall = cost 100; the reward signal never sees it

_P_RIM = 0.05                           # per step on the gorge rim. Students never see this number.
_P_UNDERMINED = 0.02                    # per step on undermined ground. Or this one.
_MAX_STEPS = 60
GAMMA = 0.99                            # the solver's tiny impatience (breaks ties toward short routes)

C_BLUE, C_RED, C_VIOLET, C_GREEN, C_GRAY = "#4a6cd4", "#d4574e", "#764ba2", "#2e9e7a", "#9aa0b5"
C_AMBER = "#d9932c"


def _s(row, col):
    return row * N_COLS + col


def _rc(s):
    return divmod(int(s), N_COLS)


def _cell_type(rc):
    if rc == GOAL:
        return "goal"
    if rc == START:
        return "start"
    if rc in PIT_CELLS:
        return "pit"
    if rc in RIM_CELLS:
        return "rim"
    if rc in UNDERMINED_CELLS:
        return "undermined"
    if rc[0] == SERVICE_ROW:
        return "road"
    return "forest"


def _cell_bonus(rc, action):
    """Reward bonus for ENTERING cell rc via `action` (on top of the -1 move)."""
    if rc == GOAL:
        return FINISH
    if rc in PIT_CELLS:
        return 0.0
    if rc in RIM_CELLS:
        return SCENIC_BONUS if action == 1 else 0.0   # tourists tip for eastward progress only
    if rc[0] == SERVICE_ROW:
        return ROAD_BONUS                             # holds for undermined road cells too
    return 0.0


def _fall_prob(rc):
    if rc in PIT_CELLS:
        return 1.0                                    # an open hole: not a risk, a certainty
    if rc in RIM_CELLS:
        return _P_RIM
    if rc in UNDERMINED_CELLS:
        return _P_UNDERMINED
    return 0.0


def into_gorge(rc, action):
    """True if this move walks straight off the south edge, where the gorge is.
    Not a risk: a certainty, exactly like stepping into a pit."""
    return int(action) == 2 and rc[0] == N_ROWS - 1 and rc[1] in GORGE_COLS


def next_cell(s, a):
    """Where action `a` from state `s` lands you: the (row, col) of the target cell.
    Walking into a fence keeps you where you are (walking into the gorge is handled
    separately: there is no cell to land on)."""
    r, c = _rc(s)
    dr, dc = _DELTAS[int(a)]
    r2, c2 = r + dr, c + dc
    if not (0 <= r2 < N_ROWS and 0 <= c2 < N_COLS):
        return (r, c)
    return (r2, c2)


class Trail:
    """The mountain between the depot and the lodge. Movement is deterministic;
    the ground next to a hazard is not.

    step(a) -> (s_next, reward, cost, done, info)
      reward : CHF this step (effort, tips, the delivery bonus). BLIND to falls.
      cost   : damage points this step: 100.0 on a fall, else 0.0. Separate from reward.
      info   : {"fell": bool, "timeout": bool, "cell": type of the cell entered}
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.pos = START
        self.t = 0
        return _s(*self.pos)

    def step(self, action):
        self.t += 1
        action = int(action)
        r, c = self.pos
        dr, dc = _DELTAS[action]
        r2, c2 = r + dr, c + dc
        if into_gorge((r, c), action):
            # straight over the edge: no dice to roll, the trip ends here
            reward, cost, done, fell = MOVE, FALL_DAMAGE, True, True
            cell = "gorge"
        elif not (0 <= r2 < N_ROWS and 0 <= c2 < N_COLS):
            # bumped a fence: stay put, burn the effort, roll no dice
            reward, cost, done, fell = MOVE, 0.0, False, False
            cell = "bump"
        else:
            tgt = (r2, c2)
            self.pos = tgt
            reward = MOVE + _cell_bonus(tgt, action)
            cost, done, fell = 0.0, False, False
            cell = _cell_type(tgt)
            if tgt == GOAL:
                done = True
            elif self.rng.random() < _fall_prob(tgt):
                cost, done, fell = FALL_DAMAGE, True, True   # the reward line above does NOT change
        timeout = False
        if self.t >= _MAX_STEPS and not done:
            done, timeout = True, True   # shift over — the clock, not the world
        return _s(*self.pos), reward, cost, done, {"fell": fell, "timeout": timeout, "cell": cell}


# ===========================================================================
#  The MDP tensors and the off-the-shelf solver (Part 2's black box)
# ===========================================================================
_S_DONE = N_STATES          # absorbing bookkeeping state: "the trip is over"
_NS = N_STATES + 1


def mdp_tensors():
    """The mountain as matrices — the planner's food.

    Returns (P, R, C):
      P : (4, 49, 49)  P[a, s, s'] = probability that action a in state s lands in s'
      R : (49, 4)      expected CHF for taking action a in state s   (the money menu)
      C : (49, 4)      expected damage points for the same move      (the danger menu)
    State 48 is a bookkeeping state meaning "the trip is over".
    """
    P = np.zeros((N_ACTIONS, _NS, _NS))
    R = np.zeros((_NS, N_ACTIONS))
    C = np.zeros((_NS, N_ACTIONS))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            s = _s(r, c)
            if (r, c) == GOAL:
                P[:, s, _S_DONE] = 1.0
                continue
            for a, (dr, dc) in enumerate(_DELTAS):
                r2, c2 = r + dr, c + dc
                if into_gorge((r, c), a):
                    P[a, s, _S_DONE] = 1.0
                    R[s, a] = MOVE
                    C[s, a] = FALL_DAMAGE          # certain, not probable
                    continue
                if not (0 <= r2 < N_ROWS and 0 <= c2 < N_COLS):
                    P[a, s, s] = 1.0
                    R[s, a] = MOVE
                    continue
                tgt = (r2, c2)
                R[s, a] = MOVE + _cell_bonus(tgt, a)
                p_fall = _fall_prob(tgt)
                C[s, a] = p_fall * FALL_DAMAGE
                if tgt == GOAL:
                    P[a, s, _S_DONE] = 1.0
                else:
                    P[a, s, _S_DONE] += p_fall
                    P[a, s, _s(*tgt)] += 1.0 - p_fall
    P[:, _S_DONE, _S_DONE] = 1.0
    return P, R, C


def solve_mdp(P, R, mask=None, gamma=GAMMA):
    """Standard policy iteration — the same algorithm any operations-research
    toolbox ships. Feed it the map (P) and a value menu (R); it returns the
    policy that maximises expected discounted total value.

    mask : optional (48, 4) or (49, 4) boolean array; False = "this move is off
           the menu". The solver simply never considers masked moves.

    Note this is *planning*, not learning: it needs the model up front and
    computes V exactly by linear solve. No samples, no episodes, no learning
    rate. The only sampling in the notebook happens in `evaluate`, which audits
    a finished policy.
    """
    if mask is None:
        mask = np.ones((_NS, N_ACTIONS), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] == N_STATES:                       # pad the bookkeeping state
            mask = np.vstack([mask, np.ones((1, N_ACTIONS), dtype=bool)])
    # a state whose every move is masked keeps its least-bad move (unreachable in practice)
    safe_mask = np.where(mask.any(axis=1, keepdims=True), mask, True)
    R_eff = np.where(safe_mask, R, -1e9)
    policy = np.argmax(R_eff, axis=1)
    for _ in range(500):
        P_pi = P[policy, np.arange(_NS), :]
        R_pi = R_eff[np.arange(_NS), policy]
        V = np.linalg.solve(np.eye(_NS) - gamma * P_pi, R_pi)
        Q = R_eff + gamma * np.einsum("asx,x->sa", P, V)
        new_policy = np.argmax(Q, axis=1)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy


def trip_distribution(policy, max_steps=_MAX_STEPS):
    """The EXACT outcome distribution of a policy, computed from the model.

    This is distributional policy evaluation: instead of asking "what does this
    policy earn on average", we push probability through the map and keep the
    whole distribution of what one trip can end up paying and breaking. No
    sampling, no learning — the same model the planner used, asked a richer
    question.

    Returns a dict with
      "money"  : {trip profit  -> probability}
      "damage" : {trip damage  -> probability}
      "mean_money", "mean_damage", "p_fall"
    """
    policy = np.asarray(policy)
    money_dist, damage_dist = {}, {}

    def _record(money, damage, prob):
        m = round(float(money), 6)
        money_dist[m] = money_dist.get(m, 0.0) + prob
        damage_dist[damage] = damage_dist.get(damage, 0.0) + prob

    frontier = {(_s(*START), 0.0): 1.0}          # (state, money so far) -> probability
    for _ in range(max_steps):
        nxt = {}
        for (s, acc), prob in frontier.items():
            rc = _rc(s)
            a = int(policy[s])
            if into_gorge(rc, a):                # a certain fall: the trip ends here
                _record(acc + MOVE, FALL_DAMAGE, prob)
                continue
            tgt = next_cell(s, a)
            earned = acc + MOVE + (0.0 if tgt == rc else _cell_bonus(tgt, a))
            if tgt == GOAL:
                _record(earned, 0.0, prob)
                continue
            p_fall = _fall_prob(tgt)
            if p_fall > 0:
                _record(earned, FALL_DAMAGE, prob * p_fall)
            if p_fall < 1:
                key = (_s(*tgt), round(earned, 6))
                nxt[key] = nxt.get(key, 0.0) + prob * (1.0 - p_fall)
        frontier = nxt
        if not frontier:
            break
    for (s, acc), prob in frontier.items():      # ran out of shift time, still walking
        _record(acc, 0.0, prob)

    mean_money = sum(v * p for v, p in money_dist.items())
    mean_damage = sum(v * p for v, p in damage_dist.items())
    return {"money": money_dist, "damage": damage_dist,
            "mean_money": mean_money, "mean_damage": mean_damage,
            "p_fall": damage_dist.get(FALL_DAMAGE, 0.0)}


def greedy_route(policy, max_steps=_MAX_STEPS):
    """Follow a policy across the map deterministically (nobody falls) and
    return the visited (row, col) list — used for drawing routes."""
    policy = np.asarray(policy)
    rc = START
    path = [rc]
    for _ in range(max_steps):
        a = int(policy[_s(*rc)])
        nxt = next_cell(_s(*rc), a)
        if nxt == rc:
            break
        rc = nxt
        path.append(rc)
        if rc == GOAL:
            break
    return path


def route_name(policy):
    """A human name for where a policy actually walks."""
    cells = set(greedy_route(policy))
    if cells & set(RIM_CELLS):
        return "the panorama path"
    if cells & set(UNDERMINED_CELLS | PIT_CELLS):
        return "the service road"
    if GOAL in cells:
        return "the forest detour"
    return "somewhere odd (never reaches the lodge)"


ROUTE_COLORS = {"the panorama path": C_RED, "the service road": C_AMBER,
                "the forest detour": C_BLUE}


# ===========================================================================
#  Quiz machinery (identical mechanics to the other WE4 notebooks)
# ===========================================================================
def _card(inner, maxw=860):
    display(HTML(
        '<div style="font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid '
        '#e6e8ee;border-radius:14px;padding:18px;max-width:%dpx;background:#fff;'
        'color:#1b1b1b;color-scheme:light">%s</div>'
        % (maxw, inner)))


def _mc_render(title, question, options, answer_index, reveal):
    data = {"opts": list(options), "ans": int(answer_index), "reveal": reveal}
    uid = "mc_" + str(abs(hash((question, tuple(options), answer_index))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .mc-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:4px}
#__UID__ .mc-q{color:#444;font-size:13.5px;margin-bottom:12px;line-height:1.55}
#__UID__ .mc-opt{color:#1b1b1b;display:flex;align-items:flex-start;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-size:13.5px;line-height:1.5;transition:.12s}
#__UID__ .mc-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .mc-dot{width:16px;height:16px;border-radius:50%;border:2px solid #c2c7da;flex:0 0 auto;margin-top:2px}
#__UID__ .mc-txt{flex:1;min-width:0}
#__UID__ .mc-opt code{background:#f3f0ff;border-radius:5px;padding:1px 5px;font-size:12.5px}
#__UID__ .mc-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .mc-opt.sel .mc-dot{background:#764ba2;border-color:#764ba2}
#__UID__ .mc-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .mc-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .mc-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .mc-rev{font-size:13px;color:#2c2350;margin-top:10px;min-height:18px;line-height:1.6}
</style>
<div id="__UID__">
  <div class="mc-head">__TITLE__</div>
  <div class="mc-q">__Q__</div>
  <div class="mc-list"></div>
  <button class="mc-btn">Check my answer</button>
  <div class="mc-rev"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  let idx=D.opts.map((_,i)=>i);
  for(let i=idx.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  const list=root.querySelector(".mc-list");
  idx.forEach(orig=>{
    const o=document.createElement("div"); o.className="mc-opt"; o.dataset.i=orig;
    o.innerHTML='<span class="mc-dot"></span><span class="mc-txt">'+D.opts[orig]+'</span>';
    list.appendChild(o);
  });
  const opts=list.querySelectorAll(".mc-opt"); let sel=null;
  opts.forEach(o=>o.addEventListener("click",()=>{
    sel=+o.dataset.i; opts.forEach(x=>x.classList.remove("sel","ok","no")); o.classList.add("sel");
    root.querySelector(".mc-rev").textContent="";
  }));
  root.querySelector(".mc-btn").addEventListener("click",()=>{
    if(sel===null){root.querySelector(".mc-rev").textContent="Pick an option first!";return;}
    opts.forEach(o=>{const i=+o.dataset.i; o.classList.remove("sel");
      if(i===D.ans)o.classList.add("ok"); else if(i===sel)o.classList.add("no");});
    root.querySelector(".mc-rev").innerHTML=(sel===D.ans?"✅ Correct. ":"❌ Not quite. ")+D.reveal;
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__Q__", question).replace("__DATA__", _json.dumps(data)))
    display(HTML(html))


def _tf_render(title, statements,
               prompt="Click every statement you think is TRUE, then check."):
    items = [{"t": t, "ok": bool(v)} for t, v in statements]
    uid = "tf_" + str(abs(hash((title, tuple(t for t, _ in statements)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .tf-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:4px}
#__UID__ .tf-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .tf-opt{color:#1b1b1b;display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:9px 12px;margin-bottom:7px;cursor:pointer;font-size:13.5px;line-height:1.5}
#__UID__ .tf-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .tf-box{width:16px;height:16px;border-radius:4px;border:2px solid #c2c7da;flex:0 0 auto}
#__UID__ .tf-txt{flex:1;min-width:0}
#__UID__ .tf-opt.sel{border-color:#764ba2;background:#f1edff}
#__UID__ .tf-opt.sel .tf-box{background:#764ba2;border-color:#764ba2}
#__UID__ .tf-opt.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .tf-opt.no{border-color:#e07a7a;background:#fdecec}
#__UID__ .tf-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .tf-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="tf-head">__TITLE__</div>
  <div class="tf-sub">__PROMPT__</div>
  <div class="tf-list"></div>
  <button class="tf-btn">Check</button>
  <div class="tf-status"></div>
</div>
<script>
(function(){
  let DATA=__DATA__.slice();
  for(let i=DATA.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[DATA[i],DATA[j]]=[DATA[j],DATA[i]];}
  const root=document.getElementById("__UID__"), list=root.querySelector(".tf-list");
  DATA.forEach((d,i)=>{
    const row=document.createElement("div"); row.className="tf-opt"; row.dataset.i=i;
    row.innerHTML='<span class="tf-box"></span><span class="tf-txt">'+d.t+'</span>';
    row.addEventListener("click",()=>{row.classList.remove("ok","no");row.classList.toggle("sel");});
    list.appendChild(row);
  });
  root.querySelector(".tf-btn").addEventListener("click",()=>{
    let right=0; const rows=list.querySelectorAll(".tf-opt");
    rows.forEach(r=>{
      const d=DATA[+r.dataset.i], picked=r.classList.contains("sel");
      r.classList.remove("ok","no");
      if(picked===d.ok)right++; else r.classList.add("no");
      if(d.ok)r.classList.add("ok");
    });
    root.querySelector(".tf-status").textContent =
      right+" / "+DATA.length+" correct"+(right===DATA.length?" 🎉":" — green = actually true.");
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__PROMPT__", prompt).replace("__DATA__", _json.dumps(items)))
    display(HTML(html))


def _nq_render(title, questions,
               prompt="Work each number out, type it in, then check."):
    """questions: list of (question_html, answer_number, tolerance, reveal)."""
    items = [{"q": q, "a": float(a), "tol": float(tol), "rev": rev}
             for q, a, tol, rev in questions]
    uid = "nq_" + str(abs(hash((title, tuple(q for q, _, _, _ in questions)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .nq-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:4px}
#__UID__ .nq-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .nq-row{color:#1b1b1b;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;font-size:13.5px;line-height:1.55}
#__UID__ .nq-row.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .nq-row.no{border-color:#e07a7a;background:#fdecec}
#__UID__ input{width:110px;padding:5px 8px;border:1px solid #c2c7da;border-radius:7px;font-size:13px;margin-top:6px;background:#fff;color:#1b1b1b}
#__UID__ .nq-rev{font-size:12.5px;color:#3b2d6b;margin-top:6px;display:none}
#__UID__ .nq-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .nq-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="nq-head">__TITLE__</div>
  <div class="nq-sub">__PROMPT__</div>
  <div class="nq-list"></div>
  <button class="nq-btn">Check</button>
  <div class="nq-status"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__"), list=root.querySelector(".nq-list");
  D.forEach((d,i)=>{
    const row=document.createElement("div"); row.className="nq-row"; row.dataset.i=i;
    row.innerHTML=d.q+'<br><input type="text" placeholder="your answer">'
      +'<div class="nq-rev">'+d.rev+'</div>';
    list.appendChild(row);
  });
  root.querySelector(".nq-btn").addEventListener("click",()=>{
    let right=0;
    list.querySelectorAll(".nq-row").forEach(r=>{
      const d=D[+r.dataset.i], v=parseFloat(r.querySelector("input").value.replace(",","."));
      r.classList.remove("ok","no");
      const good=!isNaN(v)&&Math.abs(v-d.a)<=d.tol;
      r.classList.add(good?"ok":"no"); if(good)right++;
      r.querySelector(".nq-rev").style.display=good?"none":"block";
    });
    root.querySelector(".nq-status").textContent=
      right+" / "+D.length+" correct"+(right===D.length?" 🎉":" — the hints under the red ones should help.");
  });
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__PROMPT__", prompt).replace("__DATA__", _json.dumps(items)))
    display(HTML(html))


# ===========================================================================
#  §1  The story cards
# ===========================================================================
def _briefing_html():
    return '''
<div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
  <div style="font-size:44px;line-height:1">🤖</div>
  <div style="flex:1;min-width:280px">
    <div style="font-weight:800;font-size:16px;margin-bottom:6px;color:#1b1b1b">Your job at GaGGle Alpine</div>
    <div style="font-size:13.5px;color:#444;line-height:1.65">
      You have just taken over operations. Your predecessor's routing software maximised average
      profit per trip, so it sent the robot along the panorama path on the gorge rim, where
      tourists tip well. The quarter was a record. Then the rim gave way, the robot went into the
      gorge, and the insurer stepped in.<br><br>
      Your brief has two parts: keep the deliveries profitable, and honour the new insurance
      contract, which caps the <b>expected damage per trip</b>. Two numbers to manage, not one.
      Everything in this notebook follows from taking that seriously.
    </div>
  </div>
</div>'''


def the_briefing():
    """The story and the map, in one card: who you are, and where you are."""
    divider = ('<div style="border-top:1px solid #e6e8ee;margin:16px 0 14px 0"></div>')
    _card(_briefing_html() + divider + _trail_map_html(), maxw=900)


def _cell_visual(rc):
    """(emoji, background) for one map cell, shared by the HTML map and the animation."""
    t = _cell_type(rc)
    return {
        "start":      ("🏠", "#eef0f7"),
        "goal":       ("🏁", "#e7f7ec"),
        "pit":        ("🕳️", "#e2dbd2"),
        "rim":        ("🌅", "#fdf1ea"),
        "undermined": ("⚠️", "#fbf0dd"),
        "road":       ("🌼", "#f4f8e3"),
        "forest":     ("🌲", "#eaf2e6"),
    }[t]


def _trail_map_html():
    """The mountain as HTML: the gorge along part of the south edge, two open pits
    inland, and the undermined ground that surrounds them."""
    def cell(txt, bg, extra=""):
        return ('<td style="width:44px;height:40px;text-align:center;font-size:17px;'
                'background:%s;border:1px solid #e2e5ef;border-radius:8px;%s">%s</td>'
                % (bg, extra, txt))
    rows_html = []
    for r in range(N_ROWS):
        row = []
        for c in range(N_COLS):
            e, bg = _cell_visual((r, c))
            extra = "box-shadow:inset 0 0 0 2px #e8b661;" if (r, c) in UNDERMINED_CELLS else ""
            row.append(cell(e, bg, extra))
        rows_html.append("<tr>%s</tr>" % "".join(row))
    # the gorge is NOT a row of cells: it is terrain beyond the map's south edge,
    # drawn as one thin strip under the columns it runs beneath
    gorge = ('<tr><td colspan="%d" style="padding:0"></td>'
             '<td colspan="%d" style="padding:0"><div style="height:15px;border-radius:0 0 9px 9px;'
             'background:linear-gradient(#bcd4f0,#9dbfe8);margin-top:1px;text-align:center;'
             'font-size:10px;color:#3d6395;font-style:italic;line-height:15px">'
             'the gorge</div></td>'
             '<td colspan="%d" style="padding:0"></td></tr>'
             % (GORGE_COLS[0], len(GORGE_COLS), N_COLS - GORGE_COLS[-1] - 1))

    def legend_panel(heading, note, entries):
        rows = "".join(
            '<tr><td style="width:26px;font-size:15px;padding:3px 6px 3px 0">%s</td>'
            '<td style="font-size:12.5px;color:#1b1b1b;font-weight:600;padding:3px 12px 3px 0;'
            'white-space:nowrap">%s</td>'
            '<td style="font-size:12.5px;color:#444;padding:3px 0">%s</td></tr>' % e
            for e in entries)
        return ('<div style="flex:1;min-width:270px;border:1px solid #e6e8ee;border-radius:10px;'
                'padding:10px 14px">'
                '<div style="font-size:11px;font-weight:800;letter-spacing:.5px;color:#8a8f9c;'
                'margin-bottom:5px">%s</div>'
                '<table style="border-collapse:collapse">%s</table>'
                '<div style="font-size:11.5px;color:#666;border-top:1px solid #eef0f5;'
                'margin-top:7px;padding-top:6px">%s</div></div>'
                % (heading, rows, note))

    rewards = legend_panel(
        "REWARD, PER CELL YOU STEP ON", "…and every move costs <b>−1</b>",
        [("🌲", "forest", "0"),
         ("🌼", "service road", "+1"),
         ("🌅", "panorama path", "+5 when heading east"),
         ("🏁", "lodge", "+50, delivery done")])
    danger = legend_panel(
        "DANGER — ANY FALL: COST 100, TRIP OVER",
        "A cell is dangerous when it <b>touches a hazard</b>: the gorge or a pit. "
        "Nothing touches the depot or the lodge, so they are safe.",
        [("🕳️", "open pit", "falling is certain"),
         ("🌅", "gorge rim", "can give way"),
         ("⚠️", "undermined ground", "can collapse (reward unchanged)")])

    return '''
<div style="font-weight:800;font-size:15px;margin-bottom:2px;color:#1b1b1b">🗺️ The mountain</div>
<div style="font-size:13px;color:#555;margin-bottom:10px">
  An 8×6 map. The robot starts at the <b>depot</b> 🏠 and must reach the <b>lodge</b> 🏁.
  Actions: <b>Up · Right · Down · Left</b>, one cell per move.</div>
<table style="border-collapse:separate;border-spacing:3px;color:#1b1b1b">%s</table>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">%s%s</div>
<div style="font-size:12px;color:#666;margin-top:9px">
  The two kinds of dangerous ground are <b>not</b> equally dangerous: §1.1 gives both
  probabilities and works out what each route through them costs.</div>''' \
        % ("".join(rows_html) + gorge, rewards, danger)


def trail_map():
    """The mountain on its own, without the briefing above it."""
    _card(_trail_map_html(), maxw=900)


# ===========================================================================
#  §2  Matplotlib visuals for the mountain
# ===========================================================================
_TYPE_COLORS = {"start": "#eef0f7", "goal": "#e7f7ec", "rim": "#fdf1ea",
                "pit": "#e2dbd2", "undermined": "#fbf0dd", "road": "#f4f8e3",
                "forest": "#eaf2e6"}


def _draw_grid(ax):
    for r in range(N_ROWS):
        for c in range(N_COLS):
            y = N_ROWS - 1 - r
            ax.add_patch(plt.Rectangle((c, y), 1, 1,
                                       facecolor=_TYPE_COLORS[_cell_type((r, c))],
                                       edgecolor="#d8dbe4", linewidth=1))
            if (r, c) in UNDERMINED_CELLS:       # the ring: hatched, so it reads in print too
                ax.add_patch(plt.Rectangle((c, y), 1, 1, facecolor="none", edgecolor="#e0a13c",
                                           linewidth=1.4, hatch="///", zorder=2, alpha=0.75))
            if (r, c) in PIT_CELLS:
                ax.add_patch(plt.Circle((c + 0.5, y + 0.5), 0.26, facecolor="#4a3f33",
                                        edgecolor="#2f2823", linewidth=1.2, zorder=3))
    # the gorge: only under the columns it runs beneath
    ax.add_patch(plt.Rectangle((GORGE_COLS[0], -0.55), len(GORGE_COLS), 0.55,
                               facecolor="#c9ddf7", edgecolor="none"))
    ax.text(GORGE_COLS[0] + len(GORGE_COLS) / 2, -0.28, "the gorge", ha="center",
            va="center", fontsize=8.5, color="#5b82b5", style="italic")
    ax.text(START[1] + 0.5, N_ROWS - 1 - START[0] + 0.5, "DEPOT", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="#fff", ec="#999"))
    ax.text(GOAL[1] + 0.5, N_ROWS - 1 - GOAL[0] + 0.5, "LODGE", ha="center", va="center",
            fontsize=7, fontweight="bold", color="#1c6b45",
            bbox=dict(boxstyle="round,pad=0.2", fc="#e7f7ec", ec="#2e9e7a"))
    # name the three bands on the map itself
    ax.text(N_COLS + 0.15, 0.5, "PANORAMA PATH\n+5 per eastward stretch\nthe gorge rim",
            fontsize=8, va="center", ha="left", color="#a33d36", linespacing=1.35)
    ax.text(N_COLS + 0.15, 1.5, "OLD SERVICE ROAD\n+1 a stretch", fontsize=8,
            va="center", ha="left", color="#8a5a08", linespacing=1.35)
    ax.text(N_COLS + 0.15, 2.5, "THE PITS\nhatched = undermined", fontsize=8,
            va="center", ha="left", color="#7a5a2a", linespacing=1.35)
    ax.text(N_COLS + 0.15, 4.5, "FOREST\nquiet · slow · safe", fontsize=8,
            va="center", ha="left", color="#3f6b4f", linespacing=1.35)
    ax.set_xlim(-0.05, N_COLS + 3.0)
    ax.set_ylim(-0.6, N_ROWS + 0.05)
    ax.set_aspect("equal")
    ax.axis("off")


def _plain_grid(ax, face="#f2f4f8", index_labels=True):
    """A neutral grid with row/column indices, used by the definition figures."""
    for r in range(N_ROWS):
        for c in range(N_COLS):
            ax.add_patch(plt.Rectangle((c, N_ROWS - 1 - r), 1, 1, facecolor=face,
                                       edgecolor="#d8dbe4", linewidth=1))
    if index_labels:
        for r in range(N_ROWS):
            ax.text(-0.28, N_ROWS - 1 - r + 0.5, str(r), ha="center", va="center",
                    fontsize=8, color="#8a8f9c")
        for c in range(N_COLS):
            ax.text(c + 0.5, N_ROWS + 0.22, str(c), ha="center", va="center",
                    fontsize=8, color="#8a8f9c")
        ax.text(-0.28, N_ROWS + 0.22, "row", ha="center", va="center", fontsize=7,
                color="#b0b4c0", style="italic")
    ax.set_xlim(-0.7, N_COLS + 0.1)
    ax.set_ylim(-0.35, N_ROWS + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_states_actions():
    """S and A, drawn: the indexed grid, and the four moves available in a cell."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.9, 1]})

    ax = axes[0]
    _plain_grid(ax)
    hl = (2, 4)
    ax.add_patch(plt.Rectangle((hl[1], N_ROWS - 1 - hl[0]), 1, 1, facecolor="#ece5fa",
                               edgecolor=C_VIOLET, linewidth=2, zorder=3))
    ax.annotate("one state:  s = (row 2, col 4)",
                xy=(hl[1] + 0.5, N_ROWS - 1 - hl[0] + 0.95), xytext=(1.2, N_ROWS + 1.05),
                fontsize=9.5, color=C_VIOLET, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=C_VIOLET, lw=1.6))
    for rc, lab in ((START, "depot"), (GOAL, "lodge")):
        ax.text(rc[1] + 0.5, N_ROWS - 1 - rc[0] + 0.5, lab, ha="center", va="center",
                fontsize=7, color="#666")
    ax.set_ylim(-0.35, N_ROWS + 1.5)
    ax.set_title("S — one state per cell:  48 states", fontsize=10.5, loc="left",
                 fontweight="bold")

    ax2 = axes[1]
    ax2.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, facecolor="#f2f4f8",
                                edgecolor="#c9cdd8", linewidth=1.4))
    ax2.plot(0, 0, "o", color="#1b1b1b", markersize=9)
    for (dx, dy), name in (((0, 1), "Up"), ((1, 0), "Right"), ((0, -1), "Down"), ((-1, 0), "Left")):
        ax2.annotate("", xy=(dx * 1.15, dy * 1.15), xytext=(dx * 0.42, dy * 0.42),
                     arrowprops=dict(arrowstyle="-|>", color=C_VIOLET, lw=2.2))
        ax2.text(dx * 1.45, dy * 1.45, name, ha="center", va="center", fontsize=9.5,
                 color="#1b1b1b")
    ax2.text(0, -2.05, "the same four everywhere;\nat the map's edge a blocked move\nleaves the robot where it is",
             ha="center", va="center", fontsize=8, color="#666", linespacing=1.4)
    ax2.set_xlim(-2.2, 2.2)
    ax2.set_ylim(-2.6, 2.0)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("A — the four actions", fontsize=10.5, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.show()


def fig_transitions():
    """P, drawn: the only three shapes a move can have on this mountain."""
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))

    def box(ax, x, y, text, fc, ec, w=2.6, h=0.72, fs=9):
        ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, edgecolor=ec,
                                   linewidth=1.3, zorder=2,
                                   joinstyle="round"))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color="#1b1b1b", zorder=3)

    def arrow(ax, x0, y0, x1, y1, label, color="#555"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
        ax.text((x0 + x1) / 2 + 0.55, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=9, color=color, fontweight="bold")

    for ax in axes:
        ax.set_xlim(-2.6, 3.4)
        ax.set_ylim(-0.4, 4.1)
        ax.axis("off")

    ax = axes[0]
    box(ax, 0, 3.4, "in state $s$, do $a$", "#f2f4f8", "#c9cdd8")
    arrow(ax, 0, 3.0, 0, 1.2, "prob. 1")
    box(ax, 0, 0.7, "the next cell $s'$", "#e7f7ec", C_GREEN)
    ax.set_title("onto ordinary ground", fontsize=10, loc="left", fontweight="bold")

    ax = axes[1]
    box(ax, 0, 3.4, "in state $s$, do $a$", "#f2f4f8", "#c9cdd8")
    arrow(ax, -0.5, 3.0, -1.1, 1.2, "$1-p$")
    box(ax, -1.1, 0.7, "the next cell $s'$", "#e7f7ec", C_GREEN, w=2.4)
    arrow(ax, 0.6, 3.0, 1.9, 1.2, "$p$", color=C_RED)
    box(ax, 1.9, 0.7, "the gorge:\ntrip over", "#fdeeec", C_RED, w=2.0, h=1.0)
    ax.set_title("onto dangerous ground", fontsize=10, loc="left", fontweight="bold")

    ax = axes[2]
    box(ax, 0, 3.4, "in state $s$, do $a$", "#f2f4f8", "#c9cdd8")
    arrow(ax, 0, 3.0, 0, 1.4, "prob. 1", color=C_RED)
    box(ax, 0, 0.8, "the gorge:\ntrip over", "#fdeeec", C_RED, w=2.0, h=1.0)
    ax.set_title("into a pit, or south off the rim", fontsize=10, loc="left", fontweight="bold")

    fig.suptitle("P — every move on this mountain has one of three shapes "
                 "(the values of $p$ live in the simulator)", fontsize=10.5, y=1.02)
    plt.tight_layout()
    plt.show()


def fig_reward_cost():
    """r and c, drawn: the same map twice, labeled by what stepping on a cell
    pays and by what it can cost."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))

    def gorge_strip(ax):
        ax.add_patch(plt.Rectangle((GORGE_COLS[0], -0.42), len(GORGE_COLS), 0.42,
                                   facecolor="#c9ddf7", edgecolor="none"))
        ax.text(GORGE_COLS[0] + len(GORGE_COLS) / 2, -0.21, "the gorge", ha="center",
                va="center", fontsize=7.5, color="#5b82b5", style="italic")

    # ---- reward panel ----
    ax = axes[0]
    reward_face = {0.0: "#eef1f5", ROAD_BONUS: "#f4f8e3", SCENIC_BONUS: "#fdf1ea",
                   FINISH: "#e7f7ec"}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rc = (r, c)
            bonus = FINISH if rc == GOAL else _cell_bonus(rc, 1)   # east entry: the generous case
            ax.add_patch(plt.Rectangle((c, N_ROWS - 1 - r), 1, 1,
                                       facecolor=reward_face.get(bonus, "#eef1f5"),
                                       edgecolor="#d8dbe4", linewidth=1))
            if rc == START:
                lab, col, fs = "start", "#666", 7
            elif rc == GOAL:
                lab, col, fs = "+50", "#1c6b45", 8.5
            elif bonus == SCENIC_BONUS:
                lab, col, fs = "+5 →", "#a35d2a", 8
            elif bonus == ROAD_BONUS:
                lab, col, fs = "+1", "#6b7a2a", 8
            else:
                lab, col, fs = "0", "#9aa0b5", 8
            ax.text(c + 0.5, N_ROWS - 1 - r + 0.5, lab, ha="center", va="center",
                    fontsize=fs, color=col, fontweight="bold")
    gorge_strip(ax)
    ax.set_xlim(-0.1, N_COLS + 0.1)
    ax.set_ylim(-0.5, N_ROWS + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("r — what stepping on each cell pays (CHF)", fontsize=10.5, loc="left",
                 fontweight="bold")
    ax.text(N_COLS / 2, -1.0, "plus −1 for every move\nthe +5 is paid only entering eastward (→)",
            ha="center", va="top", fontsize=8, color="#666", linespacing=1.4)
    ax.set_ylim(-1.9, N_ROWS + 0.1)

    # ---- cost panel ----
    ax2 = axes[1]
    for r in range(N_ROWS):
        for c in range(N_COLS):
            rc = (r, c)
            p = _fall_prob(rc)
            if rc in PIT_CELLS:
                face, lab, col = "#f3c1bc", "100", "#7c2a24"
            elif p > 0:
                face, lab, col = "#fdeeec", "0/100", "#a33d36"
            else:
                face, lab, col = "#eef1f5", "0", "#9aa0b5"
            ax2.add_patch(plt.Rectangle((c, N_ROWS - 1 - r), 1, 1, facecolor=face,
                                        edgecolor="#e0a49e" if p > 0 else "#d8dbe4",
                                        linewidth=1.5 if p > 0 else 1))
            ax2.text(c + 0.5, N_ROWS - 1 - r + 0.5, lab, ha="center", va="center",
                     fontsize=6.8 if lab == "0/100" else 8, color=col, fontweight="bold")
    gorge_strip(ax2)
    ax2.set_xlim(-0.1, N_COLS + 0.1)
    ax2.set_ylim(-0.5, N_ROWS + 0.1)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("c — what stepping on each cell can cost (damage points)", fontsize=10.5,
                  loc="left", fontweight="bold")
    ax2.text(N_COLS / 2, -1.0,
             "14 dangerous cells: 4 rim + 2 pits (certain) + 8 undermined\n"
             "walking south off the rim is a certain fall too",
             ha="center", va="top", fontsize=8, color="#666", linespacing=1.4)
    ax2.set_ylim(-1.9, N_ROWS + 0.1)
    plt.tight_layout()
    plt.show()


def fig_tensors():
    """P, R and C as the planner sees them: one move, looked up in three tables."""
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5),
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    hl_row, hl_col = 2, 1                       # the display row for s = 42, column "Right"
    row_labels = ["s = 0", "⋮", "s = 42", "⋮", "s = 48"]
    col_labels = ["Up", "Right", "Down", "Left"]

    def table(ax, title, unit, value, color, face):
        w, h = 1.0, 0.62
        for i, rlab in enumerate(row_labels):
            y = -i * h
            ax.text(-0.25, y + h / 2, rlab, ha="right", va="center", fontsize=8,
                    color="#1b1b1b" if i == hl_row else "#9aa0b5",
                    fontweight="bold" if i == hl_row else "normal")
            for j in range(len(col_labels)):
                on = (i == hl_row and j == hl_col)
                ax.add_patch(plt.Rectangle((j * w, y), w, h,
                                           facecolor=face if on else "#f4f5f9",
                                           edgecolor=color if on else "#e2e5ef",
                                           linewidth=1.8 if on else 0.9, zorder=2 if on else 1))
                if on:
                    ax.text(j * w + w / 2, y + h / 2, value, ha="center", va="center",
                            fontsize=10.5, fontweight="bold", color=color, zorder=3)
                elif rlab == "⋮":
                    ax.text(j * w + w / 2, y + h / 2, "⋮", ha="center", va="center",
                            fontsize=9, color="#c2c6d2")
                else:
                    ax.text(j * w + w / 2, y + h / 2, "·", ha="center", va="center",
                            fontsize=9, color="#c2c6d2")
        for j, clab in enumerate(col_labels):
            ax.text(j * w + w / 2, h * 0.35, clab, ha="center", va="bottom", fontsize=8,
                    color="#1b1b1b" if j == hl_col else "#8a8f9c",
                    fontweight="bold" if j == hl_col else "normal")
        ax.text(0, -len(row_labels) * h - 0.22, unit, fontsize=9, color="#555")
        ax.set_xlim(-1.6, len(col_labels) * w + 0.15)
        ax.set_ylim(-len(row_labels) * h - 0.55, h * 1.25)
        ax.axis("off")
        ax.set_title(title, fontsize=10.5, loc="left", fontweight="bold", color=color)

    # ---- P: not a table of numbers but a table of destinations ----
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0.55, 2.05), 2.1, 0.62, facecolor="#f2f4f8",
                               edgecolor="#c9cdd8", linewidth=1.3))
    ax.text(1.6, 2.36, "in s = 42, go Right", ha="center", va="center", fontsize=9.5,
            color="#1b1b1b")
    ax.annotate("", xy=(0.55, 1.15), xytext=(1.25, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.8))
    ax.text(0.42, 1.62, "95%", ha="right", va="center", fontsize=9, color=C_GREEN,
            fontweight="bold")
    ax.add_patch(plt.Rectangle((-0.6, 0.5), 2.0, 0.62, facecolor="#e7f7ec",
                               edgecolor=C_GREEN, linewidth=1.3))
    ax.text(0.4, 0.81, "s' = 43", ha="center", va="center", fontsize=9.5, color="#1b1b1b")
    ax.annotate("", xy=(2.75, 1.15), xytext=(1.95, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=1.8))
    ax.text(2.85, 1.62, "5%", ha="left", va="center", fontsize=9, color=C_RED,
            fontweight="bold")
    ax.add_patch(plt.Rectangle((1.85, 0.5), 2.0, 0.62, facecolor="#fdeeec",
                               edgecolor=C_RED, linewidth=1.3))
    ax.text(2.85, 0.81, "trip over", ha="center", va="center", fontsize=9.5, color="#1b1b1b")
    ax.text(-0.6, -0.12, "one 49 × 49 table per action:\nwhere the move can land you",
            fontsize=9, color="#555", va="top", linespacing=1.5)
    ax.set_xlim(-1.0, 4.1)
    ax.set_ylim(-1.15, 3.0)
    ax.axis("off")
    ax.set_title("P — the map", fontsize=10.5, loc="left", fontweight="bold", color="#1b1b1b")

    table(axes[1], "R — the money table", "CHF earned, on average", "+4.0", C_GREEN, "#e7f7ec")
    table(axes[2], "C — the damage table", "damage points, on average", "5.0", C_RED, "#fdeeec")

    fig.suptitle("the same move looked up in all three tables: from the rim cell s = 42, step east",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()


_ROUTE_SPECS = {"forest": (4.0, "the forest detour", C_BLUE),
                "service": (1.5, "the old service road", C_AMBER),
                "panorama": (0.0, "the panorama path", C_RED)}


def fig_route_budgets(which=("forest", "service", "panorama"), budget=5.0, title=None,
                      show_math=True):
    """Where a route's damage number comes from: the route drawn on the map, its
    risky crossings circled, and the arithmetic spelled out underneath.

    which     : any subset of "forest" / "service" / "panorama", in display order.
    show_math : False drops the step-by-step arithmetic and keeps only the route,
                its headline numbers and the verdict against the cap — use it
                where the compounding has already been worked out once.
    """
    P_t, R_t, C_t = mdp_tensors()
    specs = [_ROUTE_SPECS[k] for k in which]
    n = len(specs)
    if show_math:
        figsize = (4.0 * n, 4.9) if n > 1 else (7.4, 6.0)
    else:                                  # no arithmetic underneath: much less room needed
        figsize = (4.0 * n, 3.9) if n > 1 else (7.0, 4.3)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, (lam, name, color) in zip(axes, specs):
        policy = solve_mdp(P_t, R_t - lam * C_t)
        path = greedy_route(policy)
        stats = trip_distribution(policy)
        crossed = [rc for rc in path[1:] if _fall_prob(rc) > 0]

        for r in range(N_ROWS):
            for c in range(N_COLS):
                rc = (r, c)
                p = _fall_prob(rc)
                face = "#fdeeec" if p > 0 else "#f2f4f8"
                ax.add_patch(plt.Rectangle((c, N_ROWS - 1 - r), 1, 1, facecolor=face,
                                           edgecolor="#dde0e8", linewidth=0.8))
        ax.add_patch(plt.Rectangle((GORGE_COLS[0], -0.34), len(GORGE_COLS), 0.34,
                                   facecolor="#c9ddf7", edgecolor="none"))
        for rc, lab in ((START, "start"), (GOAL, "lodge")):
            ax.text(rc[1] + 0.5, N_ROWS - 1 - rc[0] + 0.5, lab, ha="center", va="center",
                    fontsize=6.5, color="#444", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none", alpha=0.9))

        xs = [c + 0.5 for _, c in path]
        ys = [N_ROWS - 1 - r + 0.5 for r, _ in path]
        ax.plot(xs, ys, color=color, linewidth=3.0, solid_capstyle="round", zorder=5)
        for rc in crossed:                       # ring every risky cell the route enters
            ax.add_patch(plt.Circle((rc[1] + 0.5, N_ROWS - 1 - rc[0] + 0.5), 0.40,
                                    facecolor="none", edgecolor="#a33d36", linewidth=2,
                                    zorder=6))
            ax.text(rc[1] + 0.5, N_ROWS - 1 - rc[0] + 0.5, f"{_fall_prob(rc):.0%}",
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="#a33d36", zorder=7)

        if crossed:
            p = _fall_prob(crossed[0])
            k = len(crossed)
            survive = (1 - p) ** k
            fall = 1 - survive
            lines = [f"crosses {k} risky cells, each {p:.0%}",
                     f"stays dry: {1 - p:.2f}$^{k}$ = {survive:.2%}",
                     f"falls: 100% − {survive:.2%} = {fall:.2%}",
                     f"damage: 100 × {fall:.4f} = {100 * fall:.2f}"]
        else:
            fall = 0.0
            lines = ["crosses no risky cells",
                     "stays dry: 100%",
                     "falls: 0%",
                     "damage: 100 × 0 = 0.00"]
        step = 0.085 if n > 1 else 0.105       # a single wide panel needs looser line spacing
        if show_math:
            for j, line in enumerate(lines):
                weight = "bold" if j == len(lines) - 1 else "normal"
                ax.text(0.02, -0.10 - step * j, line, transform=ax.transAxes,
                        fontsize=9 if n > 1 else 10.5,
                        color="#1b1b1b" if j < len(lines) - 1 else color, fontweight=weight)
        ok = 100 * fall <= budget
        verdict_y = -0.10 - step * (len(lines) + 0.2) if show_math else -0.12
        # with the arithmetic shown, the damage number is already on the line above
        prefix = "" if show_math else f"expected damage {100 * fall:.2f}  ·  "
        ax.text(0.02, verdict_y,
                f"{prefix}{'within' if ok else 'OVER'} the cap d = {budget:g}",
                transform=ax.transAxes, fontsize=9.5 if n > 1 else 11, fontweight="bold",
                color="#1c6b45" if ok else "#a33d36")

        ax.set_xlim(-0.1, N_COLS + 0.1)
        ax.set_ylim(-0.4, N_ROWS + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(name, fontsize=11.5, color=color, fontweight="bold", loc="left", pad=20)
        # the decision numbers, right under the route's name
        ax.text(0.0, 1.015,
                f"profit {stats['mean_money']:+.2f} CHF   ·   falls on {stats['p_fall']:.2%} of trips",
                transform=ax.transAxes, fontsize=9.5, color="#444")

    if title is None:
        title = ("the three ways to the lodge, and where each one's damage number comes from"
                 if n > 1 else "where this route's damage number comes from")
    plt.suptitle(title, fontsize=11.5, y=1.0)
    # the arithmetic is drawn *below* the axes, so reserve the room explicitly:
    # tight_layout cannot fit decorations it does not know about, and warns.
    if show_math:
        top, bottom = 0.88, (0.34 if n > 1 else 0.40)
    else:                                  # a shorter figure needs a bigger share for the titles
        top, bottom = (0.86, 0.14) if n > 1 else (0.80, 0.11)
    fig.subplots_adjust(top=top, bottom=bottom, left=0.04, right=0.97)
    plt.show()


def fig_risk_charge(candidates, lam=1.5):
    """Why a price picks a route: profit, minus the risk charge, equals the score.
    candidates: list of dicts with name / avg_money / avg_damage."""
    rows = sorted(candidates, key=lambda c: -c["avg_money"])
    winner = max(rows, key=lambda c: c["avg_money"] - lam * c["avg_damage"])
    fig, ax = plt.subplots(figsize=(9.6, 3.1))
    for i, cand in enumerate(rows):
        y = len(rows) - 1 - i
        profit, dmg = cand["avg_money"], cand["avg_damage"]
        charge = lam * dmg
        score = profit - charge
        color = ROUTE_COLORS.get(cand["name"], C_VIOLET)
        ax.barh(y, profit, height=0.5, color=color, alpha=0.22)         # what it earns
        if charge > 0:                                                   # what risk costs it
            ax.barh(y, charge, left=score, height=0.5, color=color, alpha=0.55,
                    hatch="///", edgecolor="white", linewidth=0)
        ax.barh(y, 0.45, left=score - 0.22, height=0.5, color=color)     # the score marker
        ax.text(-1.2, y, cand["name"].replace("the ", ""), ha="right", va="center",
                fontsize=10, color="#1b1b1b")
        tag = "   ← best score" if cand is winner else ""
        ax.text(profit + 1.5, y,
                f"{profit:.1f} − {lam:g}×{dmg:.1f} = {score:.1f}{tag}",
                va="center", fontsize=10, color=color, fontweight="bold")
    ax.set_xlim(-14, 88)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.set_xticks(range(0, 61, 10))
    ax.set_xlabel("CHF per trip\npale bar = profit   ·   hatched = the risk charge   ·   "
                  "solid tick = the score the planner maximises", fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(f"score = expected profit − λ × expected damage      (here λ = {lam:g})",
                 fontsize=11, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.show()


def path_plot(policies, title="The routes on the map"):
    """policies: dict label -> (policy_array, color). Draws each greedy route."""
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    _draw_grid(ax)
    styles = ["-", (0, (6, 3)), (0, (1.5, 2.2))]
    for i, (label, spec) in enumerate(policies.items()):
        pol, color = spec[0], spec[1]
        style = spec[2] if len(spec) > 2 else styles[i % len(styles)]
        path = greedy_route(pol)
        xs = [c + 0.5 for _, c in path]
        ys = [N_ROWS - 1 - r + 0.5 for r, _ in path]
        off = (i - (len(policies) - 1) / 2) * 0.12
        ax.plot(xs, [y + off for y in ys], color=color, linewidth=2.8, alpha=0.95,
                linestyle=style, solid_capstyle="round", label=label, zorder=5)
        if len(xs) > 1:
            ax.annotate("", xy=(xs[-1] + 0.02, ys[-1] + off), xytext=(xs[-2], ys[-2] + off),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6), zorder=6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.04), ncol=1,
              frameon=False, fontsize=9.5, handlelength=3)
    ax.set_title(title, fontsize=11, pad=8)
    plt.tight_layout()
    plt.show()


def walk_marathon(walks, title="🎬 The same route, forty trips"):
    """Play many trips in a row on the map, with running averages of profit and damage.

    walks : list of walk logs; each log is a list of step dicts with keys
            'reward', 'cost', 'fell' and 'state' (the state index moved into).
    The first few walks play step by step; the rest flash past, each leaving a
    dot on the running profit chart. The final frame is the whole sample.
    """
    payload_walks = []
    for log in walks:
        cells = [list(_rc(int(step["state"]))) for step in log]
        payload_walks.append({
            "cells": cells,
            "profit": round(float(sum(step["reward"] for step in log)), 1),
            "damage": round(float(sum(step["cost"] for step in log)), 1),
            "fell": bool(any(step.get("fell") for step in log)),
        })
    grid = [{"e": _cell_visual((r, c))[0], "bg": _cell_visual((r, c))[1]}
            for r in range(N_ROWS) for c in range(N_COLS)]
    payload = {"walks": payload_walks, "rows": N_ROWS, "cols": N_COLS, "grid": grid,
               "start": list(START), "gorge": list(GORGE_COLS)}
    uid = "wm_" + str(abs(hash(str(payload_walks))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;
  padding:16px;max-width:820px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .wm-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:2px}
#__UID__ .wm-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.5}
#__UID__ .wm-wrap{display:flex;gap:18px;align-items:flex-start;flex-wrap:wrap}
#__UID__ .wm-map{position:relative;flex:0 0 auto}
#__UID__ .wm-cell{position:absolute;width:42px;height:42px;border-radius:8px;
  display:flex;align-items:center;justify-content:center;font-size:17px;border:1px solid #e2e5ef;
  transition:box-shadow .12s}
#__UID__ .wm-cell.trace{box-shadow:inset 0 0 0 3px rgba(118,75,162,.55)}
#__UID__ .wm-cell.tracebad{box-shadow:inset 0 0 0 3px rgba(212,87,78,.85)}
#__UID__ .wm-bot{position:absolute;width:42px;height:42px;display:flex;align-items:center;
  justify-content:center;font-size:23px;transition:left .13s linear,top .13s linear;z-index:5}
#__UID__ .wm-boom{position:absolute;width:42px;height:42px;display:flex;align-items:center;
  justify-content:center;font-size:27px;z-index:6;opacity:0;transition:opacity .1s}
#__UID__ .wm-side{flex:1;min-width:300px}
#__UID__ .wm-stats{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
#__UID__ .wm-stat{flex:1;min-width:88px;border:1px solid #e2e5ef;border-radius:10px;padding:7px 10px}
#__UID__ .wm-stat .k{font-size:10.5px;font-weight:800;color:#8a8f9c;letter-spacing:.4px}
#__UID__ .wm-stat .v{font-size:19px;font-weight:800;font-variant-numeric:tabular-nums;color:#1b1b1b}
#__UID__ canvas{display:block;width:100%;height:auto;border:1px solid #eceef4;border-radius:8px;background:#fbfcfe}
#__UID__ .wm-log{font-size:12.5px;color:#444;min-height:19px;margin-top:8px;font-variant-numeric:tabular-nums}
#__UID__ .wm-btn{cursor:pointer;border:none;border-radius:8px;padding:8px 16px;font-size:13px;
  font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:10px}
#__UID__ .wm-btn.ghost{background:#fff;color:#555;border:1px solid #d7d9e3;margin-left:6px}
</style>
<div id="__UID__">
  <div class="wm-head">__TITLE__</div>
  <div class="wm-sub">Same route, forty separate days. The first few trips play step by step, the
    rest flash past. Keep an eye on the two running averages on the right.</div>
  <div class="wm-wrap">
    <div class="wm-map"></div>
    <div class="wm-side">
      <div class="wm-stats">
        <div class="wm-stat"><div class="k">WALK</div><div class="v wm-i">0 / 0</div></div>
        <div class="wm-stat"><div class="k">AVG PROFIT</div><div class="v wm-p" style="color:#2e9e7a">—</div></div>
        <div class="wm-stat"><div class="k">AVG DAMAGE</div><div class="v wm-d" style="color:#d4574e">—</div></div>
      </div>
      <canvas width="440" height="212"></canvas>
      <div class="wm-log">Press ▶ to send the walker out.</div>
      <button class="wm-btn">▶ Replay all forty</button>
      <button class="wm-btn ghost">⏭ Skip to the end</button>
    </div>
  </div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const W=D.walks, N=W.length, PITCH=46;
  const map=root.querySelector(".wm-map");
  map.style.width=(D.cols*PITCH)+"px"; map.style.height=(D.rows*PITCH+22)+"px";
  const iEl=root.querySelector(".wm-i"), pEl=root.querySelector(".wm-p"), dEl=root.querySelector(".wm-d");
  const logEl=root.querySelector(".wm-log"), cv=root.querySelector("canvas"), cx=cv.getContext("2d");
  const cells=[];
  for(let r=0;r<D.rows;r++){
    for(let c=0;c<D.cols;c++){
      const g=D.grid[r*D.cols+c], d=document.createElement("div");
      d.className="wm-cell";
      d.style.left=(c*PITCH)+"px"; d.style.top=(r*PITCH)+"px";
      d.style.background=g.bg; d.textContent=g.e;
      map.appendChild(d); cells.push(d);
    }
  }
  {  // the gorge: a thin off-map strip, not a row of cells
    const g=document.createElement("div");
    g.style.cssText="position:absolute;height:14px;border-radius:0 0 8px 8px;"
      +"background:linear-gradient(#bcd4f0,#9dbfe8);text-align:center;font-size:9.5px;"
      +"color:#3d6395;font-style:italic;line-height:14px";
    g.style.left=(D.gorge[0]*PITCH)+"px"; g.style.top=(D.rows*PITCH+2)+"px";
    g.style.width=(D.gorge.length*PITCH-4)+"px";
    g.textContent="the gorge";
    map.appendChild(g);
  }
  const bot=document.createElement("div"); bot.className="wm-bot"; bot.textContent="\u{1F916}";
  map.appendChild(bot);
  const boom=document.createElement("div"); boom.className="wm-boom"; boom.textContent="\u{1F4A5}";
  map.appendChild(boom);
  function place(el,r,c){ el.style.left=(c*PITCH)+"px"; el.style.top=(r*PITCH)+"px"; }
  function clearTrace(){ cells.forEach(d=>d.classList.remove("trace","tracebad")); }

  // ---- chart geometry ----
  const L=44,R=12,T=16,B=26,H=cv.height,CW=cv.width;
  let lo=Infinity, hi=-Infinity;
  W.forEach(w=>{ lo=Math.min(lo,w.profit); hi=Math.max(hi,w.profit); });
  const pad=Math.max(6,(hi-lo)*0.18); lo=Math.min(0,lo-pad); hi=hi+pad;
  const X=i=>L+(N>1?(i-1)/(N-1):0.5)*(CW-L-R);
  const Y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);

  let done=0, sumP=0, sumD=0, nFell=0;
  function drawChart(){
    cx.clearRect(0,0,CW,H);
    cx.strokeStyle="#eceef4"; cx.lineWidth=1; cx.fillStyle="#8a8f9c";
    cx.font="10.5px system-ui"; cx.textAlign="right"; cx.textBaseline="middle";
    const step=(hi-lo)/4, nice=[5,10,20,25,50].find(t=>t>=step)||100;
    for(let v=Math.ceil(lo/nice)*nice; v<=hi; v+=nice){
      cx.beginPath(); cx.moveTo(L,Y(v)); cx.lineTo(CW-R,Y(v)); cx.stroke();
      cx.fillText(v, L-6, Y(v));
    }
    cx.textAlign="center"; cx.textBaseline="top";
    cx.fillText("walk number  →", (L+CW-R)/2, H-B+9);
    cx.save(); cx.translate(11,(T+H-B)/2); cx.rotate(-Math.PI/2);
    cx.textAlign="center"; cx.textBaseline="middle"; cx.fillText("profit (CHF)",0,0); cx.restore();
    if(done>0){
      const avg=sumP/done;
      cx.strokeStyle="#1b1b1b"; cx.lineWidth=1.6; cx.setLineDash([6,4]);
      cx.beginPath(); cx.moveTo(L,Y(avg)); cx.lineTo(CW-R,Y(avg)); cx.stroke(); cx.setLineDash([]);
      cx.fillStyle="#1b1b1b"; cx.font="bold 10.5px system-ui"; cx.textAlign="left"; cx.textBaseline="bottom";
      cx.fillText("average "+(avg>=0?"+":"")+avg.toFixed(1)+" CHF", L+3, Y(avg)-3);
    }
    for(let i=1;i<=done;i++){
      const w=W[i-1], x=X(i), y=Y(w.profit);
      cx.strokeStyle=w.fell?"rgba(212,87,78,.45)":"rgba(46,158,122,.35)"; cx.lineWidth=1.6;
      cx.beginPath(); cx.moveTo(x,Y(0)); cx.lineTo(x,y); cx.stroke();
      if(w.fell){
        cx.strokeStyle="#d4574e"; cx.lineWidth=2.4;
        cx.beginPath(); cx.moveTo(x-4,y-4); cx.lineTo(x+4,y+4);
        cx.moveTo(x+4,y-4); cx.lineTo(x-4,y+4); cx.stroke();
      }else{
        cx.fillStyle="#2e9e7a"; cx.beginPath(); cx.arc(x,y,3.4,0,7); cx.fill();
      }
    }
  }
  function stats(){
    iEl.textContent=done+" / "+N;
    pEl.textContent=done?((sumP/done>=0?"+":"")+(sumP/done).toFixed(1)):"—";
    dEl.textContent=done?(sumD/done).toFixed(1):"—";
  }
  function reset(){
    done=0; sumP=0; sumD=0; nFell=0; clearTrace(); boom.style.opacity=0;
    place(bot, D.start[0], D.start[1]); stats(); drawChart();
  }

  const FULL=5;                       // walks played step by step before the montage
  let timer=null, wi=0, si=0;
  function finishWalk(){
    const w=W[wi];
    done++; sumP+=w.profit; sumD+=w.damage; if(w.fell) nFell++;
    stats(); drawChart();
    logEl.innerHTML="walk "+done+": "+(w.fell
      ? "<b style='color:#d4574e'>fell into the gorge</b> · profit "+w.profit.toFixed(0)
        +" CHF · damage 100"
      : "reached the lodge · profit <b>+"+w.profit.toFixed(0)+" CHF</b> · damage 0")
      +" &nbsp;|&nbsp; "+nFell+" fall"+(nFell===1?"":"s")+" so far";
    wi++; si=0;
  }
  function schedule(ms){ timer=setTimeout(tick, ms); }
  function tick(){
    if(wi>=N){ finale(); return; }
    const w=W[wi];
    if(wi<FULL){
      if(si===0){ clearTrace(); boom.style.opacity=0; place(bot,D.start[0],D.start[1]); }
      si++;
      if(si<=w.cells.length){
        const rc=w.cells[si-1];
        place(bot, rc[0], rc[1]);
        if(w.fell && si===w.cells.length){ place(boom, rc[0], rc[1]); boom.style.opacity=1; }
        schedule(si===w.cells.length?420:150);
        return;
      }
      finishWalk(); schedule(260); return;
    }
    // montage: the whole route flashes at once
    clearTrace(); boom.style.opacity=0;
    w.cells.forEach(rc=>cells[rc[0]*D.cols+rc[1]].classList.add(w.fell?"tracebad":"trace"));
    const last=w.cells[w.cells.length-1];
    place(bot,last[0],last[1]);
    if(w.fell){ place(boom,last[0],last[1]); boom.style.opacity=1; }
    finishWalk(); schedule(230);
  }
  function finale(){
    timer=null; clearTrace(); boom.style.opacity=0;
    logEl.innerHTML="<b>Forty walks done — "+nFell+" of "+N+" ended in the gorge. "
      +"Average profit "+(sumP/N>=0?"+":"")+(sumP/N).toFixed(1)+" CHF, average damage "
      +(sumD/N).toFixed(1)+" points.</b>";
  }
  function play(){
    if(timer){ clearTimeout(timer); timer=null; }
    reset(); wi=0; si=0;
    logEl.textContent="Leaving the depot…";
    schedule(320);
  }
  root.querySelector(".wm-btn").addEventListener("click",play);
  root.querySelector(".ghost").addEventListener("click",()=>{
    if(timer){ clearTimeout(timer); timer=null; }
    reset();
    W.forEach(w=>{ done++; sumP+=w.profit; sumD+=w.damage; if(w.fell) nFell++; });
    stats(); drawChart(); place(bot, D.start[0], D.start[1]); finale();
  });
  play();
})();
</script>'''
    html = (tmpl.replace("__UID__", uid).replace("__TITLE__", title)
            .replace("__DATA__", _json.dumps(payload)))
    display(HTML(html))


def outcome_hist(runs, title="2000 evaluation walks — the profit distributions",
                 xlabel="profit of one walk (CHF)", spotlight=None):
    """runs: dict label -> (profit_array, color).
    spotlight: optional (x_lo, x_hi, text) — ring and label a region of the axis."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    lo = min(float(np.min(v)) for v, _ in runs.values()) - 4
    hi = max(float(np.max(v)) for v, _ in runs.values()) + 4
    bins = np.linspace(lo, hi, 46)
    for label, (vals, color) in runs.items():
        ax.hist(vals, bins=bins, color=color, alpha=0.55, label=label, edgecolor="none")
        m = float(np.mean(vals))
        ax.axvline(m, color=color, linestyle="--", linewidth=1.6)
        ax.text(m, ax.get_ylim()[1] * 0.97, f" mean {m:+.1f}", color=color,
                fontsize=9, ha="left", va="top", fontweight="bold")
    if spotlight is not None:
        x_lo, x_hi, text = spotlight
        ytop = ax.get_ylim()[1]
        peak = 0.0
        for vals, _ in runs.values():
            counts, _ = np.histogram(vals, bins=bins)
            inside = counts[(bins[:-1] >= x_lo) & (bins[1:] <= x_hi)]
            if len(inside):
                peak = max(peak, float(inside.max()))
        box_top = max(peak * 2.4, ytop * 0.055)
        ax.add_patch(plt.Rectangle((x_lo, -ytop * 0.012), x_hi - x_lo, box_top,
                                   fill=False, edgecolor=C_RED, linewidth=2,
                                   linestyle=(0, (5, 3)), zorder=6))
        ax.annotate(text, xy=(x_hi, box_top * 0.55),
                    xytext=(x_hi + (bins[-1] - bins[0]) * 0.05, ytop * 0.45),
                    ha="left", va="center", fontsize=10, fontweight="bold", color=C_RED,
                    arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2,
                                    connectionstyle="arc3,rad=0.2"), zorder=7)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("walks", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def outcome_panel(dists, budget=None,
                  title="What each route actually pays, and actually breaks"):
    """The exact outcome distributions of profit and damage, side by side.

    dists: dict label -> (trip_distribution dict, colour). Every bar is a real
    outcome with its real probability; the means are printed in the legend so
    no annotation can collide with the data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    labels = list(dists)
    n = max(len(labels), 1)

    ax = axes[0]
    for label, (dist, color) in dists.items():
        atoms = sorted(dist["money"].items())
        xs = [v for v, _ in atoms]
        ps = [p for _, p in atoms]
        ax.vlines(xs, 0, ps, color=color, linewidth=2.4, alpha=0.85)
        ax.plot(xs, ps, "o", color=color, markersize=6,
                label=f"{label}  (mean {dist['mean_money']:+.1f})")
    ax.set_xlabel("profit of one trip (CHF)", fontsize=10)
    ax.set_ylabel("probability", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("profit per trip", fontsize=10.5, loc="left", fontweight="bold")

    ax2 = axes[1]
    width = 7.0 / n
    for i, (label, (dist, color)) in enumerate(dists.items()):
        off = (i - (n - 1) / 2) * width * 1.15
        atoms = sorted(dist["damage"].items())
        ax2.bar([v + off for v, _ in atoms], [p for _, p in atoms], width=width,
                color=color, alpha=0.85)
        # the means live in open space on the right, so nothing can collide with a bar
        ax2.text(52, 0.92 - 0.10 * i, f"mean {dist['mean_damage']:.2f}", color=color,
                 fontsize=10, fontweight="bold", va="center")
    if budget is not None:
        ax2.axvline(budget, color=C_VIOLET, linewidth=2.0, linestyle="--")
        ax2.annotate(f"budget d = {budget:g}", xy=(budget, 0.45), xytext=(24, 0.45),
                     color=C_VIOLET, fontsize=9.5, fontweight="bold", va="center",
                     arrowprops=dict(arrowstyle="-|>", color=C_VIOLET, lw=1.6))
    ax2.set_xlabel("damage in one trip (points)", fontsize=10)
    ax2.set_xticks([0, 100])
    ax2.set_xlim(-14, 118)
    ax2.set_ylim(0, 1.08)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_title("damage per trip: only two values exist", fontsize=10.5, loc="left",
                  fontweight="bold")
    plt.suptitle(title, fontsize=12, y=1.03)
    plt.tight_layout()
    plt.show()


def lambda_ladder_plot(switches, names, lam_max=5.0,
                       title="The price ladder — which route each internal price buys"):
    """One axis, the whole λ story: coloured bands showing the route the planner
    picks at each price, with the switch points marked.

    switches: ascending list of λ values where the choice changes
    names   : the route chosen on each band (len == len(switches) + 1)
    """
    fig, ax = plt.subplots(figsize=(9.5, 2.3))
    edges = [0.0] + list(switches) + [lam_max]
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        color = ROUTE_COLORS.get(name, C_VIOLET)
        ax.axvspan(lo, hi, color=color, alpha=0.28)
        ax.text((lo + hi) / 2, 0.55, name.replace("the ", ""), ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=color)
    for s in switches:
        ax.axvline(s, color="#444", linewidth=1.5, linestyle="--")
        ax.annotate(f"λ = {s:.2f}", xy=(s, 0.06), ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color="#333",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff", ec="#bbb"))
    ax.set_xlim(0, lam_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("λ — internal price of one damage point (CHF)", fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def cvar_curve(cvar_fn, dists, alphas=None,
               title="Every question you could ask the same three distributions"):
    """CVaR against α for each route, with the crossings marked.
    cvar_fn: the student's cvar(distribution, alpha)."""
    if alphas is None:
        # fine enough that the crossover labels round to the true values (0.16, 0.58)
        alphas = np.linspace(0.01, 1.0, 400)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    curves = {}
    for label, (dist, color) in dists.items():
        ys = np.array([cvar_fn(dist["money"], float(a)) for a in alphas])
        curves[label] = ys
        ax.plot(alphas, ys, color=color, linewidth=2.4, label=label)
    # mark where the winner changes
    names = list(curves)
    stacked = np.vstack([curves[k] for k in names])
    winners = stacked.argmax(axis=0)
    for i in range(1, len(alphas)):
        if winners[i] != winners[i - 1]:
            a = float(alphas[i])
            ax.axvline(a, color="#777", linestyle=":", linewidth=1.4)
            ax.annotate(f"α ≈ {a:.2f}\n{names[winners[i]].replace('the ', '')} takes over",
                        xy=(a, ax.get_ylim()[0]), xytext=(a + 0.015, 4),
                        fontsize=9, color="#555", va="bottom")
    ax.axvspan(0.98, 1.0, color="#f0edff", zorder=0)
    ax.text(0.985, 52, "α = 1 is\nthe plain mean", fontsize=8.5, color=C_VIOLET,
            ha="right", va="top")
    ax.set_xlabel("α — the share of worst trips your score looks at "
                  "(1.0 = all of them = the plain mean)", fontsize=10)
    ax.set_ylabel("CVaR$_α$ of profit (CHF)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="center right", bbox_to_anchor=(1.0, 0.42))
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def dual_ascent_plot(lams, damages, budget):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    rounds = range(1, len(lams) + 1)
    ms = 6 if len(lams) <= 20 else 3.0        # a long run needs smaller dots
    axes[0].plot(rounds, lams, "-o", color=C_VIOLET, linewidth=2, markersize=ms)
    axes[0].annotate(f"λ = {lams[-1]:.3f}", xy=(len(lams), lams[-1]),
                     xytext=(-4, 14), textcoords="offset points", ha="right",
                     fontsize=9.5, fontweight="bold", color=C_VIOLET)
    axes[0].set_ylabel("λ — internal price of a damage point", fontsize=10)
    axes[0].set_title("the price the board charges", fontsize=11)
    axes[1].plot(rounds, damages, "-o", color=C_RED, linewidth=2, markersize=ms)
    axes[1].axhline(budget, color="#777", linestyle="--", linewidth=1.4)
    axes[1].annotate(f"budget d = {budget:g}", xy=(len(lams), budget),
                     xytext=(0, 6), textcoords="offset points", fontsize=9,
                     color="#555", va="bottom", ha="right")
    axes[1].set_ylabel("measured damage per trip (points)", fontsize=10)
    axes[1].set_title("the damage the fleet actually takes", fontsize=11)
    for ax in axes:
        ax.set_xlabel("board meeting (round)", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eceef4")
    plt.suptitle("Dual ascent — the price of risk is discovered, not decreed", fontsize=12, y=1.04)
    plt.tight_layout()
    plt.show()


def budget_menu_plot(budgets, chosen):
    """The menu of safety: what each risk budget buys.
    budgets: list of d values; chosen: list of dicts with keys name/avg_money/avg_damage."""
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    xs = np.asarray(budgets, dtype=float)
    ys = np.asarray([c["avg_money"] for c in chosen], dtype=float)
    ax.step(xs, ys, where="post", color=C_VIOLET, linewidth=2.4, zorder=3)
    seen = set()
    for x, y, c in zip(xs, ys, chosen):
        color = ROUTE_COLORS.get(c["name"], C_VIOLET)
        ax.plot(x, y, "o", color=color, markersize=8, zorder=4)
        if c["name"] not in seen:
            seen.add(c["name"])
            ax.annotate(c["name"].replace("the ", ""), (x, y), textcoords="offset points",
                        xytext=(6, 8), fontsize=9.5, fontweight="bold", color=color)
    ax.set_xlabel("the damage budget d the board signs (expected damage points per trip)", fontsize=10)
    ax.set_ylabel("best affordable expected profit (CHF)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#eceef4", linewidth=0.8)
    ax.set_title("The menu — what each risk budget buys", fontsize=11)
    plt.tight_layout()
    plt.show()


def two_offers(title="Two offers. Identical expected value."):
    """The cold open: same mean, wildly different risk. Probabilities, not samples,
    so the two means are exactly equal and a student can check them by hand."""
    A_x, A_p = [7, 9, 11, 13], [0.25] * 4
    B_x, B_p = [-50, 25], [0.20, 0.80]
    mean_a = sum(x * p for x, p in zip(A_x, A_p))

    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    ax.bar(A_x, A_p, width=1.9, color=C_BLUE, alpha=.85,
           label="Offer A — the steady one")
    ax.bar(B_x, B_p, width=1.9, color=C_RED, alpha=.85,
           label="Offer B — the bold one")

    ax.axvline(mean_a, color="#1b1b1b", linestyle="--", linewidth=2)
    ax.annotate(f"both average exactly {mean_a:+.0f} M",
                xy=(mean_a, 0.50), xytext=(-20, 0.71),
                ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#1b1b1b",
                arrowprops=dict(arrowstyle="-|>", color="#1b1b1b", lw=1.8,
                                connectionstyle="arc3,rad=-0.18"),
                bbox=dict(boxstyle="round,pad=0.32", fc="#ffffff", ec="#1b1b1b", lw=1.4))
    ax.annotate("1 year in 5,\nyou lose 50",
                xy=(-50, 0.21), xytext=(-44, 0.52),
                fontsize=10.5, fontweight="bold", color=C_RED, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=C_RED, lw=2,
                                connectionstyle="arc3,rad=-0.25"))
    ax.annotate("4 years in 5", xy=(25, 0.83), ha="center", va="bottom",
                fontsize=10.5, fontweight="bold", color=C_RED)

    ax.set_xlabel("what the offer pays you in one year (M CHF)", fontsize=10.5)
    ax.set_ylabel("probability", fontsize=10.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-58, 33)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ===========================================================================
#  §3  The interactive labs
# ===========================================================================
def shield_lab():
    """Interactive: click the cells YOU would forbid the walker to enter, then
    check your shield against the map's actual danger cells."""
    grid = []
    for r in range(N_ROWS):
        for c in range(N_COLS):
            e, bg = _cell_visual((r, c))
            grid.append({"e": e, "bg": bg, "danger": (r, c) in DANGER_CELLS})
    payload = {"rows": N_ROWS, "cols": N_COLS, "grid": grid, "gorge": list(GORGE_COLS),
               "start": list(START), "goal": list(GOAL), "ndanger": len(DANGER_CELLS)}
    uid = "sl_" + str(abs(hash(str(payload))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;
  padding:16px;max-width:640px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .sl-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:2px}
#__UID__ .sl-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .sl-wrap{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
#__UID__ .sl-map{position:relative;flex:0 0 auto}
#__UID__ .sl-cell{position:absolute;width:44px;height:44px;border-radius:8px;border:1px solid #e2e5ef;
  display:flex;align-items:center;justify-content:center;font-size:17px;cursor:pointer;user-select:none;
  transition:box-shadow .1s}
#__UID__ .sl-cell:hover{box-shadow:0 0 0 2px rgba(118,75,162,.35)}
#__UID__ .sl-cell.flag::after{content:"\1F6AB";position:absolute;font-size:15px;top:-4px;right:-4px}
#__UID__ .sl-cell.flag{box-shadow:inset 0 0 0 3px #d4574e}
#__UID__ .sl-cell.okflag{box-shadow:inset 0 0 0 3px #46b46e}
#__UID__ .sl-cell.missed{box-shadow:inset 0 0 0 3px #e0a13c;animation:none}
#__UID__ .sl-side{flex:1;min-width:220px}
#__UID__ .sl-count{font-size:13px;color:#444;margin-bottom:8px}
#__UID__ .sl-count b{color:#1b1b1b}
#__UID__ .sl-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;
  font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .sl-btn.ghost{background:#fff;color:#555;border:1px solid #d7d9e3}
#__UID__ .sl-verdict{font-size:13px;color:#1b1b1b;background:#f6f7fb;border:1px solid #e6e8ee;
  border-radius:10px;padding:10px 12px;line-height:1.6;margin-top:10px;min-height:40px}
</style>
<div id="__UID__">
  <div class="sl-head">🛡️ Build the shield: flag every cell the walker must never enter</div>
  <div class="sl-sub">Click cells to flag them 🚫 (click again to unflag). You are writing the
    safety officer's list: any move that would <i>land</i> on a flagged cell is masked out before
    the planner ever sees it. Remember the rule from Part 1: a cell is dangerous when it
    <b>touches a hazard</b>, so the pits and their whole ⚠️ ring count, and so does the rim above
    the gorge. The depot and the lodge sit on ordinary ground.</div>
  <div class="sl-wrap">
    <div class="sl-map"></div>
    <div class="sl-side">
      <div class="sl-count">flagged: <b>0</b> cells</div>
      <button class="sl-btn">Check my shield</button>
      <button class="sl-btn ghost" style="margin-left:6px">Clear</button>
      <div class="sl-verdict">Flag your suspects, then check.</div>
    </div>
  </div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const map=root.querySelector(".sl-map"), PITCH=48;
  map.style.width=(D.cols*PITCH)+"px"; map.style.height=(D.rows*PITCH)+"px";
  const countEl=root.querySelector(".sl-count"), verdict=root.querySelector(".sl-verdict");
  const cells=[];
  for(let r=0;r<D.rows;r++){
    for(let c=0;c<D.cols;c++){
      const g=D.grid[r*D.cols+c], d=document.createElement("div");
      d.className="sl-cell"; d.dataset.r=r; d.dataset.c=c;
      d.style.left=(c*PITCH)+"px"; d.style.top=(r*PITCH)+"px";
      d.style.background=g.bg; d.textContent=g.e;
      d.addEventListener("click",()=>{
        d.classList.remove("okflag","missed");
        d.classList.toggle("flag");
        countEl.innerHTML="flagged: <b>"+map.querySelectorAll(".flag").length+"</b> cells";
      });
      map.appendChild(d); cells.push(d);
    }
  }
  function maskedPairs(flagged){
    // count (cell, action) pairs whose landing cell is flagged — same clamping as the world
    const del=[[-1,0],[0,1],[1,0],[0,-1]];
    let n=0;
    for(let r=0;r<D.rows;r++)for(let c=0;c<D.cols;c++){
      for(const[dr,dc]of del){
        let r2=r+dr,c2=c+dc;
        if(r2<0||r2>=D.rows||c2<0||c2>=D.cols){r2=r;c2=c;}
        if(flagged.has(r2+","+c2))n++;
      }
    }
    return n;
  }
  root.querySelector(".sl-btn").addEventListener("click",()=>{
    const flagged=new Set();
    cells.forEach(d=>{ if(d.classList.contains("flag"))flagged.add(d.dataset.r+","+d.dataset.c); });
    let hit=0, extra=0, missed=0;
    cells.forEach(d=>{
      const key=d.dataset.r+","+d.dataset.c;
      const danger=D.grid[+d.dataset.r*D.cols+ +d.dataset.c].danger;
      d.classList.remove("okflag","missed");
      if(flagged.has(key)&&danger){hit++; d.classList.add("okflag");}
      else if(flagged.has(key)&&!danger){extra++;}
      else if(!flagged.has(key)&&danger){missed++; d.classList.add("missed");}
    });
    const total=hit+missed;
    if(missed===0&&extra===0){
      verdict.innerHTML="✅ <b>Perfect shield.</b> You flagged all "+total+" cells where the walker "
        +"can fall: the four rim cells above the gorge, the two open pits, and the eight cells of "
        +"undermined ground touching them. That list masks <b>"+maskedPairs(flagged)+"</b> of the "
        +(D.rows*D.cols*4)+" possible (cell, action) moves. Everything else stays on the menu, and "
        +"the planner will never even consider a flagged move.";
    }else{
      verdict.innerHTML="You flagged <b>"+(hit+extra)+"</b> cells: <b>"+hit+"</b> genuinely dangerous"
        +(extra?", <b>"+extra+"</b> harmless (nothing can drop the walker there, and an "
        +"over-cautious shield quietly costs profit)":"")
        +(missed?", and <b>"+missed+"</b> danger cell"+(missed===1?"":"s")+" still unflagged "
        +"(outlined in amber). Apply the rule again: a cell is dangerous when it <b>touches a "
        +"hazard</b>, so every cell of the ring around a pit counts, not just the pit itself":"")
        +". Fix the list and check again.";
    }
  });
  root.querySelector(".ghost").addEventListener("click",()=>{
    cells.forEach(d=>d.classList.remove("flag","okflag","missed"));
    countEl.innerHTML="flagged: <b>0</b> cells";
    verdict.textContent="Flag your suspects, then check.";
  });
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(payload))
    display(HTML(html))


def cvar_lab(dists, alpha_init=0.20):
    """Interactive: drag α and watch which route wins on its bad days.

    dists: dict label -> (trip_distribution dict, colour).
    Each route is drawn as its outcome curve, worst trips first, on a shared
    probability axis. The shaded strip is exactly the α-share CVaR averages.
    """
    series = []
    for label, (dist, color) in dists.items():
        atoms = sorted(dist["money"].items())
        series.append({"label": label, "color": color,
                       "atoms": [[round(float(v), 3), float(p)] for v, p in atoms],
                       "mean": round(float(dist["mean_money"]), 2)})
    payload = {"series": series, "a0": int(round(alpha_init * 100))}
    uid = "cv_" + str(abs(hash(str(payload))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;
  padding:16px;max-width:790px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .cv-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:2px}
#__UID__ .cv-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ canvas{display:block;width:100%;height:auto;border:1px solid #eceef4;border-radius:8px;background:#fbfcfe}
#__UID__ .cv-ctl{display:flex;align-items:center;gap:12px;margin:14px 0 4px;flex-wrap:wrap}
#__UID__ .cv-ctl label{font-size:13px;font-weight:700;color:#1b1b1b;white-space:nowrap}
#__UID__ input[type=range]{flex:1;min-width:210px;accent-color:#764ba2}
#__UID__ .cv-a{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:16px;
  font-weight:800;color:#764ba2;min-width:150px;text-align:right}
#__UID__ .cv-preset{cursor:pointer;border:1px solid #d7d9e3;background:#fff;color:#444;
  border-radius:999px;padding:4px 11px;font-size:12px;font-weight:700}
#__UID__ .cv-preset:hover{border-color:#764ba2;color:#764ba2}
#__UID__ .cv-cards{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
#__UID__ .cv-card{flex:1;min-width:200px;border:1.5px solid #e2e5ef;border-radius:10px;padding:10px 12px}
#__UID__ .cv-card.win{box-shadow:0 0 0 2px rgba(118,75,162,.22)}
#__UID__ .cv-name{font-size:12.5px;font-weight:700;margin-bottom:3px}
#__UID__ .cv-val{font-size:22px;font-weight:800;font-variant-numeric:tabular-nums}
#__UID__ .cv-note{font-size:11.5px;color:#666;margin-top:3px;line-height:1.5}
#__UID__ .cv-verdict{margin-top:12px;font-size:13.5px;font-weight:700;color:#1b1b1b;
  background:#f6f7fb;border:1px solid #e6e8ee;border-radius:10px;padding:10px 12px;line-height:1.6}
</style>
<div id="__UID__">
  <div class="cv-head">🎛️ The CVaR lab — drag α and watch the ranking change</div>
  <div class="cv-sub">Each route's trips are lined up worst first along the bottom axis, so the
    curve <i>is</i> its distribution. The shaded strip on the left is exactly the α-share of trips
    CVaR averages; the thick line is that average.</div>
  <canvas width="750" height="330"></canvas>
  <div class="cv-ctl">
    <label>α =</label>
    <input type="range" min="1" max="100" step="1">
    <span class="cv-a"></span>
  </div>
  <div class="cv-ctl" style="margin-top:0">
    <button class="cv-preset" data-a="2">2% — the worst 1 trip in 50</button>
    <button class="cv-preset" data-a="20">20% — the worst fifth of trips</button>
    <button class="cv-preset" data-a="100">100% — every trip (the plain average)</button>
  </div>
  <div class="cv-cards"></div>
  <div class="cv-verdict"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const cv=root.querySelector("canvas"), cx=cv.getContext("2d");
  const slider=root.querySelector("input[type=range]"), aOut=root.querySelector(".cv-a");
  const cards=root.querySelector(".cv-cards"), verdict=root.querySelector(".cv-verdict");
  const S=D.series, L=58,R=16,T=42,B=44,W=cv.width,H=cv.height;
  let lo=Infinity, hi=-Infinity;
  S.forEach(s=>s.atoms.forEach(([v])=>{ lo=Math.min(lo,v); hi=Math.max(hi,v); }));
  const pad=(hi-lo)*0.16; lo=Math.min(0,lo-pad); hi=hi+pad;
  const X=u=>L+u*(W-L-R);                 // u = cumulative probability in [0,1]
  const Y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);

  S.forEach(s=>{
    const d=document.createElement("div"); d.className="cv-card"; d.style.borderColor=s.color;
    d.innerHTML='<div class="cv-name" style="color:'+s.color+'">'+s.label+'</div>'
      +'<div class="cv-val"></div><div class="cv-note"></div>';
    cards.appendChild(d);
  });

  function cvar(atoms, a){                 // exact: walk the atoms, take partial mass
    let taken=0, out=0;
    for(const [v,p] of atoms){
      const take=Math.min(p, a-taken);
      if(take<=0) break;
      out+=v*take; taken+=take;
    }
    return out/a;
  }
  function worstShare(atoms, a){           // how much of the window is the failure atom
    let taken=0, bad=0;
    for(const [v,p] of atoms){
      const take=Math.min(p, a-taken);
      if(take<=0) break;
      if(v < atoms[atoms.length-1][0]) bad+=take;
      taken+=take;
    }
    return bad/a;
  }

  function draw(a){
    cx.clearRect(0,0,W,H);
    cx.fillStyle="rgba(118,75,162,0.10)"; cx.fillRect(L,T,X(a)-L,H-B-T);
    cx.strokeStyle="#eceef4"; cx.lineWidth=1; cx.fillStyle="#8a8f9c";
    cx.font="11px system-ui"; cx.textAlign="right"; cx.textBaseline="middle";
    const step=(hi-lo)/5, nice=[5,10,20,25,50].find(t=>t>=step)||100;
    for(let v=Math.ceil(lo/nice)*nice; v<=hi; v+=nice){
      cx.beginPath(); cx.moveTo(L,Y(v)); cx.lineTo(W-R,Y(v)); cx.stroke();
      cx.fillText(v, L-7, Y(v));
    }
    cx.textAlign="center"; cx.textBaseline="top";
    [0,0.25,0.5,0.75,1].forEach(u=>cx.fillText(Math.round(u*100)+"%", X(u), H-B+7));
    cx.fillText("the trips, worst first  →", (L+W-R)/2, H-B+23);
    cx.strokeStyle="#764ba2"; cx.lineWidth=2; cx.setLineDash([5,4]);
    cx.beginPath(); cx.moveTo(X(a),T-10); cx.lineTo(X(a),H-B); cx.stroke(); cx.setLineDash([]);

    S.forEach(s=>{
      cx.strokeStyle=s.color; cx.lineWidth=2.6; cx.beginPath();   // the quantile curve
      let u=0, started=false;
      for(const [v,p] of s.atoms){
        const y=Y(v);
        if(!started){ cx.moveTo(X(u), y); started=true; } else { cx.lineTo(X(u), y); }
        u+=p; cx.lineTo(X(Math.min(u,1)), y);
      }
      cx.stroke();
      const c=cvar(s.atoms,a);                                     // the CVaR itself
      cx.strokeStyle=s.color; cx.lineWidth=5; cx.globalAlpha=.9;
      cx.beginPath(); cx.moveTo(L,Y(c)); cx.lineTo(X(a),Y(c)); cx.stroke(); cx.globalAlpha=1;
      cx.fillStyle=s.color; cx.beginPath(); cx.arc(X(a),Y(c),4.5,0,7); cx.fill();
    });
    cx.textAlign="left"; cx.textBaseline="middle"; cx.font="bold 11.5px system-ui";
    let lx=L;
    S.forEach(s=>{
      cx.fillStyle=s.color; cx.fillRect(lx,14,11,11);
      cx.fillText(s.label, lx+16, 20);
      lx += 16 + cx.measureText(s.label).width + 20;
    });
    cx.fillStyle="#8a8f9c"; cx.font="11px system-ui";
    cx.fillText("thin = the outcome curve   ·   thick = CVaR over the shaded trips", L, 33);
  }

  function update(){
    const a=(+slider.value)/100;
    draw(a);
    aOut.textContent=(+slider.value)+"%  (worst "+(+slider.value)+" trips in 100)";
    let best=null, vals=[];
    S.forEach((s,i)=>{
      const c=cvar(s.atoms,a), bad=worstShare(s.atoms,a), card=cards.children[i];
      vals.push({label:s.label, color:s.color, c:c});
      card.querySelector(".cv-val").textContent=(c>=0?"+":"")+c.toFixed(1)+" CHF";
      card.querySelector(".cv-val").style.color=s.color;
      card.querySelector(".cv-note").innerHTML= bad>0.999
        ? "every trip in this window ended in the gorge"
        : (bad<=0.0005 ? "no failed trip is inside this window yet"
           : Math.round(bad*100)+"% of the trips in this window ended in the gorge");
      card.classList.remove("win");
      if(best===null||c>best.c) best={i:i,c:c,label:s.label,color:s.color};
    });
    cards.children[best.i].classList.add("win");
    vals.sort((x,y)=>y.c-x.c);
    verdict.innerHTML="At α = "+(+slider.value)+"%, the ranking is "
      + vals.map(v=>"<span style='color:"+v.color+"'>"+v.label.replace("the ","")+"</span>")
            .join(" &gt; ")
      + ". Best: <b>"+best.label+"</b> at "+(best.c>=0?"+":"")+best.c.toFixed(1)+" CHF."
      + (+slider.value===100 ? "  <i>(α = 100% is just the plain mean, which is what your predecessor optimised.)</i>" : "");
  }

  slider.value=D.a0;
  slider.addEventListener("input",update);
  root.querySelectorAll(".cv-preset").forEach(b=>b.addEventListener("click",()=>{
    slider.value=b.dataset.a; update(); }));
  update();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(payload))
    display(HTML(html))


def budget_lab(candidates, budget_init=5.0):
    """Interactive: drag the damage budget d and watch which routes stay affordable.

    candidates: list of dicts with keys name, avg_money, avg_damage, fall_pct
                (fall_pct in percent), optionally color.
    """
    pols = []
    for cand in candidates:
        color = cand.get("color") or ROUTE_COLORS.get(cand["name"], C_VIOLET)
        pols.append({"label": cand["name"], "color": color,
                     "money": round(float(cand["avg_money"]), 2),
                     "damage": round(float(cand["avg_damage"]), 2),
                     "fall": round(float(cand["fall_pct"]), 2)})
    payload = {"pols": pols, "d0": float(budget_init)}
    uid = "bl_" + str(abs(hash(str(payload))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;
  padding:16px;max-width:780px;background:#fff;color:#1b1b1b;color-scheme:light}
#__UID__ .bl-head{font-weight:800;font-size:15px;color:#1b1b1b;margin-bottom:2px}
#__UID__ .bl-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ canvas{display:block;width:100%;height:auto;border:1px solid #eceef4;border-radius:8px;background:#fbfcfe}
#__UID__ .bl-ctl{display:flex;align-items:center;gap:12px;margin:14px 0 4px;flex-wrap:wrap}
#__UID__ .bl-ctl label{font-size:13px;font-weight:700;color:#1b1b1b;white-space:nowrap}
#__UID__ input[type=range]{flex:1;min-width:220px;accent-color:#764ba2}
#__UID__ .bl-d{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:16px;
  font-weight:800;color:#764ba2;min-width:120px;text-align:right}
#__UID__ .bl-preset{cursor:pointer;border:1px solid #d7d9e3;background:#fff;color:#444;
  border-radius:999px;padding:4px 11px;font-size:12px;font-weight:700}
#__UID__ .bl-preset:hover{border-color:#764ba2;color:#764ba2}
#__UID__ .bl-cards{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
#__UID__ .bl-card{flex:1;min-width:200px;border:1.5px solid #e2e5ef;border-radius:10px;padding:10px 12px;transition:opacity .15s}
#__UID__ .bl-card.out{opacity:.42}
#__UID__ .bl-card.win{box-shadow:0 0 0 2px rgba(118,75,162,.22)}
#__UID__ .bl-name{font-size:12.5px;font-weight:700;margin-bottom:3px}
#__UID__ .bl-val{font-size:21px;font-weight:800;font-variant-numeric:tabular-nums;color:#1b1b1b}
#__UID__ .bl-note{font-size:11.5px;color:#666;margin-top:3px;line-height:1.5}
#__UID__ .bl-badge{display:inline-block;font-size:10.5px;font-weight:800;border-radius:999px;
  padding:2px 8px;margin-top:6px;background:#e7f7ec;color:#1c6b45;border:1px solid #cbead8}
#__UID__ .bl-card.out .bl-badge{background:#fdeeec;color:#a33d36;border-color:#f3cdc8}
#__UID__ .bl-verdict{margin-top:12px;font-size:13.5px;font-weight:700;color:#1b1b1b;
  background:#f6f7fb;border:1px solid #e6e8ee;border-radius:10px;padding:10px 12px;line-height:1.6}
</style>
<div id="__UID__">
  <div class="bl-head">🎛️ The budget lab — drag the insurer's cap and watch the menu change</div>
  <div class="bl-sub">Each dot is one route's <b>measured average damage per trip</b>. Routes to the
    <b>left</b> of the purple line fit the budget; the best affordable one (by expected profit) wins.</div>
  <canvas width="740" height="170"></canvas>
  <div class="bl-ctl">
    <label>budget d =</label>
    <input type="range" min="0" max="25" step="0.5">
    <span class="bl-d"></span>
  </div>
  <div class="bl-ctl" style="margin-top:0">
    <button class="bl-preset" data-d="0.5">0.5 — near-zero tolerance</button>
    <button class="bl-preset" data-d="5">5 — the insurer's letter</button>
    <button class="bl-preset" data-d="20">20 — damage is cheap</button>
  </div>
  <div class="bl-cards"></div>
  <div class="bl-verdict"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const cv=root.querySelector("canvas"), cx=cv.getContext("2d");
  const slider=root.querySelector("input[type=range]"), dOut=root.querySelector(".bl-d");
  const cards=root.querySelector(".bl-cards"), verdict=root.querySelector(".bl-verdict");
  const P=D.pols;
  const L=30,R=20,T=26,B=34,W=cv.width,H=cv.height;
  const X=v=>L+(v/25)*(W-L-R);

  P.forEach((p,i)=>{
    const d=document.createElement("div"); d.className="bl-card"; d.style.borderColor=p.color;
    d.innerHTML='<div class="bl-name" style="color:'+p.color+'">'+p.label+'</div>'
      +'<div class="bl-val">'+(p.money>=0?"+":"")+p.money.toFixed(2)+' CHF</div>'
      +'<div class="bl-note">average damage <b>'+p.damage.toFixed(2)+'</b> points/trip · '
      +p.fall.toFixed(2)+'% of trips end in the gorge</div>'
      +'<span class="bl-badge"></span>';
    cards.appendChild(d);
  });

  function draw(d){
    cx.clearRect(0,0,W,H);
    cx.strokeStyle="#eceef4"; cx.fillStyle="#8a8f9c"; cx.font="11px system-ui";
    cx.textAlign="center"; cx.textBaseline="top";
    for(let v=0;v<=25;v+=5){
      cx.beginPath(); cx.moveTo(X(v),T); cx.lineTo(X(v),H-B); cx.stroke();
      cx.fillText(v, X(v), H-B+7);
    }
    cx.fillText("measured average damage per trip (points)  →", (L+W-R)/2, H-B+21);
    // affordable zone
    cx.fillStyle="rgba(118,75,162,0.07)";
    cx.fillRect(L,T,X(d)-L,H-B-T);
    // budget line
    cx.strokeStyle="#764ba2"; cx.lineWidth=2.4; cx.setLineDash([6,4]);
    cx.beginPath(); cx.moveTo(X(d),T-8); cx.lineTo(X(d),H-B); cx.stroke(); cx.setLineDash([]);
    cx.fillStyle="#764ba2"; cx.font="bold 11.5px system-ui"; cx.textAlign=(d>20?"right":"left");
    cx.fillText("budget d = "+d.toFixed(1), X(d)+(d>20?-6:6), T-18);
    // route lollipops
    P.forEach((p,i)=>{
      const y=T+22+i*((H-B-T-30)/Math.max(P.length-1,1));
      const ok=p.damage<=d;
      cx.globalAlpha=ok?1:0.35;
      cx.strokeStyle=p.color; cx.lineWidth=2;
      cx.beginPath(); cx.moveTo(X(0),y); cx.lineTo(X(Math.min(p.damage,25)),y); cx.stroke();
      cx.fillStyle=p.color;
      cx.beginPath(); cx.arc(X(Math.min(p.damage,25)),y,6,0,7); cx.fill();
      cx.font="bold 11px system-ui"; cx.textAlign="left"; cx.textBaseline="middle";
      // a white halo, so a label sitting on the budget line stays readable
      cx.strokeStyle="#fff"; cx.lineWidth=3.5; cx.lineJoin="round";
      cx.strokeText(p.label.replace("the ",""), X(Math.min(p.damage,25))+11, y);
      cx.fillText(p.label.replace("the ",""), X(Math.min(p.damage,25))+11, y);
      cx.globalAlpha=1;
    });
  }

  function update(){
    const d=+slider.value;
    dOut.textContent=d.toFixed(1)+" pts/trip";
    draw(d);
    let best=null;
    P.forEach((p,i)=>{
      const ok=p.damage<=d, card=cards.children[i];
      card.classList.toggle("out",!ok);
      card.classList.remove("win");
      card.querySelector(".bl-badge").textContent= ok?"WITHIN BUDGET":"OVER BUDGET";
      if(ok&&(best===null||p.money>best.money)) best={...p,i:i};
    });
    if(best!==null){
      cards.children[best.i].classList.add("win");
      verdict.innerHTML="At d = "+d.toFixed(1)+", the best route you can afford is "
        +"<span style='color:"+best.color+"'>"+best.label+"</span> — expected profit <b>"
        +(best.money>=0?"+":"")+best.money.toFixed(2)+" CHF</b>, average damage "
        +best.damage.toFixed(2)+" points ("+best.fall.toFixed(2)+"% of trips end in the gorge).";
    }else{
      verdict.innerHTML="No route fits this budget — the walker stays in the depot. "
        +"A budget can be so strict that the only compliant policy is not operating at all.";
    }
  }

  slider.value=D.d0;
  slider.addEventListener("input",update);
  root.querySelectorAll(".bl-preset").forEach(b=>b.addEventListener("click",()=>{
    slider.value=b.dataset.d; update(); }));
  update();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(payload))
    display(HTML(html))


# ===========================================================================
#  §4  Quiz banks (the answer keys live here, not in the notebook)
# ===========================================================================
_MC_QUIZZES = {
    "two_offers": (
        "\U0001F9E0 Before any code — which offer do you take?",
        "Offer <b>A</b> pays 7, 9, 11 or 13 M, each equally likely. Offer <b>B</b> pays "
        "<b>25 M in four years out of five</b>, and <b>loses 50 M in the fifth</b>. Both average "
        "exactly +10 M a year. Which is the right answer?",
        ["A. It is obviously safer, and safer is always better.",
         "B. It pays more than double, four years out of five — the odds are clearly on its side.",
         "Neither, as a <i>calculation</i>. The expected values are identical, so expected value "
         "cannot decide this. Picking one means adding a second number — something that "
         "describes the bad years — and deciding how much of that you can survive.",
         "B, because over enough years the average converges to +10 anyway, so the risk washes out."],
        2,
        "This is the whole notebook in one question. The mean is <i>indifferent</i> here — it "
        "genuinely cannot tell A from B — yet almost nobody is indifferent. That gap is where "
        "risk-aware RL lives.<br><br>Option 4 is the seductive one: the average really does "
        "converge, but only if you are still solvent when it gets there. Sign B for a decade and "
        "roughly one year in five costs you five years of A’s earnings. The robot's panorama "
        "route was profitable on average too, right up to the morning it was not."),
    "cmdp_elements": (
        "🧠 The constrained MDP, matched to the mountain",
        "The board's sentence was: <i>“keep expected profit per trip high, and keep expected damage "
        "per trip under the insurer's cap.”</i> Which mapping of story → mathematics is right?",
        ["Profit is the <b>reward r</b> (maximise its expected total). Damage is the <b>cost c</b> — "
         "a separate signal, not negative reward. The insurer's cap is the <b>budget d</b> — a "
         "constraint someone in charge signs, not something the algorithm learns.",
         "Damage should simply be added to the reward as −100 CHF per fall — one number is tidier "
         "than two, and the mathematics comes out the same.",
         "The budget d is estimated by the agent from experience, like everything else in RL.",
         "Reward and cost must be measured in the same units (CHF), otherwise the constrained "
         "problem cannot be written down."],
        0,
        "Separation is the entire point. The moment you write −100 CHF into the reward, someone has "
        "<i>implicitly priced</i> a fall at exactly 100 CHF — without the board ever discussing it. "
        "The CMDP keeps the books apart (they do not even share units: CHF vs damage points) and "
        "turns the exchange rate into an explicit management decision. And d is the one number no "
        "amount of data can produce: it encodes what the organisation can survive."),
    "shield_limits": (
        "🧠 What did the shield actually buy?",
        "The shielded planner walks the forest detour: zero falls, ever, and about 16 CHF/trip "
        "less profit than the risk-neutral optimum. Which verdict on <b>preemptive shielding</b> "
        "is the fair one?",
        ["It made the walks safer <i>and</i> more profitable, a free lunch thanks to better planning.",
         "It is a switch, not a dial: the mask treats undermined ground, where collapses are rare, "
         "exactly like the gorge rim and the open pits. Off is off. You bought <i>all</i> the "
         "safety, paid full price for it, and no setting exists between “everything allowed” and "
         "“nothing risky allowed”.",
         "It changed the reward function, so the comparison with Part 2's first planner is unfair.",
         "It learned the fall probabilities from experience and blocked only the truly dangerous moves."],
        1,
        "The shield is honest, auditable and absolute, which is exactly what a regulator wants, but "
        "it is binary and it only knows what its <i>hand-made list</i> knows: a danger nobody drew "
        "on the map never gets masked, and a harmless cell someone flags in a panic is forbidden "
        "forever. It also cannot answer the board's actual question, which is not “is this risky?” "
        "but “<b>how much</b> risk, at <b>what price</b>?” That question needs Part 3."),
    "price_vs_budget": (
        "🧠 The two dials on the boardroom table",
        "You now have a <b>budget d</b> (expected damage per trip must stay under it) and a "
        "<b>price λ</b> (every damage point costs the planner λ CHF of imaginary money). Which "
        "statement is correct?",
        ["They are two names for the same dial, so either will do in the risk policy document.",
         "d is a <b>promise about outcomes</b> (“we will not average more than d damage per trip”) "
         "and λ is an <b>internal price</b> that makes the planner keep that promise. Management "
         "owns d; λ can then be discovered automatically — the dual-ascent loop moves the price "
         "until the chosen route is compliant with the budget.",
         "λ is the number the insurer cares about, and d is a technical detail of the solver.",
         "Once λ is found, the budget d is no longer needed and can be deleted from the policy."],
        1,
        "A budget speaks the board's language (outcomes: damage per trip); a price speaks the "
        "planner's language (CHF, comparable with profit). Dual ascent is the translator: state "
        "the promise, and the loop finds the internal price that enforces it. Delete d, though, "
        "and λ loses its anchor — it was only ever correct <i>relative to the budget</i>."),
    "transfer": (
        "🧠 Take it back to the office",
        "A bank's system approves consumer loans and is rewarded for interest income. Once in a "
        "while it approves a cluster of loans that default together and eat a year of profit. "
        "The board wants the <i>Part 3</i> treatment, not the Part 2 one. Which intervention is that?",
        ["Hand-list the forbidden segments (no loans to X, Y, Z) and filter applications before "
         "the model ever scores them.",
         "Track expected cluster-default loss as a separate cost signal, cap its expected value "
         "per portfolio at a board-approved budget, and let an internal price on default risk "
         "steer the optimiser toward the most profitable portfolio that fits the cap.",
         "Retrain on more historical data so the average return estimate gets sharper.",
         "Turn the system off during volatile quarters."],
        1,
        "Option 1 is the shield — legitimate, auditable, and exactly as blunt as it was on the "
        "mountain (it forbids whole segments regardless of how profitable or mildly risky they "
        "are). The Part 3 move keeps a separate cost signal, puts a <b>budget</b> on it, and lets a "
        "<b>price</b> do the fine-grained trading under that budget.<br><br>And the third dial is "
        "already law in this industry: Basel's <b>expected shortfall</b> is CVaR, computed on the "
        "loss distribution at α = 2.5%. A bank that reports only its average P&amp;L is doing "
        "exactly what your predecessor did on the mountain."),
}

_TF_QUIZZES = {
    "reward_vs_cost": (
        "🧠 Reward and cost, sorted",
        [("The −1 per move, the +5 panorama tips and the +50 delivery bonus are all part of the "
          "reward r.", True),
         ("A fall shows up in the reward as −100 CHF.", False),
         ("The reward cannot tell undermined ground from solid road: both pay +1.", True),
         ("Cost is measured in damage points, so expected damage per trip equals 100 × the "
          "probability that the trip ends in a fall.", True),
         ("The budget d is estimated by the planning algorithm.", False),
         ("Writing the fall into the reward as −100 CHF would silently fix the price of a "
          "disaster at 100 CHF — a management decision made by a data structure.", True)]),
    "three_dials": (
        "🧠 The three dials, sorted",
        [("The budget d and the price λ both read a single number off the damage distribution: "
          "its mean.", True),
         ("CVaR with α = 100% is exactly the ordinary mean, so it reproduces the risk-neutral "
          "answer.", True),
         ("A smaller α means a more paranoid question, because the window keeps only the worst "
          "trips.", True),
         ("α is a property of the mountain, which the planner can measure from the map.", False),
         ("Because the panorama path has the best mean, it must also have the best CVaR at every "
          "α.", False),
         ("The service road beats the panorama in the middle of the α range but loses at both "
          "ends, which is a preference no single average could produce.", True),
         ("Computing the outcome distribution required training the agent on sampled trips.",
          False)]),
    "wrapup": (
        "🏁 Final check — the whole notebook in seven claims",
        [("The risk-neutral planner chose the panorama path because it genuinely had the highest "
          "expected profit — the algorithm did exactly its job.", True),
         ("Separating cost from reward changed the panorama path's profitability.", False),
         ("Preemptive shielding filters actions before they are taken, using a hand-made list "
          "of unsafe moves — the planner never even sees them.", True),
         ("The λ-price and the budget d are connected: the dual-ascent loop moves the price until "
          "the measured damage sits near the budget.", True),
         ("With the insurer's budget of 5 damage points per trip, the best affordable route was "
          "the old service road, a route nobody hand-designed.", True),
         ("Risk-aware planning is free: the forest detour earns as much as the panorama path.", False),
         ("The budget d is a number management must choose; no amount of data will choose it "
          "for them.", True)]),
}

_NUMBER_QUIZZES = {
    "route_prices": (
        "🔢 Price the three routes yourself",
        [("A <b>clean panorama walk</b>: 7 moves of effort (−1 each), four scenic stretches of "
          "tips (+5 each), and the +50 delivery bonus. <b>How much does a clean panorama walk "
          "earn?</b>",
          63, 0.5,
          "−7 + 20 + 50 = <b>+63</b>. The good days really are good, and that is exactly what "
          "seduced your predecessor's optimizer."),
         ("The <b>old service road</b>: 9 moves of effort, and it pays +1 for each of the 8 road "
          "stretches you cross, plus +50 at the lodge. <b>How much on a clean run?</b>",
          49, 0.5,
          "−9 + 8 + 50 = <b>+49</b>. Slower and plainer than the panorama, but it still pays its "
          "way, which is what will make it interesting later."),
         ("The <b>forest detour</b>: 15 moves of effort, two road stretches on the way out and "
          "back (+1 each), +50 at the lodge, and nothing can go wrong. <b>How much, every single "
          "time?</b>",
          37, 0.5,
          "−15 + 2 + 50 = <b>+37</b>, with zero variance: the same number every trip, forever.")]),
}


def mc_quiz(key):
    _mc_render(*_MC_QUIZZES[key])


def true_false_quiz(key):
    title, statements = _TF_QUIZZES[key]
    _tf_render(title, statements)


def number_quiz(key):
    title, questions = _NUMBER_QUIZZES[key]
    _nq_render(title, questions)
