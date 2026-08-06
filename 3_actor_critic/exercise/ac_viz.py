"""Presentation & quiz helpers for the WE4 notebook 03:
"Actor–Critic — learning a critic instead of averaging a batch".

Same idea as WE0's `pdm_viz`, WE3's `we3_viz` and WE4/02's `pg_viz`: every
HTML/CSS illustration, interactive widget, quiz *answer key* and matplotlib
visual lives here, out of the notebook, so the teaching cells stay about the
*idea* and the quizzes can't be solved by reading the cell. The notebook does::

    import ac_viz
    ac_viz.campaign_recap()
    ac_viz.mc_quiz("g_is_q")

The campaign is the SAME one as notebook 02 (Owlinguo's three-day win-back
campaign), so the tables below are identical to `pg_viz`'s on purpose.

Students are told not to read this file.
"""
import json as _json

import numpy as np
from IPython.display import HTML, display

# ===========================================================================
#  The campaign vocabulary (identical to notebook 02 — same world, new method)
# ===========================================================================
ENGAGE = ["Cold", "Warm", "Hot"]
ENGAGE_EMOJI = ["😴", "🙂", "🔥"]
ENGAGE_COLOR = ["#8d93a8", "#dd8452", "#c0554e"]

ACTIONS = ["Wait", "Nudge", "Ad blast"]
ACTION_EMOJI = ["⏸️", "🔔", "📺"]
ACTION_COLOR = ["#9aa0b5", "#4a5bd0", "#2e9e7a"]

#   TRANS[s][a] = list of (next engagement, probability)
TRANS = [
    [[(0, 1.0)],              [(1, 0.7), (0, 0.3)],  [(0, 1.0)]],              # 😴 Cold
    [[(0, 0.7), (1, 0.3)],    [(2, 0.7), (1, 0.3)],  [(0, 0.8), (1, 0.2)]],    # 🙂 Warm
    [[(1, 0.7), (2, 0.3)],    [(2, 1.0)],            [(0, 0.8), (2, 0.2)]],    # 🔥 Hot
]
REWARD = [[0.0, -0.5, -0.5],
          [1.0, 0.5, 2.0],
          [3.0, 2.5, 6.0]]
START_PROBS = [0.40, 0.60, 0.00]
N_DAYS = 3

ACTOR_COLOR = "#4a5bd0"
CRITIC_COLOR = "#a3652f"


def _e(i):
    return "%s %s" % (ENGAGE_EMOJI[i], ENGAGE[i])


def _a(i):
    return "%s %s" % (ACTION_EMOJI[i], ACTIONS[i])


# ===========================================================================
#  Generic renderers  (ported from pg_viz so this module stands alone)
# ===========================================================================
def _card(inner, maxw=860):
    display(HTML(
        '<div style="font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid '
        '#e6e8ee;border-radius:14px;padding:18px;max-width:%dpx;background:#fff">%s</div>'
        % (maxw, inner)))


def _mc_render(title, question, options, answer_index, reveal):
    data = {"opts": list(options), "ans": int(answer_index), "reveal": reveal}
    uid = "mc_" + str(abs(hash((question, tuple(options), answer_index))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff}
#__UID__ .mc-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .mc-q{color:#444;font-size:13.5px;margin-bottom:12px;line-height:1.55}
#__UID__ .mc-opt{display:flex;align-items:flex-start;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;cursor:pointer;font-size:13.5px;line-height:1.5;transition:.12s}
#__UID__ .mc-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .mc-dot{width:16px;height:16px;border-radius:50%;border:2px solid #c2c7da;flex:0 0 auto;margin-top:2px}
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
  // shuffle the options so the correct one is not in a predictable slot
  let idx=D.opts.map((_,i)=>i);
  for(let i=idx.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  const list=root.querySelector(".mc-list");
  idx.forEach(orig=>{
    const o=document.createElement("div"); o.className="mc-opt"; o.dataset.i=orig;
    o.innerHTML='<span class="mc-dot"></span>'+D.opts[orig];
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
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff}
#__UID__ .tf-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .tf-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .tf-opt{display:flex;align-items:center;gap:10px;border:1px solid #e2e5ef;border-radius:10px;padding:9px 12px;margin-bottom:7px;cursor:pointer;font-size:13.5px;line-height:1.5}
#__UID__ .tf-opt:hover{border-color:#764ba2;background:#faf9ff}
#__UID__ .tf-box{width:16px;height:16px;border-radius:4px;border:2px solid #c2c7da;flex:0 0 auto}
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
    row.innerHTML='<span class="tf-box"></span>'+d.t;
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
               prompt="Work each number out, type it in, then check.", header=""):
    """questions: list of (question_html, answer_number, tolerance, reveal).
    `header` is optional HTML shown above the questions (e.g. a table to read off)."""
    items = [{"q": q, "a": float(a), "tol": float(tol), "rev": rev}
             for q, a, tol, rev in questions]
    uid = "nq_" + str(abs(hash((title, tuple(q for q, _, _, _ in questions)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:780px;background:#fff}
#__UID__ .nq-head{font-weight:800;font-size:15px;margin-bottom:4px}
#__UID__ .nq-sub{color:#666;font-size:12.5px;margin-bottom:12px}
#__UID__ .nq-row{border:1px solid #e2e5ef;border-radius:10px;padding:10px 12px;margin-bottom:8px;font-size:13.5px;line-height:1.55}
#__UID__ .nq-row.ok{border-color:#46b46e;background:#e7f7ec}
#__UID__ .nq-row.no{border-color:#e07a7a;background:#fdecec}
#__UID__ input{width:110px;padding:5px 8px;border:1px solid #c2c7da;border-radius:7px;font-size:13px;margin-top:6px}
#__UID__ .nq-rev{font-size:12.5px;color:#3b2d6b;margin-top:6px;display:none}
#__UID__ .nq-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 18px;font-size:13.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:6px}
#__UID__ .nq-status{font-size:13px;font-weight:700;color:#3b2d6b;margin-top:10px;min-height:18px}
</style>
<div id="__UID__">
  <div class="nq-head">__TITLE__</div>
  <div class="nq-sub">__PROMPT__</div>
  __HEADER__
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
            .replace("__PROMPT__", prompt).replace("__HEADER__", header)
            .replace("__DATA__", _json.dumps(items)))
    display(HTML(html))


def _reward_table_html():
    """The reward table, compact — so a quiz can be answered without scrolling back."""
    def money(v):
        col = "#1d7a46" if v > 0 else ("#b23b34" if v < 0 else "#777")
        return ('<td style="padding:5px 12px;text-align:center;font-weight:700;font-size:12.5px;'
                'color:%s">%+.1f</td>' % (col, v))
    head = ('<tr><th style="padding:4px 10px"></th>' + "".join(
        '<th style="padding:4px 12px;font-size:11px;color:%s">%s</th>' % (ACTION_COLOR[a], _a(a))
        for a in range(len(ACTIONS))) + '</tr>')
    rows = "".join(
        '<tr><td style="padding:5px 10px;font-weight:700;font-size:12.5px;color:%s">%s</td>%s</tr>'
        % (ENGAGE_COLOR[s], _e(s), "".join(money(float(REWARD[s][a])) for a in range(len(ACTIONS))))
        for s in range(len(ENGAGE)))
    return ('<div style="background:#fafbff;border:1px solid #e6e8ee;border-radius:9px;'
            'padding:9px 11px;margin-bottom:12px">'
            '<div style="font-size:11px;color:#888;font-weight:700;text-transform:uppercase;'
            'letter-spacing:.04em;margin-bottom:4px">💰 reward — revenue booked today (CHF)</div>'
            '<table style="border-collapse:collapse">%s%s</table></div>' % (head, rows))


# ===========================================================================
#  §1  Where notebook 02 left us
# ===========================================================================
def campaign_recap():
    """The same campaign as notebook 02, on one card — states, actions, money."""
    def chip(emoji, name, sub, color):
        return ('<div style="flex:1;min-width:140px;border:2px solid %s;border-radius:12px;'
                'padding:9px 11px;background:#fff">'
                '<div style="font-size:20px">%s</div>'
                '<div style="font-weight:800;font-size:13px;color:#222">%s</div>'
                '<div style="font-size:11.5px;color:#777;line-height:1.45;margin-top:2px">%s</div></div>'
                % (color, emoji, name, sub))

    states = "".join(chip(ENGAGE_EMOJI[i], ENGAGE[i], sub, ENGAGE_COLOR[i]) for i, sub in enumerate([
        "streak broken, hasn't opened the app in weeks — no lessons, no ads, <b>no revenue</b>. "
        "<b>40%</b> of the learners who enter the campaign",
        "opens it a few times a week, does the odd lesson. <b>60%</b> of them",
        "back on a daily streak — a lesson, and an ad, every morning. <i>Nobody enters here</i>"]))
    acts = "".join(chip(ACTION_EMOJI[i], ACTIONS[i], sub, ACTION_COLOR[i]) for i, sub in enumerate([
        "no contact today. Free — and streaks decay on their own",
        "a message from their coach with one lesson picked for them. Costs a little; "
        "often moves the learner <b>up</b> a level",
        "double the ad load today. Much more revenue from whoever is still learning — "
        "and most of them get fed up and <b>drop out</b>"]))

    def money(v):
        col = "#1d7a46" if v > 0 else ("#b23b34" if v < 0 else "#777")
        return ('<td style="padding:6px 12px;text-align:center;font-weight:700;font-size:12.5px;'
                'color:%s">%+.1f</td>' % (col, v))

    head = ('<tr><th></th>' + "".join('<th style="padding:5px 12px;font-size:11.5px;color:%s">%s</th>'
                                      % (ACTION_COLOR[a], _a(a)) for a in range(3)) + "</tr>")
    rrows = "".join(
        '<tr><td style="padding:6px 10px;font-weight:700;font-size:12.5px;color:%s">%s</td>%s</tr>'
        % (ENGAGE_COLOR[s], _e(s), "".join(money(float(REWARD[s][a])) for a in range(3)))
        for s in range(3))

    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:4px">'
        '📋 The same campaign as notebook 02 · Owlinguo\'s three-day win-back</div>'
        '<div style="font-size:12.5px;color:#555;line-height:1.6;margin-bottom:14px">'
        'A free language-learning app that makes money one way: <b>one ad before every lesson</b>. '
        'When a learner\'s streak breaks you have <b>three days</b> to win them back, and every morning '
        'you pick <b>one</b> move. Nothing about the world has changed since notebook 02 — what changes '
        'now is <i>how we learn from it</i>.</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">The state — how engaged the learner is</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px">%s</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">The actions — what you can do about it</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px">%s</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">💰 Reward — expected ad revenue booked today (CHF)</div>'
        '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table>'
        % (states, acts, head, rrows))


def where_we_left_off():
    """The estimator we ended notebook 02 with, and its two open problems."""
    def problem(n, title, body, fix):
        return ('<div style="flex:1;min-width:250px;border:2px solid #c0554e;border-radius:12px;'
                'padding:12px 13px;background:#fffafa">'
                '<div style="font-size:10.5px;font-weight:800;color:#c0554e;letter-spacing:.04em">'
                'PROBLEM %s</div>'
                '<div style="font-weight:800;font-size:13.5px;color:#2b2d6b;margin:3px 0 5px">%s</div>'
                '<div style="font-size:12px;color:#555;line-height:1.55">%s</div>'
                '<div style="font-size:12px;color:#1d7a46;line-height:1.55;margin-top:7px">'
                '<b>The fix →</b> %s</div></div>' % (n, title, body, fix))

    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:6px">'
        '📦 What notebook 02 ended with — and what it left unfinished</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:12px 14px;font-size:13px;'
        'color:#333;line-height:1.7;margin-bottom:14px">'
        'REINFORCE with a baseline: run a batch of complete campaigns, score each step by '
        '<b>what it still earned</b>, subtract <b>what a batch normally earns from that level</b>, '
        'push the actions accordingly. It worked — it found the optimal sheet.</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap">%s%s</div>'
        % (problem("1", "We could only learn once the campaign was over",
                   "The weight on day 0\'s action was its <b>return-to-go</b> — and that number does not "
                   "exist until day 3 has happened. Handle a learner badly on Monday and the algorithm "
                   "cannot react on Tuesday. Useless for anything that never ends.",
                   "learn from a <b>single day</b>: (s, a, r, s′)."),
           problem("2", "The baseline was a crude stand-in",
                   "We wanted “what we expect from this situation” and we settled for "
                   "<b>the average of this batch</b> — noisy, blind to the day, and thrown away every "
                   "iteration instead of accumulating knowledge.",
                   "<b>learn</b> that expectation, and keep it.")))


def score_gradient_recap():
    """The estimator from notebook 02, with its 'advantage' term flagged as fake."""
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:8px">'
        '🧾 The rule we ended notebook 02 with</div>'
        '<div style="background:#fafbff;border:1px solid #e6e8ee;border-radius:10px;padding:14px;'
        'text-align:center;font-size:15px;color:#222;line-height:2">'
        '∇<sub>θ</sub>J &nbsp;=&nbsp; 𝔼<sub>τ∼π<sub>θ</sub></sub>'
        '[ &nbsp;Σ<sub>t</sub> &nbsp;γ<sup>t</sup> &nbsp;'
        '<span style="background:#fdecec;border-radius:6px;padding:3px 8px;color:#b23b34;'
        'font-weight:700">( G<sub>t</sub> − b(s<sub>t</sub>) )</span>'
        '&nbsp; ∇<sub>θ</sub> log π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) &nbsp;]</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:14px">'
        '<div style="flex:1;min-width:230px;border:2px solid #4a5bd0;border-radius:12px;padding:11px 12px">'
        '<div style="font-weight:800;font-size:13px;color:#4a5bd0">G<sub>t</sub> — the return-to-go</div>'
        '<div style="font-size:12px;color:#555;line-height:1.55;margin-top:4px">what <b>this one</b> '
        'campaign happened to collect after step t. One roll of the dice.</div></div>'
        '<div style="flex:1;min-width:230px;border:2px solid #a3652f;border-radius:12px;padding:11px 12px">'
        '<div style="font-weight:800;font-size:13px;color:#a3652f">b(s<sub>t</sub>) — the baseline</div>'
        '<div style="font-size:12px;color:#555;line-height:1.55;margin-top:4px">the average return-to-go '
        'in the <b>current batch</b>, per engagement level. A stand-in for something we never named.'
        '</div></div></div>'
        '<div style="background:#f3f0ff;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:13px;color:#2c2350;line-height:1.65">We called that red bracket the '
        '<b>“advantage”</b> and moved on. It was not one — it was a sampled number minus a '
        'batch average. This notebook replaces <i>both</i> terms with the objects they were imitating, '
        'and the whole algorithm changes shape as a result.</div>')


# ===========================================================================
#  §2  G is a sample of Q  ·  the value functions
# ===========================================================================
def _probs(theta):
    """Softmax rows of a 3x3 logit table (accepts a torch tensor, list or array)."""
    z = np.asarray(theta.tolist() if hasattr(theta, "tolist") else theta, float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _exact_values(theta, gamma=0.9):
    """V(day, state) and Q(day, state, action) of a policy, by backward induction.
    Duplicated here only so the widgets can draw the truth; the notebook writes its
    own version as an exercise."""
    pi = _probs(theta)
    V = np.zeros((N_DAYS + 1, len(ENGAGE)))
    Q = np.zeros((N_DAYS, len(ENGAGE), len(ACTIONS)))
    for t in reversed(range(N_DAYS)):
        for st in range(len(ENGAGE)):
            for a in range(len(ACTIONS)):
                Q[t, st, a] = REWARD[st][a] + gamma * sum(p * V[t + 1][ns]
                                                          for ns, p in TRANS[st][a])
            V[t, st] = float((pi[st] * Q[t, st]).sum())
    return V[:N_DAYS], Q


def _replay(theta, day, state, first_action, gamma, n):
    """Play the campaign from (day, state) n times — forcing the first move — and
    return the discounted return-to-go collected each time."""
    pi = _probs(theta)
    rng = np.random.default_rng(0)
    out = np.zeros(n)
    for k in range(n):
        st, G, disc = state, 0.0, 1.0
        for t in range(day, N_DAYS):
            a = first_action if (t == day and first_action is not None) else int(
                rng.choice(len(ACTIONS), p=pi[st]))
            G += disc * REWARD[st][a]
            disc *= gamma
            nxt = TRANS[st][a]
            st = int(rng.choice([x for x, _ in nxt], p=[p for _, p in nxt]))
        out[k] = G
    return out


def fixed_setup(theta, day, state, action, gamma=0.9):
    """The policy we hold fixed, and the one (morning, learner, move) we are about
    to replay a few thousand times."""
    pi = _probs(theta)
    head = ('<tr><th style="padding:4px 10px"></th>' + "".join(
        '<th style="padding:4px 12px;font-size:11.5px;color:%s">%s</th>' % (ACTION_COLOR[a], _a(a))
        for a in range(len(ACTIONS))) + '</tr>')
    rows = ""
    for st in range(len(ENGAGE)):
        cells = "".join(
            '<td style="padding:6px 12px;text-align:center;font-size:12.5px;font-weight:%s;color:%s">'
            '%d%%</td>' % ("800" if pi[st][a] == pi[st].max() else "500",
                           ACTION_COLOR[a] if pi[st][a] == pi[st].max() else "#999",
                           int(round(100 * pi[st][a])))
            for a in range(len(ACTIONS)))
        rows += ('<tr><td style="padding:6px 10px;font-weight:700;font-size:12.5px;color:%s">%s</td>'
                 '%s</tr>' % (ENGAGE_COLOR[st], _e(st), cells))

    def pill(label, value, colour):
        return ('<div style="border:2px solid %s;border-radius:11px;padding:8px 14px;background:#fff">'
                '<div style="font-size:10px;color:#999;font-weight:700;text-transform:uppercase;'
                'letter-spacing:.04em">%s</div>'
                '<div style="font-weight:800;font-size:14px;color:%s;margin-top:2px">%s</div></div>'
                % (colour, label, colour, value))

    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🔒 What we hold fixed</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55">'
        'The policy below is <b>how we normally behave</b> for the whole of Part 2 — π(a|s), one row '
        'per engagement level. It never changes here; we are not learning yet, we are measuring.</div>'
        '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin:14px 0 6px">…and the one situation we replay</div>'
        '<div style="display:flex;gap:9px;flex-wrap:wrap;align-items:stretch">%s%s%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:13px;'
        'font-size:12.5px;color:#333;line-height:1.6">Every replay starts from exactly this: the same '
        'morning, the same learner, the same move. Only the dice differ — the world\'s answer to that '
        'move, and the moves the policy samples on the days after it.</div>'
        % (head, rows, pill("morning", "day %d" % day, "#8d93a8"),
           pill("learner", _e(state), ENGAGE_COLOR[state]),
           pill("move we force today", _a(action), ACTION_COLOR[action])))


def g_samples_q(theta, day, state, action, gamma=0.9, n=4000):
    """Replay one fixed (day, state, action) n times and histogram the returns-to-go,
    with the exact Q (their average) and V (the state's average over all actions)."""
    import matplotlib.pyplot as plt
    samples = _replay(theta, day, state, action, gamma, n)
    V, Q = _exact_values(theta, gamma)
    q_exact, v_exact = float(Q[day, state, action]), float(V[day, state])

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.hist(samples, bins=30, color="#4a5bd0", alpha=.55, edgecolor="white")
    ax.axvline(q_exact, color="#b23b34", lw=2.4,
               label="Q(day %d, %s, %s) = %.2f  — the exact average"
                     % (day, ENGAGE[state], ACTIONS[action], q_exact))
    ax.axvline(float(samples.mean()), color="#1d7a46", lw=1.8, ls="--",
               label="mean of the %d replays = %.2f" % (n, samples.mean()))
    ax.axvline(v_exact, color="#a3652f", lw=1.8, ls=":",
               label="V(day %d, %s) = %.2f  — the same state, averaged over ALL moves"
                     % (day, ENGAGE[state], v_exact))
    ax.set_xlabel("return-to-go $G_t$ collected on that one campaign   [CHF]")
    ax.set_ylabel("how many campaigns")
    ax.set_title("%d replays of the same morning, the same learner and the same move"
                 % n, fontsize=11.5, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()
    return samples


def value_table(V, title="V(day, engagement) — what a learner is worth from here",
                subtitle=None, fmt="%+.2f", highlight=None):
    """Render a (N_DAYS x 3) table of values, one row per day."""
    V = np.asarray(V, float)
    head = ('<tr><th style="padding:5px 10px"></th>' + "".join(
        '<th style="padding:5px 14px;font-size:12px;color:%s">%s</th>' % (ENGAGE_COLOR[s], _e(s))
        for s in range(len(ENGAGE))) + '</tr>')
    rows = []
    for t in range(V.shape[0]):
        cells = ""
        for s in range(V.shape[1]):
            hot = highlight is not None and tuple(highlight) == (t, s)
            cells += ('<td style="padding:7px 14px;text-align:center;font-weight:800;font-size:13px;'
                      'color:#2b2d6b;%s">%s</td>'
                      % ("background:#f3f0ff;border-radius:6px;" if hot else "", fmt % V[t, s]))
        rows.append('<tr><td style="padding:7px 10px;font-size:12.5px;color:#777;font-weight:700">'
                    'day %d</td>%s</tr>' % (t, cells))
    sub = subtitle or ('Read one cell: <i>a learner sitting at that engagement level, on that morning '
                       'of the campaign, is worth this much for the rest of the campaign</i> — under '
                       'the policy we are currently running.')
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:4px">📊 %s</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px;line-height:1.55">%s</div>'
          '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table>'
          % (title, sub, head, "".join(rows)), maxw=680)


def q_v_a_explorer(Q, V):
    """The three functions on one screen: pick a day and an engagement level, see
    Q(s,a) for the three actions, the V line running through them, and the
    advantages that fall out. Q: (N_DAYS,3,3), V: (N_DAYS,3)."""
    data = {"Q": np.asarray(Q, float).round(4).tolist(),
            "V": np.asarray(V, float).round(4).tolist(),
            "eng": ENGAGE, "ee": ENGAGE_EMOJI, "ec": ENGAGE_COLOR,
            "act": ACTIONS, "ae": ACTION_EMOJI, "ac": ACTION_COLOR, "days": N_DAYS}
    uid = "qva_" + str(abs(hash(str(data["Q"]))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:720px;background:#fff}
#__UID__ .q-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px}
#__UID__ .q-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .q-pick{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:14px}
#__UID__ .q-grp{font-size:11px;color:#888;font-weight:700;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
#__UID__ .q-chip{display:inline-block;cursor:pointer;border:2px solid #e2e5ef;border-radius:999px;padding:5px 12px;font-size:12.5px;font-weight:700;margin-right:6px;color:#555;background:#fff}
#__UID__ .q-chip.on{border-color:#764ba2;background:#f1edff;color:#3b2d6b}
#__UID__ .q-plot{position:relative;border-left:2px solid #e2e5ef;padding-left:10px}
#__UID__ .q-row{display:flex;align-items:center;gap:8px;margin:9px 0;font-size:12.5px}
#__UID__ .q-name{width:92px;font-weight:700}
#__UID__ .q-bar{flex:1;background:#eef0f7;border-radius:5px;height:20px;position:relative;overflow:hidden}
#__UID__ .q-bar>div{height:100%;transition:width .18s}
#__UID__ .q-val{width:52px;text-align:right;font-weight:800}
#__UID__ .q-adv{width:74px;text-align:right;font-weight:800;font-size:12.5px}
#__UID__ .q-vline{font-size:12.5px;color:#a3652f;font-weight:700;margin:10px 0 2px}
#__UID__ .q-note{font-size:12.5px;color:#444;background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:12px;line-height:1.6;min-height:56px}
</style>
<div id="__UID__">
  <div class="q-head">🔍 Q, V and A — the same numbers, three questions</div>
  <div class="q-sub">Pick a <b>morning</b> and a <b>learner</b>. The bars are <b>Q(s,a)</b>: what the
    campaign is worth from here if we take that move now and then carry on with the current policy.
    The brown marker is <b>V(s)</b>: what it is worth <i>before</i> we pick — the average of the bars,
    weighted by how often the policy picks each move. The right column is
    <b>A(s,a) = Q(s,a) − V(s)</b>.</div>
  <div class="q-pick">
    <div><div class="q-grp">morning</div><div class="q-day"></div></div>
    <div><div class="q-grp">learner</div><div class="q-eng"></div></div>
  </div>
  <div class="q-vline"></div>
  <div class="q-plot"></div>
  <div class="q-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  let day=0, eng=1;
  const dayBox=root.querySelector(".q-day"), engBox=root.querySelector(".q-eng");
  for(let t=0;t<D.days;t++){
    const c=document.createElement("span"); c.className="q-chip"; c.textContent="day "+t;
    c.addEventListener("click",()=>{day=t;draw();}); dayBox.appendChild(c);
  }
  D.eng.forEach((e,i)=>{
    const c=document.createElement("span"); c.className="q-chip";
    c.textContent=D.ee[i]+" "+e; c.style.borderColor="#e2e5ef";
    c.addEventListener("click",()=>{eng=i;draw();}); engBox.appendChild(c);
  });
  function draw(){
    [...dayBox.children].forEach((c,i)=>c.classList.toggle("on",i===day));
    [...engBox.children].forEach((c,i)=>c.classList.toggle("on",i===eng));
    const q=D.Q[day][eng], v=D.V[day][eng];
    const lo=Math.min(0,...q,v), hi=Math.max(...q,v,0.001), span=(hi-lo)||1;
    const px=x=>100*(x-lo)/span;
    root.querySelector(".q-vline").innerHTML=
      "V(day "+day+", "+D.ee[eng]+" "+D.eng[eng]+") = <b>"+v.toFixed(2)+"</b> CHF"
      +" &nbsp;<span style='color:#999;font-weight:400'>— the yardstick this state is judged against</span>";
    root.querySelector(".q-plot").innerHTML=q.map((qq,a)=>{
      const adv=qq-v;
      return '<div class="q-row"><div class="q-name" style="color:'+D.ac[a]+'">'+D.ae[a]+" "+D.act[a]+'</div>'
        +'<div class="q-bar"><div style="width:'+px(qq).toFixed(1)+'%;background:'+D.ac[a]+'"></div>'
        +'<div style="position:absolute;left:'+px(v).toFixed(1)+'%;top:0;width:2px;height:100%;background:#a3652f"></div>'
        +'</div>'
        +'<div class="q-val">'+qq.toFixed(2)+'</div>'
        +'<div class="q-adv" style="color:'+(adv>=0?"#1d7a46":"#b23b34")+'">'
        +(adv>=0?"↑ +":"↓ ")+adv.toFixed(2)+'</div></div>';
    }).join("");
    const best=q.indexOf(Math.max(...q)), worst=q.indexOf(Math.min(...q));
    let n="On this morning, with this learner, <b>"+D.ae[best]+" "+D.act[best]+"</b> is above the "
      +"yardstick (A = +"+(q[best]-v).toFixed(2)+") so the policy gets pushed <b>towards</b> it, and "
      +"<b>"+D.ae[worst]+" "+D.act[worst]+"</b> is below it (A = "+(q[worst]-v).toFixed(2)+") so it gets "
      +"pushed <b>away</b>. Note that the advantages straddle zero — that is not luck, it is arithmetic: "
      +"V is the policy's own average over those bars, so something must beat it and something must not.";
    if(day===D.days-1) n+=" <br><br>It is the <b>last</b> morning, so these numbers are just today's "
      +"revenue: there is no future left to add.";
    if(eng===2&&day===0) n+=" <br><br>Also compare across mornings: 🔥 Hot on day 0 is worth far more "
      +"than 🔥 Hot on day "+(D.days-1)+", because more campaign remains. One number per engagement "
      +"level — the batch-average baseline of notebook 02 — could not tell those apart.";
    root.querySelector(".q-note").innerHTML=n;
  }
  draw();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


def advantage_meaning():
    """One card: what the sign of A means, in the manager's language."""
    def box(sign, colour, title, body):
        return ('<div style="flex:1;min-width:220px;border:2px solid %s;border-radius:12px;'
                'padding:12px 13px;background:#fff">'
                '<div style="font-size:22px;font-weight:800;color:%s">%s</div>'
                '<div style="font-weight:800;font-size:13px;color:#2b2d6b;margin:2px 0 5px">%s</div>'
                '<div style="font-size:12px;color:#555;line-height:1.6">%s</div></div>'
                % (colour, colour, sign, title, body))
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '⚖️ A(s,a) = Q(s,a) − V(s) — “was that move better than my own habit?”</div>'
        '<div style="font-size:12.5px;color:#666;line-height:1.6;margin-bottom:12px">'
        'V(s) is not <i>the best</i> the state can do — it is what <b>this policy</b> normally gets '
        'there. So the advantage compares one move against the mix of moves we would usually make. '
        'That is exactly the comparison a manager makes when reviewing a decision: not against '
        'perfection, against <i>what we would otherwise have done</i>.</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap">%s%s%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:13px;'
        'font-size:12.5px;color:#333;line-height:1.65">Because V(s) is the policy\'s own average over '
        'its actions, the advantages in a state <b>average to exactly zero</b>. There is no state where '
        'everything is good news — which is precisely why this is a usable learning signal and a raw '
        'return is not.</div>'
        % (box("A &gt; 0", "#1d7a46", "better than my habit",
               "taking this move and then carrying on as usual beats carrying on as usual from the "
               "start. <b>Make it more likely.</b>"),
           box("A = 0", "#9aa0b5", "exactly my habit",
               "no information: this move is worth precisely what the policy already averages here. "
               "<b>Leave the probabilities alone.</b>"),
           box("A &lt; 0", "#b23b34", "worse than my habit",
               "the move costs us relative to what we would normally do in this situation. "
               "<b>Make it less likely.</b>")))


# ===========================================================================
#  §3  From (s,a,r,s′) to a learnable critic
# ===========================================================================
def one_step_diagram():
    """Why Q(s,a) = r + γ·V(s′) lets us drop Q entirely."""
    def node(txt, sub, colour, big=False):
        return ('<div style="border:2px solid %s;border-radius:12px;padding:%s;background:#fff;'
                'text-align:center;min-width:%dpx">'
                '<div style="font-weight:800;font-size:%dpx;color:%s">%s</div>'
                '<div style="font-size:11.5px;color:#666;margin-top:3px;line-height:1.45">%s</div></div>'
                % (colour, "13px 15px" if big else "10px 12px", 118 if big else 104,
                   14 if big else 13, colour, txt, sub))
    arrow = '<div style="align-self:center;font-size:19px;color:#9aa0b5">→</div>'
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🔗 One day of the campaign — and why the critic only needs V</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:14px;line-height:1.55">'
        'A single transition gives us four things. Everything the update needs is inside them.</div>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:stretch;margin-bottom:14px">'
        '%s%s%s%s%s%s%s</div>'
        '<div style="background:#fafbff;border:1px solid #e6e8ee;border-radius:10px;padding:14px;'
        'text-align:center;font-size:15.5px;color:#222;line-height:2.1">'
        'Q(s,a) &nbsp;=&nbsp; r &nbsp;+&nbsp; γ·V(s′)'
        '<div style="font-size:13px;color:#666;margin-top:4px">the money this move books today, plus '
        'what the learner is worth tomorrow</div>'
        '<div style="height:10px"></div>'
        'A(s,a) &nbsp;=&nbsp; Q(s,a) − V(s) &nbsp;=&nbsp; '
        '<span style="background:#e7f7ec;border-radius:6px;padding:3px 8px;color:#1d7a46;'
        'font-weight:700">r + γ·V(s′) − V(s)</span></div>'
        '<div style="background:#f3f0ff;border-radius:8px;padding:11px 13px;margin-top:13px;'
        'font-size:13px;color:#2c2350;line-height:1.65">The advantage is now written with '
        '<b>one unknown function</b> instead of two — and every symbol in it either came out of the '
        'transition we just observed, or is a call to V. <b>That is the whole reason a critic learns '
        'V and not Q</b>: one function, no action index, and every sample teaches it something.</div>'
        % (node("s", "the learner this morning", "#8d93a8"), arrow,
           node("a", "the move we picked", ACTOR_COLOR), arrow,
           node("r", "revenue booked today", "#1d7a46"), arrow,
           node("s′", "the learner tomorrow", "#8d93a8")))


def td_playground():
    """Watch one value estimate crawl toward its TD targets. Sliding α shows the
    bias/noise trade-off; the true value is drawn as the dashed line."""
    # a 🙂 Warm learner on day 1 under a fixed policy, with the true V drawn in
    data = {"outcomes": [  # (probability, reward today, V(tomorrow) of where we land, label)
        [0.35, 0.5, 5.00, "🔔 Nudge → 🔥 Hot"],
        [0.15, 0.5, 1.20, "🔔 Nudge → 🙂 Warm"],
        [0.21, 1.0, -0.20, "⏸️ Wait → 😴 Cold"],
        [0.09, 1.0, 1.20, "⏸️ Wait → 🙂 Warm"],
        [0.16, 2.0, -0.20, "📺 Ad blast → 😴 Cold"],
        [0.04, 2.0, 1.20, "📺 Ad blast → 🙂 Warm"]],
        "gamma": 0.9}
    uid = "td_" + str(abs(hash(str(data))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:700px;background:#fff}
#__UID__ .t-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px}
#__UID__ .t-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .t-ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:10px;font-size:12.5px}
#__UID__ .t-btn{cursor:pointer;border:none;border-radius:8px;padding:8px 15px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ input[type=range]{width:130px;vertical-align:middle}
#__UID__ .t-last{font-size:12.5px;color:#333;background:#fafbff;border:1px solid #e6e8ee;border-radius:9px;padding:10px 12px;line-height:1.9;min-height:64px}
#__UID__ .t-note{font-size:12.5px;color:#444;background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:10px;line-height:1.6;min-height:40px}
#__UID__ .t-eq{font-weight:700}
</style>
<div id="__UID__">
  <div class="t-head">🎚️ TD learning, one observation at a time</div>
  <div class="t-sub">We are estimating <b>V(day 1, 🙂 Warm)</b> and we start from a guess of
    <b>0.00</b>, knowing nothing. Each click plays <i>one</i> day: the policy picks a move, the world
    answers, and we get a target <b>r + γ·V(s′)</b> to nudge our estimate towards. Nothing waits for
    the campaign to end.</div>
  <div class="t-ctl">
    <button class="t-btn t-one">Observe one day</button>
    <button class="t-btn t-many">Run 40 days</button>
    <button class="t-btn t-rst" style="background:#9aa0b5">Reset</button>
    <span>α = <b><span class="t-av">0.30</span></b>
      <input type="range" class="t-a" min="0.02" max="1" step="0.02" value="0.3"></span>
  </div>
  <svg class="t-svg" viewBox="0 0 560 170" style="width:100%;height:170px"></svg>
  <div class="t-last">Press a button to observe your first day.</div>
  <div class="t-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const svg=root.querySelector(".t-svg"), aEl=root.querySelector(".t-a");
  const truth=D.outcomes.reduce((acc,o)=>acc+o[0]*(o[1]+D.gamma*o[2]),0);
  let V=0, hist=[0];
  function drawSvg(){
    const W=560,H=170,pad=34;
    const lo=Math.min(0,truth-1,...hist), hi=Math.max(truth+1,...hist);
    const x=i=>pad+(W-pad-8)*(hist.length<2?0:i/(hist.length-1));
    const y=v=>H-24-(H-40)*(v-lo)/((hi-lo)||1);
    const pts=hist.map((v,i)=>x(i).toFixed(1)+","+y(v).toFixed(1)).join(" ");
    svg.innerHTML=
      '<line x1="'+pad+'" y1="'+y(truth)+'" x2="'+(W-8)+'" y2="'+y(truth)+'" stroke="#b23b34" '
      +'stroke-width="1.6" stroke-dasharray="6 4"/>'
      +'<text x="'+(W-10)+'" y="'+(y(truth)-6)+'" font-size="11" fill="#b23b34" text-anchor="end">'
      +'true V = '+truth.toFixed(2)+'</text>'
      +'<polyline points="'+pts+'" fill="none" stroke="#4a5bd0" stroke-width="2.2"/>'
      +'<circle cx="'+x(hist.length-1)+'" cy="'+y(V)+'" r="4" fill="#4a5bd0"/>'
      +'<text x="4" y="'+(y(V)+4)+'" font-size="11" fill="#4a5bd0">V̂='+V.toFixed(2)+'</text>'
      +'<text x="'+pad+'" y="'+(H-6)+'" font-size="10.5" fill="#999">'+(hist.length-1)
      +' days observed</text>';
  }
  function step(show){
    const u=Math.random(); let acc=0, o=D.outcomes[D.outcomes.length-1];
    for(const c of D.outcomes){acc+=c[0]; if(u<acc){o=c;break;}}
    const a=+aEl.value, target=o[1]+D.gamma*o[2], err=target-V, before=V;
    V=V+a*err; hist.push(V);
    if(show) root.querySelector(".t-last").innerHTML=
      "<b>"+o[3]+"</b> &nbsp;·&nbsp; r = "+o[1].toFixed(2)+" &nbsp;·&nbsp; V(s′) = "+o[2].toFixed(2)+"<br>"
      +'<span class="t-eq">target</span> = r + γ·V(s′) = '+o[1].toFixed(2)+" + 0.9·"+o[2].toFixed(2)
      +" = <b>"+target.toFixed(2)+"</b> &nbsp;·&nbsp; "
      +'<span class="t-eq">TD error δ</span> = target − V̂ = <b style="color:'
      +(err>=0?"#1d7a46":"#b23b34")+'">'+(err>=0?"+":"")+err.toFixed(2)+"</b><br>"
      +'<span class="t-eq">update</span>: V̂ ← '+before.toFixed(2)+" + "+a.toFixed(2)+"·("
      +(err>=0?"+":"")+err.toFixed(2)+") = <b>"+V.toFixed(2)+"</b>";
    drawSvg(); note();
  }
  function note(){
    const a=+aEl.value, n=root.querySelector(".t-note");
    if(hist.length<3) n.innerHTML="Each observation is <b>wrong on its own</b> — it saw one outcome out "
      +"of several. We never trust it fully; we take a step towards it.";
    else if(a>0.75) n.innerHTML="<b>α close to 1:</b> the estimate more or less <i>becomes</i> the last "
      +"target it saw. It reaches the neighbourhood fast and then rattles around, because it forgets "
      +"everything it learned from the days before.";
    else if(a<0.1) n.innerHTML="<b>α very small:</b> beautifully steady, and slow. Each day moves the "
      +"estimate by a sliver, so it takes many days to travel — but it averages over all of them.";
    else n.innerHTML="Averaging by increments: the estimate drifts to where the targets <b>average out</b>, "
      +"which is the true value. And notice the targets themselves use our own current V — we are "
      +"<b>bootstrapping</b>: learning a guess from a slightly better guess.";
  }
  root.querySelector(".t-one").addEventListener("click",()=>step(true));
  root.querySelector(".t-many").addEventListener("click",()=>{for(let i=0;i<40;i++)step(i===39);});
  root.querySelector(".t-rst").addEventListener("click",()=>{
    V=0;hist=[0];root.querySelector(".t-last").innerHTML="Back to knowing nothing: V̂ = 0.00.";
    drawSvg();note();});
  aEl.addEventListener("input",()=>{root.querySelector(".t-av").textContent=(+aEl.value).toFixed(2);note();});
  drawSvg(); note();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


# ===========================================================================
#  §4  Actor + critic
# ===========================================================================
def actor_critic_diagram():
    """The two networks, what each learns, and the one number they exchange."""
    def panel(icon, who, colour, learns, loss, sees):
        return ('<div style="flex:1;min-width:255px;border:2px solid %s;border-radius:14px;'
                'padding:13px 14px;background:#fff">'
                '<div style="font-size:21px">%s</div>'
                '<div style="font-weight:800;font-size:14.5px;color:%s;margin:2px 0 6px">%s</div>'
                '<div style="font-size:12px;color:#555;line-height:1.6"><b>Learns:</b> %s</div>'
                '<div style="font-size:12px;color:#555;line-height:1.6;margin-top:5px">'
                '<b>Its loss:</b> %s</div>'
                '<div style="font-size:12px;color:#555;line-height:1.6;margin-top:5px">'
                '<b>Gets to look at:</b> %s</div></div>' % (colour, icon, colour, who, learns, loss, sees))
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🎭 The two halves of an actor–critic</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:13px;line-height:1.55">'
        'Two sets of parameters, two different jobs, trained side by side on the very same transitions.'
        '</div>'
        '<div style="display:flex;gap:11px;flex-wrap:wrap">%s%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:12px 14px;margin-top:14px;'
        'font-size:12.5px;color:#333;line-height:1.7">'
        '<b>The one thing they exchange is a number.</b> The critic hands the actor '
        '<b>δ = r + γV(s′) − V(s)</b> for each step — its estimate of the advantage — and the actor '
        'uses it as the weight on <code>log π(a|s)</code>. Nothing else crosses between them: '
        'the actor never sees the critic\'s parameters, the critic never sees the actor\'s.<br>'
        '<span style="color:#777">And notice the asymmetry in that last row: the critic may look at '
        'the day, because it never picks an action, so it cannot bias anything. The actor may not — '
        'the retention team asked for one rule per engagement level, applied every morning.</span></div>'
        % (panel("🎬", "The Actor — π<sub>θ</sub>(a|s)", ACTOR_COLOR,
                 "which move to make. This is the sheet we ship.",
                 "−A · log π(a|s), the policy gradient — <i>push up what beat the yardstick</i>",
                 "the engagement level only"),
           panel("🎓", "The Critic — V<sub>φ</sub>(s)", CRITIC_COLOR,
                 "what a learner in this situation is worth from here.",
                 "( V(s) − [r + γV(s′)] )², a plain regression — <i>predict your own future better</i>",
                 "the engagement level <b>and</b> the day")))


def policy_iteration_cycle():
    """Actor-critic as the policy iteration loop from lecture 1."""
    def ring(icon, title, body, colour):
        return ('<div style="flex:1;min-width:215px;border:2px solid %s;border-radius:12px;'
                'padding:11px 12px;background:#fff">'
                '<div style="font-size:19px">%s</div>'
                '<div style="font-weight:800;font-size:13px;color:%s;margin:1px 0 4px">%s</div>'
                '<div style="font-size:12px;color:#555;line-height:1.55">%s</div></div>'
                % (colour, icon, colour, title, body))
    arrow = '<div style="align-self:center;font-size:19px;color:#9aa0b5">→</div>'
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '♻️ You have seen this loop before — it is policy iteration</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:13px;line-height:1.55">'
        'Lecture 1 alternated <b>evaluate the policy</b> (work out its value function) with '
        '<b>improve the policy</b> (act greedily with respect to it). An actor–critic is the same '
        'two steps, except neither is ever run to completion — both are nudged a little, every batch.'
        '</div>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:stretch">%s%s%s%s%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:13px;'
        'font-size:12.5px;color:#333;line-height:1.65">Classic policy iteration solves the evaluation '
        'step <b>exactly</b>, using the transition table, before touching the policy. We have no table, '
        'so we approximate that step with regression on sampled transitions — and we take one small '
        'improvement step rather than jumping to the greedy policy. Same cycle, sampled and softened.'
        '</div>'
        % (ring("🎬", "the policy acts", "π<sub>θ</sub> runs campaigns and produces transitions",
                ACTOR_COLOR), arrow,
           ring("🎓", "the critic evaluates it", "regression on those transitions moves V<sub>φ</sub> "
                                                 "towards the true V<sup>π</sup>", CRITIC_COLOR), arrow,
           ring("📈", "the policy improves", "advantages from V<sub>φ</sub> tell the actor which moves "
                                             "beat its own habit", ACTOR_COLOR)))


def moving_target_warning():
    """Why one batch buys exactly one update."""
    _card(
        '<div style="font-weight:800;font-size:15px;color:#b23b34;margin-bottom:6px">'
        '⚠️ The catch in that cycle — and it is a real one</div>'
        '<div style="font-size:13px;color:#333;line-height:1.75">'
        'Every object in this notebook carries a hidden superscript: <b>V<sup>π</sup></b>, '
        '<b>Q<sup>π</sup></b>, <b>A<sup>π</sup></b>. They are the value <i>of the current policy</i>. '
        'So the moment the actor takes a step:'
        '<ul style="margin:8px 0 4px 18px;padding:0;line-height:1.8">'
        '<li>the critic is now evaluating a policy that <b>no longer exists</b> — its numbers are '
        'slightly stale;</li>'
        '<li>and the transitions we collected came from that same retired policy, so they are no longer '
        'a sample of what the current one does.</li></ul>'
        'That is why an algorithm of this shape gets <b>exactly one update out of each batch</b>, and '
        'then has to go and collect fresh campaigns. It is not a coding decision — it is the same '
        '<b>on-policy</b> constraint that made us throw batches away in notebook 02.</div>'
        '<div style="background:#f3f0ff;border-radius:8px;padding:11px 13px;margin-top:12px;'
        'font-size:12.5px;color:#2c2350;line-height:1.65">🔭 <b>Which is expensive</b> — collecting '
        'experience is the slow part of RL, and we are binning it after one gradient step. Squeezing '
        'several updates out of one batch <i>safely</i> is exactly what the next notebook is about: '
        '<b>GAE</b> for a better advantage, and <b>PPO</b> for permission to reuse the data.</div>')


def entropy_playground():
    """Sliders over three action probabilities → entropy, with the collapse warning."""
    data = {"act": ACTIONS, "ae": ACTION_EMOJI, "ac": ACTION_COLOR}
    uid = "ent_" + str(abs(hash("entropy_playground")) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:680px;background:#fff}
#__UID__ .e-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px}
#__UID__ .e-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .e-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:12.5px}
#__UID__ .e-name{width:96px;font-weight:700}
#__UID__ input[type=range]{width:140px}
#__UID__ .e-bar{flex:1;background:#eef0f7;border-radius:5px;height:16px;overflow:hidden}
#__UID__ .e-bar>div{height:100%;transition:width .12s}
#__UID__ .e-pct{width:44px;text-align:right;font-weight:700}
#__UID__ .e-meterwrap{margin-top:14px}
#__UID__ .e-meter{height:22px;border-radius:6px;background:#eef0f7;overflow:hidden}
#__UID__ .e-meter>div{height:100%;background:linear-gradient(90deg,#c0554e,#e0a500,#2e9e7a);transition:width .15s}
#__UID__ .e-btns{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
#__UID__ .e-btn{cursor:pointer;border:none;border-radius:8px;padding:7px 14px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .e-note{font-size:12.5px;color:#444;background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:11px;line-height:1.6;min-height:58px}
</style>
<div id="__UID__">
  <div class="e-head">🎲 How much is this policy still willing to explore?</div>
  <div class="e-sub">Entropy <b>H(π) = −Σ<sub>a</sub> π(a|s)·log π(a|s)</b> measures how undecided a
    distribution is. Drag the logits and watch it. The number underneath is easier to read:
    <b>e<sup>H</sup></b> — <i>how many actions this policy is effectively still choosing between</i>.</div>
  <div class="e-list"></div>
  <div class="e-meterwrap">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#666">
      <span>H(π) = <b><span class="e-h">1.099</span></b> nats</span>
      <span>effectively choosing between <b><span class="e-eff">3.00</span></b> of 3 moves</span></div>
    <div class="e-meter"><div style="width:100%"></div></div>
  </div>
  <div class="e-btns">
    <button class="e-btn e-uni">Undecided (uniform)</button>
    <button class="e-btn e-col">Collapsed onto one move</button>
  </div>
  <div class="e-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const L=[0,0,0], list=root.querySelector(".e-list");
  D.act.forEach((a,i)=>{
    const row=document.createElement("div"); row.className="e-row";
    row.innerHTML='<div class="e-name" style="color:'+D.ac[i]+'">'+D.ae[i]+" "+a+'</div>'
      +'<input type="range" min="-5" max="5" step="0.1" value="0" data-i="'+i+'">'
      +'<div class="e-bar"><div style="width:33%;background:'+D.ac[i]+'"></div></div>'
      +'<div class="e-pct" style="color:'+D.ac[i]+'">33%</div>';
    list.appendChild(row);
  });
  const sliders=[...list.querySelectorAll("input")];
  function probs(){const m=Math.max(...L),e=L.map(v=>Math.exp(v-m)),Z=e.reduce((x,y)=>x+y,0);return e.map(v=>v/Z);}
  function upd(){
    const p=probs(), H=-p.reduce((s,v)=>s+(v>1e-12?v*Math.log(v):0),0), eff=Math.exp(H);
    [...list.querySelectorAll(".e-row")].forEach((r,i)=>{
      r.querySelector(".e-bar>div").style.width=(p[i]*100).toFixed(1)+"%";
      r.querySelector(".e-pct").textContent=(p[i]*100).toFixed(0)+"%";
    });
    root.querySelector(".e-h").textContent=H.toFixed(3);
    root.querySelector(".e-eff").textContent=eff.toFixed(2);
    root.querySelector(".e-meter>div").style.width=(100*H/Math.log(3)).toFixed(1)+"%";
    const n=root.querySelector(".e-note");
    if(H>1.05) n.innerHTML="<b>Maximum entropy</b> (log 3 = 1.099). Every move still gets tried, so "
      +"every move keeps producing evidence. This is where training starts — and it is a terrible "
      +"policy to <i>ship</i>, because it acts at random.";
    else if(H<0.15) n.innerHTML="<b>Collapsed.</b> One move has all the probability, so the other two "
      +"are effectively never sampled again — and an action that is never sampled can never generate "
      +"the evidence that would have rehabilitated it. If the collapse happened early, on noisy "
      +"advantages, the policy is now stuck with a mistake it can no longer discover.";
    else n.innerHTML="A policy with an opinion that has not stopped listening. Adding <b>+c·H(π)</b> to "
      +"what we maximise puts a gentle thumb on this scale: it costs the actor a little to become "
      +"certain, so it only does so when the advantages keep insisting.";
  }
  sliders.forEach(s=>s.addEventListener("input",()=>{L[+s.dataset.i]=+s.value;upd();}));
  root.querySelector(".e-uni").addEventListener("click",()=>{for(let i=0;i<3;i++){L[i]=0;sliders[i].value=0;}upd();});
  root.querySelector(".e-col").addEventListener("click",()=>{
    L[0]=-4;L[1]=5;L[2]=-4; sliders[0].value=-4;sliders[1].value=5;sliders[2].value=-4; upd();});
  upd();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


def a2c_loop_diagram():
    """The A2C iteration as five boxes, with the loss spelled out."""
    steps = [
        ("1", "Collect", "run <b>N</b> campaigns in parallel with the current actor<br>"
                         "→ a pile of (s, a, r, s′)"),
        ("2", "Ask the critic", "V(s) and V(s′) for every transition<br>"
                                "→ δ = r + γV(s′) − V(s)"),
        ("3", "Build one loss", "actor: −δ·log π &nbsp;·&nbsp; critic: δ² <br>"
                                "&nbsp;·&nbsp; explorer: −c·H(π)"),
        ("4", "One step", "backward() once, then step <b>both</b><br>optimizers"),
        ("5", "Discard", "the batch came from the <i>old</i> actor<br>— collect again"),
    ]
    boxes = []
    for i, (n, name, sub) in enumerate(steps):
        boxes.append(
            '<div style="flex:1;min-width:150px;border:2px solid #4a5bd0;border-radius:12px;'
            'padding:10px 11px;background:#f7f8ff">'
            '<div style="font-size:10.5px;color:#8189c4;font-weight:800">STEP %s</div>'
            '<div style="font-weight:800;font-size:13.5px;color:#2b2d6b;margin:2px 0 4px">%s</div>'
            '<div style="font-size:11.5px;color:#555;line-height:1.5">%s</div></div>' % (n, name, sub))
        if i < len(steps) - 1:
            boxes.append('<div style="align-self:center;font-size:20px;color:#9aa0b5">→</div>')
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:10px">'
        '🔁 A2C — Advantage Actor–Critic, one iteration</div>'
        '<div style="display:flex;gap:7px;flex-wrap:wrap">%s</div>'
        '<div style="background:#fafbff;border:1px solid #e6e8ee;border-radius:10px;padding:13px;'
        'margin-top:14px;text-align:center;font-size:14.5px;color:#222;line-height:1.9">'
        'L &nbsp;=&nbsp; <span style="color:%s;font-weight:700">−δ·log π<sub>θ</sub>(a|s)</span>'
        ' &nbsp;+&nbsp; <span style="color:%s;font-weight:700">c<sub>V</sub>·( V<sub>φ</sub>(s) − '
        'target )²</span> &nbsp;−&nbsp; '
        '<span style="color:#2e9e7a;font-weight:700">c<sub>H</sub>·H(π<sub>θ</sub>(·|s))</span></div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:12px;'
        'font-size:12.5px;color:#333;line-height:1.65">The <b>“2”</b> is for <b>A</b>dvantage '
        '<b>A</b>ctor–<b>C</b>ritic. The <i>many campaigns at once</i> is not decoration: a batch of '
        'independent campaigns is what keeps the single update from being dominated by one lucky '
        'learner — and it is the shape that parallelises across cores, which is how this algorithm is '
        'actually run.</div>' % ("".join(boxes), ACTOR_COLOR, CRITIC_COLOR))


# ===========================================================================
#  §5  Training results
# ===========================================================================
def training_curve(curves, j_start=None, j_best=None,
                   title="Does the campaign earn more as we train?",
                   ylabel="expected return  J(θ)   [CHF per learner]",
                   xlabel="iteration  (one batch of campaigns each)"):
    """`curves` is a list of (label, values, colour)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for label, v, col in curves:
        ax.plot(np.asarray(v, float), color=col, lw=2.1, label=label)
    if j_best is not None:
        ax.axhline(float(j_best), ls="--", color="#2e9e7a", lw=1.7,
                   label="best possible sheet (%.2f)" % float(j_best))
    if j_start is not None:
        ax.axhline(float(j_start), ls=":", color="#9aa0b5", lw=1.5,
                   label="untrained / uniform policy (%.2f)" % float(j_start))
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11.5, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.show()


def critic_curve(errors, label="mean |V̂ − V^π| over all (day, engagement) cells"):
    """How far the critic is from the true value function of the current policy."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.plot(np.asarray(errors, float), color=CRITIC_COLOR, lw=2.1, label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("average error   [CHF]")
    ax.set_title("The critic learning to predict — while the policy keeps moving under it",
                 fontsize=11.5, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


def critic_vs_truth(V_learned, V_true):
    """Side-by-side of the critic's table and the exact V^π of the final policy."""
    V_learned = np.asarray(V_learned, float)
    V_true = np.asarray(V_true, float)
    head = ('<tr><th style="padding:5px 8px"></th>' + "".join(
        '<th style="padding:5px 12px;font-size:11.5px;color:%s">%s</th>' % (ENGAGE_COLOR[s], _e(s))
        for s in range(3)) + '</tr>')
    rows = ""
    for t in range(V_true.shape[0]):
        cells = ""
        for s in range(3):
            err = V_learned[t, s] - V_true[t, s]
            cells += ('<td style="padding:6px 12px;text-align:center;font-size:12.5px">'
                      '<div style="font-weight:800;color:%s">%.2f</div>'
                      '<div style="color:#999;font-size:11px">true %.2f</div>'
                      '<div style="color:%s;font-size:11px">%+.2f</div></td>'
                      % (CRITIC_COLOR, V_learned[t, s], V_true[t, s],
                         "#1d7a46" if abs(err) < 0.25 else "#b23b34", err))
        rows += ('<tr><td style="padding:6px 8px;font-size:12px;color:#777;font-weight:700">day %d</td>'
                 '%s</tr>' % (t, cells))
    mae = float(np.abs(V_learned - V_true).mean())
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:4px">'
          '🎓 What the critic ended up believing</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px;line-height:1.55">'
          'Big number: the critic\'s prediction. Below it: the exact V of the policy it was trained '
          'on, computed from the transition table — which the critic never saw. Then the gap.</div>'
          '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table>'
          '<div style="background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:12px;'
          'font-size:12.5px;color:#333;line-height:1.6">Average error: <b>CHF %.3f</b> per cell. '
          'The critic learned this by regression on sampled days alone — and the cells it is worst at '
          'are the ones that policy rarely visits, because nobody ever sent it evidence about them.'
          '</div>' % (head, rows, mae), maxw=700)


def action_diagram(probs, visits=None, title="The win-back playbook — hand this to the retention team",
                   subtitle=None):
    """`probs[state]` = the 3 action probabilities the trained policy gives that
    engagement level; `visits[state]` = share of campaign-days spent there."""
    probs = np.asarray(probs, float)
    vis = None if visits is None else np.asarray(visits, float)
    rows = []
    for s in range(probs.shape[0]):
        p = probs[s]
        a = int(np.argmax(p))
        conf = int(round(100 * p[a]))
        tone = ("decisive" if conf >= 85 else ("a preference" if conf >= 55 else "close to a coin flip"))
        bar = "".join(
            '<div style="height:8px;width:%d%%;background:%s"></div>'
            % (int(round(100 * p[i])), ACTION_COLOR[i]) for i in range(3))
        reached = True if vis is None else bool(vis[s] >= 0.03)
        seen = "" if vis is None else (
            '<div style="font-size:11px;color:%s;margin-top:3px">%s</div>'
            % (("#999", "seen on %d%% of campaign-days" % round(100 * vis[s])) if reached
               else ("#c08a2e", "this campaign never gets a learner here")))
        rows.append(
            '<tr>'
            '<td style="padding:8px 12px;border-top:1px solid #eef0f7">'
            '<div style="font-size:19px">%s</div>'
            '<div style="font-weight:800;font-size:13px;color:%s">%s</div>%s</td>'
            '<td style="padding:8px 12px;border-top:1px solid #eef0f7;font-size:20px;color:#c7cbd8">→</td>'
            '<td style="padding:8px 12px;border-top:1px solid #eef0f7">'
            '<div style="display:inline-block;border:2px %s %s;border-radius:11px;padding:7px 14px;'
            'opacity:%s"><span style="font-size:17px">%s</span> '
            '<span style="font-weight:800;font-size:13.5px;color:%s">%s</span></div></td>'
            '<td style="padding:8px 12px;border-top:1px solid #eef0f7;min-width:170px">'
            '<div style="display:flex;gap:2px;border-radius:3px;overflow:hidden">%s</div>'
            '<div style="font-size:11px;color:#777;margin-top:4px">%d%% confident — %s</div></td></tr>'
            % (ENGAGE_EMOJI[s], ENGAGE_COLOR[s], ENGAGE[s], seen,
               "solid" if reached else "dashed", ACTION_COLOR[a] if reached else "#c9ccd8",
               "1" if reached else ".5",
               ACTION_EMOJI[a], ACTION_COLOR[a], ACTIONS[a], bar, conf,
               tone if reached else "but no campaign ever asks — ignore this line"))
    sub = subtitle or ('One rule per engagement level, applied <b>every day</b> of the campaign. The '
                       'bar is the full distribution π(a|s) the actor learned; we ship its favourite '
                       'action, and the confidence tells you how close the call was.')
    _card('<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">📄 %s</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px">%s</div>'
          '<table style="border-collapse:collapse;width:100%%">%s</table>' % (title, sub, "".join(rows)))


def playbook(theta, n_samples=800, **kwargs):
    """Draw the sheet a trained actor is proposing: pi(a|s) per engagement level, and
    how much of the campaign is actually spent there. Returns the argmax sheet."""
    pi = _probs(theta)
    rng = np.random.default_rng(1)
    visits = np.zeros(len(ENGAGE))
    for _ in range(n_samples):
        st = int(rng.choice(len(ENGAGE), p=START_PROBS))
        for _t in range(N_DAYS):
            visits[st] += 1
            a = int(rng.choice(len(ACTIONS), p=pi[st]))
            nxt = TRANS[st][a]
            st = int(rng.choice([x for x, _ in nxt], p=[p for _, p in nxt]))
    action_diagram(pi, visits / visits.sum(), **kwargs)
    return [int(row.argmax()) for row in pi]


def online_update_demo(steps):
    """Show the first few one-step updates of a campaign as they happen, so the
    'no waiting for the ending' claim is concrete. `steps` is a list of dicts with
    keys day, s, a, r, s_next, v, v_next, delta."""
    rows = ""
    for st in steps:
        d = st["delta"]
        rows += ('<tr>'
                 '<td style="padding:7px 10px;color:#777;font-size:11.5px">day %d</td>'
                 '<td style="padding:7px 10px;font-size:12.5px">%s</td>'
                 '<td style="padding:7px 10px;font-size:12.5px;color:%s;font-weight:700">%s</td>'
                 '<td style="padding:7px 10px;text-align:right;font-size:12.5px">%+.2f</td>'
                 '<td style="padding:7px 10px;font-size:12.5px">%s</td>'
                 '<td style="padding:7px 10px;text-align:right;font-size:12px;color:#666">'
                 '%.2f → %.2f</td>'
                 '<td style="padding:7px 10px;text-align:right;font-weight:800;color:%s">%+.2f</td>'
                 '<td style="padding:7px 10px;font-size:12px;color:%s">%s</td></tr>'
                 % (st["day"], _e(st["s"]), ACTION_COLOR[st["a"]], _a(st["a"]), st["r"],
                    _e(st["s_next"]), st["v"], st["v_next"],
                    "#1d7a46" if d >= 0 else "#b23b34", d,
                    "#1d7a46" if d >= 0 else "#b23b34",
                    "push %s up" % ACTION_EMOJI[st["a"]] if d >= 0
                    else "push %s down" % ACTION_EMOJI[st["a"]]))
    head = ('<tr>' + "".join(
        '<th style="text-align:%s;padding:4px 10px;font-size:10.5px;color:#888;text-transform:uppercase;'
        'letter-spacing:.03em">%s</th>' % (al, h)
        for h, al in [("", "left"), ("s", "left"), ("a", "left"), ("r", "right"), ("s′", "left"),
                      ("V(s) → V(s′)", "right"), ("δ = r+γV(s′)−V(s)", "right"),
                      ("what the actor does", "left")]) + '</tr>')
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:4px">'
          '⚡ One campaign, judged day by day — nothing waits for the ending</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px;line-height:1.55">'
          'Each row is a complete learning signal, available the morning after the move. In notebook 02 '
          'the first row could not be scored until the third had happened.</div>'
          '<table style="border-collapse:collapse;width:100%%;font-size:13px">%s%s</table>'
          % (head, rows), maxw=820)


# ===========================================================================
#  Quiz answer keys
# ===========================================================================
_MC_QUIZZES = {
    "g_is_q": (
        "What is the return-to-go, really?",
        "We ran one campaign, and step <i>t</i> collected <code>G<sub>t</sub> = 4.7</code>. Which "
        "sentence describes that number most precisely?",
        ["The value of that situation — the campaign has told us what a learner there is worth.",
         "One draw from the many futures that could follow that move — a noisy sample of Q(s,a).",
         "The reward booked on that day, discounted back to the start of the campaign.",
         "The most the rest of the campaign could pay if we played it well from there on."],
        1,
        "This is the pivot the whole notebook turns on. Both the world and the policy rolled dice "
        "after step t, so <code>G<sub>t</sub></code> is one sample out of many possible futures. Its "
        "<i>average</i> over all of them is the definition of <b>Q(s,a)</b> — which is why replacing "
        "the sample by the function costs nothing in correctness and buys a great deal in noise."),
    "why_v": (
        "Why does the critic learn V and not Q?",
        "The advantage needs both: <code>A(s,a) = Q(s,a) − V(s)</code>. We could have trained two "
        "networks. Why is learning V alone enough?",
        ["Because Q is not a well-defined object once the policy is stochastic",
         "Because <code>Q(s,a) = r + γV(s′)</code>, and both r and s′ are handed to us by the "
         "transition we just observed — so a V-network plus the observed step reconstructs Q",
         "Because V is easier to represent than Q as it needs fewer parameters",
         "Because Q would require knowing the transition probabilities"],
        1,
        "The observed transition supplies exactly the two ingredients that turn V into Q: the reward "
        "actually booked and the state actually reached. Fewer parameters is a real side benefit — a "
        "V-network has no action index, so <i>every</i> sample trains the same output rather than one "
        "action's slice — but the reason it is <b>possible</b> is the identity."),
    "policy_iteration": (
        "Why can we only take one update per batch?",
        "The critic has just learned a good V, the actor has just used it, and we still throw the "
        "batch away. What breaks if we keep training on it?",
        ["The transitions become numerically stale and have to be recomputed",
         "Everything we learned — V, Q, A — is the value <b>of the policy that generated the data</b>. "
         "One actor update later, the critic is describing a policy that no longer exists and the data "
         "no longer comes from the current one",
         "Nothing breaks in principle; discarding is purely a memory optimisation",
         "The advantages would all become negative after the first update"],
        1,
        "V<sup>π</sup>, Q<sup>π</sup> and A<sup>π</sup> all carry that superscript. Moving θ invalidates "
        "them a little — and invalidates the batch as a sample of the current policy. It is the same "
        "<b>on-policy</b> constraint as REINFORCE, and PPO exists precisely to buy back a few extra "
        "updates without letting the two policies drift too far apart."),
    "entropy": (
        "What is the entropy bonus actually protecting us from?",
        "We add <code>+c·H(π)</code> to the objective, which pays the actor a little for staying "
        "undecided. What is the failure it prevents?",
        ["Losing the ability to represent a deterministic sheet at the end of training",
         "An early run of lucky advantages pushing one action to ~100%, after which the alternatives "
         "are never sampled and can never generate the evidence that would correct the mistake",
         "The critic's predictions drifting towards zero",
         "The gradient becoming numerically unstable when probabilities get close to 1"],
        1,
        "Collapse is self-sealing: a policy only ever learns about actions it takes. Because our "
        "advantages are noisy — especially while the critic is still bad — a few unlucky batches can "
        "shut a good action out permanently. The bonus makes certainty cost something, so the actor "
        "only becomes decisive when the advantages keep insisting. Set <code>c</code> too high and you "
        "ship a policy that deliberately acts at random, so it is a genuine dial, not a free lunch."),
}

_TF_QUIZZES = {
    "values": ("The value functions", [
        ("V(s) is the expected return-to-go from s, under the policy we are currently running.", True),
        ("Q(s,a) commits to action a now, then follows the current policy for the rest.", True),
        ("V(s) is the return of the best action available in s.", False),
        ("In our three-day campaign, the value of a state depends on which morning it is.", True),
        ("A single observed return-to-go G<sub>t</sub> is a noisy sample whose average is Q(s,a).",
         True),
        ("V and Q describe the environment, so they stay fixed while the policy changes.", False),
    ]),
    "advantage": ("The advantage function", [
        ("A(s,a) compares an action against what the current policy would normally do in that state.",
         True),
        ("In a given state, the advantages of the three actions average to zero under the policy.",
         True),
        ("An action with A(s,a) &lt; 0 is a bad action in every situation.", False),
        ("Using A instead of the raw return changes which policy the algorithm converges to.", False),
        ("Because V(s) does not depend on the action, subtracting it leaves the gradient unbiased.",
         True),
        ("If a state's best action has A = 0, the policy already puts all its probability elsewhere.",
         False),
    ]),
    "td": ("TD learning and the critic", [
        ("The TD target r + γV(s′) uses the critic's own current estimate of the next state.", True),
        ("The TD error δ is a one-sample estimate of the advantage of the action taken.", True),
        ("TD learning needs the episode to finish before it can update anything.", False),
        ("A critic trained by TD needs the transition probabilities of the environment.", False),
        ("Bootstrapping means the critic learns from a target built out of its own predictions.", True),
        ("The critic's target stays fixed throughout training, exactly like a supervised label.",
         False),
    ]),
    "a2c": ("The A2C algorithm", [
        ("The actor and the critic are trained on the same transitions, with different losses.", True),
        ("The critic's job is to pick the action when the actor is unsure.", False),
        ("The advantage multiplying log π is treated as a fixed verdict from the critic, not as a "
         "quantity the actor may improve.", True),
        ("A2C can update from a single (s, a, r, s′) without waiting for the campaign to end.", True),
        ("The entropy bonus keeps the actor from committing before the advantages are trustworthy.",
         True),
        ("A2C can safely take many gradient steps on the same batch.", False),
    ]),
}

_NUMBER_QUIZZES = {
    "vqa": ("🔢 Read V, Q and A off the reward table — last morning only", [
        ("It is <b>day 2</b>, the last morning: whatever we book today is the whole remaining return. "
         "The learner is 🔥 Hot. What is <b>Q(day 2, 🔥 Hot, 📺 Ad blast)</b>?", 6.0, 0.01,
         "On the last day there is no future to discount — Q is just the reward of that move, and "
         "REWARD[🔥 Hot][📺 Ad blast] = 6.0."),
        ("Same morning, same learner. What is <b>Q(day 2, 🔥 Hot, ⏸️ Wait)</b>?", 3.0, 0.01,
         "Again just today's reward: REWARD[🔥 Hot][⏸️ Wait] = 3.0."),
        ("Suppose the policy in 🔥 Hot is <b>50% ⏸️ Wait, 50% 📺 Ad blast</b> and never nudges. What "
         "is <b>V(day 2, 🔥 Hot)</b>?", 4.5, 0.01,
         "V is the policy's average over its own actions: 0.5·3.0 + 0.5·6.0."),
        ("So what is <b>A(day 2, 🔥 Hot, ⏸️ Wait)</b> for that policy?", -1.5, 0.01,
         "A = Q − V = 3.0 − 4.5. Waiting is worse than this policy's habit — push it down."),
    ]),
    "td": ("🔢 One TD update, by hand  (γ = 0.9)", [
        ("The critic currently believes <b>V(day 1, 🙂 Warm) = 4.00</b>. We nudge, book "
         "<b>r = 0.5</b>, and the learner turns 🔥 Hot, where the critic believes "
         "<b>V(day 2, 🔥 Hot) = 5.00</b>. What is the <b>TD target</b>?", 5.0, 0.01,
         "target = r + γ·V(s′) = 0.5 + 0.9·5.00."),
        ("What is the <b>TD error δ</b> for that step?", 1.0, 0.01,
         "δ = target − V(s) = 5.00 − 4.00. Positive: this day went better than the critic expected."),
        ("Which way does the actor move the probability of 🔔 Nudge for a 🙂 Warm learner? Type "
         "<b>1</b> for up, <b>0</b> for down.", 1.0, 0.01,
         "δ is the advantage estimate, and it is positive — the move beat the critic's expectation, "
         "so its probability goes up."),
    ]),
}


def mc_quiz(key):
    _mc_render(*_MC_QUIZZES[key])


def true_false_quiz(key):
    title, statements = _TF_QUIZZES[key]
    _tf_render(title, statements)


def number_quiz(key):
    title, questions = _NUMBER_QUIZZES[key]
    _nq_render(title, questions, header=_reward_table_html() if key == "vqa" else "")


# ===========================================================================
#  Final boss — timed true/false flash quiz with lives
# ===========================================================================
# Balanced pool (24 true / 24 false), phrased so neither answer is given away by
# the wording (no "always/never" tells, no absurd falses).
_FLASH_POOL = [
    # --- value functions ---
    ("V(s) is the expected return from s under the policy currently being run.", True),
    ("V(s) is the return of the single best action available in s.", False),
    ("Q(s,a) assumes the best available action is taken at every step, including the first.", False),
    ("Q and V belong to the environment and stay fixed as the policy changes.", False),
    ("A return-to-go collected on one campaign is a sample of Q(s,a).", True),
    ("Averaging many returns-to-go from the same (s,a) converges to V(s) rather than Q(s,a).", False),
    ("In a finite-horizon campaign the value of a state can depend on the day.", True),
    ("Knowing V for every state is enough to reconstruct Q without observing a transition.", False),
    # --- advantage ---
    ("A(s,a) = Q(s,a) − V(s).", True),
    ("The advantage measures an action against what the policy would normally do there.", True),
    ("Under the current policy, the advantages within a state average to zero.", True),
    ("An action with a negative advantage is a poor action in every state.", False),
    ("Replacing the return by the advantage changes which policy the algorithm converges to.", False),
    ("A baseline is allowed to depend on the action taken as long as it is accurate.", False),
    ("The critic's TD error is a one-sample estimate of the advantage.", True),
    # --- the one-step identity / TD ---
    ("Q(s,a) can be written as r + γV(s′) using the observed transition.", True),
    ("The TD target r + γV(s′) can be computed the morning after the move.", True),
    ("TD learning has to wait for the episode to end before updating.", False),
    ("Bootstrapping means building a learning target out of your own current estimate.", True),
    ("The TD target is treated as a constant when differentiating the critic's loss.", True),
    ("A critic trained by TD needs the environment's transition probabilities.", False),
    ("A small learning rate makes the value estimate follow the most recent target closely.", False),
    ("The critic's loss is a classification loss over the three actions.", False),
    ("A one-step advantage estimate has lower variance than a full-return one.", True),
    ("A one-step advantage estimate is unaffected by errors in the critic.", False),
    # --- actor / critic split ---
    ("The actor and the critic must share all their parameters for the advantage to be valid.", False),
    ("The critic chooses the action whenever the actor is undecided.", False),
    ("The actor's update weights log π(a|s) by the advantage estimate.", True),
    ("The advantage in front of log π is held fixed while the actor's gradient is taken.", True),
    ("The critic may condition on information the policy is not allowed to use.", True),
    ("The actor's loss is a genuine loss whose numerical value measures policy quality.", False),
    ("An actor-critic needs the environment's transition table to compute the advantage.", False),
    ("Actor-critic updates can be made from a single (s, a, r, s′) tuple.", True),
    # --- policy iteration / on-policy ---
    ("Actor-critic resembles policy iteration with both steps run to completion each cycle.", False),
    ("Evaluating the policy and improving it are interleaved rather than alternated to convergence.",
     True),
    ("After the actor updates, the critic's estimates describe a slightly outdated policy.", True),
    ("A batch collected under the old actor remains a valid sample of the new one.", False),
    ("A2C discards its batch after a single gradient step.", True),
    ("On-policy means the data has to come from the policy currently being improved.", True),
    ("Reusing a batch for many updates is safe as long as the learning rate is small.", False),
    # --- entropy & A2C ---
    ("The entropy of a uniform distribution over three actions is log 3.", True),
    ("Entropy is at its maximum when one action has all the probability.", False),
    ("The entropy bonus is added to what we maximise, so it discourages early collapse.", True),
    ("A collapsed policy stops generating evidence about the actions it abandoned.", True),
    ("A large entropy coefficient makes the shipped policy more decisive.", False),
    ("The entropy term is what makes the policy gradient unbiased.", False),
    ("Running many campaigns per update reduces the noise in a single A2C step.", True),
    ("In A2C the critic is trained on its own batch, collected separately from the actor's.", False),
]


def flash_quiz(n_to_pass=10, lives=3, seconds=10):
    """Timed true/false 'final boss'. All logic runs in the browser so nothing
    here reveals the answers to the notebook cell."""
    pool = [{"t": t, "a": bool(a)} for t, a in _FLASH_POOL]
    data = {"pool": pool, "need": int(n_to_pass), "lives0": int(lives), "secs": int(seconds)}
    uid = "flash_" + str(abs(hash(tuple(t for t, _ in _FLASH_POOL))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:16px;padding:20px;max-width:640px;background:#fff;position:relative}
#__UID__ .fq-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
#__UID__ .fq-title{font-weight:800;font-size:16px;color:#2b2d6b}
#__UID__ .fq-lives{font-size:18px;letter-spacing:2px}
#__UID__ .fq-meta{display:flex;justify-content:space-between;font-size:12.5px;color:#666;margin-bottom:10px}
#__UID__ .fq-bar{height:8px;border-radius:5px;background:#eceeffa8;overflow:hidden;margin-bottom:16px}
#__UID__ .fq-bar > div{height:100%;width:100%;background:linear-gradient(90deg,#46b46e,#e0a500,#e07a7a);transition:width .1s linear}
#__UID__ .fq-stmt{font-size:16px;line-height:1.5;color:#1c1e2a;min-height:64px;display:flex;align-items:center;padding:6px 2px}
#__UID__ .fq-btns{display:flex;gap:12px;margin-top:10px}
#__UID__ .fq-btn{flex:1;cursor:pointer;border:2px solid;border-radius:12px;padding:14px;font-size:15px;font-weight:800;background:#fff;transition:.1s}
#__UID__ .fq-true{border-color:#46b46e;color:#1d7a46}#__UID__ .fq-true:hover{background:#e7f7ec}
#__UID__ .fq-false{border-color:#e07a7a;color:#b23b34}#__UID__ .fq-false:hover{background:#fdecec}
#__UID__ .fq-flash{font-size:13px;font-weight:700;margin-top:12px;min-height:20px}
#__UID__ .fq-end{text-align:center;padding:14px 6px}
#__UID__ .fq-end h3{font-size:22px;margin:6px 0}
#__UID__ .fq-restart{cursor:pointer;border:none;border-radius:10px;padding:10px 20px;font-size:14px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:8px}
</style>
<div id="__UID__">
  <div class="fq-top">
    <div class="fq-title">🎯 Final boss — beat the clock</div>
    <div class="fq-lives"></div>
  </div>
  <div class="fq-meta"><span class="fq-prog"></span><span class="fq-time"></span></div>
  <div class="fq-bar"><div></div></div>
  <div class="fq-body">
    <div class="fq-stmt"></div>
    <div class="fq-btns">
      <button class="fq-btn fq-true">TRUE</button>
      <button class="fq-btn fq-false">FALSE</button>
    </div>
    <div class="fq-flash"></div>
  </div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const $=s=>root.querySelector(s);
  const bar=$(".fq-bar>div");
  const stmt=()=>$(".fq-stmt"), flash=()=>$(".fq-flash");
  const livesEl=$(".fq-lives"), progEl=$(".fq-prog"), timeEl=$(".fq-time");
  let order=[], ptr=0, correct=0, lives=D.lives0, timer=null, deadline=0, locked=false;
  function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function reorder(){order=shuffle(D.pool.map((_,i)=>i));ptr=0;}
  function renderHUD(){
    livesEl.textContent="❤️".repeat(lives)+"🖤".repeat(D.lives0-lives);
    progEl.textContent="Correct: "+correct+" / "+D.need;
  }
  function nextQ(){
    if(ptr>=order.length){reorder();}
    locked=false; flash().textContent=""; flash().style.color="";
    root.querySelectorAll(".fq-btn").forEach(b=>b.disabled=false);
    stmt().textContent=D.pool[order[ptr]].t;
    startTimer();
  }
  function startTimer(){
    deadline=Date.now()+D.secs*1000;
    clearInterval(timer);
    timer=setInterval(()=>{
      const left=Math.max(0,deadline-Date.now());
      bar.style.width=(100*left/(D.secs*1000))+"%";
      timeEl.textContent=(left/1000).toFixed(1)+"s";
      if(left<=0){clearInterval(timer); timeout();}
    },80);
  }
  function answer(val){
    if(locked)return; locked=true; clearInterval(timer);
    root.querySelectorAll(".fq-btn").forEach(b=>b.disabled=true);
    const q=D.pool[order[ptr]];
    if(val===q.a){correct++; flash().style.color="#1d7a46"; flash().textContent="✅ Correct!";}
    else{lives--; flash().style.color="#b23b34"; flash().textContent="❌ Wrong — it was "+(q.a?"TRUE":"FALSE")+".";}
    ptr++; renderHUD(); advance();
  }
  function timeout(){
    if(locked)return; locked=true;
    root.querySelectorAll(".fq-btn").forEach(b=>b.disabled=true);
    lives--; ptr++; flash().style.color="#b23b34"; flash().textContent="⏱️ Out of time — life lost.";
    renderHUD(); advance();
  }
  function advance(){
    if(correct>=D.need){return finish(true);}
    if(lives<=0){return finish(false);}
    setTimeout(nextQ, 850);
  }
  function finish(won){
    clearInterval(timer);
    root.querySelector(".fq-body").innerHTML=
      '<div class="fq-end"><h3>'+(won?"🎉 Passed!":"💀 Out of lives")+'</h3>'
      +'<div style="font-size:14px;color:#555">'
      +(won?("You cleared "+correct+" questions. The critic approves."):
            ("You reached "+correct+" / "+D.need+" correct. Review the notebook and try again."))
      +'</div><button class="fq-restart">Play again</button></div>';
    root.querySelector(".fq-restart").addEventListener("click",start);
    timeEl.textContent=""; bar.style.width="0%";
  }
  function start(){
    correct=0; lives=D.lives0; reorder();
    root.querySelector(".fq-body").innerHTML=
      '<div class="fq-stmt"></div><div class="fq-btns">'
      +'<button class="fq-btn fq-true">TRUE</button>'
      +'<button class="fq-btn fq-false">FALSE</button></div>'
      +'<div class="fq-flash"></div>';
    bind(); renderHUD(); nextQ();
  }
  function bind(){
    root.querySelector(".fq-true").addEventListener("click",()=>answer(true));
    root.querySelector(".fq-false").addEventListener("click",()=>answer(false));
  }
  start();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))
