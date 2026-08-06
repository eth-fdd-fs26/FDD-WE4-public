"""Presentation & quiz helpers for the WE4 notebook 02:
"Policy Gradients — the three-day retention campaign".

Same idea as WE0's `pdm_viz` and WE3's `we3_viz`: every HTML/CSS illustration,
interactive widget, quiz *answer key* and matplotlib visual lives here, out of
the notebook, so the teaching cells stay about the *idea* and the quizzes can't
be solved by reading the cell. The notebook does::

    import pg_viz
    pg_viz.campaign_overview()
    pg_viz.mc_quiz("pong_state")

Students are told not to read this file.
"""
import json as _json

import numpy as np
from IPython.display import HTML, display

# ===========================================================================
#  The campaign vocabulary (kept here so every widget labels things the same)
# ===========================================================================
ENGAGE = ["Cold", "Warm", "Hot"]
ENGAGE_EMOJI = ["😴", "🙂", "🔥"]
ENGAGE_COLOR = ["#8d93a8", "#dd8452", "#c0554e"]

ACTIONS = ["Wait", "Nudge", "Ad blast"]
ACTION_EMOJI = ["⏸️", "🔔", "📺"]
ACTION_COLOR = ["#9aa0b5", "#4a5bd0", "#2e9e7a"]

# The environment, duplicated here ONLY so the widgets can label / recompute
# things in the browser. The notebook builds these tables itself.
#   TRANS[s][a] = list of (next engagement, probability)
P_NUDGE, P_COOL, P_ANNOY = 0.7, 0.7, 0.8
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


def _e(i):
    return "%s %s" % (ENGAGE_EMOJI[i], ENGAGE[i])


def _a(i):
    return "%s %s" % (ACTION_EMOJI[i], ACTIONS[i])


# ===========================================================================
#  Generic renderers  (ported from WE0 pdm_viz / WE3 we3_viz)
# ===========================================================================
def _card(inner, maxw=860):
    display(HTML(
        '<div style="font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid '
        '#e6e8ee;border-radius:14px;padding:18px;max-width:%dpx;background:#fff">%s</div>'
        % (maxw, inner)))


def _mc_render(title, question, options, answer_index, reveal):
    data = {"opts": list(options), "ans": int(answer_index), "reveal": reveal}
    uid = "mc_" + str(abs(hash((question, tuple(options), answer_index))) % 10**8)
    rows = "".join(
        '<div class="mc-opt" data-i="%d"><span class="mc-dot"></span>%s</div>' % (i, o)
        for i, o in enumerate(options))
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
               prompt="Work each number out, type it in, then check."):
    """questions: list of (question_html, answer_number, tolerance, reveal)."""
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
#  §1  The campaign: states, actions, rewards
# ===========================================================================
def campaign_overview():
    """The scenario on one card: three engagement levels, three actions."""
    def chip(emoji, name, sub, color):
        return ('<div style="flex:1;min-width:150px;border:2px solid %s;border-radius:12px;'
                'padding:10px 12px;background:#fff">'
                '<div style="font-size:22px">%s</div>'
                '<div style="font-weight:800;font-size:13.5px;color:#222">%s</div>'
                '<div style="font-size:11.5px;color:#777;line-height:1.45;margin-top:2px">%s</div></div>'
                % (color, emoji, name, sub))

    states = "".join(chip(ENGAGE_EMOJI[i], ENGAGE[i], sub, ENGAGE_COLOR[i]) for i, sub in enumerate([
        "streak broken, hasn't opened the app in weeks. No lessons → no ads → <b>no revenue</b>. "
        "<b>40%</b> of the learners who enter the campaign",
        "opens it a few times a week, does the odd lesson. <b>60%</b> of them",
        "back on a daily streak, a lesson — and an ad — every morning. <i>Nobody enters the campaign "
        "here: this is where we are trying to get them.</i>"]))
    acts = "".join(chip(ACTION_EMOJI[i], ACTIONS[i], sub, ACTION_COLOR[i]) for i, sub in enumerate([
        "no contact today. Costs nothing — and streaks decay on their own",
        "a personalised message from their coach with one lesson picked for them. "
        "Costs a little to write and send; often moves the learner up a level",
        "double the ad load: a long unskippable break before every lesson today. Much more revenue "
        "from whoever is still learning — and most of them get fed up and stop"]))
    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:4px">'
        '📋 Owlinguo · the three-day win-back campaign</div>'
        '<div style="font-size:12.5px;color:#555;line-height:1.6;margin-bottom:14px">'
        '<b>Owlinguo</b> is a free language-learning app: about <b>2 million</b> people open it each '
        'month, and it makes money the simple way — <b>one ad before every lesson</b>. So a learner is '
        'worth exactly what they study: someone on a daily streak watches an ad a day, someone who has '
        'stopped opening the app is worth nothing. And your data says the same thing every time: once '
        'a streak breaks, you have about <b>three days</b> before they are gone for good. So the '
        'retention team runs a <b>three-day campaign</b> on each lapsing learner: every morning, look '
        'at how engaged they are and pick <b>one</b> of three moves. On day&nbsp;4 the campaign closes '
        'and you count the revenue.</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">The state — how engaged the learner is</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">%s</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">The actions — what you can do about it</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap">%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:16px;'
        'font-size:12.5px;color:#333;line-height:1.6">🎯 <b>Your deliverable.</b> At the end of this '
        'notebook you hand the retention team a single sheet with <b>three lines</b> — one per '
        'engagement level: <i>learner looks like this → do that.</i> That sheet is exactly what an RL '
        'practitioner calls a <b>policy</b>.<br>💰 <b>Money throughout is expected ad revenue per '
        'learner, in CHF.</b> A reward of <code>+6.0</code> means CHF&nbsp;6 — small per learner, and '
        'Owlinguo runs this campaign on a few hundred thousand of them a year.</div>'
        % (states, acts))


def mdp_tables(trans, reward):
    """Render the transition (probabilities!) and reward tables."""
    def head():
        return ('<tr><th style="padding:6px 10px;font-size:11.5px;color:#777;text-align:left"></th>'
                + "".join('<th style="padding:6px 12px;font-size:12px;color:%s">%s</th>'
                          % (ACTION_COLOR[a], _a(a)) for a in range(3)) + "</tr>")

    def outcomes(cell):
        return "".join(
            '<div style="font-size:11.5px;color:#444;white-space:nowrap">'
            '<b style="color:#4a3a86">%d%%</b> → %s</div>' % (int(round(100 * p)), _e(int(ns)))
            for ns, p in cell)

    trows = "".join(
        '<tr><td style="padding:7px 10px;font-weight:700;font-size:12.5px;color:%s">%s</td>%s</tr>'
        % (ENGAGE_COLOR[s], _e(s),
           "".join('<td style="padding:7px 12px;text-align:left">%s</td>' % outcomes(trans[s][a])
                   for a in range(3)))
        for s in range(3))

    def money(v):
        col = "#1d7a46" if v > 0 else ("#b23b34" if v < 0 else "#777")
        return ('<td style="padding:7px 12px;text-align:center;font-weight:700;font-size:12.5px;'
                'color:%s">%+.1f</td>' % (col, v))

    rrows = "".join(
        '<tr><td style="padding:7px 10px;font-weight:700;font-size:12.5px;color:%s">%s</td>%s</tr>'
        % (ENGAGE_COLOR[s], _e(s), "".join(money(float(reward[s][a])) for a in range(3)))
        for s in range(3))

    _card(
        '<div style="display:flex;gap:26px;flex-wrap:wrap">'
        '<div><div style="font-weight:800;font-size:13.5px;color:#2b2d6b;margin-bottom:6px">'
        '🔀 Transition — where tomorrow starts, and how likely</div>'
        '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table></div>'
        '<div><div style="font-weight:800;font-size:13.5px;color:#2b2d6b;margin-bottom:6px">'
        '💰 Reward — expected ad revenue booked today (CHF)</div>'
        '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table></div>'
        '</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:12.5px;color:#333;line-height:1.6">Read one cell: a 🙂 Warm learner who gets an '
        '📺 Ad blast earns <b>+2.0</b> today instead of the usual +1.0 — and <b>80%%</b> of the time '
        'gets fed up and drops to 😴 Cold, while <b>20%%</b> shrug it off. Same learner, 🔔 Nudge '
        'instead: you earn <b>+0.5</b> today (the usual +1.0, minus what the message costs) and '
        '<b>70%%</b> of the time they come back 🔥 Hot — worth <b>+3.0 a day</b> from then on.<br>'
        '<span style="color:#777">The reward is the <i>expected</i> revenue for that move, so only '
        '<b>where the learner ends up</b> is uncertain — one source of randomness is enough.</span>'
        '</div>' % (head(), trows, head(), rrows))


def transition_demo():
    """Walk one learner across the three days: pick an action, the world resolves it,
    and the learner moves. Three columns = three days, so the whole episode is visible.
    A switch flips between a DETERMINISTIC world (the intended effect always happens)
    and the STOCHASTIC one we actually model. Reset re-draws a random start."""
    det = [[max(cell, key=lambda o: o[1])[0] for cell in row] for row in TRANS]
    data = {"trans": TRANS, "det": det, "rew": REWARD, "eng": ENGAGE, "ee": ENGAGE_EMOJI,
            "acts": ACTIONS, "ae": ACTION_EMOJI, "ac": ACTION_COLOR, "ec": ENGAGE_COLOR,
            "start": START_PROBS, "days": N_DAYS}
    uid = "td_" + str(abs(hash(("transition_demo", str(TRANS)))) % 10**8)
    tmpl = r"""
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:820px;background:#fff}
#__UID__ .t-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:2px}
#__UID__ .t-sub{font-size:12.5px;color:#666;margin-bottom:10px;line-height:1.5}
#__UID__ .t-mode{display:flex;gap:8px;margin-bottom:12px}
#__UID__ .t-tab{cursor:pointer;border:2px solid #d7dbe8;border-radius:9px;padding:6px 12px;font-size:12.5px;font-weight:700;color:#555;background:#fff}
#__UID__ .t-tab.on{border-color:#764ba2;color:#4a3a86;background:#f1edff}
#__UID__ .t-days{display:flex;gap:10px;flex-wrap:wrap}
#__UID__ .t-day{flex:1;min-width:200px;border:2px solid #e2e5ef;border-radius:12px;padding:10px;background:#fbfcff}
#__UID__ .t-day.on{border-color:#764ba2;background:#faf7ff;box-shadow:0 0 0 3px #764ba21a}
#__UID__ .t-day.done{border-color:#dfe3ee;background:#fff}
#__UID__ .t-dl{font-size:10.5px;font-weight:800;color:#9aa0b5;text-transform:uppercase;letter-spacing:.04em}
#__UID__ .t-st{font-size:15px;font-weight:800;margin:5px 0 8px}
#__UID__ .t-btn{display:block;width:100%;box-sizing:border-box;cursor:pointer;border:2px solid;border-radius:9px;padding:6px 8px;font-size:12.5px;font-weight:700;background:#fff;margin-bottom:5px;text-align:left}
#__UID__ .t-btn:hover{filter:brightness(.97)}
#__UID__ .t-res{font-size:12px;color:#444;line-height:1.6;min-height:80px}
#__UID__ .t-dice{font-size:12px;color:#6a5aa8;background:#f4f1ff;border-radius:7px;padding:6px 8px;margin:5px 0}
#__UID__ .t-odds{font-size:11px;color:#777;margin-top:4px}
#__UID__ .t-foot{margin-top:12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
#__UID__ .t-tot{font-size:13.5px;background:#f3f0ff;border-radius:8px;padding:9px 12px;flex:1;min-width:240px}
#__UID__ .t-reset{cursor:pointer;border:none;border-radius:8px;padding:8px 16px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .t-note{font-size:12px;color:#666;margin-top:9px;line-height:1.55}
</style>
<div id="__UID__">
  <div class="t-head">🎲 Three days in the life of one learner</div>
  <div class="t-sub">Pick a move for day&nbsp;0. The world resolves it, the learner moves, and
    day&nbsp;1 opens. Then press <b>New learner</b> and play the <i>same</i> three moves again —
    in one of these two worlds you will get the same story back, and in the other you will not.</div>
  <div class="t-mode">
    <div class="t-tab" data-m="0">Deterministic world</div>
    <div class="t-tab on" data-m="1">Stochastic world &nbsp;·&nbsp; the one we model</div>
  </div>
  <div class="t-days"></div>
  <div class="t-foot">
    <div class="t-tot"></div>
    <button class="t-reset">🔄 New learner</button>
  </div>
  <div class="t-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const wrap=root.querySelector(".t-days");
  let day=0, s=0, rewards=[], mode=1;
  function label(i){return D.ee[i]+" "+D.eng[i];}
  function drawStart(){
    const u=Math.random(); let acc=0;
    for(let i=0;i<D.start.length;i++){acc+=D.start[i]; if(u<acc) return i;}
    return D.start.length-1;
  }
  function build(){
    wrap.innerHTML="";
    for(let d=0;d<D.days;d++){
      const col=document.createElement("div"); col.className="t-day"; col.dataset.d=d;
      col.innerHTML='<div class="t-dl">day '+d+'</div><div class="t-st"></div>'
        +'<div class="t-acts"></div><div class="t-res"></div>';
      wrap.appendChild(col);
    }
  }
  function render(){
    [...wrap.children].forEach((col,d)=>{
      col.classList.toggle("on", d===day);
      col.classList.toggle("done", d<day);
      const acts=col.querySelector(".t-acts");
      if(d===day){
        col.querySelector(".t-st").innerHTML=
          '<span style="color:'+D.ec[s]+'">'+label(s)+'</span>';
        acts.innerHTML="";
        D.acts.forEach((a,i)=>{
          const b=document.createElement("button"); b.className="t-btn";
          b.style.borderColor=D.ac[i]; b.style.color=D.ac[i];
          const odds = mode===1
            ? D.trans[s][i].map(o=>Math.round(o[1]*100)+"% "+D.ee[o[0]]).join(" · ")
            : "always → "+D.ee[D.det[s][i]];
          b.innerHTML=D.ae[i]+" "+a+' <span style="font-weight:600;color:#666">('
            +(D.rew[s][i]>=0?"+":"")+D.rew[s][i].toFixed(1)+')</span>'
            +'<div class="t-odds">'+odds+'</div>';
          b.addEventListener("click",()=>play(i)); acts.appendChild(b);
        });
      } else if(d>day){
        col.querySelector(".t-st").innerHTML='<span style="color:#c7cbd8">—</span>';
        acts.innerHTML='<div style="font-size:11.5px;color:#b6bac7">waiting for day '+(d-1)+'</div>';
      } else { acts.innerHTML=""; }
    });
    const tot=rewards.reduce((a,b)=>a+b,0);
    root.querySelector(".t-tot").innerHTML = day===0
      ? 'This learner entered the campaign as <b style="color:'+D.ec[s]+'">'+label(s)
        +'</b> — drawn from the mix at the top (25 / 50 / 25).'
      : 'Margin booked so far: <b>'+(tot>=0?"+":"")+tot.toFixed(1)+'</b>'
        +(day>=D.days?' &nbsp;·&nbsp; campaign over — press “New learner”.':'');
    root.querySelector(".t-note").innerHTML = mode===1
      ? "<b>Stochastic.</b> Each (state, action) gives a <i>distribution</i> over next states, so the "
        +"same three moves can tell different stories. Real learners are like this — and it is what "
        +"we model for the rest of the notebook."
      : "<b>Deterministic.</b> Every move now always has its intended effect, so the same three moves "
        +"always produce the same story. Clean — and nothing like a person.";
  }
  function play(a){
    const col=wrap.children[day], r=D.rew[s][a];
    let ns, head;
    if(mode===0){
      ns=D.det[s][a];
      head='<div class="t-dice">⚙️ no dice — the intended effect happens</div>';
    } else {
      const roll=Math.random(); let acc=0; ns=D.trans[s][a][0][0];
      for(const o of D.trans[s][a]){acc+=o[1]; if(roll<acc){ns=o[0]; break;}}
      const odds=D.trans[s][a].map(o=>Math.round(o[1]*100)+"% "+D.ee[o[0]]).join(" · ");
      head='<div class="t-dice">🎲 rolled <b>'+roll.toFixed(2)+'</b> &nbsp;in&nbsp; '+odds+'</div>';
    }
    col.querySelector(".t-acts").innerHTML=
      '<div style="font-size:12.5px;font-weight:700;color:'+D.ac[a]+'">'+D.ae[a]+" "+D.acts[a]+'</div>';
    col.querySelector(".t-res").innerHTML= head
      +label(s)+' &nbsp;→&nbsp; <b style="color:'+D.ec[ns]+'">'+label(ns)+'</b>'
      +'<br>margin today: <b style="color:'+(r>=0?"#1d7a46":"#b23b34")+'">'
      +(r>=0?"+":"")+r.toFixed(1)+'</b>';
    rewards.push(r); s=ns; day++; render();
  }
  function reset(keepStart){
    day=0; rewards=[]; if(!keepStart) s=drawStart(); build(); render();
  }
  root.querySelectorAll(".t-tab").forEach(t=>t.addEventListener("click",()=>{
    root.querySelectorAll(".t-tab").forEach(x=>x.classList.remove("on"));
    t.classList.add("on"); mode=+t.dataset.m; reset(true);
  }));
  root.querySelector(".t-reset").addEventListener("click",()=>reset(false));
  s=drawStart(); build(); render();
})();
</script>"""
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


def policy_types():
    """Deterministic vs stochastic policy, side by side, on the same state."""
    def bars(probs):
        return "".join(
            '<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            '<div style="width:88px;font-size:12px;color:%s;font-weight:700">%s</div>'
            '<div style="flex:1;background:#eef0f7;border-radius:5px;height:14px;overflow:hidden">'
            '<div style="width:%d%%;height:100%%;background:%s"></div></div>'
            '<div style="width:38px;font-size:11.5px;color:#555;text-align:right">%d%%</div></div>'
            % (ACTION_COLOR[i], _a(i), int(round(p * 100)), ACTION_COLOR[i], int(round(p * 100)))
            for i, p in enumerate(probs))

    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:10px">'
        '🧭 Two kinds of policy — the same 🙂 Warm learner</div>'
        '<div style="display:flex;gap:18px;flex-wrap:wrap">'
        '<div style="flex:1;min-width:270px;border:2px solid #9aa0b5;border-radius:12px;padding:12px">'
        '<div style="font-weight:800;font-size:13.5px">Deterministic policy &nbsp;<code>a = π(s)</code></div>'
        '<div style="font-size:12px;color:#777;margin:2px 0 10px">one state → one action, always</div>'
        '%s'
        '<div style="font-size:12px;color:#444;margin-top:8px;line-height:1.5">Exactly what the '
        'retention team can execute. But it can <b>never try anything else</b>, so it can never find '
        'out whether something else was better.</div></div>'
        '<div style="flex:1;min-width:270px;border:2px solid #764ba2;border-radius:12px;padding:12px">'
        '<div style="font-weight:800;font-size:13.5px">Stochastic policy &nbsp;<code>π(a|s)</code></div>'
        '<div style="font-size:12px;color:#777;margin:2px 0 10px">one state → a <i>distribution</i> over actions</div>'
        '%s'
        '<div style="font-size:12px;color:#444;margin-top:8px;line-height:1.5">Every action keeps some '
        'chance, so the campaign keeps <b>sampling alternatives</b> and can learn from them. This is the '
        'kind of policy the rest of the notebook trains.</div></div></div>'
        '<div style="background:#fff7e8;border-left:4px solid #e0a500;border-radius:6px;padding:11px 13px;'
        'margin-top:14px;font-size:12.5px;color:#5a4700;line-height:1.6">🤝 <b>So which one do we ship?</b> '
        'Both — at different moments. We <b>train</b> the stochastic one, because that is the only kind '
        'that generates evidence about the actions it is not yet sure about (and the only kind with a '
        'usable gradient). We <b>deploy</b> its favourite action in each state: the sheet the retention '
        'team gets is the <i>argmax</i> of the trained policy, with the probability printed next to it as '
        'a confidence. A well-trained softmax policy has usually collapsed to near-100%% on one action '
        'anyway — where it has not, that is the policy telling you the call is genuinely close.</div>'
        % (bars([0.0, 1.0, 0.0]), bars([0.15, 0.6, 0.25])))


def gamma_slider(rewards, labels=None):
    """Slide γ and watch the discounted return of ONE fixed episode change."""
    rewards = [float(r) for r in rewards]
    labels = labels or ["day %d" % t for t in range(len(rewards))]
    data = {"r": rewards, "lab": list(labels)}
    uid = "gs_" + str(abs(hash(("gamma", tuple(rewards)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:660px;background:#fff}
#__UID__ .g-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:10px}
#__UID__ label{display:block;font-size:13px;margin:8px 0;color:#333}
#__UID__ input[type=range]{width:100%}
#__UID__ table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
#__UID__ th{font-size:11px;color:#777;text-align:right;padding:4px 8px;font-weight:600}
#__UID__ td{padding:6px 8px;text-align:right}
#__UID__ .g-tot{margin-top:10px;font-size:15px;background:#f3f0ff;border-radius:8px;padding:11px 13px}
#__UID__ .g-big{font-size:22px;font-weight:800;color:#3b2d6b}
#__UID__ .g-note{font-size:12.5px;color:#666;margin-top:8px;line-height:1.5;min-height:34px}
</style>
<div id="__UID__">
  <div class="g-head">⏳ How far ahead do we care? — the discount factor γ</div>
  <label>γ = <b><span class="g-gv">0.90</span></b>
    <input type="range" class="g-g" min="0" max="1" step="0.01" value="0.9"></label>
  <table>
    <tr><th style="text-align:left">day t</th><th>reward r<sub>t</sub></th><th>weight γ<sup>t</sup></th>
        <th>γ<sup>t</sup> · r<sub>t</sub></th></tr>
    <tbody class="g-body"></tbody>
  </table>
  <div class="g-tot">Discounted return &nbsp; G = Σ γ<sup>t</sup> r<sub>t</sub> =
    <span class="g-big"><span class="g-tv"></span></span></div>
  <div class="g-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const g=root.querySelector(".g-g");
  function upd(){
    const gam=+g.value; let tot=0, rows="";
    D.r.forEach((r,t)=>{
      const w=Math.pow(gam,t), c=w*r; tot+=c;
      rows+='<tr><td style="text-align:left;color:#555">'+D.lab[t]+'</td>'
        +'<td style="color:'+(r>=0?"#1d7a46":"#b23b34")+'">'+(r>=0?"+":"")+r.toFixed(2)+'</td>'
        +'<td style="color:#777">'+w.toFixed(3)+'</td>'
        +'<td style="font-weight:700">'+(c>=0?"+":"")+c.toFixed(3)+'</td></tr>';
    });
    root.querySelector(".g-body").innerHTML=rows;
    root.querySelector(".g-gv").textContent=gam.toFixed(2);
    root.querySelector(".g-tv").textContent=(tot>=0?"+":"")+tot.toFixed(3);
    const n=root.querySelector(".g-note");
    if(gam<0.05) n.innerHTML="<b>γ ≈ 0 — totally short-sighted.</b> Only today's reward counts; every "
      +"future day is multiplied by ~0. An investment that pays off tomorrow is invisible.";
    else if(gam>0.95) n.innerHTML="<b>γ ≈ 1 — perfectly patient.</b> A franc on day 3 is worth exactly "
      +"as much as a franc today, so the campaign is happy to spend now to earn later.";
    else n.innerHTML="Each day into the future is worth γ times the previous one — a franc tomorrow "
      +"counts "+(gam*100).toFixed(0)+"% of a franc today. γ is <b>your choice</b>, not a fact about "
      +"the learner: it encodes how far ahead the business wants to look.";
  }
  g.addEventListener("input",upd); upd();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


def episode_strip(states, actions, rewards, gamma=None, title="One episode"):
    """Pretty 3-day filmstrip of a single rolled-out episode."""
    cells = []
    for t, (s, a, r) in enumerate(zip(states, actions, rewards)):
        disc = "" if gamma is None else (
            '<div style="font-size:10.5px;color:#777;margin-top:3px">γ<sup>%d</sup>·r = %+.3f</div>'
            % (t, (gamma ** t) * r))
        cells.append(
            '<div style="flex:1;min-width:135px;border:1px solid #e2e5ef;border-radius:11px;padding:10px">'
            '<div style="font-size:10.5px;color:#999;font-weight:700;text-transform:uppercase">day %d</div>'
            '<div style="font-size:13px;font-weight:800;color:%s;margin:3px 0">%s</div>'
            '<div style="font-size:12.5px;color:%s;font-weight:700">%s</div>'
            '<div style="font-size:13px;font-weight:800;color:%s;margin-top:5px">%+.1f</div>%s</div>'
            % (t, ENGAGE_COLOR[s], _e(s), ACTION_COLOR[a], _a(a),
               "#1d7a46" if r >= 0 else "#b23b34", r, disc))
    tot = sum((1.0 if gamma is None else gamma ** t) * r for t, r in enumerate(rewards))
    foot = ('<div style="margin-top:12px;font-size:14px;background:#f3f0ff;border-radius:8px;'
            'padding:10px 13px">%s = <b style="font-size:17px">%+.3f</b></div>'
            % ("Undiscounted total" if gamma is None
               else "Discounted return &nbsp;G = Σ γ<sup>t</sup> r<sub>t</sub>&nbsp; (γ = %.2f)" % gamma,
               tot))
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:10px">🎬 %s</div>'
          '<div style="display:flex;gap:10px;flex-wrap:wrap">%s</div>%s'
          % (title, "".join(cells), foot))


# ===========================================================================
#  §2  Parametrizing the policy — softmax over the three actions
# ===========================================================================
def why_parametrise():
    """Value-based (score every action, then argmax) vs policy-based (output the
    knobs of a distribution, then sample) — and why the second one survives when
    the actions stop being a short list."""
    def panel(title, sub, steps, note, colour, bad):
        rows = "".join(
            '<div style="display:flex;gap:8px;align-items:flex-start;margin:6px 0">'
            '<div style="flex:0 0 20px;height:20px;border-radius:50%%;background:%s;color:#fff;'
            'font-size:11px;font-weight:800;display:flex;align-items:center;justify-content:center">'
            '%d</div><div style="font-size:12.5px;color:#333;line-height:1.5">%s</div></div>'
            % (colour, i + 1, txt) for i, txt in enumerate(steps))
        return ('<div style="flex:1;min-width:290px;border:2px solid %s;border-radius:13px;padding:13px">'
                '<div style="font-weight:800;font-size:14px;color:%s">%s</div>'
                '<div style="font-size:11.5px;color:#888;margin:2px 0 9px">%s</div>%s'
                '<div style="background:%s;border-radius:8px;padding:9px 11px;margin-top:10px;'
                'font-size:12px;color:%s;line-height:1.55">%s</div></div>'
                % (colour, colour, title, sub, rows,
                   "#fdecec" if bad else "#e7f7ec", "#8a2f28" if bad else "#1d6b3a", note))

    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🧭 Why bother turning the policy into numbers?</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55">'
        'Two ways to decide what to do. They look equally reasonable on our three actions — and they '
        'stop being equally reasonable the moment the actions are not a short list.</div>'
        '<div style="display:flex;gap:12px;flex-wrap:wrap">%s%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:12.5px;color:#333;line-height:1.65">'
        '<b>Where this bites.</b> Suppose the action were <i>“how many ads, as a number between 0 and '
        '1”</i>, or a robot&rsquo;s steering angle. The value-based route now has to solve '
        '<code>max<sub>a</sub> Q(s,a)</code> over a continuum — an optimisation problem <i>inside every '
        'single decision</i>. The policy route does not care: θ can output the <b>mean and spread of a '
        'bell curve</b> instead of three logits, and acting is still one draw.<br><br>'
        '🤖 <b>And this is exactly what an LLM is.</b> A language model already ends in a softmax over '
        'its vocabulary — it <i>is</i> a parametrised policy, whose "state" is the text so far and whose '
        '"action" is the next token. There is no argmax over all possible answers to be had; but there '
        'is a gradient on those logits. That is why the method in this notebook, and not Q-learning, is '
        'the one used to fine-tune them.</div>'
        % (panel("Value-based &nbsp;<span style='font-weight:600;font-size:12px'>(e.g. Q-learning)</span>",
                 "learn how good each action is, then take the best",
                 ["Learn a score <code>Q(s,a)</code> for every action in every state",
                  "To act: compute <b>max<sub>a</sub> Q(s,a)</b> — check them all, keep the winner",
                  "The policy is never stored; it is re-derived at each step by that max"],
                 "You must be able to <b>enumerate the actions</b> to take that max. Three actions: "
                 "trivial. A continuous action: an optimisation problem every time you act.",
                 "#a3652f", True),
           panel("Policy-based &nbsp;<span style='font-weight:600;font-size:12px'>(what we are doing)</span>",
                 "output the knobs of a distribution, then draw from it",
                 ["Keep parameters <code>θ</code> that <i>describe</i> a distribution over actions",
                  "To act: push the state through θ and <b>sample</b> — no comparison, no max",
                  "Improve θ directly with a gradient on how well the sampled actions did"],
                 "Nothing here needs the actions to be listable. Change what θ describes and the same "
                 "machinery covers a handful of buttons, a continuous dial, or a vocabulary.",
                 "#4a5bd0", False)))


def softmax_playground():
    """Three logit sliders → live action probabilities, plus sampling."""
    data = {"acts": ACTIONS, "ae": ACTION_EMOJI, "ac": ACTION_COLOR}
    uid = "sp_" + str(abs(hash("softmax_playground")) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:680px;background:#fff}
#__UID__ .s-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:10px}
#__UID__ .s-row{display:flex;align-items:center;gap:10px;margin:7px 0;font-size:12.5px}
#__UID__ .s-name{width:96px;font-weight:700}
#__UID__ input[type=range]{width:150px}
#__UID__ .s-bar{flex:1;background:#eef0f7;border-radius:5px;height:16px;overflow:hidden}
#__UID__ .s-bar>div{height:100%;transition:width .12s}
#__UID__ .s-pct{width:46px;text-align:right;font-weight:700}
#__UID__ .s-btns{margin-top:12px;display:flex;gap:8px;flex-wrap:wrap}
#__UID__ .s-btn{cursor:pointer;border:none;border-radius:8px;padding:7px 14px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .s-samp{margin-top:10px;font-size:19px;letter-spacing:2px;min-height:26px}
#__UID__ .s-cnt{font-size:12px;color:#666;min-height:18px}
#__UID__ .s-note{font-size:12.5px;color:#444;background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:10px;line-height:1.55;min-height:38px}
</style>
<div id="__UID__">
  <div class="s-head">🎛️ Logits → probabilities: the softmax playground</div>
  <div class="s-list"></div>
  <div class="s-btns">
    <button class="s-btn s-s1">Sample 20 actions</button>
    <button class="s-btn s-add">Add +1 to ALL logits</button>
    <button class="s-btn s-rst">Reset</button>
  </div>
  <div class="s-samp"></div>
  <div class="s-cnt"></div>
  <div class="s-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const L=[0,0,0], list=root.querySelector(".s-list");
  D.acts.forEach((a,i)=>{
    const row=document.createElement("div"); row.className="s-row";
    row.innerHTML='<div class="s-name" style="color:'+D.ac[i]+'">'+D.ae[i]+" "+a+'</div>'
      +'<input type="range" min="-4" max="4" step="0.1" value="0" data-i="'+i+'">'
      +'<div style="width:52px;color:#777">θ='+'<span class="s-lv">0.0</span></div>'
      +'<div class="s-bar"><div style="width:33%;background:'+D.ac[i]+'"></div></div>'
      +'<div class="s-pct" style="color:'+D.ac[i]+'">33%</div>';
    list.appendChild(row);
  });
  const sliders=[...list.querySelectorAll("input")];
  function probs(){
    const m=Math.max(...L), e=L.map(v=>Math.exp(v-m)), Z=e.reduce((x,y)=>x+y,0);
    return e.map(v=>v/Z);
  }
  function upd(){
    const p=probs(), rows=[...list.querySelectorAll(".s-row")];
    rows.forEach((r,i)=>{
      r.querySelector(".s-lv").textContent=L[i].toFixed(1);
      r.querySelector(".s-bar>div").style.width=(p[i]*100).toFixed(1)+"%";
      r.querySelector(".s-pct").textContent=(p[i]*100).toFixed(0)+"%";
    });
    const gap=Math.max(...L)-Math.min(...L);
    const n=root.querySelector(".s-note");
    if(gap<0.15) n.innerHTML="All three logits are equal → a <b>uniform</b> policy: every action equally "
      +"likely. This is where an untrained policy starts.";
    else if(gap>3.5) n.innerHTML="One logit dominates → the policy is <b>almost deterministic</b>. It "
      +"will keep picking the same action and stop discovering anything new.";
    else n.innerHTML="Softmax turns any three real numbers into three positive numbers that sum to 1. "
      +"Only the <b>gaps</b> between logits matter — try the “add +1 to ALL” button.";
  }
  sliders.forEach(s=>s.addEventListener("input",()=>{L[+s.dataset.i]=+s.value;upd();}));
  root.querySelector(".s-add").addEventListener("click",()=>{
    for(let i=0;i<3;i++){L[i]=Math.min(4,L[i]+1); sliders[i].value=L[i];}
    upd();
    root.querySelector(".s-note").innerHTML="Every logit moved by the same +1 — and the probabilities "
      +"did <b>not</b> change. Softmax only reads the <b>differences</b>, so the logits are pinned down "
      +"only up to a shared constant.";
  });
  root.querySelector(".s-rst").addEventListener("click",()=>{
    for(let i=0;i<3;i++){L[i]=0; sliders[i].value=0;} upd();
    root.querySelector(".s-samp").textContent=""; root.querySelector(".s-cnt").textContent="";
  });
  root.querySelector(".s-s1").addEventListener("click",()=>{
    const p=probs(), c=[0,0,0]; let strip="";
    for(let k=0;k<20;k++){
      const u=Math.random(); let acc=0, a=2;
      for(let i=0;i<3;i++){acc+=p[i]; if(u<acc){a=i;break;}}
      c[a]++; strip+=D.ae[a];
    }
    root.querySelector(".s-samp").textContent=strip;
    root.querySelector(".s-cnt").innerHTML="counts: "+c.map((v,i)=>D.ae[i]+" "+v).join(" · ")
      +" &nbsp;—&nbsp; expected ≈ "+p.map(v=>(v*20).toFixed(1)).join(" / ");
  });
  upd();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


def trajectory_gallery(trajs, gamma, title="A handful of sampled campaigns"):
    """Compact list of sampled trajectories: actions taken, states visited, return."""
    rows = []
    for k, (states, actions, rewards) in enumerate(trajs):
        G = sum((gamma ** t) * r for t, r in enumerate(rewards))
        chips = "".join(
            '<span style="display:inline-block;margin-right:6px;padding:3px 8px;border-radius:999px;'
            'font-size:11.5px;background:#f3f4fb;color:%s;font-weight:600">%s %s</span>'
            % (ACTION_COLOR[a], ENGAGE_EMOJI[s], ACTION_EMOJI[a])
            for s, a in zip(states, actions))
        rows.append(
            '<tr><td style="padding:6px 10px;color:#999;font-size:11.5px">#%d</td>'
            '<td style="padding:6px 10px">%s</td>'
            '<td style="padding:6px 10px;font-size:12px;color:#777">%s</td>'
            '<td style="padding:6px 10px;text-align:right;font-weight:800;color:%s">%+.2f</td></tr>'
            % (k + 1, chips, " → ".join(ACTIONS[a] for a in actions),
               "#1d7a46" if G >= 0 else "#b23b34", G))
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:8px">🎞️ %s</div>'
          '<table style="border-collapse:collapse;width:100%%;font-size:13px">%s</table>'
          '<div style="font-size:12.5px;color:#666;margin-top:10px;line-height:1.55">Same policy, '
          'different campaigns — because the policy <b>samples</b>. The spread of that last column is '
          'exactly what makes learning from returns noisy.</div>'
          % (title, "".join(rows)))


# ===========================================================================
#  §3  The objective — enumerate every trajectory
# ===========================================================================
def enumerate_trajectories(gamma=0.9):
    """Every trajectory this campaign can produce, with the WORLD's part of its
    probability (the start draw times the transitions that happened) and its
    discounted return. Walking the tree is bookkeeping, so we do it for you."""
    out = []

    def walk(state, states, actions, rewards, p_world):
        if len(actions) == N_DAYS:
            G = sum((gamma ** t) * r for t, r in enumerate(rewards))
            out.append({"s": states, "a": actions, "r": rewards,
                        "p_world": p_world, "G": round(G, 4)})
            return
        for action in range(len(ACTIONS)):
            for nxt, p in TRANS[state][action]:
                walk(nxt, states + [state], actions + [action],
                     rewards + [REWARD[state][action]], p_world * p)

    for start, p_start in enumerate(START_PROBS):
        if p_start > 0:
            walk(start, [], [], [], p_start)
    return out


def trajectory_tree(all_traj):
    """Where the branching comes from: who walks in, what we pick, what the world
    answers — and how that multiplies up to the full set of possible stories."""
    starts = "".join(
        '<span style="display:inline-block;margin-right:8px;padding:4px 10px;border-radius:999px;'
        'font-size:12.5px;font-weight:700;background:#f3f4fb;color:%s">%s %d%%</span>'
        % (ENGAGE_COLOR[i], _e(i), int(round(100 * p)))
        for i, p in enumerate(START_PROBS) if p > 0)

    # how many next states each (state, action) can lead to
    cells = []
    for st in range(len(ENGAGE)):
        row = "".join(
            '<td style="padding:5px 10px;text-align:center;font-size:12.5px;color:%s">%s</td>'
            % ("#4a3a86" if len(TRANS[st][a]) > 1 else "#9aa0b5",
               ("2 ways" if len(TRANS[st][a]) > 1 else "1 way")) for a in range(len(ACTIONS)))
        cells.append('<tr><td style="padding:5px 10px;font-weight:700;font-size:12.5px;color:%s">%s'
                     '</td>%s</tr>' % (ENGAGE_COLOR[st], _e(st), row))
    head = ('<tr><td></td>' + "".join(
        '<td style="padding:4px 10px;font-size:11.5px;color:%s;text-align:center">%s</td>'
        % (ACTION_COLOR[a], _a(a)) for a in range(len(ACTIONS))) + '</tr>')

    def box(title, body, color):
        return ('<div style="flex:1;min-width:190px;border:2px solid %s;border-radius:12px;'
                'padding:11px 12px;background:#fff">'
                '<div style="font-weight:800;font-size:12.5px;color:%s;margin-bottom:5px">%s</div>'
                '<div style="font-size:12px;color:#444;line-height:1.55">%s</div></div>'
                % (color, color, title, body))

    arrow = ('<div style="align-self:center;font-size:20px;color:#9aa0b5;padding:0 2px">×</div>')

    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🌳 Where all the possible stories come from</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55">'
        'A campaign is three rounds of <i>we choose, then the world answers</i>. Both of those branch, '
        'and the branches multiply.</div>'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:stretch">%s%s%s%s%s</div>'
        '<div style="margin-top:14px;font-size:12.5px;color:#444">And the world does not always branch '
        'the same amount — some moves have only one possible outcome:</div>'
        '<table style="border-collapse:collapse;margin-top:6px;background:#fafbff;border-radius:8px">'
        '%s%s</table>'
        '<div style="background:#f3f0ff;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:13px;color:#2c2350;line-height:1.6">Multiply it all out over the three days and '
        'there are <b style="font-size:15px">%d</b> complete stories this campaign can produce. That is '
        'few enough to <b>write them all down</b> — so for once we can compute the objective, and its '
        'true gradient, <i>exactly</i>, and check every estimate we make against the real answer.</div>'
        % (box("who walks in", starts, "#8d93a8"), arrow,
           box("what we pick", "one of <b>3 moves</b>, sampled from the policy — this is the part "
                               "we control", "#4a5bd0"), arrow,
           box("what the world answers", "the learner moves — <b>1 or 2</b> possible next states, "
                                         "with the probabilities in the table above", "#a3652f"),
           head, "".join(cells), len(all_traj)))


def reward_table():
    """Just the reward table again — for the moment in Part 4 where we need to see
    that from a Hot learner every move pays well."""
    head = ('<tr><td></td>' + "".join(
        '<td style="padding:5px 14px;font-size:12px;color:%s;text-align:center">%s</td>'
        % (ACTION_COLOR[a], _a(a)) for a in range(len(ACTIONS))) + '</tr>')
    rows = ""
    for st in range(len(ENGAGE)):
        cells = ""
        for a in range(len(ACTIONS)):
            v = float(REWARD[st][a])
            col = "#1d7a46" if v > 0 else ("#b23b34" if v < 0 else "#777")
            bg = "#eef7f0" if st == 2 else "#fff"
            cells += ('<td style="padding:7px 14px;text-align:center;font-weight:700;font-size:13px;'
                      'color:%s;background:%s">%+.1f</td>' % (col, bg, v))
        rows += ('<tr><td style="padding:7px 12px;font-weight:700;font-size:12.5px;color:%s">%s</td>'
                 '%s</tr>' % (ENGAGE_COLOR[st], _e(st), cells))
    _card('<div style="font-weight:800;font-size:14px;color:#2b2d6b;margin-bottom:8px">'
          '💰 The reward table again (expected ad revenue today, CHF)</div>'
          '<table style="border-collapse:collapse;background:#fafbff;border-radius:8px">%s%s</table>'
          '<div style="font-size:12.5px;color:#444;margin-top:10px;line-height:1.6">Look at the '
          '<b>green row</b>: once a learner is 🔥 Hot, <b>every</b> move pays well — +2.5, +3.0, +6.0. '
          'A campaign that reaches Hot looks like a triumph whatever it did there.</div>'
          % (head, rows), maxw=560)


def trajectory_enumerator(rows, theta, gamma, focus_state=0, show=40):
    """Every trajectory the campaign can produce, with P(τ) split into the part
    the POLICY controls and the part the WORLD controls — plus sliders on one
    state's logits so students can watch probability mass (and J) move.

    `rows`: list of dicts {"a": [...], "s": [...], "p_world": float, "G": float}.
    `theta`: nested list [state][action] of logits.
    """
    data = {"rows": rows, "theta": theta, "gamma": float(gamma), "show": int(show),
            "ae": ACTION_EMOJI, "ee": ENGAGE_EMOJI, "acts": ACTIONS,
            "ac": ACTION_COLOR, "sf": int(focus_state), "eng": ENGAGE}
    uid = "te_" + str(abs(hash(("enum", len(rows), str(theta)[:60]))) % 10**8)
    tmpl = r"""
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:840px;background:#fff}
#__UID__ .e-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px}
#__UID__ .e-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .e-row{display:flex;align-items:center;gap:9px;margin:6px 0;font-size:12.5px}
#__UID__ .e-name{width:104px;font-weight:700}
#__UID__ input[type=range]{width:130px}
#__UID__ .e-J{font-size:15px;background:#f3f0ff;border-radius:8px;padding:11px 13px;margin:12px 0}
#__UID__ .e-big{font-size:23px;font-weight:800;color:#3b2d6b}
#__UID__ table{border-collapse:collapse;width:100%}
#__UID__ th{font-size:10.5px;color:#888;text-transform:uppercase;letter-spacing:.03em;padding:4px 6px;text-align:right;font-weight:700}
#__UID__ td{padding:4px 6px;text-align:right;border-top:1px solid #f0f1f6;font-size:12px}
#__UID__ .e-scroll{max-height:330px;overflow-y:auto;border:1px solid #eef0f7;border-radius:9px}
#__UID__ .e-pol{color:#4a3a86;font-weight:700}
#__UID__ .e-wor{color:#a3652f}
</style>
<div id="__UID__">
  <div class="e-head">🌳 Every story this campaign can produce</div>
  <div class="e-sub">A trajectory now has <b>two</b> sources of randomness, and its probability
    factorises into exactly those two: what <span class="e-pol">the policy</span> chose, times what
    <span class="e-wor">the world</span> did (including which learner walked in). Move the logits
    for a <b>__SNAME__</b> learner: only the <span class="e-pol">purple</span> column moves — the
    world's column is not ours to change.</div>
  <div class="e-sliders"></div>
  <div class="e-J">Objective &nbsp; J(θ) = Σ<sub>τ</sub> P(τ|θ) · G(τ) =
    <span class="e-big"><span class="e-Jv"></span></span>
    <span style="font-size:12px;color:#777">&nbsp;(every trajectory's return, weighted by how likely
    it is — summed over all <span class="e-n"></span> of them)</span></div>
  <div class="e-scroll"><table>
    <thead><tr><th style="text-align:left">trajectory (day 0 → 1 → 2)</th>
      <th class="e-pol">policy</th><th class="e-wor">world</th><th>P(τ)</th>
      <th></th><th>G(τ)</th><th>P · G</th></tr></thead>
    <tbody class="e-body"></tbody></table></div>
  <div style="font-size:11.5px;color:#888;margin-top:7px">Showing the __SHOW__ largest contributors,
    sorted by P·G; J above sums <b>all</b> of them.</div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const TH=JSON.parse(JSON.stringify(D.theta));
  const sl=root.querySelector(".e-sliders");
  D.acts.forEach((a,i)=>{
    const row=document.createElement("div"); row.className="e-row";
    row.innerHTML='<div class="e-name" style="color:'+D.ac[i]+'">'+D.ae[i]+" "+a+'</div>'
      +'<input type="range" min="-4" max="4" step="0.1" data-i="'+i+'" value="'+TH[D.sf][i]+'">'
      +'<div style="width:118px;color:#777">logit = <b class="e-lv">'
      +(+TH[D.sf][i]).toFixed(1)+'</b></div>'
      +'<div style="flex:1;background:#eef0f7;border-radius:5px;height:13px;overflow:hidden">'
      +'<div class="e-pb" style="width:33%;height:100%;background:'+D.ac[i]+'"></div></div>'
      +'<div class="e-pv" style="width:44px;text-align:right;font-weight:700;color:'+D.ac[i]+'">33%</div>';
    sl.appendChild(row);
  });
  function soft(v){const m=Math.max(...v),e=v.map(x=>Math.exp(x-m)),Z=e.reduce((a,b)=>a+b,0);
    return e.map(x=>x/Z);}
  function upd(){
    const pf=soft(TH[D.sf]);
    [...sl.querySelectorAll(".e-row")].forEach((r,i)=>{
      r.querySelector(".e-lv").textContent=TH[D.sf][i].toFixed(1);
      r.querySelector(".e-pb").style.width=(pf[i]*100).toFixed(1)+"%";
      r.querySelector(".e-pv").textContent=(pf[i]*100).toFixed(0)+"%";
    });
    let J=0; const items=D.rows.map(r=>{
      let pp=1;
      for(let t=0;t<r.a.length;t++) pp*=soft(TH[r.s[t]])[r.a[t]];
      const p=pp*r.p_world; J+=p*r.G; return {r:r,pp:pp,p:p};
    });
    items.sort((x,y)=>y.p*y.r.G-x.p*x.r.G);
    const top=items.slice(0,D.show), mx=Math.max(...top.map(i=>i.p));
    root.querySelector(".e-Jv").textContent=(J>=0?"+":"")+J.toFixed(3);
    root.querySelector(".e-n").textContent=D.rows.length;
    root.querySelector(".e-body").innerHTML=top.map(it=>{
      const r=it.r, path=r.a.map((a,t)=>D.ee[r.s[t]]+D.ae[a]).join(" → ");
      return '<tr><td style="text-align:left">'+path+'</td>'
        +'<td class="e-pol">'+(it.pp*100).toFixed(1)+'%</td>'
        +'<td class="e-wor">'+(r.p_world*100).toFixed(1)+'%</td>'
        +'<td>'+(it.p*100).toFixed(2)+'%</td>'
        +'<td style="width:70px"><div style="background:#eef0f7;border-radius:4px;height:10px;overflow:hidden">'
        +'<div style="height:100%;background:#764ba2;width:'+(100*it.p/mx).toFixed(1)+'%"></div></div></td>'
        +'<td style="color:'+(r.G>=0?"#1d7a46":"#b23b34")+';font-weight:700">'
        +(r.G>=0?"+":"")+r.G.toFixed(2)+'</td>'
        +'<td style="font-weight:700">'+(it.p*r.G>=0?"+":"")+(it.p*r.G).toFixed(3)+'</td></tr>';
    }).join("");
  }
  sl.querySelectorAll("input").forEach(s=>s.addEventListener("input",()=>{
    TH[D.sf][+s.dataset.i]=+s.value; upd();}));
  upd();
})();
</script>"""
    html = (tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
            .replace("__SNAME__", "%s %s" % (ENGAGE_EMOJI[focus_state], ENGAGE[focus_state]))
            .replace("__SHOW__", str(show)))
    display(HTML(html))


def expectation_recap():
    """One-card reminder of what an expectation is, in campaign terms."""
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:6px">'
        '🎲 Expectation, in one sentence</div>'
        '<div style="font-size:13px;color:#333;line-height:1.65">An <b>expected value</b> weights each '
        'possible outcome by how likely it is: &nbsp;<code>𝔼[V] = Σ value × probability</code>.<br>'
        'Here the outcomes are <b>every story the campaign could produce</b>, their values are the '
        'discounted returns '
        '<b>G(τ)</b>, and their probabilities come from the policy. So <b>J(θ)</b> is the answer to: '
        '<i>“if we ran this policy on very many learners, what would the average campaign earn?”</i> '
        'That average — not any single lucky campaign — is what we want to push up.</div>'
        '<div style="background:#fff7e8;border-left:4px solid #e0a500;border-radius:6px;padding:10px 12px;'
        'margin-top:12px;font-size:12.5px;color:#5a4700;line-height:1.55">⚠️ Careful: those stories are '
        '<b>not</b> equally likely, so J is <i>not</i> the plain average of their returns. Change the '
        'policy and the weights change — which is exactly the lever we have.</div>')


def objective_landscape(f_j, lo=-3.0, hi=3.0, n=161):
    """A 1-D slice of J(θ): how the objective moves as ONE logit changes.
    `f_j(x)` must return J when that logit is set to x."""
    import matplotlib.pyplot as plt
    xs = np.linspace(lo, hi, n)
    js = np.array([float(f_j(float(x))) for x in xs])
    best = int(np.argmax(js))
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(xs, js, color="#4a5bd0", lw=2.6)
    ax.plot(xs[best], js[best], marker="*", ms=19, color="#e0a500", zorder=6)
    ax.text(xs[best], js[best] + 0.05, "best value of this one logit",
            ha="center", fontsize=9, color="#8a6d1f")
    i = int(np.argmin(np.abs(xs - (lo + 0.35 * (hi - lo)))))
    slope = float((js[i + 1] - js[i - 1]) / (xs[i + 1] - xs[i - 1]))
    tx = np.array([xs[i] - 0.8, xs[i] + 0.8])
    ax.plot(tx, js[i] + slope * (tx - xs[i]), ls="--", color="#2e9e7a", lw=1.8)
    ax.annotate("slope = ∂J/∂θ\nwe walk UPHILL (ascent)", (xs[i], js[i]),
                textcoords="offset points", xytext=(10, -52), fontsize=9, color="#1d6b3a",
                arrowprops=dict(arrowstyle="-|>", color="#1d6b3a", lw=1.2, alpha=.8))
    ax.set_xlabel("one logit  θ  (state Cold, action Nudge)")
    ax.set_ylabel("objective  J(θ)   [expected margin]")
    ax.set_title("The objective really is a function of the logits\n"
                 "(a 1-D slice: every other logit held fixed)",
                 fontsize=11, color="#2b2d6b", fontweight="bold")
    plt.tight_layout(); plt.show()


# ===========================================================================
#  §4  Policy-gradient intuition — push up, push down, baselines
# ===========================================================================
def push_pull_viz(rows, baseline=None, k=8):
    """For a sample of campaigns: their return, the weight R (or R − b), and
    whether one REINFORCE step makes them MORE or LESS likely."""
    rows = list(rows)[:k]
    b = 0.0 if baseline is None else float(baseline)
    mx = max(abs(r["G"] - b) for r in rows) or 1.0
    body = []
    for r in rows:
        w = r["G"] - b
        up = w >= 0
        col = "#1d7a46" if up else "#b23b34"
        bar = ('<div style="flex:1;display:flex;align-items:center;gap:4px">'
               '<div style="width:50%%;display:flex;justify-content:flex-end">%s</div>'
               '<div style="width:1px;height:16px;background:#c2c7da"></div>'
               '<div style="width:50%%">%s</div></div>'
               % ('<div style="height:12px;width:%d%%;background:%s;border-radius:3px 0 0 3px"></div>'
                  % (int(100 * abs(w) / mx), col) if not up else "",
                  '<div style="height:12px;width:%d%%;background:%s;border-radius:0 3px 3px 0"></div>'
                  % (int(100 * abs(w) / mx), col) if up else ""))
        path = " ".join("%s%s" % (ENGAGE_EMOJI[s], ACTION_EMOJI[a])
                        for s, a in zip(r["s"], r["a"]))
        body.append(
            '<div style="display:flex;align-items:center;gap:10px;margin:5px 0;font-size:12.5px">'
            '<div style="width:120px">%s</div>'
            '<div style="width:58px;text-align:right;font-weight:700;color:%s">%+.2f</div>'
            '%s<div style="width:150px;font-size:11.5px;color:%s;font-weight:700">%s</div></div>'
            % (path, col, w, bar, col,
               "↑ make these actions MORE likely" if up else "↓ make them LESS likely"))
    head = ("weight = G(τ)" if baseline is None
            else "weight = G(τ) − b &nbsp;(b = %.2f)" % b)
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:4px">'
          '↕️ What one policy-gradient step does to each campaign</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px">%s — its <b>sign</b> decides the '
          'direction, its <b>size</b> decides how hard we push.</div>%s'
          % (head, "".join(body)))


def baseline_effect():
    """Why a baseline matters: the same ranking, but every campaign pushed up."""
    G = [11.8, 9.4, 7.6, 6.2, 4.5, 2.7]
    labels = ["📺📺", "🔔📺", "📺⏸️", "⏸️📺", "🔔⏸️", "⏸️⏸️"]
    data = {"G": G, "lab": labels}
    uid = "bl_" + str(abs(hash(("baseline", tuple(G)))) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:700px;background:#fff}
#__UID__ .b-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px}
#__UID__ .b-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ label{display:block;font-size:13px;margin:8px 0;color:#333}
#__UID__ input[type=range]{width:100%}
#__UID__ .b-row{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:12.5px}
#__UID__ .b-lab{width:70px;font-size:15px}
#__UID__ .b-g{width:52px;text-align:right;color:#666}
#__UID__ .b-track{flex:1;display:flex;align-items:center;height:18px}
#__UID__ .b-half{width:50%;display:flex}
#__UID__ .b-tag{width:34px;font-size:15px;text-align:center}
#__UID__ .b-note{font-size:12.5px;color:#444;background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:12px;line-height:1.55;min-height:52px}
</style>
<div id="__UID__">
  <div class="b-head">📏 Good compared to <i>what</i>? — choosing b(🔥 Hot)</div>
  <div class="b-sub">Six times the campaign reached a <b>🔥 Hot</b> learner. Each row is what happened
    <i>from that point on</i> — the moves taken, and the return-to-go they collected. The bar is the
    weight that step gets in the update: <b>G<sub>t</sub> − b(🔥 Hot)</b>. Slide the baseline for Hot
    and watch which of them get pushed up (green, right) and which get pushed down (red, left).</div>
  <label>b(🔥 Hot) = <b><span class="b-bv">0.00</span></b>
    <input type="range" class="b-b" min="0" max="13" step="0.1" value="0"></label>
  <div class="b-list"></div>
  <div class="b-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const b=root.querySelector(".b-b"), list=root.querySelector(".b-list");
  D.G.forEach((g,i)=>{
    const row=document.createElement("div"); row.className="b-row"; row.dataset.i=i;
    row.innerHTML='<div class="b-lab">'+D.lab[i]+'</div><div class="b-g">'+g.toFixed(2)+'</div>'
      +'<div class="b-track"><div class="b-half" style="justify-content:flex-end">'
      +'<div class="b-neg" style="height:13px;width:0%;background:#c0554e;border-radius:3px 0 0 3px"></div></div>'
      +'<div style="width:1px;height:16px;background:#c2c7da"></div>'
      +'<div class="b-half"><div class="b-pos" style="height:13px;width:0%;background:#2e9e7a;border-radius:0 3px 3px 0"></div></div></div>'
      +'<div class="b-tag"></div>';
    list.appendChild(row);
  });
  const mean=D.G.reduce((a,c)=>a+c,0)/D.G.length;
  function upd(){
    const bb=+b.value;
    const w=D.G.map(g=>g-bb), mx=Math.max(...w.map(Math.abs))||1;
    [...list.querySelectorAll(".b-row")].forEach((r,i)=>{
      const v=w[i];
      r.querySelector(".b-pos").style.width=(v>0?100*v/mx:0).toFixed(1)+"%";
      r.querySelector(".b-neg").style.width=(v<0?-100*v/mx:0).toFixed(1)+"%";
      r.querySelector(".b-tag").textContent=v>=0?"↑":"↓";
    });
    root.querySelector(".b-bv").textContent=bb.toFixed(2);
    const n=root.querySelector(".b-note");
    if(bb<=Math.min(...D.G)+0.05) n.innerHTML="<b>b too low.</b> Every one of them — including the "
      +"mediocre ones — gets pushed <i>up</i>, simply because reaching Hot pays well no matter what you "
      +"do there. The good ones are pushed harder, so it still works, but most of the signal is “Hot is "
      +"great”, repeated loudly. That part is pure noise.";
    else if(Math.abs(bb-mean)<0.25) n.innerHTML="<b>b ≈ what a Hot learner normally still earns ("
      +mean.toFixed(1)+").</b> Now the weights are centred: better-than-usual outcomes go up, "
      +"worse-than-usual ones go down. Same ranking, much smaller numbers → a <b>far less noisy</b> "
      +"gradient estimate.";
    else if(bb>Math.max(...D.G)-0.05) n.innerHTML="<b>b too high.</b> Everything gets pushed <i>down</i> "
      +"— even the best of them. Still unbiased on average, still badly centred.";
    else n.innerHTML="An outcome is only good or bad <b>relative to what you normally expect in that "
      +"situation</b>. The baseline is that expectation — subtracting it does not change which outcomes "
      +"were best, only how big the numbers are.";
  }
  b.addEventListener("input",upd); upd();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))


def measure_baseline(sample_episode, returns_to_go, theta, bonus=0.0, n_episodes=2000):
    """Measure b(engagement) — the average return-to-go seen from each engagement
    level — on a batch of campaigns, draw it, and hand the numbers back."""
    sums = np.zeros(len(ENGAGE))
    counts = np.zeros(len(ENGAGE))
    for _ in range(n_episodes):
        states, _, rewards, _ = sample_episode(theta)
        G = returns_to_go([r + bonus for r in rewards])
        for t, st in enumerate(states):
            sums[st] += G[t]; counts[st] += 1
    b = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    mx = max(abs(v) for v in b) or 1.0
    rows = "".join(
        '<div style="display:flex;align-items:center;gap:10px;margin:5px 0">'
        '<div style="width:92px;font-weight:700;font-size:12.5px;color:%s">%s</div>'
        '<div style="flex:1;background:#eef0f7;border-radius:5px;height:16px;overflow:hidden">'
        '<div style="height:100%%;width:%d%%;background:%s"></div></div>'
        '<div style="width:62px;text-align:right;font-weight:800;font-size:13px">%+.2f</div>'
        '<div style="width:132px;font-size:11px;color:#999">from %d campaign-days</div></div>'
        % (ENGAGE_COLOR[i], _e(i), int(100 * abs(b[i]) / mx), ENGAGE_COLOR[i], b[i], int(counts[i]))
        for i in range(len(ENGAGE)))
    _card('<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
          '📏 The baseline, measured — b(engagement level)</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px;line-height:1.55">'
          'Ran <b>%d campaigns</b> with the current policy and averaged the return-to-go seen from each '
          'engagement level. No cleverness: this <i>is</i> “what we normally expect from here”, read '
          'straight off experience.</div>%s'
          '<div style="background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:12px;'
          'font-size:12.5px;color:#333;line-height:1.6">From here on, an outcome is scored as '
          '<b>G<sub>t</sub> − b(s<sub>t</sub>)</b>: what this step actually collected, minus what a '
          'learner at that level normally collects.</div>' % (n_episodes, rows))
    return b


def measure_day_baseline(sample_episode, returns_to_go, theta, bonus=0.0, n_episodes=2000):
    """The finer baseline b(day, engagement) — same measurement, one number per
    (day, engagement) pair. Returned quietly; we only use it for comparison."""
    sums = np.zeros((N_DAYS, len(ENGAGE)))
    counts = np.zeros((N_DAYS, len(ENGAGE)))
    for _ in range(n_episodes):
        states, _, rewards, _ = sample_episode(theta)
        G = returns_to_go([r + bonus for r in rewards])
        for t, st in enumerate(states):
            sums[t, st] += G[t]; counts[t, st] += 1
    fallback = (sums.sum(axis=1) / np.maximum(counts.sum(axis=1), 1))[:, None]
    return np.where(counts > 0, sums / np.maximum(counts, 1), fallback)


def variance_experiment(sample_episode, returns_to_go, theta, exact_grad, gamma,
                        baseline_state, baseline_day_state, bonus=0.0,
                        watch=(0, 1), n_batches=300, n_episodes=16):
    """Estimate ONE gradient component many times over, with three different
    baselines, and show how wide each estimator's spread is. Everything here is
    the REINFORCE estimator you already met — only the baseline changes."""
    import torch

    def estimate(b, per_day):
        surrogate = torch.zeros(())
        for _ in range(n_episodes):
            states, _, rewards, log_probs = sample_episode(theta)
            G = returns_to_go([r + bonus for r in rewards])
            for t in range(N_DAYS):
                base = 0.0 if b is None else (b[t, states[t]] if per_day else b[states[t]])
                surrogate = surrogate + (gamma ** t) * (G[t] - base) * log_probs[t]
        grad, = torch.autograd.grad(surrogate / n_episodes, theta)
        return float(grad[watch])

    runs = [("no baseline at all", None, False, "#dd8452"),
            ("b(engagement)", baseline_state, False, "#4a5bd0"),
            ("b(day, engagement)", baseline_day_state, True, "#2e9e7a")]
    out = []
    for label, b, per_day, col in runs:
        out.append((label, [estimate(b, per_day) for _ in range(n_batches)], col))

    truth = float(exact_grad)
    lines = "".join(
        '<tr><td style="padding:5px 12px;font-weight:700;color:%s">%s</td>'
        '<td style="padding:5px 12px;text-align:right">%+.4f</td>'
        '<td style="padding:5px 12px;text-align:right;font-weight:800">%.3f</td>'
        '<td style="padding:5px 12px;text-align:right;color:#666">%.0f×</td></tr>'
        % (col, label, float(np.mean(v)), float(np.std(v)), np.std(out[0][1]) / np.std(v))
        for label, v, col in out)
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:6px">'
          '🎯 Same gradient component, estimated 300× over — how far does each land from the truth?</div>'
          '<table style="border-collapse:collapse;font-size:12.5px;width:100%%">'
          '<tr><th style="text-align:left;padding:4px 12px;font-size:10.5px;color:#888">estimator</th>'
          '<th style="text-align:right;padding:4px 12px;font-size:10.5px;color:#888">average</th>'
          '<th style="text-align:right;padding:4px 12px;font-size:10.5px;color:#888">spread (std)</th>'
          '<th style="text-align:right;padding:4px 12px;font-size:10.5px;color:#888">vs none</th></tr>'
          '%s</table>'
          '<div style="font-size:12.5px;color:#444;margin-top:10px;line-height:1.6">The exact answer is '
          '<b>%+.4f</b>, and all three estimators are <b>aiming at it</b> — a baseline introduces no '
          'bias. What differs is the spread. And look at the first row: with a spread that wide, even '
          '300 batches are not enough for its <i>average</i> to settle near the truth. That is what '
          'variance costs you — not a wrong destination, just far more experience needed to see which '
          'way it lies.</div>' % (lines, truth), maxw=680)

    variance_histogram(out, truth,
                       title="Three estimators of the same number, %d batches each" % n_batches)


def variance_histogram(series, exact, title=None):
    """Histograms of the SAME gradient component under several estimators.
    `series` is a list of (label, values, colour). They all centre on the exact
    gradient; the better-baselined ones are far tighter."""
    import matplotlib.pyplot as plt
    series = [(lab, np.asarray(v, float), col) for lab, v, col in series]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    lo = min(v.min() for _, v, _ in series); hi = max(v.max() for _, v, _ in series)
    bins = np.linspace(lo, hi, 46)
    for lab, v, col in series:
        ax.hist(v, bins=bins, color=col, alpha=.6, edgecolor="white",
                label="%s   (std = %.3f)" % (lab, v.std()))
    ax.axvline(float(exact), color="#c0554e", ls="--", lw=2.2)
    ax.text(float(exact), ax.get_ylim()[1] * 0.96, "  exact ∂J/∂θ",
            color="#a3352f", fontsize=9.5, va="top")
    ax.set_xlabel("estimate of one gradient component (one batch of episodes each)")
    ax.set_ylabel("how often")
    ax.set_title(title or "Both estimators aim at the same place — one is far less jumpy",
                 fontsize=11, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


def why_baseline_helps():
    """The concrete failure: four sampled steps that all happened at a Hot learner.
    Every return-to-go is positive, so without a baseline every action taken there
    gets pushed UP — including the two that were worse than usual."""
    rows_data = [(2, 6.0), (0, 3.0), (1, 2.5), (2, 6.0)]     # (action, return-to-go)
    b = sum(g for _, g in rows_data) / len(rows_data)
    mx = max(max(g for _, g in rows_data), max(abs(g - b) for _, g in rows_data))

    def bar(v, ref):
        w = int(round(100 * abs(v) / ref))
        col = "#2e9e7a" if v >= 0 else "#c0554e"
        if v >= 0:
            return ('<div style="display:flex"><div style="width:50%%"></div>'
                    '<div style="width:50%%"><div style="height:12px;width:%d%%;background:%s;'
                    'border-radius:0 3px 3px 0"></div></div></div>' % (w, col))
        return ('<div style="display:flex"><div style="width:50%%;display:flex;justify-content:flex-end">'
                '<div style="height:12px;width:%d%%;background:%s;border-radius:3px 0 0 3px"></div></div>'
                '<div style="width:50%%"></div></div>' % (w, col))

    body = ""
    for a, g in rows_data:
        adv = g - b
        body += ('<tr>'
                 '<td style="padding:6px 10px;font-weight:700;font-size:12.5px;color:%s">%s</td>'
                 '<td style="padding:6px 10px;text-align:right;font-size:12.5px">%+.2f</td>'
                 '<td style="padding:6px 8px;width:150px">%s</td>'
                 '<td style="padding:6px 10px;font-size:11.5px;color:#1d6b3a;font-weight:700">↑ up</td>'
                 '<td style="padding:6px 10px;text-align:right;font-size:12.5px">%+.2f</td>'
                 '<td style="padding:6px 8px;width:150px">%s</td>'
                 '<td style="padding:6px 10px;font-size:11.5px;font-weight:700;color:%s">%s</td>'
                 '</tr>'
                 % (ACTION_COLOR[a], _a(a), g, bar(g, mx), adv, bar(adv, mx),
                    "#1d6b3a" if adv >= 0 else "#8a2f28", "↑ up" if adv >= 0 else "↓ down"))

    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">'
        '🔥 Four sampled steps that all happened at a Hot learner (last day)</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:10px;line-height:1.55">'
        'These are real numbers from the reward table. Left: the weight each step gets with '
        '<b>no baseline</b>. Right: the same steps weighted by <b>G<sub>t</sub> − b(🔥 Hot)</b>, '
        'with b = %.2f — the average of what was actually observed here.</div>'
        '<table style="border-collapse:collapse;width:100%%">'
        '<tr><th style="font-size:10.5px;color:#888;text-align:left;padding:3px 10px">action taken</th>'
        '<th colspan="3" style="font-size:10.5px;color:#a3652f;padding:3px 10px">weight = G<sub>t</sub>'
        ' &nbsp;(no baseline)</th>'
        '<th colspan="3" style="font-size:10.5px;color:#4a3a86;padding:3px 10px">weight = G<sub>t</sub>'
        ' − b(🔥 Hot)</th></tr>%s</table>'
        '<div style="background:#fdecec;border-left:4px solid #c0554e;border-radius:6px;'
        'padding:10px 12px;margin-top:12px;font-size:12.5px;color:#8a2f28;line-height:1.6">'
        '<b>Left column, the problem:</b> ⏸️ Wait and 🔔 Nudge were the <i>wrong</i> moves here — and '
        'they still get pushed <b>up</b>, because reaching a Hot learner pays well whatever you do next. '
        'The update is not wrong on average (📺 is pushed hardest), it is just shouting three times and '
        'whispering the difference.</div>'
        '<div style="background:#e7f7ec;border-left:4px solid #2e9e7a;border-radius:6px;'
        'padding:10px 12px;margin-top:8px;font-size:12.5px;color:#1d6b3a;line-height:1.6">'
        '<b>Right column, the fix:</b> measured against what a Hot learner normally earns, the two weak '
        'moves come out <b>negative</b> and get pushed down immediately — from the very same four '
        'campaigns.</div>' % (b, body))


def grading_analogy():
    """Ranking students from few papers on tests of unequal difficulty: the raw
    mark misleads, the mark relative to the test average does not."""
    data = [  # (student, which paper they sat, their mark, that paper's average mark)
        ("Ana",   "easy paper", 5.4, 4.6), ("Ben",  "hard paper", 4.0, 3.2),
        ("Chloé", "easy paper", 5.1, 4.6), ("Dan",  "hard paper", 3.4, 3.2),
        ("Elin",  "hard paper", 4.4, 3.2), ("Femi", "easy paper", 4.5, 4.6)]
    payload = [{"n": n, "t": t, "m": m, "avg": a, "rel": round(m - a, 2)} for n, t, m, a in data]
    uid = "ga_" + str(abs(hash(str(data))) % 10**8)
    tmpl = r"""
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:680px;background:#fff}
#__UID__ .g-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px}
#__UID__ .g-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .g-row{display:flex;align-items:center;gap:10px;margin:5px 0;font-size:13px;transition:.25s}
#__UID__ .g-rank{width:22px;color:#9aa0b5;font-weight:800;font-size:12px}
#__UID__ .g-name{width:70px;font-weight:700}
#__UID__ .g-test{width:88px;font-size:11.5px;padding:2px 8px;border-radius:999px;text-align:center}
#__UID__ .g-easy{background:#fff3e8;color:#a3652f}
#__UID__ .g-hard{background:#eef1fd;color:#3b4a9e}
#__UID__ .g-bar{flex:1;background:#eef0f7;border-radius:5px;height:14px;overflow:hidden}
#__UID__ .g-bar>div{height:100%;background:#764ba2;transition:width .25s}
#__UID__ .g-val{width:58px;text-align:right;font-weight:800}
#__UID__ .g-btn{cursor:pointer;border:none;border-radius:8px;padding:9px 16px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:12px}
#__UID__ .g-note{font-size:12.5px;color:#444;background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:10px;line-height:1.6;min-height:52px}
</style>
<div id="__UID__">
  <div class="g-head">🎓 Ranking six students from one paper each</div>
  <div class="g-sub">You must rank six students by how good they are at maths. You do not have their
    whole year — you have <b>one paper each</b>, and they did not all sit the same test.</div>
  <div class="g-list"></div>
  <button class="g-btn">Subtract each paper's average →</button>
  <div class="g-note"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const list=root.querySelector(".g-list"), btn=root.querySelector(".g-btn");
  let rel=false;
  function draw(){
    const rows=D.slice().sort((a,b)=> (rel? b.rel-a.rel : b.m-a.m));
    const mx=Math.max(...rows.map(r=> rel? Math.abs(r.rel):r.m));
    list.innerHTML=rows.map((r,i)=>
      '<div class="g-row"><div class="g-rank">'+(i+1)+'</div>'
      +'<div class="g-name">'+r.n+'</div>'
      +'<div class="g-test '+(r.t==="easy paper"?"g-easy":"g-hard")+'">'+r.t+'</div>'
      +'<div class="g-bar"><div style="width:'+(100*(rel?Math.abs(r.rel):r.m)/mx).toFixed(0)+'%;'
      +'background:'+(rel&&r.rel<0?"#c0554e":"#764ba2")+'"></div></div>'
      +'<div class="g-val">'+(rel?(r.rel>=0?"+":"")+r.rel.toFixed(2):r.m.toFixed(1))+'</div></div>').join("");
    root.querySelector(".g-note").innerHTML = rel
      ? "Now every mark is read <b>against the paper it came from</b>. Elin sat the hard paper and moves "
        +"to the top; Femi's 4.5 on the easy paper turns out to be <i>below</i> what that paper normally "
        +"produced. Same six papers, same information — a ranking you can act on."
      : "The top three all sat the <b>easy</b> paper. Nothing here says they are better at maths — it "
        +"says they got the easy paper. Collect enough papers and this washes out; with one paper each, "
        +"it decides the ranking.";
    btn.textContent = rel ? "← back to raw marks" : "Subtract each paper's average →";
  }
  btn.addEventListener("click",()=>{rel=!rel;draw();});
  draw();
})();
</script>"""
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(payload))
    display(HTML(html))


def reinforce_loop_diagram():
    """The REINFORCE loop as a flow of five boxes."""
    steps = [
        ("1", "Collect", "run <b>N</b> campaigns with the <i>current</i> policy<br>(sampling the actions)"),
        ("2", "Score", "for every step, the <b>return-to-go</b> G<sub>t</sub><br>— what happened <i>after</i> it"),
        ("3", "Centre", "subtract a <b>baseline</b> b<br>→ advantage A<sub>t</sub> = G<sub>t</sub> − b"),
        ("4", "Push", "loss = −Σ A<sub>t</sub> · log π(a<sub>t</sub>|s<sub>t</sub>)<br>one gradient step on θ"),
        ("5", "Repeat", "throw the episodes away —<br>they came from the <i>old</i> policy"),
    ]
    boxes = []
    for i, (n, name, sub) in enumerate(steps):
        boxes.append(
            '<div style="flex:1;min-width:145px;border:2px solid #4a5bd0;border-radius:12px;'
            'padding:10px 11px;background:#f7f8ff">'
            '<div style="font-size:10.5px;color:#8189c4;font-weight:800">STEP %s</div>'
            '<div style="font-weight:800;font-size:13.5px;color:#2b2d6b;margin:2px 0 4px">%s</div>'
            '<div style="font-size:11.5px;color:#555;line-height:1.5">%s</div></div>' % (n, name, sub))
        if i < len(steps) - 1:
            boxes.append('<div style="align-self:center;font-size:20px;color:#9aa0b5">→</div>')
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:10px">'
        '🔁 The REINFORCE loop</div>'
        '<div style="display:flex;gap:7px;flex-wrap:wrap">%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:12.5px;color:#333;line-height:1.6">This answers the two questions left open: '
        '<b>how often do we update?</b> once per batch of complete campaigns — we need the whole '
        'episode before we can score any step in it. And <b>what do we do with each (s, a, r) '
        'sample?</b> it contributes one term A<sub>t</sub>·log π(a<sub>t</sub>|s<sub>t</sub>) to the '
        'loss, then it is discarded: after the update the data no longer comes from the policy we '
        'are improving.</div>' % "".join(boxes))


def training_curve(js, j_start=None, j_best=None, label="J(θ) (exact, by enumeration)"):
    """Learning curve of the exact objective across REINFORCE iterations."""
    import matplotlib.pyplot as plt
    js = np.asarray(js, float)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(js, color="#4a5bd0", lw=2.2, label=label)
    if j_best is not None:
        ax.axhline(float(j_best), ls="--", color="#2e9e7a", lw=1.8,
                   label="best possible campaign (%.2f)" % float(j_best))
    if j_start is not None:
        ax.axhline(float(j_start), ls=":", color="#9aa0b5", lw=1.6,
                   label="untrained / uniform policy (%.2f)" % float(j_start))
    ax.set_xlabel("REINFORCE iteration  (one batch of campaigns each)")
    ax.set_ylabel("expected return  J(θ)   [CHF of ad revenue per learner]")
    ax.set_title("The policy learns to earn", fontsize=11.5, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout(); plt.show()


# ===========================================================================
#  §5  The deliverable — the action diagram
# ===========================================================================
def action_diagram(probs, visits=None, title="The win-back playbook — hand this to the retention team",
                   subtitle=None):
    """`probs[state]` = the 3 action probabilities the trained policy gives that
    engagement level. `visits[state]` = share of campaign-days spent at that level.
    One row per engagement level: the whole deliverable is three lines."""
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
                       'bar is the full distribution π(a|s) the policy learned; we ship its favourite '
                       'action, and the confidence tells you how close the call was.')
    _card('<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:4px">📄 %s</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px">%s</div>'
          '<table style="border-collapse:collapse;width:100%%">%s</table>' % (title, sub, "".join(rows)))

#  Quiz answer keys
# ===========================================================================
_MC_QUIZZES = {
    "pong_state": (
        "What is “the state” in a game of Pong?",
        "Your agent controls one paddle. Several of the options below could be fed to it. Which one "
        "is the <b>most useful choice of state</b> — enough to act well, without carrying anything "
        "the decision does not need?",
        ["The current score line, e.g. 7–4",
         "The positions of both paddles and of the ball, plus the ball's direction and speed",
         "A single number: the ball's horizontal position",
         "The full history of every frame since the game started"],
        1,
        "Enough to decide where to move the paddle <i>next</i>, and nothing more. The score does not "
        "tell you where the ball is; the ball's position alone hides where it is <i>going</i>; the "
        "full history is not wrong, just enormous — everything that matters is already summarised in "
        "the current positions and velocity. <b>State is a modelling choice</b>, and this one is the "
        "useful compromise."),
    "policy_kind": (
        "Why train a *stochastic* policy?",
        "Our final deliverable is a single recommended action per situation — which sounds "
        "deterministic. So why does the <i>learning</i> use a policy that samples?",
        ["Because customers behave randomly, so the policy must too",
         "Because a policy that always picks the same action never generates evidence about the "
         "alternatives — and because probabilities are what we can smoothly differentiate",
         "Because sampling makes the code run faster",
         "Because a deterministic policy cannot be written down as a table"],
        1,
        "Two reasons, both essential. <b>Evidence:</b> we only ever learn about actions we actually "
        "try. <b>Smoothness:</b> “argmax” jumps in steps, so it has no useful gradient — but the "
        "probability of an action changes smoothly with the logits, and that is exactly what we "
        "differentiate."),
    "log_trick": (
        "What did the log-derivative trick actually buy us?",
        "Differentiating gave us <code>Σ<sub>τ</sub> ∇P(τ|θ) · G(τ)</code>, and the trick rewrote it "
        "as <code>𝔼<sub>τ</sub>[ G(τ) ∇log P(τ|θ) ]</code>. In one sentence: <b>why is that second "
        "form the one we can work with?</b>",
        ["Because it is an <b>expectation</b> — an average over campaigns drawn from our own policy — "
         "and an average is something we can estimate by running campaigns and averaging",
         "Because it makes the return G(τ) differentiable",
         "Because it removes the need to know each campaign's return",
         "Because it makes the gradient smaller, and small gradients are more stable"],
        0,
        "That is the whole point, and nothing else changed. The first form is weighted by "
        "<code>∇P</code>, which is not a probability — so it is not the average of anything we know "
        "how to draw, and sampling cannot help us. Putting <code>P</code> back in front makes it a "
        "genuine expectation over trajectories from π<sub>θ</sub>, and “run some campaigns, take the "
        "average” becomes a valid way to estimate it."),
    "rtg": (
        "Why replace the full return by the return-to-go?",
        "In the estimator we can swap the whole-campaign return G(τ) for the <b>return-to-go</b> "
        "G<sub>t</sub> — only the rewards from step t onwards. Why is that legitimate <i>and</i> "
        "better?",
        ["It makes the estimate biased, but the bias is small enough to ignore",
         "The action at step t cannot have caused rewards that were already banked before it, so "
         "those terms only add noise — dropping them keeps the estimator correct and calms it down",
         "It is only valid when the transitions are deterministic",
         "Because rewards before step t are always zero in practice"],
        1,
        "Crediting an action for money earned <i>before</i> it was taken is not just useless, it is "
        "noise: on average those terms contribute nothing, but in any finite sample they wobble. "
        "Dropping them leaves the expected gradient unchanged and shrinks its variance."),
    "onpolicy": (
        "Why throw the episodes away after the update?",
        "REINFORCE collects a batch of campaigns, takes one gradient step, and then discards them "
        "instead of reusing them. Why?",
        ["Because storing them would use too much memory",
         "Because the estimator is an average over campaigns drawn from the <i>current</i> policy — "
         "once θ has moved, the old campaigns come from the wrong distribution",
         "Because each campaign can only be used once for legal reasons",
         "Because the returns become stale and have to be recomputed"],
        1,
        "The whole derivation reads “expectation under π<sub>θ</sub>”. After the update, π<sub>θ</sub> "
        "is a different distribution, so the old samples no longer represent it. That is what makes "
        "REINFORCE an <b>on-policy</b> method — and what later algorithms (importance ratios, PPO) "
        "work hard to relax."),
}

_TF_QUIZZES = {
    "state": ("What counts as a state?", [
        ("The state is whatever we <i>choose</i> to track, as long as it is enough to act on.", True),
        ("Every problem has one objectively correct state — you can look it up.", False),
        ("Two different learners at the same engagement level are, to our model, the same state.", True),
        ("Adding more variables to the state can only ever help the agent learn.", False),
        ("A state must carry every variable we are able to measure about the situation.", False),
    ]),
    "transition": ("Transition functions", [
        ("A deterministic transition function maps a (state, action) pair to exactly one next state.",
         True),
        ("Because our transitions are stochastic, our policy is forced to be stochastic too.", False),
        ("Under a stochastic transition function, repeating the same action from the same state can "
         "land you somewhere else.", True),
        ("The transition function describes what the <i>world</i> does; the policy describes what "
         "<i>we</i> do.", True),
        ("The reward a move books is drawn at random too, just like the next state.", False),
        ("If transitions are stochastic, the policy must be stochastic too.", False),
        ("In our campaign, the same action from the same state can lead to different next states.", True),
    ]),
    "gamma": ("The discount factor γ", [
        ("γ is part of the objective <i>we</i> choose, not a property of the learner.", True),
        ("With γ = 0 the policy still weighs tomorrow's reward, just a little less.", False),
        ("A larger γ makes the policy more willing to pay a cost now for a bigger payoff later.",
         True),
        ("γ = 1 is forbidden — the return would always be infinite.", False),
        ("Changing γ can change which policy is optimal.", True),
    ]),
    "objective": ("The objective J(θ)", [
        ("J(θ) is an <i>expected</i> return: each campaign counts in proportion to how likely the "
         "policy makes it.", True),
        ("J(θ) is the return of the single most likely campaign.", False),
        ("A trajectory's probability factorises into a part the policy controls and a part the world "
         "controls.", True),
        ("We want to <i>maximise</i> J, so we step along +∇J instead of −∇J.", True),
        ("If one campaign has the highest return, the optimal policy must give it probability 1 to "
         "maximise J.", False),
    ]),
    "baseline": ("Return-to-go and baselines", [
        ("Subtracting a baseline that does not depend on the action leaves the expected gradient "
         "unchanged.", True),
        ("A good baseline is roughly “what we normally expect from this situation”.", True),
        ("Subtracting the batch mean return can flip an above-average campaign into being pushed "
         "down.", False),
        ("Without a baseline, a batch where every return is positive pushes every action up.", True),
        ("The baseline exists to make the algorithm converge to a different, better policy.", False),
    ]),
}

_NUMBER_QUIZZES = {
    "returns": ("🔢 Read three discounted returns off by hand", [
        ("With γ = 0.9, a campaign collects r = [<b>−0.5, 0.5, 6.0</b>]. What is its discounted "
         "return G = Σ γ<sup>t</sup> r<sub>t</sub>?", 4.81, 0.02,
         "G = −0.5 + 0.9·0.5 + 0.81·6.0 — remember the first day is <i>not</i> discounted (γ⁰ = 1)."),
        ("Same rewards r = [−0.5, 0.5, 6.0], but now γ = <b>0</b>. What is G?", -0.5, 0.01,
         "With γ = 0 every term after the first is multiplied by 0 — only today survives."),
        ("Same rewards, γ = <b>1</b>. What is G?", 6.0, 0.01,
         "γ = 1 means no discounting at all: just add the three rewards."),
    ]),
    "probs": ("🔢 Splitting one trajectory's probability", [
        ("Along one trajectory the policy picked its action with probability 0.5 on day 0, 0.5 on "
         "day 1 and 0.6 on day 2. What is the <b>policy's part</b> of P(τ)?", 0.15, 0.005,
         "Multiply the three action probabilities: 0.5 · 0.5 · 0.6."),
        ("For that same trajectory the <b>world's part</b> — the start draw times the two "
         "transitions that happened — is 0.20. What is P(τ) in total?", 0.03, 0.002,
         "The two parts multiply: 0.15 · 0.20. Policy and world each roll their own dice."),
        ("Under a <i>uniform</i> policy (each of 3 actions equally likely, three days), what is the "
         "policy's part of P(τ) for any trajectory? <b>Give it as a decimal rounded to 3 places</b>, "
         "e.g. 0.123", 0.037, 0.004,
         "(1/3)³ = 1/27 ≈ 0.037 — and note this one does <b>not</b> depend on which trajectory."),
    ]),
}


def mc_quiz(key):
    _mc_render(*_MC_QUIZZES[key])


def true_false_quiz(key):
    title, statements = _TF_QUIZZES[key]
    _tf_render(title, statements)


def number_quiz(key):
    title, questions = _NUMBER_QUIZZES[key]
    _nq_render(title, questions)


# ===========================================================================
#  Final boss — timed true/false flash quiz with lives
# ===========================================================================
# Balanced pool (25 true / 25 false), phrased so neither answer is given away by
# the wording (no "always/never" tells, no absurd falses).
_FLASH_POOL = [
    # --- state / actions / transitions ---
    ("The state is a modelling choice: what we decide is worth knowing in order to act.", True),
    ("Every environment has exactly one correct definition of its state.", False),
    ("A transition function maps a state and an action to the next state.", True),
    ("Under a deterministic transition function, the same action from the same state always lands "
     "in the same place.", True),
    ("A stochastic transition function returns a single guaranteed next state.", False),
    ("The transition function describes the environment, not the agent's choices.", True),
    ("A policy that cannot see the day must apply the same rule on every day of the campaign.", True),
    ("Our campaign uses stochastic transitions.", True),
    # --- policy ---
    ("A deterministic policy maps each state to one action.", True),
    ("A stochastic policy assigns a probability to each action in a state.", True),
    ("Deploying a trained softmax policy means rolling its dice for every real customer.", False),
    ("A stochastic policy keeps generating evidence about actions it does not currently prefer.",
     True),
    ("A softmax policy can assign a negative probability to an unattractive action.", False),
    ("Adding the same constant to all logits in a state changes the action probabilities.", False),
    ("Softmax probabilities in a state sum to one.", True),
    ("Increasing the gap between logits pushes a policy closer to uniform.", False),
    ("The argmax of a set of logits is a smooth, differentiable function of them.", False),
    # --- reward / return / gamma ---
    ("The discount factor γ expresses how much we value future rewards relative to today's.", True),
    ("γ is measured from the environment rather than chosen by the modeller.", False),
    ("With γ = 0 the objective reduces to the immediate reward.", True),
    ("A larger γ makes an agent less willing to pay a cost now for a later payoff.", False),
    ("Changing γ can change which policy is optimal.", True),
    ("The first reward of an episode is discounted by γ.", False),
    ("The return-to-go at step t sums the rewards from step t onwards.", True),
    ("The return-to-go at step t includes rewards collected before step t.", False),
    # --- objective ---
    ("J(θ) is the expected discounted return of the policy.", True),
    ("J(θ) is the return of the single most likely trajectory.", False),
    ("A trajectory's probability multiplies the policy's action probabilities by the world's "
     "transition probabilities.", True),
    ("With three actions and three steps there are 9 possible action sequences.", False),
    ("We improve the policy by stepping in the direction of −∇J.", False),
    ("Maximising J is the same problem as minimising −J.", True),
    ("The parameters θ appear inside the return G(τ) of a fixed trajectory.", False),
    ("θ influences J by changing how likely each trajectory is.", True),
    # --- policy gradient ---
    ("The log-derivative trick rewrites ∇P(τ|θ) as P(τ|θ)·∇log P(τ|θ).", True),
    ("The policy gradient makes high-return trajectories more likely and low-return ones less "
     "likely.", True),
    ("The policy gradient requires knowing the environment's transition probabilities.", False),
    ("The world's transition probabilities survive the gradient of log P(τ|θ).", False),
    ("Estimating a transition table needs repeated tries per state-action pair when the world is "
     "stochastic.", True),
    ("The probability of a trajectory is the sum of the probabilities of its actions.", False),
    ("Replacing the full return by the return-to-go changes what the gradient converges to.", False),
    ("Dropping rewards earned before an action reduces the variance of the estimate.", True),
    ("A policy-gradient estimate computed from sampled episodes is exact rather than noisy.", False),
    ("Averaging over more episodes per update leaves the gradient estimate just as noisy.", False),
    # --- baseline ---
    ("Subtracting a state-dependent baseline that ignores the action leaves the gradient unbiased.",
     True),
    ("A baseline changes which policy the algorithm is aiming at.", False),
    ("A useful baseline is a number that depends on which action was sampled.", False),
    ("If every return in a batch is positive, an estimator without a baseline pushes every sampled "
     "action up.", True),
    ("Subtracting the batch mean keeps every advantage positive.", False),
    ("The main purpose of a baseline is to speed up each individual gradient computation.", False),
    # --- REINFORCE loop ---
    ("REINFORCE can score the first step of an episode before the episode has finished.", False),
    ("REINFORCE can reuse the same batch of episodes for many updates without any correction.",
     False),
    ("REINFORCE is an on-policy method: its data must come from the current policy.", True),
    ("The REINFORCE loss is the negative of the log-probabilities weighted by the advantages.",
     True),
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
      +(won?("You cleared "+correct+" questions. Policy gradients are yours."):
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
