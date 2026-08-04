"""Presentation & quiz helpers for the WE4 notebook 05:
"RL in LLMs — teaching a seven-word tutor to answer".

Same idea as WE0's `pdm_viz` and notebook 02's `pg_viz`: every HTML/CSS
illustration, interactive widget, quiz *answer key* and matplotlib visual lives
here, out of the notebook, so the teaching cells stay about the *idea* and the
quizzes can't be solved by reading the cell. The notebook does::

    import llm_rl_viz as viz
    viz.tutor_overview()
    viz.mc_quiz("why_not_mle")

Students are told not to read this file.
"""
import json as _json

import numpy as np
from IPython.display import HTML, display

# ===========================================================================
#  Owly's world (kept here so every widget labels things the same way)
# ===========================================================================
PROMPTS = ["how do you say hello?",
           "i keep forgetting words",
           "i finished the lesson!"]
PROMPT_EMOJI = ["🗣️", "😴", "🎉"]
PROMPT_COLOR = ["#4a5bd0", "#dd8452", "#2e9e7a"]

# The whole vocabulary. "PREMIUM" is the upsell word the growth team loved.
VOCAB = ["hola", "!", "keep", "going", "great", "work", "PREMIUM"]
REPLY_LEN = 2

# The one reply a human tutor would give to each message — this is the verifier.
TARGETS = [[0, 1],   # hola !
           [2, 3],   # keep going
           [4, 5]]   # great work

# What a reply is really worth to a learner (used only to *grade* Part 5 —
# the policy never sees it, and neither does the reward model).
TRUE_UTILITY = {"correct": 1.0, "friendly": 0.35, "upsell": -0.6, "nonsense": 0.0}

# Frozen numbers for the exposure-bias widget, so the picture is the same for
# everyone: a model that is decent at the first word and lost after a wrong one.
_EB_P1 = [0.05, 0.03, 0.62, 0.05, 0.10, 0.05, 0.10]          # step 1, from the prompt
_EB_P2GOLD = [0.02, 0.03, 0.05, 0.80, 0.04, 0.03, 0.03]      # step 2, given the GOLD "keep"
_EB_P2 = [                                                    # step 2, given whatever was said
    [0.10, 0.55, 0.08, 0.07, 0.06, 0.06, 0.08],   # after "hola"
    [0.14, 0.14, 0.14, 0.16, 0.14, 0.14, 0.14],   # after "!"      → lost
    [0.02, 0.03, 0.05, 0.80, 0.04, 0.03, 0.03],   # after "keep"   → on track
    [0.14, 0.16, 0.14, 0.14, 0.14, 0.14, 0.14],   # after "going"  → lost
    [0.10, 0.08, 0.07, 0.07, 0.06, 0.54, 0.08],   # after "great"
    [0.15, 0.14, 0.14, 0.14, 0.15, 0.14, 0.14],   # after "work"   → lost
    [0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.56],   # after "PREMIUM" → doubles down
]


def _decode_id(reply_id):
    """Reply index (0 … 7**2-1) → the tuple of token ids it stands for."""
    V = len(VOCAB)
    return tuple((int(reply_id) // V**(REPLY_LEN - 1 - t)) % V for t in range(REPLY_LEN))


def _say(reply_id):
    return " ".join(VOCAB[t] for t in _decode_id(reply_id))

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
#  §1  The tutor: prompts, vocabulary, replies
# ===========================================================================
def tutor_overview():
    """The scenario on one card: three learner messages, one seven-word tutor."""
    def chip(emoji, name, sub, color):
        return ('<div style="flex:1;min-width:150px;border:1px solid #e6e8ee;border-left:5px solid %s;'
                'border-radius:10px;padding:10px 12px">'
                '<div style="font-size:15px;font-weight:800;color:#2b2d6b">%s %s</div>'
                '<div style="font-size:12.5px;color:#555;margin-top:3px;line-height:1.5">%s</div></div>'
                % (color, emoji, name, sub))
    prompts = "".join(chip(PROMPT_EMOJI[i], "“%s”" % PROMPTS[i],
                           "Owly should reply <code>%s</code>" % " ".join(
                               VOCAB[t] for t in TARGETS[i]),
                           PROMPT_COLOR[i]) for i in range(len(PROMPTS)))
    def word_chip(w):
        style = (("#c0554e", "#c0554e", "#fdecec") if w == "PREMIUM"
                 else ("#d7dae6", "#3b2d6b", "#f7f7fb"))
        return ('<span style="display:inline-block;border:1px solid %s;color:%s;border-radius:8px;'
                'padding:4px 10px;margin:3px;font-size:13px;font-weight:700;background:%s">%s</span>'
                % (style + (w,)))

    vocab = "".join(word_chip(w) for w in VOCAB)
    _card(
        '<div style="font-size:16px;font-weight:800;color:#2b2d6b;margin-bottom:4px">'
        'Owly — the Owlinguo tutor</div>'
        '<div style="font-size:13px;color:#555;margin-bottom:12px;line-height:1.6">'
        'A learner writes one message. Owly answers with <b>exactly two words</b>, one at a time, '
        'each drawn from a seven-word vocabulary. That is the whole language model.</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px">%s</div>'
        '<div style="font-size:12.5px;color:#666;margin-bottom:4px">The vocabulary '
        '(7 words → %d possible replies per message):</div><div>%s</div>' % (
            prompts, len(VOCAB) ** REPLY_LEN, vocab))


def logs_table(rows, title="A sample of the chat logs Owly was trained on"):
    """rows: list of (prompt_index, [token ids], tag) — tag in {'good','upsell'}."""
    body = ""
    for p, toks, tag in rows:
        bad = tag == "upsell"
        body += (
            '<tr>'
            '<td style="padding:7px 10px;border-bottom:1px solid #eef0f6;font-size:13px">%s “%s”</td>'
            '<td style="padding:7px 10px;border-bottom:1px solid #eef0f6;font-size:13px;'
            'font-weight:700;color:%s">%s</td>'
            '<td style="padding:7px 10px;border-bottom:1px solid #eef0f6;font-size:12.5px;color:#777">'
            '%s</td></tr>'
            % (PROMPT_EMOJI[p], PROMPTS[p], "#c0554e" if bad else "#2e9e7a",
               " ".join(VOCAB[t] for t in toks),
               "written by the growth team 😬" if bad else "written by a tutor"))
    _card('<div style="font-size:15px;font-weight:800;color:#2b2d6b;margin-bottom:10px">%s</div>'
          '<table style="border-collapse:collapse;width:100%%">%s</table>'
          '<div style="font-size:12.5px;color:#666;margin-top:10px;line-height:1.6">'
          'Maximum likelihood does not know which rows you are proud of. It raises the probability '
          'of <b>every</b> row it is shown.</div>' % (title, body), maxw=760)


def mdp_mapping():
    """The RL vocabulary translated into LM vocabulary — the Part 2 bridge slide."""
    rows = [("agent", "the language model", "#4a5bd0"),
            ("state  s<sub>t</sub>", "the prompt + the words written so far &nbsp;<code>(x, y<sub>&lt;t</sub>)</code>", "#4a5bd0"),
            ("action  a<sub>t</sub>", "write the next word &nbsp;<code>y<sub>t</sub></code>", "#2e9e7a"),
            ("policy  &pi;<sub>&theta;</sub>(a|s)", "<code>p<sub>&theta;</sub>(y<sub>t</sub> | y<sub>&lt;t</sub>, x)</code> — the model's softmax", "#2e9e7a"),
            ("transition", "glue the word on the end. <b>Deterministic.</b> No dice.", "#dd8452"),
            ("reward  r", "one number, once the reply is finished", "#c0554e"),
            ("episode", "one complete reply", "#8d93a8")]
    body = "".join(
        '<tr><td style="padding:8px 12px;border-bottom:1px solid #eef0f6;font-size:13.5px;'
        'font-weight:800;color:%s;white-space:nowrap">%s</td>'
        '<td style="padding:8px 12px;border-bottom:1px solid #eef0f6;font-size:13.5px;'
        'line-height:1.55">%s</td></tr>' % (c, k, v) for k, v, c in rows)
    _card('<div style="font-size:15px;font-weight:800;color:#2b2d6b;margin-bottom:4px">'
          'Generating text <i>is</i> a Markov decision process</div>'
          '<div style="font-size:12.5px;color:#666;margin-bottom:10px">Every row is something you '
          'already met in notebook 02 — only the right-hand column changed.</div>'
          '<table style="border-collapse:collapse;width:100%%">%s</table>' % body, maxw=780)


def rollout_strip(prompt_idx, tokens, probs, reward=None, title=None):
    """Draw one reply as a left-to-right strip: prompt → word → word → reward."""
    cells = ('<div style="border:1px solid %s;background:%s;border-radius:10px;padding:8px 12px;'
             'font-size:13px;font-weight:700;color:#2b2d6b">%s “%s”</div>'
             % (PROMPT_COLOR[prompt_idx], "#fbfbfe", PROMPT_EMOJI[prompt_idx], PROMPTS[prompt_idx]))
    for t, (tok, p) in enumerate(zip(tokens, probs)):
        cells += ('<div style="font-size:18px;color:#c2c7da;align-self:center">→</div>'
                  '<div style="border:1px solid #d7dae6;border-radius:10px;padding:8px 12px;'
                  'text-align:center;background:#fff">'
                  '<div style="font-size:14px;font-weight:800;color:%s">%s</div>'
                  '<div style="font-size:11.5px;color:#777;margin-top:2px">p = %.2f</div></div>'
                  % ("#c0554e" if VOCAB[tok] == "PREMIUM" else "#3b2d6b", VOCAB[tok], p))
    if reward is not None:
        good = reward > 0.5
        cells += ('<div style="font-size:18px;color:#c2c7da;align-self:center">⇒</div>'
                  '<div style="border:2px solid %s;background:%s;border-radius:10px;padding:8px 14px;'
                  'font-size:14px;font-weight:800;color:%s">r = %.2f</div>'
                  % (("#46b46e", "#e7f7ec", "#1d7a46") if good else ("#e07a7a", "#fdecec", "#b23b34"),
                     reward))
    _card('%s<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:stretch">%s</div>'
          % ('<div style="font-size:14px;font-weight:800;color:#2b2d6b;margin-bottom:10px">%s</div>'
             % title if title else "", cells), maxw=820)


def exposure_bias_demo():
    """Teacher forcing vs. free running, on the same model, side by side."""
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:18px;max-width:820px;background:#fff}
#__UID__ h4{margin:0 0 4px 0;font-size:15px;color:#2b2d6b}
#__UID__ .sub{font-size:12.5px;color:#666;margin-bottom:14px;line-height:1.6}
#__UID__ .cols{display:flex;gap:14px;flex-wrap:wrap}
#__UID__ .col{flex:1;min-width:300px;border:1px solid #e6e8ee;border-radius:12px;padding:12px}
#__UID__ .ttl{font-weight:800;font-size:13.5px;margin-bottom:8px}
#__UID__ .step{display:flex;gap:8px;align-items:center;margin-bottom:7px;font-size:13px}
#__UID__ .box{border:1px solid #d7dae6;border-radius:8px;padding:5px 9px;background:#fbfbfe;font-weight:700}
#__UID__ .gold{border-color:#46b46e;background:#e7f7ec;color:#1d7a46}
#__UID__ .own{border-color:#e0a500;background:#fff8e6;color:#8a6100}
#__UID__ .bad{border-color:#e07a7a;background:#fdecec;color:#b23b34}
#__UID__ .note{font-size:12px;color:#666;margin-top:8px;line-height:1.55}
#__UID__ .btn{cursor:pointer;border:none;border-radius:8px;padding:8px 16px;font-size:13px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2);margin-top:12px}
</style>
<div id="__UID__">
  <h4>Training conditions on the truth. Generation conditions on itself.</h4>
  <div class="sub">Both columns use the <b>same</b> model on the prompt
  <b>“i keep forgetting words”</b>, whose correct reply is <code>keep going</code>. The only
  difference is what the second step is allowed to look at.</div>
  <div class="cols">
    <div class="col">
      <div class="ttl" style="color:#1d7a46">✅ Training (teacher forcing)</div>
      <div class="step">step 1 → predict <span class="box" id="__UID__t1">?</span></div>
      <div class="step">step 2 sees <span class="box gold">keep</span>
        <span style="color:#999">← the <i>gold</i> word, whatever the model said</span></div>
      <div class="step">step 2 → predict <span class="box" id="__UID__t2">?</span></div>
      <div class="note">The model is never asked to recover from its own mistake, because its
      mistake is thrown away before the next step.</div>
    </div>
    <div class="col">
      <div class="ttl" style="color:#8a6100">🎲 Deployment (free running)</div>
      <div class="step">step 1 → sampled <span class="box own" id="__UID__f1">?</span></div>
      <div class="step">step 2 sees <span class="box own" id="__UID__f2">?</span>
        <span style="color:#999">← whatever it just said</span></div>
      <div class="step">step 2 → sampled <span class="box" id="__UID__f3">?</span></div>
      <div class="note" id="__UID__verdict"></div>
    </div>
  </div>
  <button class="btn">Sample another reply</button>
</div>
<script>
(function(){
  const root=document.getElementById("__UID__");
  const V=__VOCAB__, P1=__P1__, P2GOLD=__P2GOLD__, P2=__P2__;
  function draw(p){let u=Math.random(),c=0;for(let i=0;i<p.length;i++){c+=p[i];if(u<=c)return i;}return p.length-1;}
  function run(){
    const g1=draw(P1);
    root.querySelector("#__UID__t1").textContent=V[g1];
    root.querySelector("#__UID__t2").textContent=V[draw(P2GOLD)];
    const f1=draw(P1);
    root.querySelector("#__UID__f1").textContent=V[f1];
    const b2=root.querySelector("#__UID__f2"); b2.textContent=V[f1];
    b2.className="box "+(V[f1]==="keep"?"gold":"bad");
    const f2=draw(P2[f1]);
    const b3=root.querySelector("#__UID__f3"); b3.textContent=V[f2];
    const ok=(V[f1]==="keep"&&V[f2]==="going");
    b3.className="box "+(ok?"gold":"bad");
    root.querySelector("#__UID__verdict").innerHTML = ok
      ? "✅ Right first word, so step 2 was on familiar ground — and it finished the job."
      : "❌ The first word was off. Step 2 is now conditioned on a prefix the model <b>never saw in "
        +"training</b>, so it has no idea how to rescue the sentence. That is <b>exposure bias</b>.";
  }
  root.querySelector(".btn").addEventListener("click",run); run();
})();
</script>'''
    uid = "eb_demo"
    html = (tmpl.replace("__UID__", uid)
            .replace("__VOCAB__", _json.dumps(VOCAB))
            .replace("__P1__", _json.dumps(_EB_P1))
            .replace("__P2GOLD__", _json.dumps(_EB_P2GOLD))
            .replace("__P2__", _json.dumps(_EB_P2)))
    display(HTML(html))


# ===========================================================================
#  §3  Group advantages — the GRPO widget
# ===========================================================================
def group_advantages(rewards, normalize=False, title="One group, four replies"):
    """Show r → r − mean → (÷ std) for a single group of sampled replies."""
    r = np.asarray(rewards, float)
    centred = r - r.mean()
    if normalize:
        z = r.std()
        adv = centred / z if z > 1e-8 else centred * 0.0
    else:
        adv = centred
    dead = float(r.std()) < 1e-8

    def row(label, vals, colour=True):
        cells = ""
        for v in vals:
            col = "#777"
            if colour:
                col = "#1d7a46" if v > 1e-9 else ("#b23b34" if v < -1e-9 else "#999")
            cells += ('<td style="padding:7px 12px;text-align:center;font-size:13.5px;'
                      'font-weight:700;color:%s;border-bottom:1px solid #eef0f6">%+.2f</td>' % (col, v))
        return ('<tr><td style="padding:7px 12px;font-size:13px;color:#555;white-space:nowrap;'
                'border-bottom:1px solid #eef0f6">%s</td>%s</tr>' % (label, cells))

    head = "".join('<td style="padding:7px 12px;text-align:center;font-size:12px;color:#888">'
                   'reply %d</td>' % (i + 1) for i in range(len(r)))
    banner = ""
    if dead:
        banner = ('<div style="margin-top:12px;border:2px solid #e07a7a;background:#fdecec;'
                  'border-radius:10px;padding:10px 12px;font-size:13px;color:#b23b34;line-height:1.6">'
                  '<b>Every advantage is zero.</b> All four replies scored the same, so the group '
                  'mean <i>is</i> every reward. This prompt contributes <b>nothing</b> to the '
                  'gradient — you paid for four generations and bought no learning signal.</div>')
    _card('<div style="font-size:15px;font-weight:800;color:#2b2d6b;margin-bottom:10px">%s</div>'
          '<table style="border-collapse:collapse;width:100%%">'
          '<tr><td></td>%s</tr>%s%s%s</table>%s'
          % (title, head,
             row("reward &nbsp;r", r, colour=False),
             row("− group mean (%.2f)" % r.mean(), centred),
             row("÷ group std (%.2f)" % r.std(), adv) if normalize else "",
             banner), maxw=760)


def live_groups_bar(fracs, labels, title="How much of each batch actually teaches anything?"):
    """Horizontal bars: fraction of groups with non-zero advantage."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 0.62 * len(fracs) + 1.5))
    y = np.arange(len(fracs))[::-1]
    cols = ["#2e9e7a" if f > 0.45 else ("#dd8452" if f > 0.15 else "#c0554e") for f in fracs]
    ax.barh(y, [100 * f for f in fracs], color=cols, height=0.55)
    for yi, f in zip(y, fracs):
        ax.text(100 * f + 1.5, yi, "%.0f%%" % (100 * f), va="center", fontsize=10.5,
                color="#3b2d6b", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 112); ax.set_xlabel("share of prompts with a non-zero advantage  (“live groups”)")
    ax.set_title(title, fontsize=11.5, color="#2b2d6b", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()


def clip_picture(eps=0.2):
    """The PPO clipped objective, both branches, as a picture."""
    import matplotlib.pyplot as plt
    rho = np.linspace(0.0, 2.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=False)
    for ax, A, ttl in [(axes[0], +1.0, "A > 0  ·  a reply worth encouraging"),
                       (axes[1], -1.0, "A < 0  ·  a reply worth discouraging")]:
        unclipped = rho * A
        clipped = np.clip(rho, 1 - eps, 1 + eps) * A
        obj = np.minimum(unclipped, clipped)
        ax.plot(rho, unclipped, ls=":", lw=1.6, color="#9aa0b5", label="ρ·A  (no clip)")
        ax.plot(rho, obj, lw=2.6, color="#4a5bd0", label="PPO objective")
        ax.axvspan(1 - eps, 1 + eps, color="#e7f7ec", zorder=0)
        ax.axvline(1.0, color="#c2c7da", lw=1)
        ax.set_title(ttl, fontsize=10.5, color="#2b2d6b", fontweight="bold")
        ax.set_xlabel("ratio  ρ = p_new / p_old")
        ax.legend(fontsize=8.5, loc="lower right")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("objective")
    plt.suptitle("The clip only bites in the direction the advantage wants to go",
                 fontsize=11.5, color="#2b2d6b", fontweight="bold")
    plt.tight_layout(); plt.show()


# ===========================================================================
#  §4-5  Training curves and the reward-hacking story
# ===========================================================================
def training_curve(series, labels=None, ylabel="accuracy of Owly's replies",
                   title="Does the tutor get better?", hlines=()):
    import matplotlib.pyplot as plt
    if not isinstance(series[0], (list, tuple, np.ndarray)):
        series = [series]
    labels = labels or ["run %d" % (i + 1) for i in range(len(series))]
    cols = ["#4a5bd0", "#c0554e", "#2e9e7a", "#dd8452"]
    fig, ax = plt.subplots(figsize=(8.0, 4.3))
    for i, (s, lab) in enumerate(zip(series, labels)):
        ax.plot(np.asarray(s, float), lw=2.2, color=cols[i % len(cols)], label=lab)
    for val, lab, col in hlines:
        ax.axhline(val, ls="--", lw=1.6, color=col, label=lab)
    ax.set_xlabel("training step  (one batch of prompts each)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11.5, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()


def variance_histogram(no_baseline, with_baseline, exact, title=None):
    import matplotlib.pyplot as plt
    a = np.asarray(no_baseline, float); b = np.asarray(with_baseline, float)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    bins = np.linspace(lo, hi, 44)
    ax.hist(a, bins=bins, alpha=0.62, color="#c0554e", label="plain REINFORCE  (sd %.3f)" % a.std())
    ax.hist(b, bins=bins, alpha=0.62, color="#2e9e7a", label="with a baseline  (sd %.3f)" % b.std())
    ax.axvline(exact, color="#2b2d6b", lw=2.2, ls="--", label="the exact gradient (%.3f)" % exact)
    ax.set_xlabel("one component of the estimated gradient")
    ax.set_ylabel("how often")
    ax.set_title(title or "Same target, same average — very different aim", fontsize=11.5,
                 color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()


def hacking_curve(rm_score, true_score, kl=None,
                  title="The reward goes up. The tutor gets worse."):
    """The Gao-et-al picture, reproduced in miniature."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(rm_score))
    ax.plot(x, rm_score, lw=2.4, color="#c0554e", label="what the reward model says  (the proxy)")
    ax.plot(x, true_score, lw=2.4, color="#2e9e7a", label="what learners actually get  (the truth)")
    k = int(np.argmax(true_score))
    ax.axvline(k, ls=":", lw=1.8, color="#8d93a8")
    ax.annotate("peak true quality\n— everything after this is over-optimization",
                xy=(k, true_score[k]), xytext=(k + max(1, len(x) * 0.08), true_score[k]),
                fontsize=9, color="#3b2d6b",
                arrowprops=dict(arrowstyle="->", color="#8d93a8", lw=1.2))
    ax.set_xlabel("training step")
    ax.set_ylabel("score  (normalised)")
    ax.set_title(title, fontsize=11.5, color="#2b2d6b", fontweight="bold")
    ax.legend(fontsize=9, loc="center right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if kl is not None:
        ax2 = ax.twinx()
        ax2.plot(x, kl, lw=1.6, ls="--", color="#9aa0b5", label="KL from the SFT model")
        ax2.set_ylabel("KL(p_θ ‖ p_SFT)", color="#8d93a8", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#8d93a8")
        ax2.spines["top"].set_visible(False)
    plt.tight_layout(); plt.show()


def kl_frontier(kls, trues, betas):
    """Reward-vs-KL frontier: one point per β."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(kls, trues, "-o", lw=2.0, color="#4a5bd0", ms=7)
    for k, t, b in zip(kls, trues, betas):
        ax.annotate("β=%g" % b, xy=(k, t), xytext=(4, 6), textcoords="offset points",
                    fontsize=9.5, color="#3b2d6b", fontweight="bold")
    ax.set_xlabel("distance travelled from the SFT model    KL(p_θ ‖ p_SFT)")
    ax.set_ylabel("true quality for learners")
    ax.set_title("The honest way to report an RLHF run: quality against distance",
                 fontsize=11.5, color="#2b2d6b", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    plt.tight_layout(); plt.show()


def playbook(probs, title="Owly's reply policy — the deliverable",
             subtitle="For each learner message: the most likely reply, and how sure the tutor is."):
    """probs[p] = full distribution over the V**REPLY_LEN possible replies for prompt p."""
    rows = ""
    for p in range(len(PROMPTS)):
        dist = np.asarray(probs[p], float)
        order = np.argsort(-dist)[:3]
        best = order[0]
        conf = dist[best]
        tone = "decisive" if conf >= .85 else ("a preference" if conf >= .5 else "close to a guess")
        target_ok = list(_decode_id(best)) == list(TARGETS[p])
        alts = " &nbsp;·&nbsp; ".join(
            '<span style="color:#888">%s <span style="font-size:11px">(%.0f%%)</span></span>'
            % (" ".join(VOCAB[t] for t in _decode_id(int(j))), 100 * dist[j]) for j in order[1:])
        bar = ('<div style="height:8px;border-radius:5px;background:#eef0f6;overflow:hidden;'
               'margin-top:6px"><div style="height:100%%;width:%.1f%%;background:%s"></div></div>'
               % (100 * conf, "#2e9e7a" if target_ok else "#c0554e"))
        rows += (
            '<div style="border:1px solid #e6e8ee;border-left:5px solid %s;border-radius:10px;'
            'padding:11px 13px;margin-bottom:9px">'
            '<div style="font-size:13px;color:#555">%s “%s”</div>'
            '<div style="font-size:16px;font-weight:800;color:%s;margin-top:5px">%s %s</div>'
            '<div style="font-size:12px;color:#777;margin-top:3px">%.0f%% — %s &nbsp;|&nbsp; '
            'runners-up: %s</div>%s</div>'
            % (PROMPT_COLOR[p], PROMPT_EMOJI[p], PROMPTS[p],
               "#1d7a46" if target_ok else "#b23b34",
               " ".join(VOCAB[t] for t in _decode_id(int(best))),
               "✅" if target_ok else "❌", 100 * conf, tone, alts, bar))
    _card('<div style="font-size:15px;font-weight:800;color:#2b2d6b">%s</div>'
          '<div style="font-size:12.5px;color:#666;margin:4px 0 12px">%s</div>%s'
          % (title, subtitle, rows), maxw=760)


# ===========================================================================
#  Quiz banks
# ===========================================================================
_MC_QUIZZES = {
    "why_not_mle": (
        "Why can't we fix Owly by collecting more chat logs?",
        "Owly upsells because the logs it imitated contain upsell replies. Suppose you clean the "
        "logs and fine-tune again. What does that <b>still</b> not solve?",
        ["Nothing — clean data solves it completely",
         "Cleaning removes the bad rows, but imitation can still only copy what a human already "
         "wrote: it has no way to express “this reply is <i>worse</i> than that one”, and it never "
         "trains Owly on prefixes Owly itself produces",
         "It fails because cross-entropy is not differentiable",
         "It fails because the vocabulary is too small"],
        1,
        "Cleaning the data is worth doing and it is not enough. Maximum likelihood has exactly one "
        "verb — <b>make this more likely</b>. It cannot rank, cannot penalise, and cannot teach "
        "recovery from a bad first word, because in training the first word is always the gold one."),
    "nondiff": (
        "Why can't we just backpropagate through the reward?",
        "We want ∇<sub>θ</sub> 𝔼[r(x, y)] where y is the reply Owly generated. What exactly is in "
        "the way?",
        ["r is too expensive to evaluate",
         "y is a <b>sampled, discrete</b> object — θ does not appear inside r(x,y) at all, it only "
         "sets how likely each reply is, so the gradient has to act on the probabilities",
         "The reward is between 0 and 1, and that range is too narrow to differentiate",
         "PyTorch cannot differentiate through a softmax"],
        1,
        "The reward of a <i>given</i> reply is a fixed number — nudging θ does not change what "
        "<code>keep going</code> scores. What θ changes is the <b>probability</b> of producing it. "
        "That is why we need the log-derivative trick, exactly as in notebook 02."),
    "dead_group": (
        "A prompt where all four sampled replies are wrong — what does it teach?",
        "You sample K = 4 replies for one prompt and the verifier scores them <code>0, 0, 0, 0</code>. "
        "What does that prompt contribute to the GRPO update?",
        ["A strong negative signal — all four replies get pushed down",
         "<b>Nothing.</b> Every reward equals the group mean, so every advantage is 0 and the "
         "gradient contribution is exactly zero — the four generations were paid for and wasted",
         "It contributes, but only half as much as a mixed group",
         "It makes the update unstable"],
        1,
        "This is the single most practical consequence of the group baseline. Advantages are "
        "<i>relative to the group</i>, so a uniform group — all right or all wrong — is silent. "
        "Real pipelines filter prompts by difficulty precisely to avoid paying for silence."),
    "ratio": (
        "What is the probability ratio in the PPO objective actually correcting for?",
        "The GRPO/PPO loss multiplies the advantage by ρ = p<sub>new</sub>/p<sub>old</sub>. If you "
        "took exactly one gradient step per batch of samples, what would ρ be?",
        ["It would still matter — it is what keeps the update small",
         "It would be exactly <b>1</b>: ρ is an importance-sampling correction for reusing samples "
         "drawn from an <i>older</i> policy, and with one step per batch there is nothing to correct",
         "It would be 0, and the loss would vanish",
         "It depends on the learning rate but is never 1"],
        1,
        "Generation is the expensive part, so we take several gradient steps on one batch of "
        "replies. After the first step the samples come from the wrong distribution and ρ repairs "
        "the estimator. Strictly on-policy, ρ = 1 and the whole thing collapses back to REINFORCE — "
        "which is why <code>clip</code> is protecting the <i>reuse</i>, not the algorithm."),
    "kl_why": (
        "Why does the KL penalty stop reward hacking, when clipping does not?",
        "PPO's clip already keeps each update small. So why is a separate KL term needed?",
        ["It isn't — the KL term is just a second safety margin doing the same job",
         "They constrain different things: clipping keeps <b>this step</b> near the <b>previous "
         "step</b>, while KL keeps the <b>policy</b> near the <b>SFT model</b> — and a policy can "
         "walk arbitrarily far in small, perfectly-clipped steps",
         "Because KL is cheaper to compute than clipping",
         "Because clipping only works for verifiable rewards"],
        1,
        "This is the confusion worth killing early. Clipping is about optimisation stability, one "
        "update at a time. KL is about staying inside the region where the <i>reward model was "
        "trained</i> and is therefore still meaningful. Small steps in a consistent direction still "
        "get you a long way from home."),
    "rlvr_vs_rlhf": (
        "Why can we be so much more aggressive with the verifiable reward?",
        "In Part 4 we optimised the verifier hard, with no KL penalty, and Owly just got better. In "
        "Part 5 the same treatment destroyed it. What is the difference?",
        ["Part 5 used a bigger learning rate",
         "The verifier is an <b>exact function</b> — there is no gap between it and what we want, "
         "so there is nothing to over-optimise. The reward model is a <b>learned approximation</b>, "
         "and it stops being informative off the data it was trained on",
         "The verifier is differentiable and the reward model is not",
         "Because the reward model was trained on too few parameters"],
        1,
        "Exactly the reason modern RLVR pipelines routinely run with β = 0 while RLHF cannot. The "
        "KL leash is not dogma — it is a repair for a specific defect: your reward is a model, and "
        "models are only trustworthy near their training distribution."),
}

_TF_QUIZZES = {
    "mdp": ("🧠 Text generation as an MDP", [
        ("The state is the prompt plus the words generated so far.", True),
        ("The transition function is stochastic: the same prefix can lead to different next states.",
         False),
        ("The action space is the vocabulary, so a real LM chooses between tens of thousands of "
         "actions at every step.", True),
        ("Because the transitions are deterministic and known, model-based RL would be the natural "
         "choice here.", False),
        ("All of the randomness in a rollout comes from the policy's own sampling.", True),
        ("The reward arrives once, at the end of the reply.", True),
    ]),
    "reinforce": ("🧠 The estimator", [
        ("The REINFORCE gradient is the cross-entropy gradient on the model's own sample, weighted "
         "by the reward.", True),
        ("Subtracting a baseline that does not depend on the action changes what the gradient "
         "converges to.", False),
        ("Subtracting a baseline reduces the variance of the estimate.", True),
        ("A reply with a negative advantage has its probability pushed down.", True),
        ("REINFORCE needs to know the transition probabilities of the environment.", False),
    ]),
    "grpo": ("🧠 Group-relative advantages", [
        ("GRPO estimates the baseline from several replies to the <i>same</i> prompt.", True),
        ("GRPO needs a separately trained value network.", False),
        ("If all replies in a group get the same reward, that prompt produces no gradient.", True),
        ("Dividing by the group standard deviation is required for the advantages to be unbiased.",
         False),
        ("Generating K replies per prompt instead of 1 makes each step roughly K times more "
         "expensive.", True),
    ]),
    "hacking": ("🧠 Reward models and hacking", [
        ("A Bradley–Terry reward model is trained to score preferred replies above dispreferred "
         "ones.", True),
        ("Only the <i>differences</i> between reward-model scores are identified — the overall "
         "scale and offset are arbitrary.", True),
        ("A rising reward curve is evidence that the policy is genuinely improving.", False),
        ("Reward hacking means the policy found inputs where the reward model is wrong.", True),
        ("The reward model is reliable everywhere, because it is a neural network.", False),
    ]),
}

_NUMBER_QUIZZES = {
    "advantages": ("🔢 Work out one group's advantages by hand", [
        ("A group of K = 4 replies scores r = [<b>1, 0, 0, 0</b>]. What is the group mean?",
         0.25, 0.005, "One correct out of four: (1+0+0+0)/4."),
        ("Same group. What advantage does the <b>correct</b> reply get, before any division by the "
         "standard deviation?", 0.75, 0.005, "A = r − mean = 1 − 0.25."),
        ("Same group. What advantage does <b>each wrong</b> reply get?", -0.25, 0.005,
         "A = 0 − 0.25. Three replies share the push-down; one gets all the push-up."),
    ]),
    "probs": ("🔢 The probability of one reply", [
        ("Owly writes <code>keep</code> with probability 0.6, then <code>going</code> with "
         "probability 0.5. What is p(<code>keep going</code> | prompt)?", 0.30, 0.005,
         "Autoregressive means multiply: 0.6 × 0.5. Nothing else enters — the transitions are "
         "deterministic, so the world contributes no probability of its own."),
        ("Under a <b>uniform</b> policy over 7 words and a 2-word reply, what is the probability of "
         "any particular reply? Give it to 3 decimals.", 0.020, 0.004,
         "(1/7)² = 1/49 ≈ 0.020."),
        ("How many distinct replies can Owly produce for one prompt?", 49, 0.5,
         "7 words, 2 slots, so 7² = 49. Small enough that we can enumerate every one of them and "
         "compute the objective exactly."),
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
_FLASH_POOL = [
    # --- why RL at all ---
    ("Maximum likelihood training raises the probability of every example it is shown.", True),
    ("Cross-entropy fine-tuning can express that one reply is worse than another.", False),
    ("Fine-tuning on a corpus caps the model near the quality of the demonstrations in it.", True),
    ("Exposure bias comes from training on gold prefixes but generating from the model's own.", True),
    ("At training time the model is asked to recover from its own mistakes.", False),
    ("The objective we care about, such as “is the answer correct”, is defined on a complete "
     "sequence.", True),
    ("Reward functions used in RL fine-tuning are usually differentiable in the model's weights.",
     False),
    # --- the MDP ---
    ("In the LM as an MDP, the state is the prompt plus the tokens generated so far.", True),
    ("Appending a token to a prefix is a deterministic transition.", True),
    ("The action space of a language-model policy is its vocabulary.", True),
    ("In the one-step formulation, the whole reply counts as a single action.", True),
    ("The token-level formulation makes it possible to give different tokens different rewards.",
     True),
    ("Generating a reply involves environment randomness as well as policy randomness.", False),
    ("Reward in the basic RLVR setup arrives at every token.", False),
    # --- REINFORCE ---
    ("The score-function estimator differentiates the distribution rather than the reward.", True),
    ("REINFORCE requires knowing the environment's transition probabilities.", False),
    ("The REINFORCE update is a reward-weighted cross-entropy gradient on the model's own samples.",
     True),
    ("A baseline that depends only on the prompt leaves the expected gradient unchanged.", True),
    ("Adding a baseline is a bias-variance trade: it introduces bias to cut variance.", False),
    ("An action with a negative advantage gets its probability reduced.", True),
    # --- GRPO ---
    ("GRPO estimates its baseline from a group of replies to the same prompt.", True),
    ("GRPO removes the need for a learned value network.", True),
    ("A group in which every reply scores identically contributes no gradient.", True),
    ("Dividing group-centred rewards by the group standard deviation is needed for unbiasedness.",
     False),
    ("Sampling K replies per prompt makes generation roughly K times more expensive.", True),
    ("Prompts that the model always solves are as useless for GRPO as prompts it never solves.",
     True),
    # --- ratio / clipping ---
    ("The probability ratio corrects for taking several gradient steps on one batch of samples.",
     True),
    ("Under strictly on-policy training the probability ratio equals 1.", True),
    ("Clipping the ratio prevents the policy from drifting far from the SFT model over many steps.",
     False),
    ("PPO's clip caps how much a single update can move in the direction the advantage favours.",
     True),
    ("PPO returns to a first-order method while keeping TRPO's trust-region idea.", True),
    # --- rewards ---
    ("A verifiable reward is a function you can write down and check.", True),
    ("Fraction-of-tests-passed is a denser signal than a 0/1 correctness check.", True),
    ("A Bradley–Terry reward model is trained on pairwise preferences.", True),
    ("Only differences of reward-model scores are identified; the scale is arbitrary.", True),
    ("A learned reward model is equally trustworthy on inputs far from its training data.", False),
    ("Reward hacking means the optimiser found where the reward model is wrong.", True),
    ("A rising reward curve alone is evidence that quality is improving.", False),
    ("An exact verifier can be over-optimised in the same way a learned reward model can.", False),
    # --- KL ---
    ("The KL penalty keeps the trained policy close to the model it started from.", True),
    ("A larger β gives the policy more freedom to chase reward.", False),
    ("In practice the KL penalty is often folded into the reward.", True),
    ("Using a KL penalty requires keeping a frozen copy of the reference model.", True),
    ("The optimal KL-regularised policy is the reference policy reweighted by exponentiated reward.",
     True),
    ("Modern RLVR pipelines often run with no KL penalty at all.", True),
    # --- practice ---
    ("Generation usually dominates the wall-clock time of an RL fine-tuning run.", True),
    ("Classic PPO-based RLHF may hold four models at once: policy, reference, reward and value.",
     True),
    ("Entropy collapse means the policy's output distribution loses diversity.", True),
    ("Improving pass@1 guarantees pass@k improves too.", False),
    ("Evaluating with the same reward model you trained against is a valid check.", False),
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