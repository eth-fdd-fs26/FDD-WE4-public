"""Presentation, environment & quiz helpers for the WE4 notebook 01:
"Reinforcement Learning — the interview you are not qualified for".

Same idea as WE0's `pdm_viz`, WE4's `pg_viz` and `ac_viz`: every HTML/CSS
illustration, interactive widget, quiz *answer key* and matplotlib visual lives
here, out of the notebook, so the teaching cells stay about the *idea*.

It also hides one thing on purpose: **the interviewer's brain**. The transition
probabilities live in this file and are never printed by the notebook — that is
what makes the exercise *model-free*. The notebook only ever gets to call
`Interviewer().step(action)` and look at what comes back.

    import intro_viz as iv
    iv.the_setup()
    env = iv.Interviewer()

Students are told not to read this file. (You are, presumably, not a student.)
"""
import json as _json
import itertools as _it

import numpy as np
from IPython.display import HTML, display

# ===========================================================================
#  The interview vocabulary (kept here so every widget labels things the same)
# ===========================================================================
MOODS = ["Furrowed", "Bored", "Neutral", "Smiling", "Delighted"]
MOOD_EMOJI = ["\U0001F620", "\U0001F971", "\U0001F610", "\U0001F642", "\U0001F604"]
MOOD_COLOR = ["#c0554e", "#9aa0b5", "#7d84a0", "#dd8452", "#2e9e7a"]
N_MOODS = 5

HIRED, REJECTED = 5, 6          # the two terminal faces
TERM_EMOJI = {HIRED: "\U0001F973", REJECTED: "\U0001F494"}
TERM_NAME = {HIRED: "Hired", REJECTED: "Rejected"}
TERM_LINE = {
    HIRED: "“You are taken. Welcome to GaGGle.”",
    REJECTED: "“After careful consideration, we would like to move on "
              "to the next candidate.”",
}

ACTIONS = ["Smile", "Jargon", "Compliment", "Agree hard"]
ACTION_EMOJI = ["\U0001F601", "\U0001F9E0", "\U0001F490", "\U0001F44F"]
ACTION_COLOR = ["#e0a500", "#4a5bd0", "#c9548f", "#2e9e7a"]
ACTION_BLURB = [
    "Beam at them. Costs nothing, says nothing, defuses a lot.",
    "Deploy words you have heard but never understood, in an order that sounds load-bearing.",
    "Praise the office, the questions, the mug — anything, warmly and specifically.",
    "Punctuate whatever they just said with “that’s actually very smart”.",
]
N_ACTIONS = 4

# What you actually say. Same action, different line every time — purely cosmetic,
# because watching yourself say these things is most of the fun.
ACTION_LINES = [
    [  # 😁 Smile
        "You beam. It is a wide, unbroken, slightly-too-long smile.",
        "You smile with your whole face, teeth included, for four full seconds.",
        "You give a warm, closed-lip smile and one small understanding nod.",
        "You smile the way people smile in stock photos about teamwork.",
        "You smile. That is the entire contribution. Just… smiling.",
        "You smile so hard your ears move. You did not know they could do that.",
    ],
    [  # 🧠 Jargon
        "“Fundamentally it’s a sparse attention manifold over a low-rank latent prior.”",
        "“We basically distilled the retrieval head into a quantised mixture-of-experts router.”",
        "“Right — so it’s really a contrastive objective sitting on a diffusion backbone.”",
        "“I’d argue the bottleneck is KV-cache topology, not parameter count.”",
        "“It’s gradient descent, but Bayesian. And on a graph. Non-Euclidean, obviously.”",
        "“The embeddings are fine, it’s the manifold curvature that’s under-provisioned.”",
    ],
    [  # 💐 Compliment
        "“This is a beautiful office. The light in here is doing something special.”",
        "“Can I just say — your questions are unusually well constructed.”",
        "“Honestly, the research here changed how I think about the whole field.”",
        "“That mug is fantastic. Where does one even acquire a mug like that?”",
        "“You have an extremely calming interviewing presence.”",
        "“I love that you took notes by hand. Nobody does that any more.”",
    ],
    [  # 👏 Agree hard
        "“That’s actually very smart.” You nod slowly, as if only now understanding.",
        "“Mmh. That’s a really good point.” You write nothing down, meaningfully.",
        "“Exactly — that’s the part most people miss.”",
        "“See, that’s the right question to ask.”",
        "“Interesting. I genuinely hadn’t framed it that way.”",
        "“Right, right, right. Yes. Completely.”",
    ],
]

# --- the three things you can see across the desk --------------------------
FACE_LABEL = ["eyebrows furrowed", "visibly bored", "neutral", "smiling", "delighted"]
HANDS_EMOJI = ["✋", "✊", "\U0001F44D", "\U0001F91D"]
HANDS_LABEL = ["resting flat on the desk", "clenched into a fist",
               "giving a small thumbs-up", "extended to shake yours"]
PEN_EMOJI = ["\U0001F58A️", "✍️", "\U0001F442", "\U0001F4D5"]
PEN_LABEL = ["capped, on the desk", "uncapped, in hand",
             "parked behind their ear", "put away, notebook closed"]

FEATURES = {
    "face":  {"title": "Their face", "emoji": MOOD_EMOJI, "labels": FACE_LABEL,
              "n": 5, "note": "the whole point of having a face"},
    "hands": {"title": "Their hands", "emoji": HANDS_EMOJI[:3], "labels": HANDS_LABEL[:3],
              "n": 3, "note": "people do fidget when it is going badly"},
    "pen":   {"title": "Their pen", "emoji": PEN_EMOJI[:3], "labels": PEN_LABEL[:3],
              "n": 3, "note": "a pen is a pen"},
}
FEATURE_ORDER = ["face", "hands", "pen"]

# ===========================================================================
#  The interviewer's brain.  NOT printed by the notebook — that is the point.
# ===========================================================================
#  _MOOD_TRANS[mood][action] = list of (next mood-or-terminal, probability)
#  Optimal sheet: 😠 Smile · 🥱 Agree · 😐 Compliment · 🙂 Agree · 😄 Jargon
#  Random play gets an offer ~30% of the time (G = -0.24); optimal ~80% (G = +0.40).
_MOOD_TRANS = [
    # 😠 Furrowed — genuinely dangerous, every option can end you
    [[(1, .40), (0, .35), (REJECTED, .25)],                 # 😁 smile   — the least bad
     [(REJECTED, .60), (0, .30), (1, .10)],                 # 🧠 jargon  — catastrophic
     [(1, .30), (0, .35), (REJECTED, .35)],                 # 💐 compliment — transparent
     [(0, .35), (REJECTED, .50), (1, .15)]],                # 👏 agree   — reads as mockery
    # 🥱 Bored
    [[(2, .35), (1, .45), (0, .15), (REJECTED, .05)],
     [(REJECTED, .35), (0, .35), (2, .30)],
     [(2, .45), (1, .30), (0, .20), (REJECTED, .05)],
     [(2, .50), (3, .25), (1, .15), (0, .10)]],             # 👏 wakes them up
    # 😐 Neutral
    [[(3, .25), (2, .55), (1, .20)],
     [(0, .35), (2, .30), (3, .30), (REJECTED, .05)],
     [(3, .55), (4, .10), (2, .25), (1, .10)],              # 💐 the reliable climber
     [(3, .35), (2, .35), (1, .30)]],
    # 🙂 Smiling
    [[(3, .55), (4, .15), (2, .30)],
     [(4, .25), (3, .20), (2, .15), (REJECTED, .30), (HIRED, .10)],
     [(4, .20), (3, .45), (2, .35)],
     [(4, .60), (3, .25), (2, .15)]],                       # 👏 best climber
    # 😄 Delighted
    [[(4, .55), (3, .40), (HIRED, .05)],
     [(HIRED, .45), (4, .30), (3, .15), (REJECTED, .10)],   # 🧠 the closer
     [(3, .55), (4, .30), (2, .15)],                        # 💐 now it reads as grovelling
     [(4, .50), (3, .42), (HIRED, .08)]],
]

#  The SECOND interviewer — the final round, used only for the race in Part 4.
#  Deliberately a different person: allergic to flattery, energised by technical
#  talk, and unimpressed by agreement until they already like you.
#  Optimal sheet: 😠 Jargon · 🥱 Jargon · 😐 Smile · 🙂 Agree · 😄 Agree
#  — only ONE action in common with the sheet learned upstairs, and playing the
#  first sheet here scores WORSE than acting at random. Copying will not save you.
_MOOD_TRANS_2 = [
    # 😠 Furrowed
    [[(0, .45), (1, .25), (REJECTED, .30)],
     [(1, .50), (0, .30), (REJECTED, .20)],                 # 🧠 content earns patience
     [(0, .30), (REJECTED, .55), (1, .15)],
     [(0, .35), (REJECTED, .45), (1, .20)]],
    # 🥱 Bored
    [[(1, .50), (2, .30), (0, .15), (REJECTED, .05)],
     [(2, .55), (3, .20), (1, .20), (REJECTED, .05)],       # 🧠 finally, a real conversation
     [(0, .35), (1, .40), (REJECTED, .25)],
     [(1, .45), (2, .25), (0, .25), (REJECTED, .05)]],
    # 😐 Neutral
    [[(3, .50), (2, .35), (1, .15)],                        # 😁 warmth lands here
     [(3, .35), (2, .30), (0, .25), (REJECTED, .10)],
     [(1, .45), (2, .30), (REJECTED, .25)],
     [(2, .45), (3, .25), (1, .30)]],
    # 🙂 Smiling
    [[(3, .45), (4, .25), (2, .30)],
     [(4, .30), (3, .25), (2, .20), (REJECTED, .25)],
     [(2, .50), (3, .25), (REJECTED, .25)],
     [(4, .55), (3, .30), (2, .15)]],                       # 👏 now agreement works
    # 😄 Delighted
    [[(4, .50), (3, .35), (HIRED, .15)],
     [(HIRED, .30), (4, .25), (3, .15), (REJECTED, .30)],
     [(3, .60), (2, .20), (REJECTED, .20)],
     [(HIRED, .40), (4, .35), (3, .25)]],                   # 👏 closes it
]

#  P(hands | mood) — correlated with the mood, but far from a giveaway
_HAND_PROBS = [[.20, .75, .05],
               [.60, .35, .05],
               [.65, .25, .10],
               [.55, .15, .30],
               [.35, .05, .60]]
#  P(pen | mood) — identical in every mood. A pen is a pen.
_PEN_PROBS = [1 / 3.0, 1 / 3.0, 1 / 3.0]

_START_PROBS = [0.00, 0.25, 0.50, 0.20, 0.05]
MAX_TURNS = 30
R_HIRED, R_REJECTED = +1.0, -1.0


def _draw(pairs, rng):
    u = rng.random()
    acc = 0.0
    for value, p in pairs:
        acc += p
        if u < acc:
            return value
    return pairs[-1][0]


class Interviewer(object):
    """The environment. You may call `reset()` and `step(action)`. That is all
    you get — exactly as much as you get from a real interviewer.

    An observation is a dict::

        {"face": 0..4 (or 5 hired / 6 rejected), "hands": 0..3, "pen": 0..3,
         "turn": int, "done": bool}
    """

    def __init__(self, seed=None, deterministic=False):
        self.rng = np.random.default_rng(seed)
        self.deterministic = bool(deterministic)
        self._mood = None
        self._turn = 0
        self.done = True

    # -- what you can see, given how they actually feel ---------------------
    def _observe(self, mood):
        if mood in (HIRED, REJECTED):
            return {"face": mood, "hands": 3, "pen": 3,
                    "turn": self._turn, "done": True}
        hands = int(self.rng.choice(3, p=_HAND_PROBS[mood]))
        pen = int(self.rng.choice(3, p=_PEN_PROBS))
        return {"face": int(mood), "hands": hands, "pen": pen,
                "turn": self._turn, "done": False}

    def reset(self):
        """A new interviewer, a new costume, a fresh start. Returns the first observation."""
        self._mood = int(self.rng.choice(N_MOODS, p=_START_PROBS))
        self._turn = 0
        self.done = False
        return self._observe(self._mood)

    def step(self, action):
        """Take one action. Returns (observation, reward, done)."""
        if self.done:
            raise RuntimeError("This interview is over. Call reset() for a new one.")
        pairs = _MOOD_TRANS[self._mood][int(action)]
        if self.deterministic:
            nxt = max(pairs, key=lambda o: o[1])[0]
        else:
            nxt = _draw(pairs, self.rng)
        self._turn += 1

        if nxt in (HIRED, REJECTED):
            self._mood, self.done = nxt, True
            reward = R_HIRED if nxt == HIRED else R_REJECTED
            return self._observe(nxt), reward, True
        if self._turn >= MAX_TURNS:            # they have another meeting. It is a no.
            self._mood, self.done = REJECTED, True
            return self._observe(REJECTED), R_REJECTED, True
        self._mood = int(nxt)
        return self._observe(self._mood), 0.0, False


# ===========================================================================
#  Generic renderers  (shared with pdm_viz / pg_viz / ac_viz)
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
#  §1  The setting: the story, the actions, the reward
# ===========================================================================
def five_words():
    """The five objects of RL, and who decides each one."""
    rows = [
        ("\U0001F440", "State", "s", "what you look at before you speak",
         "yours", "#764ba2",
         "The description of the situation your decision is allowed to depend on. "
         "Not a fact about the room — a <b>choice</b>."),
        ("\U0001F5E3️", "Action", "a", "what you can actually do",
         "given", "#e0a500",
         "The closed list of moves available. Four sentences, in your case."),
        ("\U0001F3B2", "Transition", "P(s′|s,a)", "how the interviewer reacts",
         "the world's", "#9aa0b5",
         "Where each move leaves you, with probabilities. The world's business — "
         "and in this notebook you never get to see it."),
        ("\U0001F3AF", "Reward", "r", "what counts as success",
         "yours", "#2e9e7a",
         "The number the world pays you. +1 for an offer, −1 for a rejection, "
         "0 the rest of the time. You wrote it, so it is on you."),
        ("\U0001F4CB", "Policy", "π(s) → a", "the rule you act by",
         "learned", "#4a5bd0",
         "The map from situation to move. This is the object being learned — "
         "everything else exists to make this one good."),
    ]
    owner_style = {
        "yours": ("#764ba2", "#f3f0ff", "you decide"),
        "given": ("#a07800", "#fff8e5", "given by the problem"),
        "the world's": ("#6b7186", "#f2f3f8", "the world decides"),
        "learned": ("#2a3ea8", "#eef1ff", "learned from experience"),
    }
    body = "".join(
        '<tr>'
        '<td style="padding:9px 10px;border-top:1px solid #eef0f7;font-size:21px;'
        'text-align:center;width:38px">%s</td>'
        '<td style="padding:9px 10px;border-top:1px solid #eef0f7;white-space:nowrap">'
        '<div style="font-weight:800;font-size:13.5px;color:%s">%s</div>'
        '<div style="font-family:ui-monospace,Menlo,monospace;font-size:11.5px;color:#888">%s</div>'
        '</td>'
        '<td style="padding:9px 10px;border-top:1px solid #eef0f7;font-size:12.5px;color:#333;'
        'line-height:1.55"><b>%s</b><br><span style="color:#777;font-size:11.5px">%s</span></td>'
        '<td style="padding:9px 10px;border-top:1px solid #eef0f7;text-align:right">'
        '<span style="display:inline-block;background:%s;color:%s;border-radius:20px;'
        'padding:4px 11px;font-size:11px;font-weight:800;white-space:nowrap">%s</span></td>'
        '</tr>'
        % (emoji, col, name, sym, short, long_,
           owner_style[owner][1], owner_style[owner][0], owner_style[owner][2])
        for emoji, name, sym, short, owner, col, long_ in rows)
    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:3px">'
        '\U0001F9E9 The five objects — and who gets to choose each one</div>'
        '<div style="font-size:12.5px;color:#666;line-height:1.6;margin-bottom:10px">'
        'People usually meet these as a list of definitions to memorise. Meet them instead as five '
        'decisions somebody had to make about your interview — and note the right-hand column, '
        'because <b>two of the five are yours</b>, and getting those two wrong is the most common '
        'way real RL projects fail.</div>'
        '<table style="border-collapse:collapse;width:100%%">%s</table>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:12.5px;color:#333;line-height:1.6">'
        'One sentence ties them together, and it is the whole field:<br>'
        '<b style="font-size:13.5px;color:#2b2d6b">“Learn a policy π that picks the action a which '
        'earns the most reward r, given the state s I am in — without ever being told P.”</b></div>'
        % body, maxw=800)


def the_setup():
    """The scenario on one card."""
    def chip(emoji, name, sub, color):
        return ('<div style="flex:1;min-width:170px;border:2px solid %s;border-radius:12px;'
                'padding:10px 12px;background:#fff">'
                '<div style="font-size:22px">%s</div>'
                '<div style="font-weight:800;font-size:13.5px;color:#222">%s</div>'
                '<div style="font-size:11.5px;color:#777;line-height:1.45;margin-top:2px">%s</div></div>'
                % (color, emoji, name, sub))

    acts = "".join(chip(ACTION_EMOJI[i], ACTIONS[i], ACTION_BLURB[i], ACTION_COLOR[i])
                   for i in range(N_ACTIONS))
    outcomes = (
        chip("\U0001F973", "“You are taken.”",
             "The interview ends. <b>Reward +1.</b> You have a job, a badge, and a free lunch.",
             "#2e9e7a")
        + chip("\U0001F494", "“…move on to the next candidate.”",
               "The interview ends. <b>Reward −1.</b> You change costume in the corridor and "
               "queue up again.", "#c0554e")
        + chip("\U0001F92B", "anything else they say",
               "The interview continues. <b>Reward 0.</b> Nothing has been decided yet — which "
               "is not the same as nothing having happened.", "#9aa0b5"))
    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:4px">'
        '\U0001F3E2 GaGGle · Interview room 4B</div>'
        '<div style="font-size:12.5px;color:#555;line-height:1.6;margin-bottom:14px">'
        'You graduated last month. Unfortunately, four years of letting a chatbot do the thinking '
        'have left your brain with the structural integrity of a wet croissant, and in eleven minutes '
        'you have a technical interview at <b>GaGGle</b> — best pay in the industry, best research in '
        'the world.<br><br>Two things are in your favour. The interviewer is of the older generation '
        'and their hearing and eyesight are, let us say, <i>generous</i>. And you own an unreasonable '
        'number of costumes. So if this interview goes badly, you put on a moustache and get back in '
        'the queue.<br><br>You will not learn machine learning before 11 a.m. But you can learn '
        '<b>how to act</b>.</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">Everything you are capable of doing</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">%s</div>'
        '<div style="font-size:11px;font-weight:700;color:#4a3a86;text-transform:uppercase;'
        'letter-spacing:.04em;margin-bottom:6px">How the interview can end</div>'
        '<div style="display:flex;gap:10px;flex-wrap:wrap">%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:16px;'
        'font-size:12.5px;color:#333;line-height:1.6">\U0001F3AF <b>Your deliverable.</b> By the end '
        'of this notebook you will hand yourself a small table: <i>interviewer looks like this → do '
        'that.</i> That table has a name in reinforcement learning — a <b>policy</b> — and you are '
        'going to learn it from experience, one humiliating interview at a time.</div>'
        % (acts, outcomes))


# ===========================================================================
#  §2  The playable interview  (the "Pokémon battle" scene)
# ===========================================================================
_REACTIONS = {
    0: ["They frown. Whatever you just said has been taken personally.",
        "Their eyebrows descend like a garage door.",
        "A silence forms. It has texture."],
    1: ["They glance at the clock. Twice.",
        "They exhale through the nose and check something on their phone.",
        "They say “mm-hm” in the key of someone thinking about lunch."],
    2: ["They nod, professionally, revealing absolutely nothing.",
        "“Okay.” That is the entire response. “Okay.”",
        "They write one word down. You cannot read it upside down."],
    3: ["The corners of their mouth move upward. This is progress.",
        "“Ha — right.” They sit back a little.",
        "They smile, and for one second the room is warm."],
    4: ["They laugh. An actual, involuntary laugh, and lean in.",
        "“You know, that is *exactly* how I see it too.”",
        "They start telling you about their PhD. You are winning."],
}
_SCENE_DATA = {
    "trans": _MOOD_TRANS, "hand": _HAND_PROBS, "pen": _PEN_PROBS, "start": _START_PROBS,
    "me": MOOD_EMOJI, "mc": MOOD_COLOR, "mn": MOODS, "fl": FACE_LABEL,
    "he": HANDS_EMOJI, "hl": HANDS_LABEL, "pe": PEN_EMOJI, "pl": PEN_LABEL,
    "acts": ACTIONS, "ae": ACTION_EMOJI, "ac": ACTION_COLOR, "lines": ACTION_LINES,
    "react": _REACTIONS, "term": {str(HIRED): TERM_LINE[HIRED], str(REJECTED): TERM_LINE[REJECTED]},
    "te": {str(HIRED): TERM_EMOJI[HIRED], str(REJECTED): TERM_EMOJI[REJECTED]},
    "hired": HIRED, "rejected": REJECTED, "maxturns": MAX_TURNS,
}

_SCENE_CSS = r'''
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:16px;padding:0;max-width:__W__px;background:#fff;overflow:hidden}
#__UID__ .g-top{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff}
#__UID__ .g-title{font-weight:800;font-size:14px}
#__UID__ .g-meta{font-size:12px;opacity:.92}
#__UID__ .g-modes{display:flex;gap:8px;padding:10px 16px 0}
#__UID__ .g-tab{cursor:pointer;border:2px solid #d7dbe8;border-radius:9px;padding:5px 11px;font-size:12px;font-weight:700;color:#555;background:#fff}
#__UID__ .g-tab.on{border-color:#764ba2;color:#4a3a86;background:#f1edff}
#__UID__ .g-room{position:relative;height:225px;background:linear-gradient(#eef1fa,#e2e7f5 62%,#d7ddef 62%);overflow:hidden}
#__UID__ .g-win{position:absolute;top:16px;left:18px;width:96px;height:64px;border-radius:6px;background:linear-gradient(#bfe0f5,#e9f4fb);border:4px solid #b9c0d6}
#__UID__ .g-poster{position:absolute;top:16px;right:18px;width:78px;height:64px;border-radius:6px;background:#fff;border:3px solid #b9c0d6;font-size:10px;color:#8a90a8;display:flex;align-items:center;justify-content:center;text-align:center;line-height:1.25;padding:3px;font-weight:700}
#__UID__ .g-person{position:absolute;left:50%;top:24px;transform:translateX(-50%);text-align:center;width:220px}
#__UID__ .g-face{font-size:62px;line-height:1;transition:.18s;display:inline-block}
#__UID__ .g-face.pop{transform:scale(1.16)}
#__UID__ .g-body{width:132px;height:74px;margin:-6px auto 0;border-radius:44px 44px 8px 8px;transition:.3s}
#__UID__ .g-desk{position:absolute;left:0;right:0;bottom:0;height:84px;background:linear-gradient(#a9763f,#8d5f30);border-top:5px solid #c08c4e}
#__UID__ .g-hands{position:absolute;bottom:52px;left:0;right:0;text-align:center;font-size:27px;letter-spacing:26px;text-indent:26px}
#__UID__ .g-pen{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);font-size:22px}
#__UID__ .g-mug{position:absolute;bottom:20px;left:calc(50% - 96px);font-size:22px}
#__UID__ .g-obs{position:absolute;right:10px;bottom:8px;font-size:10.5px;color:#f4ead9;line-height:1.45;text-align:right;opacity:.9}
#__UID__ .g-dlg{border-top:1px solid #e6e8ee;padding:11px 16px;min-height:74px;font-size:13px;line-height:1.6}
#__UID__ .g-you{color:#2b2d6b}
#__UID__ .g-them{color:#555;font-style:italic}
#__UID__ .g-end{font-weight:800;font-size:14.5px;margin-top:5px}
#__UID__ .g-btns{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 16px 14px}
#__UID__ .g-btn{cursor:pointer;border:2px solid;border-radius:11px;padding:9px 11px;font-size:13px;font-weight:800;background:#fff;text-align:left;transition:.1s}
#__UID__ .g-btn:hover{filter:brightness(.96);transform:translateY(-1px)}
#__UID__ .g-btn[disabled]{opacity:.35;cursor:default;transform:none}
#__UID__ .g-foot{display:flex;align-items:center;gap:10px;padding:0 16px 14px;flex-wrap:wrap}
#__UID__ .g-again{cursor:pointer;border:none;border-radius:9px;padding:8px 15px;font-size:12.5px;font-weight:700;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .g-tally{font-size:12px;color:#666}
#__UID__ .g-note{font-size:11.5px;color:#777;padding:0 16px 14px;line-height:1.5}
'''

_SCENE_JS = r'''
  function pick(arr){return arr[Math.floor(Math.random()*arr.length)];}
  function drawFrom(pairs){var u=Math.random(),acc=0;
    for(var i=0;i<pairs.length;i++){acc+=pairs[i][1]; if(u<acc) return pairs[i][0];}
    return pairs[pairs.length-1][0];}
  function drawIdx(probs){var u=Math.random(),acc=0;
    for(var i=0;i<probs.length;i++){acc+=probs[i]; if(u<acc) return i;}
    return probs.length-1;}
  function observe(mood){
    if(mood===D.hired||mood===D.rejected) return {face:mood,hands:3,pen:3};
    return {face:mood, hands:drawIdx(D.hand[mood]), pen:drawIdx(D.pen)};}
  function bestOf(pairs){var b=pairs[0];for(var i=1;i<pairs.length;i++){if(pairs[i][1]>b[1])b=pairs[i];}return b[0];}
'''


def interview_game(mode_switch=False, height=None, title="\U0001F3AE The interview",
                   note=""):
    """The playable interview. Click an action, watch what it does to the interviewer.

    mode_switch=True adds a DETERMINISTIC / STOCHASTIC toggle and a
    “replay the same moves” button — used later to make the point about
    stochastic transitions.
    """
    uid = "ivg_" + str(abs(hash(("game", mode_switch, title))) % 10**8)
    data = dict(_SCENE_DATA)
    data["modesw"] = bool(mode_switch)
    modes = ('<div class="g-modes">'
             '<div class="g-tab" data-m="0">Deterministic world</div>'
             '<div class="g-tab on" data-m="1">Stochastic world · the real one</div>'
             '</div>') if mode_switch else ''
    replay = ('<button class="g-again g-replay" style="background:#fff;color:#4a3a86;'
              'border:2px solid #764ba2">↻ Replay the same moves</button>') if mode_switch else ''
    tmpl = (r'<style>' + _SCENE_CSS + r'''</style>
<div id="__UID__">
  <div class="g-top"><div class="g-title">__TITLE__</div>
    <div class="g-meta"><span class="g-turn"></span></div></div>
  __MODES__
  <div class="g-room">
    <div class="g-win"></div>
    <div class="g-poster">GaGGle<br>DO THE<br>RIGHT THING</div>
    <div class="g-person"><div class="g-face">😐</div><div class="g-body"></div></div>
    <div class="g-desk">
      <div class="g-hands"></div>
      <div class="g-mug">☕</div>
      <div class="g-pen">🖊️</div>
      <div class="g-obs"></div>
    </div>
  </div>
  <div class="g-dlg"></div>
  <div class="g-btns"></div>
  <div class="g-foot">
    <button class="g-again">🥸 New costume, new interview</button>
    __REPLAY__
    <div class="g-tally"></div>
  </div>
  __NOTE__
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
  const $=s=>root.querySelector(s);
''' + _SCENE_JS + r'''
  let mood=2, startMood=null, turn=0, done=false, mode=1, played=[];
  let tries=0, hires=0;

  const btns=$(".g-btns");
  D.acts.forEach((a,i)=>{
    const b=document.createElement("button"); b.className="g-btn"; b.dataset.i=i;
    b.style.borderColor=D.ac[i]; b.style.color=D.ac[i];
    b.innerHTML=D.ae[i]+"  "+a;
    b.addEventListener("click",()=>{ if(!done){ played.push(i); act(i); } });
    btns.appendChild(b);
  });

  function paint(obs, extra){
    const f=obs.face, term=(f===D.hired||f===D.rejected);
    const face=$(".g-face");
    face.textContent = term ? D.te[""+f] : D.me[f];
    face.classList.add("pop"); setTimeout(()=>face.classList.remove("pop"),180);
    $(".g-body").style.background = term ? "#8a90a8" : D.mc[f];
    $(".g-hands").textContent = D.he[obs.hands];
    $(".g-pen").textContent   = D.pe[obs.pen];
    $(".g-turn").textContent  = "turn " + turn + " / " + D.maxturns;
    $(".g-obs").innerHTML = "face: <b>"+(term?(f===D.hired?"hired":"rejected"):D.fl[f])
      +"</b><br>hands: <b>"+D.hl[obs.hands]+"</b><br>pen: <b>"+D.pl[obs.pen]+"</b>";
    $(".g-tally").textContent = tries?("interviews: "+tries+"  ·  offers: "+hires):"";
    root.querySelectorAll(".g-btn").forEach(b=>b.disabled=done);
  }
  function reset(keepMoves){
    // A replay — and a switch between the two worlds — has to start from the SAME
    // interviewer, or you are comparing two different stories. Only "new costume"
    // draws a fresh starting mood.
    if(!keepMoves || startMood===null) startMood=drawFrom(D.start.map((p,i)=>[i,p]));
    mood=startMood; turn=0; done=false;
    if(!keepMoves) played=[];
    $(".g-dlg").innerHTML='<span class="g-them">You sit down. They look up. '
      +'Somewhere a printer is having a difficult morning.</span>';
    paint(observe(mood));
  }
  function act(a){
    const pairs=D.trans[mood][a];
    const nxt = (mode===0) ? bestOf(pairs) : drawFrom(pairs);
    turn++;
    let html='<div class="g-you">'+D.ae[a]+' <b>'+pick(D.lines[a])+'</b></div>';
    if(nxt===D.hired||nxt===D.rejected){
      done=true; tries++; if(nxt===D.hired)hires++;
      mood=nxt;
      html+='<div class="g-end" style="color:'+(nxt===D.hired?"#1d7a46":"#b23b34")+'">'
           +D.te[""+nxt]+" "+D.term[""+nxt]+'  ·  reward '+(nxt===D.hired?"+1":"−1")+'</div>';
    } else if(turn>=D.maxturns){
      done=true; tries++; mood=D.rejected;
      html+='<div class="g-end" style="color:#b23b34">💔 “Sorry — I have another meeting.” '
           +'Thirty turns is, itself, an answer.  ·  reward −1</div>';
    } else {
      mood=nxt;
      html+='<div class="g-them">'+pick(D.react[mood])+'</div>';
    }
    $(".g-dlg").innerHTML=html;
    paint(observe(mood));
  }

  root.querySelectorAll(".g-tab").forEach(t=>t.addEventListener("click",()=>{
    root.querySelectorAll(".g-tab").forEach(x=>x.classList.remove("on"));
    t.classList.add("on"); mode=+t.dataset.m; reset(true);
  }));
  $(".g-again").addEventListener("click",()=>reset(false));
  const rp=$(".g-replay");
  if(rp) rp.addEventListener("click",()=>{
    const moves=played.slice(); if(!moves.length) return;
    reset(true); played=moves;
    $(".g-dlg").innerHTML='<span class="g-them">Same interviewer, same opening mood, same '
      +moves.length+' sentences. Watch what the world does with them this time.</span>';
    let k=0;
    function nextMove(){
      if(k>=moves.length||done) return;
      act(moves[k++]); setTimeout(nextMove, 620);
    }
    setTimeout(nextMove, 900);   // hold on the starting interviewer so you can see it is the same
  });
  reset(false);
})();
</script>''')
    html = (tmpl.replace("__UID__", uid).replace("__W__", str(height or 560))
            .replace("__TITLE__", title).replace("__MODES__", modes)
            .replace("__REPLAY__", replay)
            .replace("__NOTE__", ('<div class="g-note">%s</div>' % note) if note else "")
            .replace("__DATA__", _json.dumps(data)))
    display(HTML(html))


# ===========================================================================
#  §3  State — the modelling choice
# ===========================================================================
def what_you_can_see():
    """Everything visible across the desk, laid out value by value."""
    def row(key, informative):
        f = FEATURES[key]
        chips = "".join(
            '<div style="border:1px solid #e2e5ef;border-radius:10px;padding:7px 10px;'
            'background:#fbfcff;text-align:center;min-width:92px">'
            '<div style="font-size:24px">%s</div>'
            '<div style="font-size:10.5px;color:#666;line-height:1.35;margin-top:2px">%s</div>'
            '<div style="font-size:10px;color:#aab;margin-top:3px">value <b>%d</b></div></div>'
            % (e, lab, i) for i, (e, lab) in enumerate(zip(f["emoji"], f["labels"])))
        return ('<div style="margin-bottom:14px">'
                '<div style="font-weight:800;font-size:13px;color:#2b2d6b">%s '
                '<span style="font-weight:600;font-size:11.5px;color:#888">· %d possible values '
                '· <code>obs["%s"]</code></span></div>'
                '<div style="font-size:11.5px;color:#777;margin:2px 0 7px">%s</div>'
                '<div style="display:flex;gap:8px;flex-wrap:wrap">%s</div></div>'
                % (f["title"], f["n"], key, informative, chips))

    term = "".join(
        '<div style="border:2px solid %s;border-radius:10px;padding:7px 12px;background:#fff;'
        'text-align:center;min-width:140px"><div style="font-size:24px">%s</div>'
        '<div style="font-size:11px;color:#444;font-weight:700;margin-top:2px">%s</div>'
        '<div style="font-size:10px;color:#888;margin-top:2px">face = %d · hands = 3 · pen = 3</div>'
        '</div>' % (col, TERM_EMOJI[t], TERM_NAME[t], t)
        for t, col in [(HIRED, "#2e9e7a"), (REJECTED, "#c0554e")])

    _card(
        '<div style="font-weight:800;font-size:16px;color:#2b2d6b;margin-bottom:10px">'
        '\U0001F440 Everything you can see across that desk</div>'
        + row("face", "Their expression. It moves when you say things.")
        + row("hands", "What their hands are doing while their face does whatever it does.")
        + row("pen", "Where the pen is. It is a pen. It has a life of its own.")
        + '<div style="font-weight:800;font-size:13px;color:#2b2d6b;margin-bottom:6px">'
          'And the two ways it ends <span style="font-weight:600;font-size:11.5px;color:#888">'
          '· note that <i>everything</i> changes when the interview ends — even the pen</span></div>'
          '<div style="display:flex;gap:10px;flex-wrap:wrap">%s</div>'
          '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:15px;'
          'font-size:12.5px;color:#333;line-height:1.6">\U0001F9E9 <b>You must now choose.</b> '
          'The <b>state</b> is not what the world is — it is what you decide to look at before you '
          'act. Take all three and you carry more information, but you also have far more situations '
          'to learn about, each visited far less often. Take one and you learn fast — about possibly '
          'the wrong thing.</div>' % term)


def state_space_card(features):
    """How big is the state space this choice creates?"""
    n = 1
    for f in features:
        n *= FEATURES[f]["n"]
    total = n + 2
    if not features:
        desc = ('You chose to look at <b>nothing</b>. Every non-terminal moment of every interview '
                'is, to you, the same situation — so your "policy" is a single row: one preferred '
                'action, forever. That is not reinforcement learning, that is a habit.')
    elif features == ["face"]:
        desc = ('The face is the part that actually <i>moves</i> when you speak. Five situations to '
                'learn about, each visited constantly — this will learn fast.')
    else:
        desc = ('Every extra component multiplies the number of situations you must learn about '
                'separately. Your experience does not multiply with it: the same interviews now have '
                'to fill %d rows instead of 5.' % n)
    chips = " × ".join('<b>%s</b> (%d)' % (FEATURES[f]["title"].split()[-1], FEATURES[f]["n"])
                       for f in features) or "<b>nothing</b> (1)"
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:8px">'
        '\U0001F9ED Your state space</div>'
        '<div style="font-size:13px;color:#333;line-height:1.7">%s &nbsp;=&nbsp; '
        '<b style="color:#4a3a86;font-size:16px">%d</b> situations to act in, '
        '<span style="color:#888">+ 2 terminal ones you never act in</span> = <b>%d</b> states '
        'in total.</div>'
        '<div style="font-size:12.5px;color:#555;line-height:1.6;margin-top:10px">%s</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:12px;'
        'font-size:12.5px;color:#333;line-height:1.6">Your table of “what to do here” will have '
        '<b>%d rows × %d actions = %d numbers</b> to learn. Hold on to that number — you will watch '
        'them fill in.</div>' % (chips, n, total, desc, n, N_ACTIONS, n * N_ACTIONS), maxw=780)


def transition_graph(features, seed=0):
    """Draw the chosen state space and the arrows between its states — with every
    probability left as ???, because nobody gave us the interviewer's brain."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    names = [state_text(s, features) for s in enumerate_states(features)]
    n = len(names)
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.axis("off")

    named = n <= 10        # beyond that, labelling every node is illegible — which is the point
    if named:                                      # ring layout
        ang = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n, endpoint=False)
        pos = np.c_[2.4 * np.cos(ang), 1.55 * np.sin(ang)]
    else:                                          # grid of anonymous nodes
        cols = int(np.ceil(np.sqrt(n * 1.7)))
        rows = int(np.ceil(n / cols))
        pos = np.array([[(i % cols) - (cols - 1) / 2.0,
                         -((i // cols) - (rows - 1) / 2.0)] for i in range(n)], dtype=float)
        pos[:, 0] *= 5.4 / max(cols, 1)
        pos[:, 1] *= 3.4 / max(rows, 1)
    term = np.array([[3.9, 0.95], [3.9, -0.95]])

    if named:
        for (x, y), lab in zip(pos, names):
            ax.text(x, y, lab, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.34", fc="#f3f0ff", ec="#764ba2", lw=1.4))
    else:
        ax.scatter(pos[:, 0], pos[:, 1], s=125, marker="o", c="#f3f0ff",
                   edgecolors="#764ba2", linewidths=1.5, zorder=3)
        ax.text(pos[:, 0].min(), pos[:, 1].max() + 0.55,
                "one circle = one situation you must learn about separately",
                fontsize=9, color="#764ba2", style="italic", ha="left")
    for (x, y), lab, col in zip(term, ["HIRED\n+1", "REJECTED\n-1"], ["#2e9e7a", "#c0554e"]):
        ax.text(x, y, lab, ha="center", va="center", fontsize=11, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.42", fc="#fff", ec=col, lw=2.2))

    # a representative handful of arrows — all of them would be a hairball, and
    # that is exactly the point being made
    src = list(range(min(n, 6)))
    for i in src:
        picks = list(rng.choice([j for j in range(n) if j != i],
                                size=min(2, max(n - 1, 1)), replace=False)) if n > 1 else []
        for j in picks:
            ax.add_patch(FancyArrowPatch(pos[i], pos[j], connectionstyle="arc3,rad=0.22",
                                         arrowstyle="-|>", mutation_scale=11,
                                         color="#9aa0b5", lw=1.0, shrinkA=17, shrinkB=17))
            mid = (pos[i] + pos[j]) / 2 + np.array([0.0, 0.13])
            ax.text(mid[0], mid[1], "???", fontsize=7.5, color="#8a6fd0", ha="center")
    for k, t in enumerate(term):
        i = int(rng.integers(n))
        ax.add_patch(FancyArrowPatch(pos[i], t, connectionstyle="arc3,rad=0.15",
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=["#2e9e7a", "#c0554e"][k], lw=1.3,
                                     shrinkA=17, shrinkB=26, alpha=.75))

    ax.set_xlim(-3.4, 5.0)
    ax.set_ylim(-2.15, 2.15)
    ax.set_title("Your state space: %d states you act in, 2 you don't.\n"
                 "Every arrow needs a probability — and every one of them is ???"
                 % n, fontsize=12, color="#2b2d6b", fontweight="bold")
    ax.text(0.5, -0.02,
            "A SKETCH, not the real graph: only a handful of the arrows are drawn, and which ones "
            "is arbitrary.\nIn truth every state has 4 actions leading almost anywhere — including, "
            "from most of them, straight to an ending.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#8a90a8",
            style="italic")
    plt.tight_layout()
    plt.show()


def model_free_card():
    _card(
        '<div style="display:flex;gap:14px;flex-wrap:wrap">'
        '<div style="flex:1;min-width:250px;border:2px solid #9aa0b5;border-radius:12px;padding:12px">'
        '<div style="font-size:20px">\U0001F9E0\U0001F50C</div>'
        '<div style="font-weight:800;font-size:13.5px;color:#333;margin:3px 0 5px">'
        'If you had a cable into their head</div>'
        '<div style="font-size:12px;color:#555;line-height:1.6">You would know '
        '<code>P(s′ | s, a)</code> for every arrow. That is a <b>known MDP</b>, and it is barely a '
        'learning problem any more — with the whole table of probabilities in front of you the best '
        'possible policy can simply be <i>calculated</i>, on paper, without ever entering the '
        'room.</div></div>'
        '<div style="flex:1;min-width:250px;border:2px solid #764ba2;border-radius:12px;padding:12px;'
        'background:#faf7ff">'
        '<div style="font-size:20px">\U0001F576️</div>'
        '<div style="font-weight:800;font-size:13.5px;color:#3b2d6b;margin:3px 0 5px">'
        'What you actually have</div>'
        '<div style="font-size:12px;color:#555;line-height:1.6">A human being, a chair, and as many '
        'costumes as it takes. You never see a probability — you see one interview, then another. '
        'Learning to act <b>without ever writing down the arrows</b> is called <b>model-free</b> RL, '
        'and it is the setting for the rest of this notebook, and the rest of the week.</div></div>'
        '</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:12.5px;color:#333;line-height:1.6">\U0001F4A1 Model-free is not a compromise you '
        'make because the maths is hard. It is what you do when the environment is a person, a market '
        'or a user — things that have dynamics but do not come with documentation.</div>')


# ===========================================================================
#  State bookkeeping — shared by every widget that has to name a state
# ===========================================================================
def enumerate_states(features):
    """Every non-terminal state that this choice of features creates."""
    return [tuple(c) for c in _it.product(*[range(FEATURES[f]["n"]) for f in features])]


def state_label(s, features):
    """A short emoji label for a state, whatever the student chose to look at."""
    if len(s) == 2 and s[0] == "end":
        return "%s %s" % (TERM_EMOJI[s[1]], TERM_NAME[s[1]])
    if not features:
        return "\U0001F441️ (one state)"
    return " ".join(FEATURES[f]["emoji"][v] for f, v in zip(features, s))


_state_label = state_label


def state_text(s, features):
    """Plain-text state label — matplotlib has no emoji font, so charts use this."""
    if len(s) == 2 and s[0] == "end":
        return "%s\n%+d" % (TERM_NAME[s[1]].upper(), 1 if s[1] == HIRED else -1)
    if not features:
        return "(one\nstate)"
    return "\n".join(FEATURES[f]["labels"][v].split(",")[0] for f, v in zip(features, s))


def state_tooltip(s, features):
    if not features:
        return "the only state there is"
    return ", ".join("%s %s" % (FEATURES[f]["title"].split()[-1], FEATURES[f]["labels"][v])
                     for f, v in zip(features, s))


# ===========================================================================
#  §4  Return, V and Q
# ===========================================================================
def gamma_ladder():
    """What the discounted return *is* when the only rewards are ±1 at the end."""
    uid = "gl_" + str(abs(hash("gamma_ladder")) % 10**8)
    tmpl = r'''
<style>
#__UID__{font-family:system-ui,Segoe UI,Roboto,sans-serif;border:1px solid #e6e8ee;border-radius:14px;padding:16px;max-width:800px;background:#fff}
#__UID__ .gl-head{font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:3px}
#__UID__ .gl-sub{font-size:12.5px;color:#666;margin-bottom:12px;line-height:1.55}
#__UID__ .gl-ctl{display:flex;align-items:center;gap:12px;margin-bottom:14px;flex-wrap:wrap}
#__UID__ input[type=range]{flex:1;min-width:220px;accent-color:#764ba2}
#__UID__ .gl-g{font-size:14px;font-weight:800;color:#4a3a86;min-width:78px}
#__UID__ .gl-rows{display:flex;flex-direction:column;gap:4px}
#__UID__ .gl-row{display:flex;align-items:center;gap:9px;font-size:12px}
#__UID__ .gl-k{width:112px;color:#555;flex:0 0 auto}
#__UID__ .gl-track{flex:1;height:17px;background:#f2f3f8;border-radius:5px;position:relative;overflow:hidden}
#__UID__ .gl-fill{position:absolute;left:0;top:0;bottom:0;background:linear-gradient(90deg,#7fd0a6,#2e9e7a);transition:width .12s}
#__UID__ .gl-v{width:58px;text-align:right;font-weight:800;color:#1d7a46;flex:0 0 auto}
#__UID__ .gl-note{background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:13px;font-size:12.5px;color:#333;line-height:1.6}
</style>
<div id="__UID__">
  <div class="gl-head">⏳ What one interview is worth, if the offer comes on turn k</div>
  <div class="gl-sub">Every reward before the end is <b>0</b>, so the whole discounted return
    <b>G = Σ γ<sup>t</sup> r<sub>t</sub></b> collapses to a single surviving term:
    <b>γ<sup>k−1</sup> · (+1)</b> for an offer on turn <b>k</b>. Slide γ and watch how much
    a slow “yes” is worth compared with a fast one.</div>
  <div class="gl-ctl">
    <span class="gl-g"></span>
    <input type="range" min="0" max="100" value="90">
  </div>
  <div class="gl-rows"></div>
  <div class="gl-note"></div>
</div>
<script>
(function(){
  const root=document.getElementById("__UID__"), rows=root.querySelector(".gl-rows");
  const sl=root.querySelector("input"), lab=root.querySelector(".gl-g"), note=root.querySelector(".gl-note");
  const KS=[1,2,3,5,8,12,20];
  KS.forEach(k=>{
    const r=document.createElement("div"); r.className="gl-row"; r.dataset.k=k;
    r.innerHTML='<span class="gl-k">offer on turn '+k+'</span>'
      +'<span class="gl-track"><span class="gl-fill"></span></span><span class="gl-v"></span>';
    rows.appendChild(r);
  });
  function draw(){
    const g=(+sl.value)/100; lab.textContent="γ = "+g.toFixed(2);
    rows.querySelectorAll(".gl-row").forEach(r=>{
      const k=+r.dataset.k, v=Math.pow(g,k-1);
      r.querySelector(".gl-fill").style.width=(v*100).toFixed(1)+"%";
      r.querySelector(".gl-v").textContent=v.toFixed(3);
    });
    let msg;
    if(g>=0.995) msg="γ = 1 — <b>an offer is an offer.</b> Turn 1 and turn 20 are worth exactly the "
      +"same, so nothing in the objective asks you to hurry. Stalling costs you nothing.";
    else if(g<=0.05) msg="γ ≈ 0 — <b>only this turn exists.</b> The only thing worth anything is an "
      +"offer <i>right now</i>; a plan that pays off in two turns is worth literally nothing, so no "
      +"amount of building up the interviewer can ever be justified.";
    else msg="With γ = "+g.toFixed(2)+", an offer on turn 5 is worth <b>"+Math.pow(g,4).toFixed(3)
      +"</b> and one on turn 12 only <b>"+Math.pow(g,11).toFixed(3)+"</b>. Nobody wrote “be quick” "
      +"into the reward — but γ &lt; 1 <b>makes speed valuable for free</b>. And symmetrically: a "
      +"rejection on turn 12 costs only −"+Math.pow(g,11).toFixed(3)+", so γ also quietly makes "
      +"<i>stalling a disaster</i> less terrible than walking into one immediately.";
    note.innerHTML=msg;
  }
  sl.addEventListener("input",draw); draw();
})();
</script>'''
    display(HTML(tmpl.replace("__UID__", uid)))


def g_to_v_to_q():
    """The one card that has to land: G → V → Q, and the two identities."""
    def box(tag, title, formula, body, color):
        return ('<div style="border:2px solid %s;border-radius:12px;padding:12px;margin-bottom:10px">'
                '<div style="display:flex;align-items:baseline;gap:9px;flex-wrap:wrap">'
                '<span style="font-size:11px;font-weight:800;color:%s;text-transform:uppercase;'
                'letter-spacing:.05em">%s</span>'
                '<span style="font-weight:800;font-size:13.5px;color:#222">%s</span></div>'
                '<div style="font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:%s;'
                'background:#fafbff;border-radius:7px;padding:7px 10px;margin:7px 0">%s</div>'
                '<div style="font-size:12.5px;color:#555;line-height:1.6">%s</div></div>'
                % (color, color, tag, title, color, formula, body))

    _card(
        box("one run", "Return G(τ) — how <i>this</i> interview went",
            "G(τ) = r₀ + γ·r₁ + γ²·r₂ + … = γ<sup>k−1</sup> · (±1)",
            "A single number for a single interview. Useless on its own: the same behaviour in the "
            "same room produces a different G every time, because the interviewer rolls dice.",
            "#9aa0b5")
        + box("one state", "Value V<sup>π</sup>(s) — what <i>this situation</i> is worth",
              "V<sup>π</sup>(s) = 𝔼<sub>π</sub>[ G | start from s ]",
              "Average G over <i>all</i> the interviews that pass through s, while behaving as π. "
              "Now it is a property of the situation, not of one unlucky Tuesday. Walk in and they "
              "are already 😄 delighted — that is worth a lot. Walk in to 😠 furrowed — that is worth "
              "much less, <b>before you have done anything at all</b>.",
              "#4a5bd0")
        + box("one state + one move", "Action-value Q<sup>π</sup>(s, a) — what <i>this move here</i> "
              "is worth",
              "Q<sup>π</sup>(s, a) = 𝔼<sub>π</sub>[ G | start from s, do a first, then follow π ]",
              "Same average, but you pin down the first move. This is the only one of the three that "
              "compares <b>actions</b>, which is the only thing you actually need in order to act.",
              "#764ba2")
        + '<div style="background:#f3f0ff;border-radius:10px;padding:12px 14px;font-size:13px;'
          'color:#2c2350;line-height:1.9">'
          '<b>And the three are the same object seen from three distances:</b><br>'
          '<code>V<sup>π</sup>(s) = Σ<sub>a</sub> π(a|s) · Q<sup>π</sup>(s, a)</code> '
          '<span style="color:#777">— a state is worth the average of its moves, weighted by how '
          'often you play them</span><br>'
          '<code>V*(s) = max<sub>a</sub> Q*(s, a)</code> '
          '<span style="color:#777">— if you play optimally, a state is worth its <i>best</i> move'
          '</span><br>'
          '<code>π*(s) = argmax<sub>a</sub> Q*(s, a)</code> '
          '<span style="color:#777">— and the best policy is “look up the row, take the biggest '
          'number”</span></div>')


def the_wall():
    """Same question — 'which action?' — asked of V* and of Q*. Only one is answerable."""
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:3px">'
        '\U0001F9F1 You have a perfect value function. Now pick a sentence.</div>'
        '<div style="font-size:12.5px;color:#666;line-height:1.6;margin-bottom:12px">'
        'Both boxes below answer exactly the same question — <i>which action should I take in state '
        's?</i> — and both are correct. Read what each one needs you to know.</div>'
        '<div style="display:flex;gap:14px;flex-wrap:wrap">'
        '<div style="flex:1;min-width:280px;border:2px solid #c0554e;border-radius:12px;padding:12px;'
        'background:#fff8f7">'
        '<div style="font-weight:800;font-size:13px;color:#b23b34;margin-bottom:6px">'
        'Acting from V* — the situation-scorer</div>'
        '<div style="font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#b23b34;'
        'background:#fff;border-radius:7px;padding:9px;line-height:1.8">'
        'π*(s) = argmax<sub>a</sub> <b style="background:#ffe3e0">Σ<sub>s′</sub> P(s′|s,a)</b> '
        '[ R(s,a,s′) + γ V*(s′) ]</div>'
        '<div style="font-size:12.5px;color:#555;line-height:1.65;margin-top:9px">'
        'V* scores <b>situations</b>. To turn that into a <b>move</b> you have to ask, for each of '
        'your four sentences, <i>where would this one leave them?</i> — and weigh the situations it '
        'could lead to by how likely they are.<br><br>'
        'That is the highlighted term: <b style="color:#b23b34">P(s′|s,a)</b>, the interviewer\'s '
        'brain. <b>You have never had it and you never will.</b> A perfect V*, handed to you for '
        'free, and you are still sitting in that chair unable to open your mouth.</div></div>'
        '<div style="flex:1;min-width:280px;border:2px solid #2e9e7a;border-radius:12px;padding:12px;'
        'background:#f6fdfa">'
        '<div style="font-weight:800;font-size:13px;color:#1d7a46;margin-bottom:6px">'
        'Acting from Q* — the move-scorer</div>'
        '<div style="font-family:ui-monospace,Menlo,monospace;font-size:12px;color:#1d7a46;'
        'background:#fff;border-radius:7px;padding:9px;line-height:1.8">'
        'π*(s) = argmax<sub>a</sub> Q*(s, a)</div>'
        '<div style="font-size:12.5px;color:#555;line-height:1.65;margin-top:9px">'
        'That is the entire formula. No sum, no P, no reward function — <b>read the row, take the '
        'biggest number.</b><br><br>'
        'Nothing was hidden: Q* contains exactly the same information, but the '
        '“where would this leave them” part has <b>already been folded into each entry</b>. Someone '
        'did the sum for you, once, per action. And “someone” turns out to be the thousands of '
        'interviews you are about to sit through.</div></div></div>'
        '<div style="background:#f3f0ff;border-radius:8px;padding:11px 13px;margin-top:14px;'
        'font-size:12.5px;color:#2c2350;line-height:1.65">\U0001F511 <b>That is the whole reason Q '
        'exists.</b> V is one number per situation, and choosing with it costs you a model. Q is one '
        'number per situation <i>and move</i> — four times the table, and choosing with it costs you '
        'nothing. In a world where nobody will show you P, that trade is not close.</div>')


def td_diagram():
    """One TD update, drawn."""
    _card(
        '<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:10px">'
        '\U0001F501 One update, from one turn of one interview</div>'
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;'
        'font-size:12.5px;margin-bottom:14px">'
        '<div style="border:2px solid #764ba2;border-radius:10px;padding:8px 12px;background:#faf7ff">'
        '<b>s</b><br><span style="color:#777;font-size:11.5px">what you saw</span></div>'
        '<div style="font-size:19px;color:#9aa0b5">→</div>'
        '<div style="border:2px solid #e0a500;border-radius:10px;padding:8px 12px;background:#fffdf5">'
        '<b>a</b><br><span style="color:#777;font-size:11.5px">what you said</span></div>'
        '<div style="font-size:19px;color:#9aa0b5">→</div>'
        '<div style="border:2px solid #2e9e7a;border-radius:10px;padding:8px 12px;background:#f6fdfa">'
        '<b>r</b><br><span style="color:#777;font-size:11.5px">0, +1 or −1</span></div>'
        '<div style="font-size:19px;color:#9aa0b5">→</div>'
        '<div style="border:2px solid #764ba2;border-radius:10px;padding:8px 12px;background:#faf7ff">'
        '<b>s′</b><br><span style="color:#777;font-size:11.5px">how they look now</span></div>'
        '</div>'
        '<div style="font-family:ui-monospace,Menlo,monospace;font-size:13px;background:#f3f0ff;'
        'border-radius:9px;padding:12px 14px;color:#2c2350;line-height:1.9">'
        'target &nbsp;= &nbsp;r + γ · max<sub>a′</sub> Q(s′, a′)<br>'
        '<span style="color:#8a7ab8">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
        '…but if r was the LAST reward — they just hired or rejected you — there is no s′ to be '
        'worth anything,<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;so that whole '
        'second half is dropped and the target is simply &nbsp;<b>target = r</b>.</span><br>'
        'error &nbsp;&nbsp;= &nbsp;target − Q(s, a) &nbsp;&nbsp;'
        '<span style="color:#8a7ab8">← the “TD error”: how wrong we were</span><br>'
        'Q(s, a) &nbsp;← &nbsp;Q(s, a) + α · error</div>'
        '<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:13px;'
        'font-size:12.5px;color:#333;line-height:1.6">'
        'Read it as a sentence: <b>“I thought this move was worth Q(s,a). What actually happened was '
        'r, and I now find myself somewhere whose best move I currently rate at max Q(s′,·). Let me '
        'move my old opinion a fraction α of the way towards that.”</b><br>'
        'Nothing in there is P(s′|s,a). The transition probabilities never appear — they are '
        '<i>sampled</i> instead, one interview at a time. That single substitution is what turns '
        'a calculation that needs the interviewer\'s brain into a method that needs only the '
        'interviewer.</div>')


# ===========================================================================
#  §5  Looking at what was learned
# ===========================================================================
def _heat(v, lo=-1.0, hi=1.0):
    t = (float(v) - lo) / (hi - lo)
    t = min(1.0, max(0.0, t))
    if t < 0.5:                                   # red → grey
        k = t / 0.5
        rgb = (int(217 - 40 * k), int(150 + 70 * k), int(145 + 75 * k))
    else:                                         # grey → green
        k = (t - 0.5) / 0.5
        rgb = (int(177 - 90 * k), int(220 + 5 * k), int(220 - 40 * k))
    return "rgb(%d,%d,%d)" % rgb


def v_bars(V, features, title="V(s) — what each situation is worth"):
    """A value function, one bar per state. V may be a dict or anything indexable."""
    states = [s for s in enumerate_states(features) if s in V] or list(V.keys())
    vals = [float(V[s]) for s in states]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    span = max(hi - lo, 1e-9)
    rows = "".join(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:5px;font-size:12px">'
        '<div style="width:78px;font-size:17px;text-align:right" title="%s">%s</div>'
        '<div style="flex:1;height:20px;background:#f2f3f8;border-radius:5px;position:relative">'
        '<div style="position:absolute;left:%.1f%%;width:%.1f%%;top:0;bottom:0;background:%s;'
        'border-radius:4px"></div>'
        '<div style="position:absolute;left:%.1f%%;top:0;bottom:0;width:1px;background:#c2c7da">'
        '</div></div>'
        '<div style="width:62px;text-align:right;font-weight:800;color:%s">%+.3f</div></div>'
        % (state_tooltip(s, features), state_label(s, features),
           100 * (min(v, 0) - lo) / span, 100 * abs(v) / span,
           "#2e9e7a" if v >= 0 else "#c0554e",
           100 * (0 - lo) / span, "#1d7a46" if v >= 0 else "#b23b34", v)
        for s, v in zip(states, vals))
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:10px">%s</div>'
          '%s<div style="font-size:11.5px;color:#888;margin-top:8px">The thin grey line is zero. '
          'A state above it is one you are, on balance, glad to be in.</div>' % (title, rows),
          maxw=640)


def q_table_view(Q, features, title="Q(s, a) — what each move is worth, in each situation",
                 show_v=True, max_rows=48):
    """The learned table, as a table. Best action in each row gets a ring."""
    states = enumerate_states(features)
    head = ("".join('<th style="padding:6px 10px;font-size:12px;color:%s;font-weight:800">%s %s</th>'
                    % (ACTION_COLOR[a], ACTION_EMOJI[a], ACTIONS[a]) for a in range(N_ACTIONS)))
    rows = []
    for s in states[:max_rows]:
        q = np.asarray(Q[s], dtype=float)
        best = int(np.argmax(q))
        cells = "".join(
            '<td style="padding:4px 6px"><div style="background:%s;border-radius:7px;padding:6px 4px;'
            'text-align:center;font-size:12.5px;font-weight:%s;color:#1c1e2a;%s">%+.3f</div></td>'
            % (_heat(q[a]), "800" if a == best else "500",
               "box-shadow:0 0 0 2.5px #2b2d6b inset;" if a == best else "", q[a])
            for a in range(N_ACTIONS))
        vcell = ('<td style="padding:4px 10px;font-size:12.5px;font-weight:800;color:#2b2d6b;'
                 'text-align:center">%+.3f</td>' % q.max()) if show_v else ""
        pcell = ('<td style="padding:4px 8px;font-size:15px;text-align:center">%s</td>'
                 % ACTION_EMOJI[best])
        rows.append('<tr><td style="padding:4px 10px;font-size:16px;white-space:nowrap" title="%s">'
                    '%s</td>%s%s%s</tr>'
                    % (state_tooltip(s, features), state_label(s, features), cells, vcell, pcell))
    extra = ("" if len(states) <= max_rows else
             '<div style="font-size:11.5px;color:#888;margin-top:6px">… and %d more rows '
             '(your state space is large — that is the whole point).</div>'
             % (len(states) - max_rows))
    _card('<div style="font-weight:800;font-size:14.5px;color:#2b2d6b;margin-bottom:9px">%s</div>'
          '<div style="overflow-x:auto"><table style="border-collapse:collapse">'
          '<tr><th style="padding:6px 10px;font-size:11.5px;color:#888;text-align:left">state</th>%s'
          '%s<th style="padding:6px 8px;font-size:11.5px;color:#888">π</th></tr>%s</table></div>%s'
          % (title, head,
             '<th style="padding:6px 10px;font-size:12px;color:#2b2d6b;font-weight:800">'
             'V(s) = max<sub>a</sub> Q</th>' if show_v else "",
             "".join(rows), extra), maxw=820)


def policy_card(Q, features, title="\U0001F4CB Your policy — the sheet you walk in with"):
    """The deliverable: one line per state."""
    states = enumerate_states(features)
    rows = "".join(
        '<div style="display:flex;align-items:center;gap:12px;border:1px solid #e2e5ef;'
        'border-radius:10px;padding:8px 12px;margin-bottom:6px;background:#fbfcff">'
        '<div style="font-size:20px;min-width:60px">%s</div>'
        '<div style="font-size:12px;color:#777;flex:1;min-width:150px">%s</div>'
        '<div style="font-size:18px;color:#9aa0b5">→</div>'
        '<div style="font-weight:800;font-size:13px;color:%s;min-width:130px">%s %s</div>'
        '<div style="font-size:11.5px;color:#888">worth %+.3f</div></div>'
        % (state_label(s, features), state_tooltip(s, features),
           ACTION_COLOR[int(np.argmax(Q[s]))], ACTION_EMOJI[int(np.argmax(Q[s]))],
           ACTIONS[int(np.argmax(Q[s]))], float(np.max(Q[s])))
        for s in states[:24])
    _card('<div style="font-weight:800;font-size:15px;color:#2b2d6b;margin-bottom:9px">%s</div>%s'
          '<div style="background:#f6f7fb;border-radius:8px;padding:10px 12px;margin-top:9px;'
          'font-size:12.5px;color:#333;line-height:1.6">Every line is <code>argmax<sub>a</sub> '
          'Q(s,a)</code>, and the number on the right is <code>V(s) = max<sub>a</sub> Q(s,a)</code> '
          '— how much this whole situation is worth once you commit to acting well from here on.'
          '</div>' % (title, rows), maxw=800)


def training_curve(hires, returns, window=200, turns=None):
    """How the agent got better, interview after humiliating interview."""
    import matplotlib.pyplot as plt
    hires, returns = np.asarray(hires, float), np.asarray(returns, float)
    n = len(hires)
    w = max(5, min(window, n // 5))
    ker = np.ones(w) / w

    def smooth(x):
        return np.convolve(x, ker, mode="valid")

    xs = np.arange(w - 1, n)
    ncols = 3 if turns is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 3.5))
    axes[0].plot(xs, smooth(hires) * 100, color="#2e9e7a", lw=2)
    axes[0].axhline(50, color="#9aa0b5", ls=":", lw=1.2)
    axes[0].set_title("Offers, %% of interviews\n(rolling %d)" % w, fontsize=11)
    axes[0].set_ylabel("% hired")
    axes[1].plot(xs, smooth(returns), color="#764ba2", lw=2)
    axes[1].axhline(0, color="#9aa0b5", ls=":", lw=1.2)
    axes[1].set_title("Discounted return G\n(rolling %d)" % w, fontsize=11)
    axes[1].set_ylabel("G")
    if turns is not None:
        axes[2].plot(xs, smooth(np.asarray(turns, float)), color="#e0a500", lw=2)
        axes[2].set_title("Turns per interview\n(rolling %d)" % w, fontsize=11)
        axes[2].set_ylabel("turns")
    for ax in axes:
        ax.set_xlabel("interview")
        ax.grid(alpha=.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.show()


def scoreboard(results):
    """results: dict {label: (offer_rate, mean_turns, mean_G)} — one entry per state choice."""
    import matplotlib.pyplot as plt
    labels = list(results.keys())
    offer = [results[k][0] * 100 for k in labels]
    turns = [results[k][1] for k in labels]
    gs = [results[k][2] for k in labels]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 0.62 * len(labels) + 2.4), sharey=True)
    best = int(np.argmax(gs))
    cols = ["#764ba2" if i == best else "#c2c7da" for i in range(len(labels))]
    for ax, vals, ttl, fmt in [
            (axes[0], gs, "Discounted return G  ← what we optimise", "%.3f"),
            (axes[1], offer, "Offers (% of interviews)", "%.0f%%"),
            (axes[2], turns, "Turns until it ends  (lower = faster)", "%.1f")]:
        lo, hi = min(vals + [0.0]), max(vals + [0.0])
        span = max(hi - lo, 1e-9)
        # a negative return is a real outcome here (you mostly got rejected) — show it
        bar_cols = [c if v >= 0 else "#c0554e" for c, v in zip(cols, vals)]
        ax.barh(y, vals, color=bar_cols, height=.62)
        for yy, v in zip(y, vals):
            ax.text(v + np.sign(v or 1) * span * .025, yy, fmt % v, va="center", fontsize=9.5,
                    color="#444", ha="left" if v >= 0 else "right")
        ax.set_title(ttl, fontsize=10.5)
        ax.set_xlim(lo - span * .16, hi + span * .18)
        if lo < 0:
            ax.axvline(0, color="#555", lw=.9)
        ax.grid(axis="x", alpha=.25)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10)
    fig.suptitle("Same algorithm, same budget, same interviewer — only the choice of STATE differs",
                 fontsize=12, color="#2b2d6b", fontweight="bold")
    plt.tight_layout()
    plt.show()


# ===========================================================================
#  §6  The race
# ===========================================================================
_RACE_CSS = r'''
#__UID__ .rc-wrap{display:flex;flex-wrap:wrap}
#__UID__ .rc-side{flex:1;min-width:340px;border-right:1px solid #e6e8ee}
#__UID__ .rc-side:last-child{border-right:none}
#__UID__ .rc-name{font-weight:800;font-size:12.5px;padding:9px 14px;color:#fff;line-height:1.35}
#__UID__ .rc-name small{font-weight:600;opacity:.85;font-size:11px}
#__UID__ .rc-room{position:relative;height:172px;background:linear-gradient(#eef1fa,#e2e7f5 60%,#d7ddef 60%);overflow:hidden}
#__UID__ .rc-face{font-size:46px;line-height:1;position:absolute;left:50%;top:22px;transform:translateX(-50%);transition:.15s}
#__UID__ .rc-body{position:absolute;left:50%;top:70px;transform:translateX(-50%);width:104px;height:54px;border-radius:34px 34px 8px 8px}
#__UID__ .rc-desk{position:absolute;left:0;right:0;bottom:0;height:60px;background:linear-gradient(#a9763f,#8d5f30);border-top:5px solid #c08c4e}
#__UID__ .rc-hands{position:absolute;bottom:34px;left:0;right:0;text-align:center;font-size:21px;letter-spacing:20px;text-indent:20px}
#__UID__ .rc-pen{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);font-size:17px}
#__UID__ .rc-hud{position:absolute;top:6px;left:8px;font-size:10.5px;font-weight:800;color:#4a3a86;background:#ffffffcc;border-radius:6px;padding:3px 7px}
#__UID__ .rc-eps{position:absolute;top:6px;right:8px;font-size:10.5px;font-weight:700;color:#7a5fbf;background:#ffffffcc;border-radius:6px;padding:3px 7px}
#__UID__ .rc-dlg{padding:8px 14px;font-size:11.8px;line-height:1.5;height:62px;overflow:hidden;color:#444;border-top:1px solid #eef0f6}
#__UID__ .rc-btns{display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:0 12px 10px}
#__UID__ .rc-btn{cursor:pointer;border:2px solid;border-radius:9px;padding:7px 8px;font-size:12px;font-weight:800;background:#fff;text-align:left}
#__UID__ .rc-btn[disabled]{opacity:.32;cursor:default}
#__UID__ .rc-stats{padding:0 14px 12px;font-size:11.5px;color:#555}
#__UID__ .rc-track{height:15px;border-radius:5px;background:#f2f3f8;position:relative;overflow:hidden;margin:5px 0 4px}
#__UID__ .rc-fill{position:absolute;left:0;top:0;bottom:0;width:0;border-radius:5px;transition:width .3s}
#__UID__ .rc-goal{position:absolute;top:-2px;bottom:-2px;width:2px;background:#2b2d6b}
#__UID__ .rc-spark{display:flex;gap:2px;align-items:flex-end;height:26px;margin-top:5px}
#__UID__ .rc-spark div{flex:1;border-radius:1px;min-height:2px}
#__UID__ .rc-bar{padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;border-top:1px solid #e6e8ee}
#__UID__ .rc-go{cursor:pointer;border:none;border-radius:9px;padding:10px 20px;font-size:13.5px;font-weight:800;color:#fff;background:linear-gradient(135deg,#667eea,#764ba2)}
#__UID__ .rc-status{font-size:12.5px;color:#555}
#__UID__ .rc-verdict{padding:0 16px 16px;font-size:13px;line-height:1.65}
'''


def race(agent_delay=200, gamma=0.9, target=0.35, window=5,
         alpha=0.35, eps_episodes=60, eps_min=0.08):
    """Part 4 — the final round, against a DIFFERENT interviewer.

    Left: you, learning by playing. Right: a Q-learning agent starting from a
    blank table and learning online, in the browser, right now — and it does not
    stop while you think. First to average `target` discounted return over its
    last `window` interviews wins.

    The dynamics here are NOT the ones the notebook trained on: the sheet learned
    upstairs scores *worse than random* in this room, so copying it is a trap.
    """
    data = {
        "trans": _MOOD_TRANS_2, "hand": _HAND_PROBS, "pen": _PEN_PROBS,
        "start": _START_PROBS, "maxturns": MAX_TURNS,
        "me": MOOD_EMOJI, "mc": MOOD_COLOR, "fl": FACE_LABEL,
        "he": HANDS_EMOJI, "pe": PEN_EMOJI,
        "acts": ACTIONS, "ae": ACTION_EMOJI, "ac": ACTION_COLOR, "lines": ACTION_LINES,
        "react": _REACTIONS,
        "term": {str(HIRED): TERM_LINE[HIRED], str(REJECTED): TERM_LINE[REJECTED]},
        "te": {str(HIRED): TERM_EMOJI[HIRED], str(REJECTED): TERM_EMOJI[REJECTED]},
        "hired": HIRED, "rejected": REJECTED,
        "delay": int(agent_delay), "gamma": float(gamma), "target": float(target),
        "K": int(window), "alpha": float(alpha), "epsN": int(eps_episodes),
        "epsMin": float(eps_min),
    }
    uid = "race_" + str(abs(hash(("race2", agent_delay, target))) % 10**8)
    tmpl = (r'<style>' + _SCENE_CSS.replace("__W__", "1020") + _RACE_CSS + r'''</style>
<div id="__UID__">
  <div class="g-top">
    <div class="g-title">🏁 The final round — a different interviewer, and neither of you has met them</div>
    <div class="g-meta">first to average G ≥ __TARGET__ over __K__ interviews</div>
  </div>
  <div class="rc-wrap">
    <div class="rc-side" data-who="0">
      <div class="rc-name" style="background:#4a5bd0">🧑 You
        <small>— you get to think. That is your whole advantage.</small></div>
      <div class="rc-room"><div class="rc-hud"></div><div class="rc-face">😐</div>
        <div class="rc-body"></div><div class="rc-desk"><div class="rc-hands"></div>
        <div class="rc-pen"></div></div></div>
      <div class="rc-dlg"></div>
      <div class="rc-btns"></div>
      <div class="rc-stats"></div>
    </div>
    <div class="rc-side" data-who="1">
      <div class="rc-name" style="background:#2e9e7a">🤖 Q-learning, from a blank table
        <small>— it does not think. It just does not stop.</small></div>
      <div class="rc-room"><div class="rc-hud"></div><div class="rc-eps"></div>
        <div class="rc-face">😐</div>
        <div class="rc-body"></div><div class="rc-desk"><div class="rc-hands"></div>
        <div class="rc-pen"></div></div></div>
      <div class="rc-dlg"></div>
      <div class="rc-btns"></div>
      <div class="rc-stats"></div>
    </div>
  </div>
  <div class="rc-bar"><button class="rc-go">🏁 Start the race</button>
    <div class="rc-status"></div></div>
  <div class="rc-verdict"></div>
</div>
<script>
(function(){
  const D=__DATA__, root=document.getElementById("__UID__");
''' + _SCENE_JS + r'''
  const sides=[...root.querySelectorAll(".rc-side")];
  let running=false, finished=false, timers=[];
  const P=[null,null];                       // P[0] = you, P[1] = the agent
  let Q=null, ep=0;                          // the agent's table, built live in this page

  function later(fn,ms){ const t=setTimeout(fn,ms); timers.push(t); return t; }
  function clearAll(){ timers.forEach(clearTimeout); timers=[]; }
  function mean(a){ return a.reduce((x,y)=>x+y,0)/a.length; }
  function score(p){ return p.hist.length>=D.K ? mean(p.hist.slice(-D.K)) : null; }

  function blank(){ return {mood:0, turn:0, done:true, hist:[], obs:null, n:0}; }

  function paint(i){
    const p=P[i], side=sides[i], o=p.obs; if(!o) return;
    const term=(o.face===D.hired||o.face===D.rejected);
    side.querySelector(".rc-face").textContent = term? D.te[""+o.face] : D.me[o.face];
    side.querySelector(".rc-body").style.background = term? "#8a90a8" : D.mc[o.face];
    side.querySelector(".rc-hands").textContent = D.he[o.hands];
    side.querySelector(".rc-pen").textContent   = D.pe[o.pen];
    side.querySelector(".rc-hud").textContent   = "interview "+p.n+" · turn "+p.turn;
    side.querySelectorAll(".rc-btn").forEach(b=>b.disabled = p.done||!running||finished||i===1);
    const s=score(p), pct=v=>Math.max(0,Math.min(100,(v+1)/2*100));
    const spark=p.hist.slice(-18).map(g=>
      '<div style="height:'+(6+Math.abs(g)*20)+'px;background:'+(g>0?"#2e9e7a":"#c0554e")+'"></div>'
    ).join("");
    side.querySelector(".rc-stats").innerHTML=
      '<b>'+p.hist.length+'</b> interviews finished · last '+D.K+' average G: '
      +'<b style="color:'+(s===null?"#888":(s>=D.target?"#1d7a46":"#4a3a86"))+'">'
      +(s===null? ("need "+(D.K-p.hist.length)+" more") : s.toFixed(3))+'</b>'
      +'<div class="rc-track"><div class="rc-fill" style="width:'+(s===null?0:pct(s))
      +'%;background:'+(s!==null&&s>=D.target?"#2e9e7a":"#8fa0e8")+'"></div>'
      +'<div class="rc-goal" style="left:'+pct(D.target)+'%"></div></div>'
      +'<div class="rc-spark">'+spark+'</div>';
  }

  function begin(i){
    const p=P[i];
    p.mood=drawFrom(D.start.map((q,k)=>[k,q])); p.turn=0; p.done=false; p.n++;
    p.obs=observe(p.mood);
    sides[i].querySelector(".rc-dlg").innerHTML="<i>A new candidate sits down. They look up.</i>";
    paint(i);
  }

  // one turn; returns the reward (0 unless the interview ended)
  function step(i,a){
    const p=P[i]; if(p.done) return null;
    const before=p.mood;
    let nxt=drawFrom(D.trans[before][a]); p.turn++;
    if(nxt!==D.hired && nxt!==D.rejected && p.turn>=D.maxturns) nxt=D.rejected;
    const ended=(nxt===D.hired||nxt===D.rejected);
    const r = ended ? (nxt===D.hired?1:-1) : 0;
    let html='<div style="color:#2b2d6b">'+D.ae[a]+' '+pick(D.lines[a])+'</div>';
    if(ended){
      p.done=true; p.mood=nxt; p.obs=observe(nxt);
      const g=Math.pow(D.gamma,p.turn-1)*r; p.hist.push(g);
      html+='<div style="font-weight:800;color:'+(r>0?"#1d7a46":"#b23b34")+'">'
           +D.te[""+nxt]+' G = '+g.toFixed(3)+'</div>';
    } else { p.mood=nxt; p.obs=observe(nxt);
      html+='<div style="font-style:italic;color:#666">'+pick(D.react[nxt])+'</div>'; }
    sides[i].querySelector(".rc-dlg").innerHTML=html;
    paint(i);
    return {before:before, after:p.mood, r:r, ended:ended};
  }

  function check(){
    if(finished) return;
    const a=score(P[0]), b=score(P[1]);
    const youWin = a!==null && a>=D.target, botWin = b!==null && b>=D.target;
    if(!youWin && !botWin) return;
    finished=true; running=false; clearAll();
    paint(0); paint(1);
    let msg;
    if(youWin && botWin) msg='<b>🤝 Dead heat.</b> You both crossed on the same interview. '
      +'Statistically suspicious, and frankly a good result for you.';
    else if(youWin) msg='<b style="color:#1d7a46">🎉 You got there first.</b> You worked out a new '
      +'interviewer in '+P[0].hist.length+' interviews; the agent needed more than '
      +P[1].hist.length+' and was still exploring. <b>This is the thing humans are actually good '
      +'at</b> — you brought priors. You assumed a person who likes technical talk probably keeps '
      +'liking it. The agent assumes nothing, and pays for it in interviews.';
    else msg='<b style="color:#b23b34">🤖 The table got there first.</b> It played '
      +P[1].hist.length+' interviews to your '+P[0].hist.length+' — and that is the entire story. '
      +'It is not smarter than you; it is just never tired, never embarrassed, and never stops. '
      +'Notice what it did NOT need: no explanation of who this person is, no transfer from the '
      +'last one, no idea what a job is.';
    root.querySelector(".rc-verdict").innerHTML=msg
      +'<div style="background:#f6f7fb;border-radius:8px;padding:11px 13px;margin-top:11px;'
      +'font-size:12.5px;color:#333;line-height:1.6">🧠 <b>And the sheet from Part 3 was a trap.</b> '
      +'This interviewer is a different person: flattery reads as transparent, and technical talk — '
      +'which was suicide upstairs — is what earns their patience. Four of the five lines you '
      +'learned are wrong here. Playing the old sheet in this room scores <b>worse than acting at '
      +'random</b>. That is the honest limit of everything in this notebook: <b>a policy is only '
      +'ever a policy for the environment it was learned in.</b></div>';
    root.querySelector(".rc-go").textContent="🏁 Race again";
    root.querySelector(".rc-status").textContent="";
  }

  // ---- the agent: epsilon-greedy Q-learning, one turn every D.delay ms ----
  function agentTurn(){
    if(!running||finished) return;
    const p=P[1];
    if(p.done){ ep++; begin(1); sides[1].querySelector(".rc-eps").textContent=
        "ε = "+Math.max(D.epsMin,1-ep/D.epsN).toFixed(2); later(agentTurn, 420); return; }
    const eps=Math.max(D.epsMin, 1-ep/D.epsN);
    const s=p.mood;
    let a;
    if(Math.random()<eps) a=Math.floor(Math.random()*D.acts.length);
    else { a=0; for(let k=1;k<Q[s].length;k++) if(Q[s][k]>Q[s][a]) a=k; }
    const out=step(1,a);
    // Q(s,a) <- Q(s,a) + alpha * [ r + gamma*max_a' Q(s',a')  -  Q(s,a) ]
    let best=0; if(!out.ended){ best=Q[out.after][0];
      for(let k=1;k<Q[out.after].length;k++) if(Q[out.after][k]>best) best=Q[out.after][k]; }
    const target = out.ended ? out.r : out.r + D.gamma*best;
    Q[s][a] += D.alpha*(target - Q[s][a]);
    if(out.ended) check();
    if(!finished) later(agentTurn, D.delay);
  }

  // ---- you ----
  function humanMove(a){
    if(!running||finished||P[0].done) return;
    const out=step(0,a);
    if(out.ended){
      check();
      if(!finished) later(()=>{ if(running&&!finished) begin(0); }, 1400);
    }
  }

  sides.forEach((side,i)=>{
    const box=side.querySelector(".rc-btns");
    D.acts.forEach((name,k)=>{
      const b=document.createElement("button"); b.className="rc-btn";
      b.style.borderColor=D.ac[k]; b.style.color=D.ac[k];
      b.innerHTML=D.ae[k]+" "+name; b.disabled=true;
      if(i===0) b.addEventListener("click",()=>humanMove(k));
      box.appendChild(b);
    });
  });

  root.querySelector(".rc-go").addEventListener("click",()=>{
    clearAll(); finished=false; running=true; ep=0;
    Q=[]; for(let m=0;m<D.trans.length;m++) Q.push(D.acts.map(()=>0));
    P[0]=blank(); P[1]=blank();
    root.querySelector(".rc-verdict").innerHTML="";
    root.querySelector(".rc-status").innerHTML=
      "The agent is already playing. It will not wait for you.";
    begin(0); begin(1);
    later(agentTurn, D.delay);
  });

  P[0]=blank(); P[1]=blank(); P[0].obs=observe(2); P[1].obs=observe(2);
  paint(0); paint(1);
  root.querySelector(".rc-status").textContent="Press start. Then act fast, and act well.";
})();
</script>''')
    html = (tmpl.replace("__UID__", uid)
            .replace("__TARGET__", "%.2f" % target).replace("__K__", str(window))
            .replace("__DATA__", _json.dumps(data)))
    display(HTML(html))


# ===========================================================================
#  §7  Quiz banks   (options are shuffled at render time, so order means nothing)
# ===========================================================================
_MC_QUIZZES = {
    "what_is_state": (
        "Which of these is the best <i>state</i> for this problem?",
        "You have to hand your future self one description of “the situation I am in”, and it will "
        "be the only thing your policy is ever allowed to look at. Which one would you write down?",
        ["Everything measurable in the room: their face, hands, pen, the temperature, the time, "
         "your own heart rate",
         "Whatever summary of the room is small enough to learn about and still tells you enough "
         "to pick a sentence",
         "The number of turns that have gone by, since that is objective and easy to measure",
         "Nothing — an interview is pure chance, so any state is as good as any other"],
        1,
        "A state is a <b>modelling decision</b>, and it trades two things off. Too little and no "
        "policy on earth can act well, because the situations you cannot distinguish need different "
        "answers. Too much and every situation becomes rare, so you never see any of them often "
        "enough to learn what works. Everything measurable is not free: each extra component "
        "multiplies your table and divides your experience."),
    "modelfree": (
        "What exactly is it that you do <i>not</i> have here?",
        "This notebook keeps insisting the setting is <b>model-free</b>. What is the missing “model”?",
        ["The neural network — model-free means learning without deep learning",
         "P(s′ | s, a): the probabilities telling you where each thing you could say would leave "
         "the interviewer",
         "The reward function — model-free means you do not know what you are being paid for",
         "A trained policy — model-free means starting from scratch rather than from a pre-trained "
         "one"],
        1,
        "The “model” in model-free is the <b>environment's dynamics</b>, P(s′|s,a) — and, in general, "
        "the reward function too. You <i>do</i> know when you have been hired; what you cannot do is "
        "predict, before speaking, where a sentence lands them. A model-free method never writes "
        "those probabilities down: it samples them by living through them."),
}

_TF_QUIZZES = {
    "values": ("Return, V and Q", [
        ("V(s) is an average over many possible interviews, while G is what happened in one of them.",
         True),
        ("Q(s,a) fixes the first action and then assumes we carry on behaving as usual.", True),
        ("V*(s) is the largest entry of the row Q*(s, ·).", True),
        ("Q needs one number per state; V needs one number per state and action.", False),
        ("Knowing Q* is enough to act optimally without knowing anything about the environment's "
         "dynamics.", True),
        ("V(s) can be read off a single interview that passed through s.", False),
        ("A state with a high V is one from which things tend to go well, whatever we did to get "
         "there.", True),
        ("Because the reward is zero on ordinary turns, V is zero in every non-terminal state.",
         False),
    ]),
    "qlearning": ("Q-learning and the TD update", [
        ("The update uses one transition (s, a, r, s′) and does not wait for the interview to end.",
         True),
        ("The TD error is the gap between what we expected the move to be worth and what the world "
         "plus our current estimate now suggest.", True),
        ("The learning rate α decides how much of one surprise we absorb into the estimate.", True),
        ("When s′ is terminal the target is just r, because there is no future left to discount.",
         True),
        ("Q-learning estimates the transition probabilities first and then plans with them.", False),
        ("Acting greedily from the start makes learning faster, since no turns are wasted.", False),
    ]),
}

_NUMBER_QUIZZES = {
    "returns": ("🔢 Three returns you can do in your head — or with a calculator, nobody is watching", [
        ("γ = 0.9. The interviewer says “you are taken” on turn <b>1</b> — that is a reward of +1 at "
         "t = 0, and nothing else, ever. What is G?", 1.0, 0.01,
         "The first reward is <i>not</i> discounted: γ⁰ = 1. So G = 1."),
        ("γ = 0.9, same +1 but the offer only arrives on turn <b>4</b> — rewards are "
         "[0, 0, 0, +1]. What is G?", 0.729, 0.01,
         "Only the last term survives: G = γ³ · 1 = 0.9³ = 0.729."),
        ("γ = 0.9 again, but this time it is a rejection on turn <b>3</b>: rewards [0, 0, −1]. "
         "What is G?", -0.81, 0.01,
         "G = γ² · (−1) = −0.81. Discounting shrinks bad news too — which is worth thinking about."),
    ]),
    "vq": ("🔢 Moving between Q and V", [
        ("The <b>optimal</b> action-values in one state are "
         "Q*(s, ·) = [0.20, −0.30, 0.55, 0.51]. What is V*(s)?", 0.55, 0.005,
         "V*(s) = max<sub>a</sub> Q*(s,a) — if you act optimally from here, the situation is worth "
         "whatever its best move is worth."),
        ("Same row. Which action does π*(s) pick? Give its index (0, 1, 2 or 3).", 2.0, 0.001,
         "π*(s) = argmax<sub>a</sub> Q*(s,a) — the <i>position</i> of the biggest number, not the "
         "number itself."),
        ("Now a policy π that is <b>not</b> optimal: in this state it plays action 2 with "
         "probability 0.5 and action 3 with probability 0.5, and never plays 0 or 1. Its "
         "action-values there are Q<sup>π</sup>(s, ·) = [0.20, −0.30, 0.60, 0.40]. "
         "What is V<sup>π</sup>(s)?", 0.50, 0.005,
         "V<sup>π</sup>(s) = Σ<sub>a</sub> π(a|s)·Q<sup>π</sup>(s,a). The two actions it never "
         "plays contribute nothing, so it is just 0.5·0.60 + 0.5·0.40 = 0.50. Note this is "
         "<i>below</i> the best available move — averaging, not maximising, is what makes it "
         "V<sup>π</sup> rather than V*."),
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
#  §8  Final boss — timed true/false flash quiz with lives
# ===========================================================================
# Balanced pool (25 true / 25 false), phrased so neither answer is given away by
# the wording — no "always/never" tells, no absurd falses.
_FLASH_POOL = [
    # --- state -------------------------------------------------------------
    ("The state is what we decide to look at before acting, not everything that exists.", True),
    ("Each environment comes with one correct definition of its state.", False),
    ("Two situations we cannot tell apart must receive the same action from our policy.", True),
    ("Adding a component to the state multiplies the number of situations to learn about.", True),
    ("Adding an uninformative component to the state leaves learning speed unchanged.", False),
    ("A state that is too coarse can make good behaviour unreachable for any policy.", True),
    ("The number of turns elapsed is part of the state in our interview.", False),
    # --- action / transition ----------------------------------------------
    ("An action is one of the concrete moves the agent can make.", True),
    ("A transition function maps a state and an action to what comes next.", True),
    ("Under a stochastic transition function, the same move from the same state can land elsewhere.",
     True),
    ("A deterministic transition function returns a distribution over next states.", False),
    ("The transition function describes the environment rather than the agent's preferences.", True),
    ("Stochastic transitions force the policy to be stochastic as well.", False),
    # --- reward / return / gamma ------------------------------------------
    ("In our interview the reward is zero on every turn except the last one.", True),
    ("A reward of zero on a turn means that turn had no effect on the outcome.", False),
    ("The discount factor expresses how much a future reward is worth relative to one now.", True),
    ("γ is measured from the environment rather than chosen by the modeller.", False),
    ("With γ = 1 an offer on turn 20 is worth as much as an offer on turn 1.", True),
    ("With γ = 0 the agent can still justify building the interviewer up over several turns.", False),
    ("The first reward of an episode is discounted by one factor of γ.", False),
    ("Changing γ can change which policy is optimal.", True),
    ("A discount factor below 1 makes a rejection later less costly than the same rejection now.",
     True),
    # --- V and Q -----------------------------------------------------------
    ("V(s) is the expected discounted return from state s.", True),
    ("V(s) is the return of the single most likely interview starting at s.", False),
    ("Q(s,a) pins down the first action and averages over everything after it.", True),
    ("V*(s) equals the largest entry in the row Q*(s, ·).", True),
    ("The greedy policy takes the argmax of a Q row.", True),
    ("A Q-table has one number per state; a V-table has one per state and action.", False),
    ("Knowing V* alone is enough to choose actions in a model-free setting.", False),
    ("Knowing Q* alone is enough to choose actions in a model-free setting.", True),
    # --- model-free / MDP --------------------------------------------------
    ("Knowing P(s′|s,a) for every pair turns the problem into one that can be solved by planning.",
     True),
    ("“Model-free” means the agent never writes down the environment's transition probabilities.",
     True),
    ("“Model-free” means the agent does not use a neural network.", False),
    ("An MDP requires that the next state depends only on the current state and action.", True),
    ("If our chosen state hides information the dynamics depend on, the process stays Markov in "
     "that state.", False),
    # --- TD / Q-learning ---------------------------------------------------
    ("A TD update can be applied before the episode has finished.", True),
    ("Monte-Carlo estimation of V requires waiting for the episode to end.", True),
    ("The TD error is the difference between the target and the current estimate.", True),
    ("The Q-learning target for a terminal transition is just the reward.", True),
    ("The Q-learning target contains a max over the actions available in the next state.", True),
    ("The max in the Q-learning target picks which action the agent takes next.", False),
    ("Q-learning converges to the value of whatever behaviour collected the data.", False),
    ("Q-learning can learn the optimal table from data collected while acting randomly.", True),
    ("A learning rate of 1 replaces the old estimate with the new target entirely.", True),
    ("A larger learning rate reduces how much single unlucky outcomes move the estimate.", False),
    ("ε-greedy exists so that actions the table currently dislikes still get tried.", True),
    ("Once ε reaches zero the table can still discover that a neglected action was better.", False),
    ("Rows of the Q-table that are rarely visited are estimated as reliably as frequent ones.",
     False),
    ("Q-learning needs the transition probabilities in order to compute its target.", False),
    # --- odds and ends -----------------------------------------------------
    ("The reward function here gives a small penalty each turn to encourage speed.", False),
    ("A policy is part of the environment's specification rather than the agent's.", False),
    ("The discounted return is the plain sum of an episode's rewards, unweighted.", False),
    ("Q(s,a) assumes the agent keeps choosing at random after the first move.", False),
    ("Value iteration can be run when the transition probabilities are unknown.", False),
    ("Terminal states need their own row of action values, since a choice is made there too.", False),
    ("Exploration is free, so a well-designed agent should keep exploring at the same rate for ever.",
     False),
    ("An episode is a single state-action pair.", False),
    ("The greedy policy read off a Q-table is stochastic.", False),
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
  const livesEl=$(".fq-lives"), progEl=$(".fq-prog"), timeEl=$(".fq-time");
  let order=[], ptr=0, correct=0, lives=D.lives0, timer=null, deadline=0, locked=false;
  function shuffle(a){for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]];}return a;}
  function reorder(){order=shuffle(D.pool.map((_,i)=>i));ptr=0;}
  function renderHUD(){
    livesEl.textContent="❤️".repeat(lives)+"🖤".repeat(D.lives0-lives);
    progEl.textContent="Correct: "+correct+" / "+D.need;
  }
  function stop(){ if(timer){clearInterval(timer);timer=null;} }
  function nextQ(){
    if(ptr>=order.length) reorder();
    locked=false;
    const d=D.pool[order[ptr]];
    $(".fq-stmt").textContent=d.t;
    $(".fq-flash").textContent="";
    deadline=Date.now()+D.secs*1000;
    stop();
    timer=setInterval(()=>{
      const left=Math.max(0,deadline-Date.now());
      bar.style.width=(100*left/(D.secs*1000))+"%";
      timeEl.textContent=(left/1000).toFixed(1)+"s";
      if(left<=0){ stop(); answer(null); }
    },70);
    renderHUD();
  }
  function answer(v){
    if(locked) return; locked=true; stop();
    const d=D.pool[order[ptr]]; ptr++;
    if(v===null){ lives--; $(".fq-flash").innerHTML='<span style="color:#b23b34">⏱️ Out of time — the answer was '+(d.a?"TRUE":"FALSE")+'.</span>'; }
    else if(v===d.a){ correct++; $(".fq-flash").innerHTML='<span style="color:#1d7a46">✅ Correct.</span>'; }
    else { lives--; $(".fq-flash").innerHTML='<span style="color:#b23b34">❌ No — that one is '+(d.a?"TRUE":"FALSE")+'.</span>'; }
    renderHUD();
    if(lives<=0){ setTimeout(()=>finish(false),900); return; }
    if(correct>=D.need){ setTimeout(()=>finish(true),900); return; }
    setTimeout(nextQ,900);
  }
  function finish(won){
    stop();
    $(".fq-body").innerHTML='<div class="fq-end"><div style="font-size:40px">'+(won?"🏆":"💀")+'</div>'
      +'<h3 style="color:'+(won?"#1d7a46":"#b23b34")+'">'+(won?"Hired. Notebook cleared.":"“We will keep your CV on file.”")+'</h3>'
      +'<div style="font-size:13px;color:#666">'+correct+' correct, '+(D.lives0-lives)+' mistakes.</div>'
      +'<button class="fq-restart">Try again</button></div>';
    $(".fq-restart").addEventListener("click",start);
  }
  function start(){
    correct=0; lives=D.lives0; reorder();
    $(".fq-body").innerHTML='<div class="fq-stmt"></div>'
      +'<div class="fq-btns"><button class="fq-btn fq-true">TRUE</button>'
      +'<button class="fq-btn fq-false">FALSE</button></div><div class="fq-flash"></div>';
    $(".fq-true").addEventListener("click",()=>answer(true));
    $(".fq-false").addEventListener("click",()=>answer(false));
    nextQ();
  }
  start();
})();
</script>'''
    html = tmpl.replace("__UID__", uid).replace("__DATA__", _json.dumps(data))
    display(HTML(html))





