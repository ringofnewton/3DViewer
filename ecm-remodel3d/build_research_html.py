#!/usr/bin/env python3
"""Build a self-contained, light-mode HTML view of the four research upgrades.

Embeds the four result figures as base64 data URIs so the page opens anywhere
(no server, no external files). Run:  python build_research_html.py
"""
from __future__ import annotations
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "output_validation", "figures")


def img(name):
    with open(os.path.join(FIG, name), "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


CARDS = [
    dict(tag="A", color="#2563eb", title="고변형 레올로지용 floppy-mode 인지 솔버",
         sub="trust-region Newton (Steihaug truncated-CG)",
         limit="고변형(γ가 클 때) 구간에서 FIRE 솔버가 sub-isostatic 네트워크의 "
               "near-singular·indefinite Hessian에 막혀 <b>force balance(힘 평형)에 "
               "도달하지 못함</b> — stiffening 곡선에 “수렴 안 됨” caveat가 붙어 있었음.",
         add="floppy/buckling(곡률이 0 이하인) 방향을 trust-region 경계로 보내 0으로 "
             "나누는 걸 피하는 <b>신뢰영역 뉴턴 솔버</b>를 추가. Hessian-벡터 곱은 "
             "해석적 gradient의 유한차분(matrix-free, numpy만)으로 계산. 도중에 "
             "Lees-Edwards 전단에서 <code>np.mod</code> 래핑이 어긋나는 버그도 "
             "발견·수정(<code>_le_wrap</code>).",
         results=[("FIRE 선형 너머 force-balance", "0 / 10"),
                  ("Newton force-balance (γ_c≈0.15까지)", "6 / 10"),
                  ("force-balanced 선형 modulus G₀", "0.11 ± 0.03 Pa"),
                  ("강성 전이에서 응력 점프 (Δγ=0.01)", "~2000×")],
         caveat="전이 위(γ&gt;γ_c) <b>강성 분지</b>는 near-singular라 두 솔버 모두 완전 "
                "수렴은 못 함(Newton 잔차가 FIRE의 ~1/10). 이 부분은 sparse-Hessian "
                "뉴턴 스텝이 필요 — 정직하게 남겨둔 한계.",
         fig="rheology_periodic.png",
         cap="좌: 4 seed force-balanced 선형 modulus · 중: FIRE vs Newton 잔차(로그) — "
             "FIRE는 즉시 발산, Newton은 γ_c까지 ftol 아래 · 우: 응력 σ(γ)와 강성 전이 위치"),

    dict(tag="B", color="#059669", title="KS/EMD GOF용 디지타이즈 데이터셋",
         sub="binned KS / Wasserstein-1 vs 실측 형식 히스토그램",
         limit="적합도(GOF) 비교의 참조가 <b>합성 로그정규 분포</b>였음 — 즉 “우리가 "
               "가정한 분포”와 비교한 것이지 실제 데이터와 비교한 게 아니었음.",
         add="WebPlotDigitizer 출력 그대로의 <b>디지타이즈 히스토그램 형식</b>"
             "(<code>bin_low, bin_high, count</code>) CSV + 로더 + <b>binned KS/EMD</b>"
             "(빈 내부 균일 가정의 조각선형 CDF). 실제 논문 그림을 디지타이즈하면 "
             "열 3개만 바꿔 끼우는 drop-in 교체.",
         results=[("모델 pore 중앙값 vs 참조", "22.5 vs 22 µm"),
                  ("GOF: KS 통계량 D / p", "0.087 / 0.003"),
                  ("EMD (지구이동거리)", "2.73 µm"),
                  ("검정력 대조(dense vs sparse) p", "&lt; 1e-4")],
         caveat="중앙값(스케일)은 일치하지만 <b>모양은 구분됨</b>(p&lt;0.05) — 평균이 "
                "맞는 것과 분포가 맞는 건 다르다는 걸 정확히 보여줌. 참조 빈도는 아직 "
                "문헌 기반 재구성(형식은 실측 그대로); 특정 논문 그림으로 바꾸면 "
                "paper-grade 검증.",
         fig="distribution_compare.png",
         cap="좌: 모델 pore 분포(초록) vs 디지타이즈 참조 히스토그램(빨강 윤곽) · "
             "우: dense vs sparse CDF 분리 — 검정이 실제 차이를 잡아냄(검정력 확인)"),

    dict(tag="C", color="#7c3aed", title="더 풍부한 추론: 4-파라미터 MCMC",
         sub="에뮬레이터 + affine-invariant 앙상블 MCMC",
         limit="추론이 <b>파라미터 2개, 5×5 격자</b>뿐 — 거칠고, posterior가 넓은 게 "
               "진짜 불확실성인지 격자 탓인지 구분 불가, 나머지 파라미터는 고정.",
         add="비싼 시뮬레이터 보정의 표준 절차: Latin-hypercube 설계 → "
             "<b>Gaussian-RBF 에뮬레이터</b>(LOO 교차검증) → <b>affine-invariant "
             "앙상블 MCMC</b>(Goodman–Weare)를 4개 파라미터(buckle_ratio, "
             "stiffen_alpha, k_bend, traction)에 대해 실행 + 수렴 진단 + held-out 예측 검증. "
             "에뮬레이터 오차는 likelihood에 반영.",
         results=[("에뮬레이터가 설명한 bridge 분산", "~87%"),
                  ("MCMC 수렴 (max R-hat / ESS)", "1.03 / ~1400"),
                  ("buckle_ratio (식별됨, 참값 0.02)", "0.09 ± 0.08"),
                  ("예측 검증 (held-out 0.819)", "0.792 ± 0.038 ✓")],
         caveat="단일 관측량(bridge)으론 buckle_ratio만 잘 제약되고 나머지는 넓게만 "
                "추정됨 — 다만 <b>강한 degeneracy는 없음</b>(|corr|&lt;0.3)이라 넓이는 "
                "트레이드오프가 아니라 정보 한계. 격자가 못 주는 joint posterior·"
                "identifiability·수렴 진단·외삽 예측을 모두 제공.",
         fig="mcmc_infer.png",
         cap="좌: buckle_ratio 주변 posterior(참값 빨강 점선) · 중: 두 비선형 "
             "파라미터의 joint posterior · 우: held-out aster 예측 분포가 실측값을 덮음"),

    dict(tag="D", color="#db2777", title="3D 결합 모델에서 세포 응집",
         sub="cell-cell chemotaxis (확산성 유인물질 기울기)",
         limit="순수 durotaxis에서는 각 세포가 <b>자기가 만든</b> 강성 기울기만 따라가 "
               "서로를 향하지 않고 <b>흩어짐</b> — 응집(자기조직화)이 안 일어남.",
         add="세포-세포 <b>주화성(chemotaxis)</b> 추가: 세포가 분비하는 확산성 유인물질의 "
             "차폐장 c(x) ~ Σⱼ exp(−|x−xⱼ|/λ), 그 기울기는 이웃 세포를 향함. 같은 seed로 "
             "<b>대조군(durotaxis만) vs 처리군(+chemotaxis)</b> 비교 + 행렬-장력 매개 "
             "유도 대안(<code>mechano_intercell_dir</code>)도 구현.",
         results=[("대조군: 평균 세포간 거리", "54.7 → 65.2 µm"),
                  ("처리군(+chemotaxis): 거리", "54.7 → 18.6 µm"),
                  ("응집 발생", "대조 ✗ / 처리 ✓"),
                  ("강화 네트워크 생성", "양쪽 모두 ✓")],
         caveat="화학 신호가 더해지자 세포들이 <b>자기가 강화한 inter-cell 콜라겐을 "
                "따라</b> 서로에게 수렴 — 순수 역학 감지가 놓쳤던 응집을 재현. wound "
                "healing·형태형성의 화학+역학 유도와 일치하는, 깔끔한 양성 결과.",
         fig="coupled_3d.png",
         cap="좌: 대조군(durotaxis만) — 세포 궤적이 바깥으로 발산 · 중: 처리군"
             "(+chemotaxis) — 안쪽으로 수렴·응집 · 우: 시작/끝 세포간 거리 막대"),
]

CARD_TMPL = """
  <section class="card" style="--c:__COLOR__">
    <div class="card-head">
      <span class="badge">__TAG__</span>
      <div><h2>__TITLE__</h2><p class="sub">__SUB__</p></div>
    </div>
    <div class="grid">
      <div class="text">
        <p><span class="lbl lim">지금까지의 한계</span> __LIMIT__</p>
        <p><span class="lbl add">추가한 것</span> __ADD__</p>
        <table class="res">__RESROWS__</table>
        <p class="caveat"><span class="lbl cav">의미 · 정직한 한계</span> __CAVEAT__</p>
      </div>
      <figure><img src="__IMG__" alt="__TITLE__"><figcaption>__CAP__</figcaption></figure>
    </div>
  </section>
"""


def build():
    cards_html = ""
    for c in CARDS:
        rows = "".join(
            f"<tr><td>{k}</td><td class='num'>{v}</td></tr>" for k, v in c["results"])
        html = (CARD_TMPL
                .replace("__COLOR__", c["color"]).replace("__TAG__", c["tag"])
                .replace("__TITLE__", c["title"]).replace("__SUB__", c["sub"])
                .replace("__LIMIT__", c["limit"]).replace("__ADD__", c["add"])
                .replace("__RESROWS__", rows).replace("__CAVEAT__", c["caveat"])
                .replace("__IMG__", img(c["fig"])).replace("__CAP__", c["cap"]))
        cards_html += html

    page = PAGE.replace("__CARDS__", cards_html)
    out = os.path.join(HERE, "research_upgrades.html")
    with open(out, "w") as fh:
        fh.write(page)
    print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")


PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ECM 시뮬레이션 — 연구급 업그레이드 4選</title>
<style>
  :root{ --bg:#eef1f6; --card:#ffffff; --ink:#1f2933; --muted:#5b6776;
         --line:#e2e8f0; --soft:#f1f5f9; }
  *{ box-sizing:border-box }
  body{ margin:0; background:var(--bg); color:var(--ink);
        font-family:'Apple SD Gothic Neo','Noto Sans KR',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        line-height:1.65; -webkit-font-smoothing:antialiased; }
  header{ max-width:1080px; margin:0 auto; padding:54px 24px 14px; }
  header h1{ font-size:30px; margin:0 0 8px; letter-spacing:-.02em; }
  header p.lede{ font-size:17px; color:var(--muted); margin:0; max-width:760px; }
  .pill{ display:inline-block; font-size:12.5px; font-weight:600; color:#0b7285;
         background:#d7f0f3; border-radius:999px; padding:3px 11px; margin-bottom:16px; }
  main{ max-width:1080px; margin:0 auto; padding:18px 24px 70px; }
  .card{ background:var(--card); border:1px solid var(--line); border-left:5px solid var(--c);
         border-radius:14px; padding:24px 26px; margin:22px 0;
         box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.05); }
  .card-head{ display:flex; align-items:center; gap:15px; margin-bottom:16px; }
  .badge{ flex:none; width:40px; height:40px; border-radius:11px; background:var(--c);
          color:#fff; font-weight:800; font-size:20px; display:grid; place-items:center; }
  .card-head h2{ font-size:20px; margin:0; letter-spacing:-.01em; }
  .card-head .sub{ margin:2px 0 0; color:var(--muted); font-size:13.5px;
                   font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .grid{ display:grid; grid-template-columns:1fr 1fr; gap:26px; align-items:start; }
  @media(max-width:820px){ .grid{ grid-template-columns:1fr; } }
  .text p{ margin:0 0 13px; font-size:15px; }
  .lbl{ display:inline-block; font-size:11.5px; font-weight:700; letter-spacing:.02em;
        text-transform:uppercase; padding:2px 8px; border-radius:6px; margin-right:7px;
        vertical-align:1px; }
  .lim{ background:#fee2e2; color:#b91c1c; }
  .add{ background:#e0f2fe; color:#0369a1; }
  .cav{ background:#ede9fe; color:#6d28d9; }
  table.res{ width:100%; border-collapse:collapse; margin:6px 0 14px; }
  table.res td{ padding:7px 8px; border-top:1px solid var(--line); font-size:14px;
                vertical-align:top; }
  table.res td.num{ text-align:right; font-weight:700; white-space:nowrap;
                    font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:var(--c); }
  .caveat{ background:var(--soft); border-radius:9px; padding:12px 14px; font-size:14px;
           color:#334155; }
  figure{ margin:0; }
  figure img{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }
  figcaption{ font-size:12.5px; color:var(--muted); margin-top:9px; }
  code{ background:#eef2f7; padding:1px 6px; border-radius:5px; font-size:13px;
        font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  footer{ max-width:1080px; margin:0 auto; padding:0 24px 60px; color:var(--muted);
          font-size:13.5px; }
  footer .box{ background:var(--card); border:1px solid var(--line); border-radius:12px;
               padding:18px 20px; }
  footer code{ font-size:12.5px; }
</style>
</head>
<body>
<header>
  <div class="pill">research-grade upgrades · 정량 검증을 향해</div>
  <h1>ECM 시뮬레이션 — 남아 있던 4가지 과제, 모두 처리</h1>
  <p class="lede">모델을 “방향이 맞다(qualitative)”에서 “얼마나 맞는지(quantitatively
  predictive)”로 끌어올리는 네 가지 업그레이드. 각 항목은 실행 가능한 검증 스크립트와
  결과 그림, 그리고 아직 풀리지 않은 부분에 대한 정직한 서술을 함께 담았습니다.</p>
</header>
<main>
__CARDS__
</main>
<footer>
  <div class="box">
    한 줄 요약 — <b>솔버</b>는 레올로지의 신뢰도, <b>실측 데이터</b>는 검증의 설득력,
    <b>MCMC</b>는 불확실성의 정직함, <b>chemotaxis</b>는 자기조직화의 완결성을 각각
    올립니다. 재현:
    <code>python rheology_periodic.py</code> ·
    <code>python distribution_compare.py</code> ·
    <code>python mcmc_infer.py</code> ·
    <code>python coupled_sim.py</code>. 전체 서술은
    <code>RESEARCH_UPGRADES.md</code>.
  </div>
</footer>
</body>
</html>
"""

if __name__ == "__main__":
    build()
