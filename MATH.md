# The mathematics of Bay Mile

A map of every mathematical operation to the file that implements it.

| Stage | Object | File |
|-------|--------|------|
| Environment | legal moves, make/unmake, `perft` | `src/rules.ts` |
| Featurisation | $f:(s,m)\mapsto\mathbf{x}_m\in\mathbb{R}^{F}$ | `src/features.ts` |
| Policy | $\pi_\theta(\cdot\mid s)$ (MLP + Attention) | `src/policy.ts` |
| Value | $V_\phi(s)\in(-1,1)$ | `src/value.ts` |
| Search | PUCT MCTS | `@euriklis/mcts` + `src/mcts-player.ts` |
| RL training | REINFORCE actor–critic | `src/selfplay.ts` |
| AlphaZero | MCTS distillation | `src/alphazero.ts` |
| Optimiser | Adam | `@euriklis/mathematics` (`Tensor/optim/adam`) |
| Autodiff | reverse-mode on `Tensor` | `@euriklis/mathematics` |

Notation: state $s$, legal move $m$, $L=\lvert\mathcal A(s)\rvert$ legal moves, mover colour $c\in\{+1,-1\}$.

---

## 1. Environment — `src/rules.ts`

Not learned. Produces the legal action set $\mathcal A(s)$, transitions $s\cdot m$, and terminal detection. Move generation is verified by **perft**:

$$
\operatorname{perft}(s,d)=
\begin{cases}
\lvert\mathcal A(s)\rvert, & d=1,\\[2pt]
\displaystyle\sum_{m\in\mathcal A(s)}\operatorname{perft}(s\cdot m,\,d-1), & d>1.
\end{cases}
$$

---

## 2. Move featurisation — `src/features.ts`

Each legal move is mapped to a fixed-length vector $\mathbf x_m=f(s,m)\in\mathbb R^{F}$, $F=24$ (`FEATURE_DIM`). Stacking all moves gives the input matrix

$$
X=\begin{bmatrix}\mathbf x_{m_1}^\top\\ \vdots\\ \mathbf x_{m_L}^\top\end{bmatrix}\in\mathbb R^{L\times F}.
$$

Components (mover-relative, so weights are colour-symmetric): one-hot piece type, normalised from/to file & rank, indicators $\mathbb 1[\text{capture}],\,\mathbb 1[\text{check}],\,\mathbb 1[\text{castle}],\,\mathbb 1[\text{promotion}]$, captured/own piece value, **centre control** $\tfrac18\!\sum_{q\in\mathcal C}\mathbb 1[\text{piece attacks }q]$ over the 8 central squares $\mathcal C$, post-move mobility, lands-attacked / lands-defended, develops-minor, opens-line.

---

## 3. Policy network $\pi_\theta(a\mid s)$ — `src/policy.ts`

### MLPPolicy — each move scored independently

$$
H=\operatorname{ReLU}\!\big(XW_1+\mathbf 1_L b_1\big),\quad
W_1\in\mathbb R^{F\times d_h},\ b_1\in\mathbb R^{1\times d_h},
$$
$$
\mathbf z=HW_2+\mathbf 1_L b_2\in\mathbb R^{L},\qquad
\pi_i=\operatorname{softmax}(\mathbf z)_i=\frac{e^{z_i}}{\sum_{j=1}^{L}e^{z_j}}.
$$

The softmax is taken over the $L$ candidate moves (a distribution over the *action set*), not over a fixed vocabulary.

### AttentionPolicy — moves attend to one another (set-to-policy)

$$
E=XW_{\text{in}}+\mathbf 1_L b_{\text{in}}\in\mathbb R^{L\times d},\quad
C=\operatorname{TransformerBlock}(E),\quad
\mathbf z=CW_{\text{out}}+\mathbf 1_L b_{\text{out}},\quad \pi=\operatorname{softmax}(\mathbf z).
$$

The (non-causal, pre-LN) block lives in `@euriklis/mathematics` (`Tensor/nn/transformerBlock`):

$$
\tilde E=\operatorname{LN}(E),\quad
\text{att}=\operatorname{MHA}(\tilde E W_Q,\tilde E W_K,\tilde E W_V)\,W_O,\quad
\hat E=E+\text{att},
$$
$$
\operatorname{out}=\hat E+\operatorname{ReLU}\!\big(\operatorname{LN}(\hat E)W^{(1)}+b^{(1)}\big)W^{(2)}+b^{(2)},
$$

with scaled dot-product attention (per head)

$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V .
$$

---

## 4. Value network $V_\phi(s)$ — `src/value.ts`

Position features $\boldsymbol\varphi(s)\in\mathbb R^{P}$, $P=10$ (`POS_DIM`): material balance, per-piece-type count differences, mobility, in-check flag, game phase, net centre control — all from the mover's perspective. Then

$$
V_\phi(s)=\tanh\!\Big(\operatorname{ReLU}\!\big(\boldsymbol\varphi(s)\,W_1+b_1\big)W_2+b_2\Big)\in(-1,1),
$$

the critic, bounded to match the $\{-1,0,+1\}$ outcome scale.

---

## 5. Monte-Carlo Tree Search (PUCT) — `@euriklis/mcts`, `src/mcts-player.ts`

Each edge stores visit count $N(s,a)$, total value $W(s,a)$, prior $P(s,a)$, with mean action value $Q(s,a)=W(s,a)/N(s,a)$. Each simulation **selects** by the PUCT rule

$$
a^\star=\arg\max_a\left[\,Q(s,a)+c_{\text{puct}}\,P(s,a)\,\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\right].
$$

At a leaf, the **neural evaluator** (`src/mcts-player.ts`) supplies priors $P(s,\cdot)=\pi_\theta(\cdot\mid s)$ and value $v=V_\phi(s)$; a terminal node returns reward $r\in\{-1,0\}$ (side-to-move mated / draw). **Backup** is negamax — the value flips each ply:

$$
v\leftarrow -v,\qquad N\mathrel{+}=1,\qquad W\mathrel{+}=v .
$$

Optional root exploration uses Dirichlet noise $P\leftarrow(1-\varepsilon)P+\varepsilon\,\boldsymbol\eta,\ \boldsymbol\eta\sim\operatorname{Dir}(\alpha)$. The **search policy** (also the AlphaZero training target) is

$$
\pi_{\text{MCTS}}(a\mid s)=\frac{N(s,a)^{1/\tau}}{\sum_b N(s,b)^{1/\tau}} .
$$

---

## 6. Training objectives

### (a) REINFORCE actor–critic — `src/selfplay.ts`

Let $b_k$ be the White-perspective material after ply $k$, $z\in\{-1,0,+1\}$ the result, and a decision taken at ply $p$ by mover $c$. The **material-shaped discounted return** (`computeReturns`) is

$$
G_p=\sum_{k=p+1}^{T}\gamma^{\,k-p-1}\Big[\,c\,(b_k-b_{k-1})\,\lambda_{\text{pawn}}+[k{=}T]\,c\,z\,\lambda_{\text{win}}\Big].
$$

Advantage $A_p=G_p-V_\phi(s_p)$ (baseline = critic). The losses (`trainStep`):

$$
\mathcal L_\pi=-\frac1N\sum_p A_p\,\log\pi_\theta(a_p\mid s_p),\qquad
\mathcal L_V=\frac1N\sum_p\big(V_\phi(s_p)-G_p\big)^2,
$$
$$
\mathcal L_H=\frac{\beta}{N}\sum_p\sum_a \pi_a\log\pi_a\quad(\text{negative entropy — minimising it raises exploration}),
$$
$$
\boxed{\;\mathcal L=\mathcal L_\pi+c_V\,\mathcal L_V+\mathcal L_H\;}
$$

$A_p$ enters the policy term as a **constant coefficient** (stop-gradient); the score-function estimator $\nabla_\theta\mathbb E[G]=\mathbb E[\,G\,\nabla_\theta\log\pi_\theta\,]$ is what makes this a policy gradient.

### (b) AlphaZero distillation — `src/alphazero.ts`

Self-play with MCTS produces samples $\big(s,\ \pi_{\text{MCTS}}(\cdot\mid s),\ z\big)$. The network is trained to **imitate the search** (policy cross-entropy) and **predict the outcome** (value MSE):

$$
\boxed{\;\mathcal L=\underbrace{-\frac1N\sum_{s}\sum_a \pi_{\text{MCTS}}(a\mid s)\,\log\pi_\theta(a\mid s)}_{\text{policy cross-entropy to visit counts}}
\;+\;c\,\underbrace{\frac1N\sum_{s}\big(V_\phi(s)-z\big)^2}_{\text{value MSE}}\;}
$$

The value target $z$ is the (material-softened) game outcome from the mover's perspective.

### (c) Optimiser — Adam — `@euriklis/mathematics` (`Tensor/optim/adam`)

For parameter gradient $g_t=\nabla_\theta\mathcal L$:

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^{2},
$$
$$
\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\quad
\hat v_t=\frac{v_t}{1-\beta_2^{\,t}},\qquad
\theta\leftarrow\theta-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

---

## 7. Gradients (reverse-mode autodiff) — `@euriklis/mathematics` `Tensor`

Every loss above is a composition of differentiable `Tensor` ops; gradients flow by reverse-mode autodiff. Two that do the heavy lifting:

- **Softmax** backward, for upstream $g$ and output $\pi$:
$$
\frac{\partial\mathcal L}{\partial \mathbf z}=\pi\odot\big(g-(\pi^\top g)\,\mathbf 1\big).
$$

- **Softmax + cross-entropy** fused gradient (used implicitly by the policy losses):
$$
\frac{\partial\mathcal L}{\partial \mathbf z}=\pi-\pi_{\text{target}} .
$$

These, together with matmul backward (via transposed products) and the LayerNorm / attention backward in `@euriklis/mathematics`, give $\nabla_\theta\mathcal L$ for the Adam step.
