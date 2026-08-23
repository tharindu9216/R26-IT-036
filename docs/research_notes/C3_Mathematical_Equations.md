# Component 3 Mathematical Equations

This note contains the main equations used by the C3 stress and
cognitive-distortion component. The first four equations are the most useful
for the six-page research paper. The remaining equations are useful for the
methodology explanation, presentation, and viva.

## Notation

| Symbol | Meaning |
|---|---|
| $x$ | Input text |
| $y_s$ | Binary stress label |
| $y_r$ | Ten-class subreddit-proxy label |
| $y_c$ | Binary cognitive-distortion label |
| $M$ | Number of ensemble members |
| $K$ | Number of classes |
| $z_k$ | Logit for class $k$ |
| $p_k$ | Predicted probability for class $k$ |
| $w_m$ | Ensemble weight for model $m$ |
| $\tau$ | Binary decision threshold |

## 1. Transformer representationssh chanupa@100.105.155.88

For tokenized text $x$, the contextual representation is obtained from the
transformer's classification token:

$$
h_x = \operatorname{Transformer}(x)_{[CLS]}.
$$

The shared representation used by the classification head is:

$$
u_x = \operatorname{Dropout}\!\left(\operatorname{LayerNorm}(h_x)\right).
$$

A two-layer classification head can be written as:

$$
z = W_2\,\operatorname{Dropout}\!\left(
\operatorname{GELU}(W_1u_x+b_1)
\right)+b_2.
$$

The attention operation inside a transformer is:

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

## 2. Softmax probability

The probability assigned to class $k$ is:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}.
$$

For a binary task, the model produces:

$$
p(y=1\mid x)=p_1, \qquad p(y=0\mid x)=1-p_1.
$$

## 3. Class-weighted, label-smoothed cross-entropy

With label-smoothing factor $\varepsilon$, the smoothed target for class $k$
is:

$$
q_k=(1-\varepsilon)\mathbb{1}[k=y]+\frac{\varepsilon}{K}.
$$

The class-weighted loss is:

$$
\mathcal{L}_{CE}
=-\sum_{k=1}^{K}\omega_k q_k\log(p_k),
$$

where $\omega_k$ is the training weight for class $k$. A common balanced
weight is:

$$
\omega_k=\frac{N}{K n_k},
$$

where $N$ is the total number of training examples and $n_k$ is the number of
examples in class $k$. C3 uses $\varepsilon=0.1$.

## 4. Stress dual-head loss

Each stress transformer jointly learns binary stress detection and the
ten-class subreddit-proxy task. Its combined loss is:

$$
\boxed{
\mathcal{L}_{stress}
=\alpha\mathcal{L}_{binary}
+(1-\alpha)\mathcal{L}_{subreddit}
}
$$

The selected loss weights are approximately:

$$
\alpha_{BERT}=\alpha_{MentalBERT}=0.4799,
\qquad
\alpha_{DeBERTa}=0.7863.
$$

The CBT models use one binary loss:

$$
\mathcal{L}_{CBT}=\mathcal{L}_{binary\ distortion}.
$$

## 5. Layer-wise learning-rate decay

Lower transformer layers receive smaller learning rates:

$$
\eta_l=\eta_{base}\gamma^{L-l},
$$

where $L$ is the number of encoder layers, $l$ is the current layer, and
$\gamma$ is the decay factor. The task-specific classification head uses:

$$
\eta_{head}=m\eta_{base},
$$

with $m=10$ in C3.

## 6. Weighted soft-voting ensemble

The ensemble probability is the weighted average of the individual model
probabilities:

$$
\boxed{
p_{ens}(y=k\mid x)
=\sum_{m=1}^{M}w_m p_m(y=k\mid x),
\qquad
\sum_{m=1}^{M}w_m=1
}
$$

### Reported stress ensemble

The reported BERT + DeBERTa-v3 result uses equal probability averaging:

$$
p_{stress}(y=1\mid x)
=\frac{p_{BERT}(y=1\mid x)+p_{DeBERTa}(y=1\mid x)}{2}.
$$

### FastAPI stress ensemble

The backend loads three models with equal weights:

$$
p_{stress}^{backend}
=\frac{p_{BERT}+p_{MentalBERT}+p_{DeBERTa}}{3}.
$$

### CBT ensemble

The out-of-fold-selected CBT ensemble is:

$$
\boxed{
p_{CBT}
=0.10p_{BERT}
+0.40p_{MentalBERT}
+0.50p_{DeBERTa}
}
$$

## 7. Threshold-based binary decision

The final binary prediction is:

$$
\hat{y}=
\begin{cases}
1, & p_{ens}(y=1\mid x)\ge\tau,\\
0, & p_{ens}(y=1\mid x)<\tau.
\end{cases}
$$

The backend stress threshold is $\tau_s=0.50$, and the CBT ensemble threshold
is $\tau_c=0.37$.

The CBT weights and threshold are selected using training-only out-of-fold
predictions:

$$
(w^*,\tau^*)
=\underset{w,\tau}{\arg\max}\;
F1_{macro}^{OOF}(w,\tau).
$$

## 8. TF-IDF and traditional baselines

For term $t$ in document $d$, a smoothed TF-IDF representation is:

$$
\operatorname{tfidf}(t,d)
=\operatorname{tf}(t,d)
\left[
\log\!\left(\frac{N+1}{df(t)+1}\right)+1
\right].
$$

Logistic Regression estimates the positive probability as:

$$
p(y=1\mid x)=\sigma(\beta^\top x+b)
=\frac{1}{1+e^{-(\beta^\top x+b)}}.
$$

A linear SVM learns the decision function:

$$
f(x)=w^\top x+b,
$$

by minimizing regularized hinge loss:

$$
\min_{w,b}\frac{1}{2}\lVert w\rVert^2
+C\sum_{i=1}^{N}\max\left(0,1-y_i(w^\top x_i+b)\right).
$$

## 9. Classification metrics

Using true positives $TP$, true negatives $TN$, false positives $FP$, and
false negatives $FN$:

$$
Accuracy=\frac{TP+TN}{TP+TN+FP+FN},
$$

$$
Precision=\frac{TP}{TP+FP},
\qquad
Recall=\frac{TP}{TP+FN},
$$

$$
Specificity=\frac{TN}{TN+FP},
$$

$$
F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall},
$$

$$
Balanced\ Accuracy=\frac{Recall+Specificity}{2}.
$$

For $K$ classes, macro-F1 gives every class equal importance:

$$
F1_{macro}=\frac{1}{K}\sum_{k=1}^{K}F1_k.
$$

The Matthews correlation coefficient is:

$$
MCC=
\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
$$

ROC-AUC can be interpreted as:

$$
ROC\text{-}AUC
=P\!\left(s(x^+)>s(x^-)\right),
$$

the probability that a randomly selected positive example receives a higher
score than a randomly selected negative example. PR-AUC is the area under the
precision-recall curve:

$$
PR\text{-}AUC=\int_0^1 Precision(Recall)\,d(Recall).
$$

## 10. Five-fold mean and standard deviation

For fold metric values $m_1,\ldots,m_K$:

$$
\bar{m}=\frac{1}{K}\sum_{i=1}^{K}m_i,
$$

$$
\sigma=\sqrt{\frac{1}{K}\sum_{i=1}^{K}(m_i-\bar{m})^2}.
$$

The saved C3 summaries use this population standard deviation. Report the
cross-validation result as $\bar{m}\pm\sigma$ and keep the final held-out test
result separate.

## 11. Integrated Gradients

For input embedding $x$, baseline $x'$, model output $F$, and feature $i$:

$$
IG_i(x)
=(x_i-x'_i)
\int_0^1
\frac{\partial F\!\left(x'+a(x-x')\right)}{\partial x_i}\,da.
$$

The practical approximation with $S$ interpolation steps is:

$$
IG_i(x)\approx
(x_i-x'_i)\frac{1}{S}
\sum_{s=1}^{S}
\frac{\partial F\!\left(x'+\frac{s}{S}(x-x')\right)}{\partial x_i}.
$$

## 12. SHAP

The Shapley attribution for feature $i$ is:

$$
\phi_i
=\sum_{S\subseteq N\setminus\{i\}}
\frac{|S|!(|N|-|S|-1)!}{|N|!}
\left[f(S\cup\{i\})-f(S)\right].
$$

It measures the average marginal contribution of a feature across possible
feature coalitions.

## 13. LIME

LIME fits a simple local explanation model $g$ around input $x$:

$$
g^*
=\underset{g\in G}{\arg\min}
\left[
\mathcal{L}(f,g,\pi_x)+\Omega(g)
\right],
$$

where $f$ is the original classifier, $\pi_x$ gives higher weight to
perturbations near $x$, and $\Omega(g)$ penalizes explanation complexity.

## 14. C3 XAI consensus and agreement

For method $m$ and word $i$, C3 first normalizes the attribution by the
largest absolute attribution from that method:

$$
\tilde{a}_{m,i}
=\frac{a_{m,i}}{\max_j|a_{m,j}|}.
$$

The equal-method consensus across IG, SHAP, and LIME is:

$$
\boxed{
c_i=\frac{
\tilde{a}_{IG,i}+\tilde{a}_{SHAP,i}+\tilde{a}_{LIME,i}
}{3}
}
$$

Let $A_i$ be the set of non-zero method scores for word $i$. Directional
agreement is:

$$
Agreement_i
=\frac{1}{|A_i|}
\sum_{a\in A_i}
\mathbb{1}\!\left[\operatorname{sign}(a)=\operatorname{sign}(c_i)\right].
$$

These values describe agreement among explanation methods for one
representative model. They are not an explanation of the full ensemble and do
not prove causality.

## 15. Counterfactual objective

A minimal counterfactual searches for a small text change that flips the
predicted class:

$$
x_{cf}^*
=\underset{x'}{\arg\min}\;d(x,x')
\quad\text{subject to}\quad
\hat{y}(x')\ne\hat{y}(x).
$$

Here, $d(x,x')$ measures the number or cost of word edits. A class flip shows
model sensitivity; it is not a causal psychological conclusion.

## 16. BERTopic equations

The cosine similarity between document embedding $u$ and topic embedding $v$
is:

$$
\cos(u,v)=\frac{u\cdot v}{\lVert u\rVert\lVert v\rVert}.
$$

The base class-based TF-IDF representation for term $t$ and topic/class $c$
can be expressed as:

$$
c\text{-}TFIDF_{t,c}
=tf_{t,c}\log\!\left(1+\frac{A}{f_t}\right),
$$

where $tf_{t,c}$ is the frequency of term $t$ in the aggregated topic
document, $f_t$ is its frequency across all topic documents, and $A$ is the
average number of words per topic document. C3 additionally enables BERTopic's
BM25 weighting option.

For a document $i$, let $a_i$ be its mean within-topic distance and $b_i$ the
smallest mean distance to another topic. The silhouette score is:

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)},
\qquad -1\le s_i\le1.
$$

Topic diversity over the top $n$ terms is:

$$
Topic\ Diversity
=\frac{\left|\bigcup_{c\ne-1}T_c^{(n)}\right|}
{\sum_{c\ne-1}\left|T_c^{(n)}\right|},
$$

where $T_c^{(n)}$ is the available set of top-$n$ terms for non-outlier topic
$c$.

The HDBSCAN outlier rate is:

$$
Outlier\ Rate
=\frac{\#\{i:topic_i=-1\}}{N}.
$$

## 17. Deterministic decision fusion

Let

$$
s=\mathbb{1}[p_{stress}\ge\tau_s],
\qquad
d=\mathbb{1}[p_{CBT}\ge\tau_c].
$$

The fusion state is:

$$
F(s,d)=
\begin{cases}
\texttt{no\_signal}, & (s,d)=(0,0),\\
\texttt{stress\_only}, & (s,d)=(1,0),\\
\texttt{distortion\_only}, & (s,d)=(0,1),\\
\texttt{stress\_with\_distortion}, & (s,d)=(1,1).
\end{cases}
$$

This is a rule, not a learned probabilistic fusion model.

## 18. Calibration equations for future evaluation

The Brier score for binary predictions is:

$$
Brier=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)^2.
$$

Expected Calibration Error with confidence bins $B_1,\ldots,B_J$ is:

$$
ECE
=\sum_{j=1}^{J}\frac{|B_j|}{N}
\left|accuracy(B_j)-confidence(B_j)\right|.
$$

These equations are useful for future work, but calibrated confidence should
not be claimed until the values are produced by a committed evaluation.

## Recommended equations for the six-page paper

To save space, include only:

1. the stress dual-head loss;
2. the weighted ensemble probability;
3. the threshold decision rule;
4. the XAI consensus equation, only if space remains.

The standard metric equations can remain in these research notes and be used
during the viva instead of occupying paper space.
