"""Ćwiczenie: scaled dot-product attention, maska przyczynowa i tabela kształtów.

Część 1 – ogólny przepływ kształtów (jedna sekwencja, bez wymiaru batcha).
Część 2 – ćwiczenie liczbowe z UPROSZCZENIEM Q = K = V = X (bez W_Q, W_K, W_V i bez P),
          żeby dało się policzyć elementy ręcznie.
"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax z odjęciem maksimum; wiersze z -inf dostają wagę 0."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def sinusoidal_positions(T: int, d_model: int) -> np.ndarray:
    pos = np.arange(T)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    P = np.zeros((T, d_model))
    P[:, 0::2] = np.sin(angle[:, 0::2])
    P[:, 1::2] = np.cos(angle[:, 1::2])
    return P


def attention(Q, K, V, causal: bool):
    """Zwraca: scores, weights_before_mask, weights (po masce), output."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)                      # (T, T)
    weights_before = softmax(scores)                     # bez maski
    if causal:
        T = scores.shape[0]
        future = np.triu(np.ones((T, T), dtype=bool), k=1)   # j > i
        masked_scores = np.where(future, -np.inf, scores)
        weights = softmax(masked_scores)                 # po masce (renormalizacja)
    else:
        weights = weights_before
    return scores, weights_before, weights, weights @ V


# =====================================================================
# CZĘŚĆ 1: ogólny przepływ kształtów (jedna sekwencja)
# =====================================================================
rng = np.random.default_rng(0)
T, d_model, d_k, d_v = 4, 6, 3, 3

X = rng.normal(size=(T, d_model))                 # embeddingi tokenów
P = sinusoidal_positions(T, d_model)              # kodowanie pozycji
H = X + P                                         # wejście do uwagi
W_Q = rng.normal(size=(d_model, d_k))
W_K = rng.normal(size=(d_model, d_k))
W_V = rng.normal(size=(d_model, d_v))
Q, K, V = H @ W_Q, H @ W_K, H @ W_V
scores, w_before, weights, out = attention(Q, K, V, causal=True)

general = {
    "X": X, "P": P, "H": H, "W_Q": W_Q, "W_K": W_K, "W_V": W_V,
    "Q": Q, "K": K, "V": V, "scores": scores,
    "weights": weights, "output": out,
}

# =====================================================================
# CZĘŚĆ 2: ćwiczenie liczbowe z uproszczeniem Q = K = V = X
# =====================================================================
Xn = np.array([[1.0, 0.0],
               [0.0, 1.0],
               [1.0, 1.0],
               [2.0, 0.0]])                       # (4, 2): 4 tokeny, d = 2
Qn = Kn = Vn = Xn                                 # UPROSZCZENIE: Q = K = V = X
sc, wb, wa, on = attention(Qn, Kn, Vn, causal=True)

print("=" * 70)
print("TABELA KSZTAŁTÓW (jedna sekwencja)")
print(f"{'obiekt':<10}{'ogólnie':<22}{'część 1 (T=4,d_model=6,d_k=d_v=3)':<38}{'ćw. liczbowe (T=4,d=2)'}")
rows = [
    ("X",       "(T, d_model)",  general["X"].shape,       Xn.shape),
    ("P",       "(T, d_model)",  general["P"].shape,       "brak (pomijamy)"),
    ("H = X+P", "(T, d_model)",  general["H"].shape,       "brak (H = X)"),
    ("W_Q",     "(d_model, d_k)", general["W_Q"].shape,    "brak (I)"),
    ("W_K",     "(d_model, d_k)", general["W_K"].shape,    "brak (I)"),
    ("W_V",     "(d_model, d_v)", general["W_V"].shape,    "brak (I)"),
    ("Q",       "(T, d_k)",      general["Q"].shape,       Qn.shape),
    ("K",       "(T, d_k)",      general["K"].shape,       Kn.shape),
    ("V",       "(T, d_v)",      general["V"].shape,       Vn.shape),
    ("scores",  "(T, T)",        general["scores"].shape,  sc.shape),
    ("weights", "(T, T)",        general["weights"].shape, wa.shape),
    ("output",  "(T, d_v)",      general["output"].shape,  on.shape),
]
for name, sym, s1, s2 in rows:
    print(f"{name:<10}{sym:<22}{str(s1):<38}{s2}")

assert Q.shape == (T, d_k) and scores.shape == (T, T) and out.shape == (T, d_v)
assert Qn is Xn and Kn is Xn and Vn is Xn      # uproszczenie Q = K = V = X

print("\n" + "=" * 70)
print("ĆWICZENIE LICZBOWE (Q = K = V = X, causal)")
print("X =\n", Xn)
print("scores = Q K^T / sqrt(d) =\n", sc)
print("wagi PRZED maską =\n", wb)
print("wagi PO masce =\n", wa)
print("wektory wynikowe (wagi @ V) =\n", on)

# --- asercja: wagi przyszłych pozycji = 0 oraz wiersze sumują się do 1 ---
future = np.triu(np.ones_like(wa, dtype=bool), k=1)
assert np.all(wa[future] == 0.0), "Waga przyszłej pozycji musi być równa 0!"
assert np.allclose(wa.sum(axis=1), 1.0)
assert np.any(wb[future] > 0), "Bez maski przyszłość ma niezerowe wagi"
print("\nOK: wagi przyszłych pozycji = 0, wiersze sumują się do 1.")

# --- ręczne obliczenie jednego elementu scores: scores[2, 1] ---
# q_2 = [1, 1], k_1 = [0, 1]  ->  q_2 . k_1 = 1*0 + 1*1 = 1
# scores[2, 1] = 1 / sqrt(2) = 0.7071
manual = (1 * 0 + 1 * 1) / np.sqrt(2)
print(f"\nRęcznie: scores[2,1] = (1*0 + 1*1)/sqrt(2) = {manual:.4f}; kod: {sc[2, 1]:.4f}")
assert np.isclose(manual, sc[2, 1])

# --- porównanie dwóch sekwencji: różnią się TYLKO pozycjami 2 i 3 ---
X_a = Xn.copy()
X_b = Xn.copy()
X_b[2:] = [[-3.0, 2.0], [0.5, -4.0]]              # inna "przyszłość"
assert np.array_equal(X_a[:2], X_b[:2])

_, _, _, out_a_nomask = attention(X_a, X_a, X_a, causal=False)
_, _, _, out_b_nomask = attention(X_b, X_b, X_b, causal=False)
_, _, _, out_a_mask = attention(X_a, X_a, X_a, causal=True)
_, _, _, out_b_mask = attention(X_b, X_b, X_b, causal=True)

print("\nWynik pozycji 0:")
print("  bez maski, sekwencja A:", out_a_nomask[0])
print("  bez maski, sekwencja B:", out_b_nomask[0])
print("  z maską,   sekwencja A:", out_a_mask[0])
print("  z maską,   sekwencja B:", out_b_mask[0])
assert not np.allclose(out_a_nomask[0], out_b_nomask[0])   # zależy od przyszłości
assert np.allclose(out_a_mask[0], out_b_mask[0])           # z maską: niezależny
assert np.allclose(out_a_mask[:2], out_b_mask[:2])         # pozycje 0 i 1 też
print("OK: bez maski pozycja 0 zależy od późniejszych wektorów, z maską nie.")