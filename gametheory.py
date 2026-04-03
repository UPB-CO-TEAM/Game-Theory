import streamlit as st
import numpy as np
import pandas as pd
from simplex import f, pregateste_forma_standard, ruleaza_iteratii_simplex, validare_solutie

# --- CONFIGURARE PAGINA SI DESIGN ---
st.set_page_config(page_title="Teoria Jocurilor", layout="wide")

st.markdown("""
    <style>
    .title-box { background-color: #E1BEE7; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); }
    .title-text { color: #4A148C; font-size: 60px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .authors-box { color: #CE93D8; text-align: right; font-family: 'Segoe UI', sans-serif; margin-bottom: 40px; }
    .authors-title { color: #CE93D8; font-weight: bold; font-style: italic; font-size: 20px; margin-bottom: 8px; }
    .authors-names { color: #CE93D8; line-height: 1.6; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <div class="title-box">
        <p class="title-text">💻🎲 Joc de 2 persoane cu sumă nulă 🎲💻 </p>
    </div>
''', unsafe_allow_html=True)

st.markdown('''
    <div class="authors-box">
        <div class="authors-title">Facultatea de Științe Aplicate</div>
        <div class="authors-names">
            Dedu Anișoara-Nicoleta, 1333a<br>
            Dumitrescu Andreea Mihaela, 1333a<br>
            Iliescu Daria-Gabriela, 1333a<br>
            Lungu Ionela-Diana, 1333a
        </div>
    </div>
''', unsafe_allow_html=True)

st.divider()

# --- LOGICA TEORIA JOCURILOR ---

def analiza_strategii_pure(Q):
    # Calculul valorilor Maximin (v1) și Minimax (v2) pentru a detecta punctul șa
    alpha = np.min(Q, axis=1) # Minime pe linii
    beta = np.max(Q, axis=0)  # Maxime pe coloane
    v1 = np.max(alpha)        # Maximin
    v2 = np.min(beta)         # Minimax
    
    # Dacă v1 == v2, jocul are punct șa (strategii pure)
    if abs(v1 - v2) < 1e-10:
        idx_linie = np.where(alpha == v1)[0][0]
        idx_col = np.where(beta == v2)[0][0]
        return True, v1, (idx_linie, idx_col), alpha, beta
    return False, (v1, v2), None, alpha, beta

# --- INTERFATA UTILIZATOR ---

st.sidebar.header("⚙️ Configurare Joc")
n_linii = st.sidebar.number_input("Strategii Jucător A (Linii)", 2, 6, 3)
n_coloane = st.sidebar.number_input("Strategii Jucător B (Coloane)", 2, 6, 3)

st.markdown("<h3 style='color: #CE93D8;'>1. Definirea Matricei de Câștig Q</h3>", unsafe_allow_html=True)

matrice_default = [[1, 1, 2], [3, 2, 1], [2, 4, 5]]
input_data = np.zeros((n_linii, n_coloane))
for i in range(min(n_linii, len(matrice_default))):
    for j in range(min(n_coloane, len(matrice_default[0]))):
        input_data[i, j] = matrice_default[i][j]

df_edit = pd.DataFrame(input_data, columns=[f"b{j+1}" for j in range(n_coloane)], index=[f"a{i+1}" for i in range(n_linii)])
edited_df = st.data_editor(df_edit)
Q = edited_df.values

if st.button("🚀 Calculează Soluția Optimă", type="primary", use_container_width=True):
    st.divider()
    
    # --- PASUL 1: Analiza Strategiilor Pure ---
    st.markdown("<h3 style='color: #CE93D8;'>2. Pasul 1: Calculăm: α, β, v1, v2</h3>", unsafe_allow_html=True)
    are_sa, val_sa, pos, alpha, beta = analiza_strategii_pure(Q)
    v1, v2 = np.max(alpha), np.min(beta)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Vectorul minimelor pe linii (α):**", [f(x) for x in alpha])
        st.write(f"➡️ **Valoarea inferioară (Maximin) $v_1 = {f(v1)}$**")
    with col2:
        st.write("**Vectorul maximelor pe coloane (β):**", [f(x) for x in beta])
        st.write(f"➡️ **Valoarea superioară (Minimax) $v_2 = {f(v2)}$**")

    if are_sa:
        st.success(f"✅ PUNCT ȘA DETECTAT: a{pos[0]+1}, b{pos[1]+1}")
        st.metric("Valoarea Jocului (v)", f(val_sa))
    else:
        # --- PASUL 2: Strategii Mixte prin Programare Liniara ---
        st.warning(f"⚠️ $v_1 < v_2$: Jocul nu are punct șa. Trecem la Strategii Mixte.")
        
        # Pozitivarea matricei (Transformarea jocului pentru a asigura v > 0)
        k = 0
        if np.min(Q) <= 0:
            k = abs(np.min(Q)) + 1
           #Regula: Translație matrice pentru v > 0: k=abs(Q)+1
        
        Q_ajustat = Q + k
        
        st.markdown("<h3 style='color: #CE93D8;'>3. Pasul 2: Rezolvarea prin Algoritmul Simplex Primal</h3>", unsafe_allow_html=True)
        # Construire model dual: Maximizează suma probabilităților normalizate (1/v)
        A_pl, b_pl, c_pl = Q_ajustat, [1]*n_linii, [1]*n_coloane
        semne, tip_x = ['<=']*n_linii, ['>=0']*n_coloane
        
        # Pregătire forma standard (R1: variab. >=0, R2: egalități cu variabile ecart/artificiale)
        TS_init, b_lucru, Cj_std, nume_v, baza_init, mapare = pregateste_forma_standard(A_pl, b_pl, c_pl, semne, tip_x, 'MAX', 1000)
        
        # Salvare T0 pentru validarea V3 (Relația S * XB = b_initial)
        A_prim_init, b_backup = TS_init.copy(), b_lucru.copy()
        
        # Rulare iterații Simplex (Pivotare Gauss-Jordan până la optim)
        XB_f, Z_f, Dj_f, baza_f, TS_f = ruleaza_iteratii_simplex(TS_init, b_lucru.copy(), Cj_std, baza_init, nume_v, 'MAX')
        
        # Validare logică PL (verificare Dj, nenegativitate și ecuație de bază S)
        validare_solutie(XB_f, Z_f, Dj_f, baza_f, TS_f, A_prim_init, b_backup, c_pl, mapare, nume_v, 'MAX')
        
        # --- PASUL 3: Recuperare Strategii Mixte ---
        st.markdown("<h3 style='color: #CE93D8;'>4. Pasul 3: Rezultate Finale</h3>", unsafe_allow_html=True)
        
        # Revenirea la valoarea reală: v = (1/Z) - k
        v_joc = (1 / Z_f) - k 
        
        # Extracția probabilităților: X_opt din Delta variabilelor duale, Y_opt din variabilele din baza
        val_tab_final = {nume_v[i]: 0.0 for i in range(len(nume_v))}
        for i in range(len(XB_f)): val_tab_final[nume_v[baza_f[i]]] = XB_f[i]
        Y_opt = [val_tab_final[f"x{j+1}"] * (1/Z_f) for j in range(n_coloane)]
        X_opt = [abs(Dj_f[n_coloane + i]) * (1/Z_f) for i in range(n_linii)]

        res1, res2, res3 = st.columns(3)
        res1.metric("Valoarea Jocului (v)", f(v_joc))
        res2.write("**Strategii Mixte A ($X_0$):**"); res2.write([f(x) for x in X_opt])
        res3.write("**Strategii Mixte B ($Y_0$):**"); res3.write([f(y) for y in Y_opt])
        
        # --- PASUL 4: Validare Teoria Jocurilor (V3: v = X0 * Q * Y0^T) ---
        st.markdown("---")
        st.markdown("<h3 style='color: #CE93D8; text-align: center;'> Verificări Specifice </h3>", unsafe_allow_html=True)
        
        val_col1, val_col2 = st.columns(2)
        with val_col1:
            st.markdown("**V1 & V2: Normalizare (Σ=1) și încadrare (v1 ≤ v ≤ v2)**")
            st.success(f"ΣX={f(sum(X_opt))}, ΣY={f(sum(Y_opt))}")
            st.success(f"v1({f(v1)}) ≤ v({f(v_joc)}) ≤ v2({f(v2)})")
        with val_col2:
            st.markdown("**V3: Relația fundamentală v = X0 * Q * Y0^T**")
            val_calc = np.dot(np.dot(X_opt, Q), Y_opt)
            if abs(val_calc - v_joc) < 1e-5: st.success(f"✅ {f(val_calc)} == {f(v_joc)}")
            else: st.error(f"❌ {f(val_calc)} != {f(v_joc)}")
