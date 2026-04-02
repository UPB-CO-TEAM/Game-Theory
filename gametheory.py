import streamlit as st
import numpy as np
import pandas as pd
from simplex import f, pregateste_forma_standard, ruleaza_iteratii_simplex, validare_solutie

# --- CONFIGURARE PAGINA & DESIGN ---
st.set_page_config(page_title="Teoria Jocurilor", layout="wide")

st.markdown("""
    <style>
    .title-box {
        background-color: #E1BEE7; /* Mov Lavanda deschis */
        border-radius: 10px;
        padding: 25px; 
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .title-text {
        color: #4A148C; /* Mov inchis pentru text */
        font-size: 60px; 
        font-weight: 900;
        margin: 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .authors-box {
        color: #CE93D8;
        text-align: right;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 40px;
    }
    .authors-title {
        color: #CE93D8; 
        font-weight: bold;
        font-style: italic;
        font-size: 20px;
        margin-bottom: 8px;
    }
    .authors-names {
        color: #CE93D8; 
        line-height: 1.6;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('''
    <div class="title-box">
        <p class="title-text">Joc de 2 persoane cu sumă nulă 💻 🎲</p>
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

# --- LOGICĂ TEORIA JOCURILOR ---

def analiza_strategii_pure(Q):
    alpha = np.min(Q, axis=1)
    beta = np.max(Q, axis=0)
    v1 = np.max(alpha)
    v2 = np.min(beta)
    
    if abs(v1 - v2) < 1e-10:
        idx_linie = np.where(alpha == v1)[0][0]
        idx_col = np.where(beta == v2)[0][0]
        return True, v1, (idx_linie, idx_col), alpha, beta
    return False, (v1, v2), None, alpha, beta

# --- INTERFAȚĂ UTILIZATOR ---

st.sidebar.header("⚙️ Configurare Joc")
n_linii = st.sidebar.number_input("Strategii Jucător A (Linii)", 2, 6, 3)
n_coloane = st.sidebar.number_input("Strategii Jucător B (Coloane)", 2, 6, 3)

st.markdown("<h3 style='color: #CE93D8;'>1. Definirea Matricei de Câștig Q</h3>", unsafe_allow_html=True)
st.info("Introduceți valorile matricei. Am preîncărcat problema implicită din curs.")

# Incarcare Matrice Default ceruta de tine
matrice_default = [
    [1, 1, 2],
    [3, 2, 1],
    [2, 4, 5]
]

input_data = np.zeros((n_linii, n_coloane))
for i in range(min(n_linii, len(matrice_default))):
    for j in range(min(n_coloane, len(matrice_default[0]))):
        input_data[i, j] = matrice_default[i][j]

df_edit = pd.DataFrame(input_data, 
                       columns=[f"b{j+1}" for j in range(n_coloane)], 
                       index=[f"a{i+1}" for i in range(n_linii)])
edited_df = st.data_editor(df_edit)
Q = edited_df.values

if st.button("🚀 Calculează Soluția Optimă", type="primary", use_container_width=True):
    st.divider()
    
    # --- PASUL 1: Strategii Pure ---
    st.markdown("<h3 style='color: #CE93D8;'>2. Pasul 1: Analiza în strategii pure</h3>", unsafe_allow_html=True)
    are_sa, val_sa, pos, alpha, beta = analiza_strategii_pure(Q)
    v1 = np.max(alpha)
    v2 = np.min(beta)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Vectorul minimelor pe linii (α):**")
        st.write([f(x) for x in alpha])
        st.write(f"➡️ **Valoarea inferioară (Maximin) $v_1 = {f(v1)}$**")
    with col2:
        st.write("**Vectorul maximelor pe coloane (β):**")
        st.write([f(x) for x in beta])
        st.write(f"➡️ **Valoarea superioară (Minimax) $v_2 = {f(v2)}$**")

    if are_sa:
        st.success(f"✅ PUNCT ȘA DETECTAT la poziția (a{pos[0]+1}, b{pos[1]+1})")
        st.write(f"**Soluția optimă:** Jucătorul A joacă mereu $a_{pos[0]+1}$, Jucătorul B joacă $b_{pos[1]+1}$.")
        st.metric("Valoarea Jocului (v)", f(val_sa))
    else:
        # --- PASUL 2: Strategii Mixte ---
        st.warning(f"⚠️ $v_1 ({f(v1)}) < v_2 ({f(v2)})$: Jocul nu are punct șa. Trecem la Strategii Mixte (Simplex).")
        
        k = 0
        if np.min(Q) <= 0:
            k = abs(np.min(Q)) + 1
            st.write(f"**Regulă:** Pozitivăm matricea adunând constanta $k = {k}$ la toate elementele.")
        
        Q_ajustat = Q + k
        
        st.markdown("<h3 style='color: #CE93D8;'>3. Pasul 2: Rezolvarea prin Algoritmul Simplex</h3>", unsafe_allow_html=True)
        A_pl = Q_ajustat
        b_pl = [1] * n_linii
        c_pl = [1] * n_coloane
        semne = ['<='] * n_linii
        tip_x = ['>=0'] * n_coloane
        
        TS_init, b_lucru, Cj_std, nume_v, baza_init, mapare = pregateste_forma_standard(A_pl, b_pl, c_pl, semne, tip_x, 'MAX', 1000)
        
        # Salvam copii pentru validarea Simplex (matricea S * XB)
        A_prim_init = TS_init.copy()
        b_backup = b_lucru.copy()
        
        XB_f, Z_f, Dj_f, baza_f, TS_f = ruleaza_iteratii_simplex(TS_init, b_lucru.copy(), Cj_std, baza_init, nume_v, 'MAX')
        
        # --- VALIDARE 1: Programare Liniară (din simplex.py) ---
        validare_solutie(XB_f, Z_f, Dj_f, baza_f, TS_f, A_prim_init, b_backup, c_pl, mapare, nume_v, 'MAX')
        
        # --- PASUL 3: Solutia Teoria Jocurilor ---
        st.markdown("<h3 style='color: #CE93D8;'>4. Pasul 3: Rezultatele Finale (Interpretare)</h3>", unsafe_allow_html=True)
        
        v_joc = (1 / Z_f) - k 
        
        Y_opt = np.zeros(n_coloane)
        val_tab_final = {nume_v[i]: 0.0 for i in range(len(nume_v))}
        for i in range(len(XB_f)): val_tab_final[nume_v[baza_f[i]]] = XB_f[i]
        for j in range(n_coloane): Y_opt[j] = val_tab_final[f"x{j+1}"] * (1/Z_f)
        
        X_opt = np.zeros(n_linii)
        for i in range(n_linii):
            idx_ecart = n_coloane + i 
            X_opt[i] = abs(Dj_f[idx_ecart]) * (1/Z_f)

        res1, res2, res3 = st.columns(3)
        res1.metric("Valoarea Jocului (v)", f(v_joc))
        res2.write("**Strategii Mixte A ($X_0$):**")
        res2.write([f(x) for x in X_opt])
        res3.write("**Strategii Mixte B ($Y_0$):**")
        res3.write([f(y) for y in Y_opt])
        
        # --- VALIDARE 2: Specifica Teoriei Jocurilor ---
        st.markdown("---")
        st.markdown("<h3 style='color: #CE93D8; text-align: center;'>✨ Verificări Specifice (Teoria Jocurilor) ✨</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        val_col1, val_col2 = st.columns(2)
        
        with val_col1:
            st.markdown("**V1. Suma probabilităților este 1**")
            sum_x = sum(X_opt)
            sum_y = sum(Y_opt)
            if abs(sum_x - 1) < 1e-5: st.success(f"✅ $\sum x_i = {f(sum_x)}$")
            else: st.error(f"❌ $\sum x_i = {f(sum_x)}$")
            
            if abs(sum_y - 1) < 1e-5: st.success(f"✅ $\sum y_j = {f(sum_y)}$")
            else: st.error(f"❌ $\sum y_j = {f(sum_y)}$")

            st.markdown("**V2. Încadrarea valorii jocului ($v_1 \le v \le v_2$)**")
            if v1 - 1e-5 <= v_joc <= v2 + 1e-5:
                st.success(f"✅ ${f(v1)} \le {f(v_joc)} \le {f(v2)}$")
            else:
                st.error(f"❌ Valoarea {f(v_joc)} nu este între {f(v1)} și {f(v2)}")

        with val_col2:
            st.markdown("**V3. Verificarea valorii așteptate ($v = X_0 \cdot Q \cdot Y_0^T$)**")
            
            # Calcul matematic: (X0 * Q) * Y0
            val_asteptata = np.dot(np.dot(X_opt, Q), Y_opt)
            
            st.info("Formula demonstrează că dacă ambii jucători aplică strategiile optime, câștigul mediu este exact valoarea jocului.")
            
            if abs(val_asteptata - v_joc) < 1e-5:
                st.success(f"✅ Rezultat calcul: **{f(val_asteptata)}** == **{f(v_joc)}** ($v$)")
            else:
                st.error(f"❌ Rezultat calcul: {f(val_asteptata)} != {f(v_joc)}")
