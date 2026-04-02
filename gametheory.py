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
st.info("Introduceți valorile matricei. Jucătorul A câștigă, Jucătorul B pierde.")

input_data = np.zeros((n_linii, n_coloane))
df_edit = pd.DataFrame(input_data, 
                       columns=[f"b{j+1}" for j in range(n_coloane)], 
                       index=[f"a{i+1}" for i in range(n_linii)])
edited_df = st.data_editor(df_edit)
Q = edited_df.values

if st.button("🚀 Calculează Soluția Optimă", type="primary", use_container_width=True):
    st.divider()
    
    st.markdown("<h3 style='color: #CE93D8;'>2. Pasul 1: Analiza în strategii pure</h3>", unsafe_allow_html=True)
    are_sa, val_sa, pos, alpha, beta = analiza_strategii_pure(Q)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Vectorul minimelor pe linii (α):**")
        st.write([f(x) for x in alpha])
        st.write(f"➡️ **Valoarea inferioară (Maximin) $v_1 = {f(np.max(alpha))}$**")
    with col2:
        st.write("**Vectorul maximelor pe coloane (β):**")
        st.write([f(x) for x in beta])
        st.write(f"➡️ **Valoarea superioară (Minimax) $v_2 = {f(np.min(beta))}$**")

    if are_sa:
        st.success(f"✅ PUNCT ȘA DETECTAT la poziția (a{pos[0]+1}, b{pos[1]+1})")
        st.write(f"**Soluția optimă (strategii pure):** Jucătorul A joacă mereu $a_{pos[0]+1}$, Jucătorul B joacă $b_{pos[1]+1}$.")
        st.metric("Valoarea Jocului (v)", f(val_sa))
    else:
        st.warning("⚠️ $v_1 < v_2$: Jocul nu are punct șa. Trecem la Strategii Mixte prin Programare Liniară.")
        
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
        
        # PREGATIREA
        TS_init, b_lucru, Cj_std, nume_v, baza_init, mapare = pregateste_forma_standard(A_pl, b_pl, c_pl, semne, tip_x, 'MAX', 1000)
        
        # BACKUP PENTRU VALIDARE V3
        A_prim_init = TS_init.copy()
        b_backup = b_lucru.copy()
        
        # EXECUTIA SIMPLEX
        XB_f, Z_f, Dj_f, baza_f, TS_f = ruleaza_iteratii_simplex(TS_init, b_lucru.copy(), Cj_std, baza_init, nume_v, 'MAX')
        
        # VALIDAREA SOLUTIEI
        validare_solutie(XB_f, Z_f, Dj_f, baza_f, TS_f, A_prim_init, b_backup, c_pl, mapare, nume_v, 'MAX')
        
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
