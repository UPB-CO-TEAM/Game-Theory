import streamlit as st
import numpy as np
import pandas as pd
from simplex import f, pregateste_forma_standard, ruleaza_simplex_interactiv

# --- LOGICĂ TEORIA JOCURILOR (Conform curs/seminar) ---

def analiza_strategii_pure(Q):
    # Pas 1: Calcul minime pe linii (alpha) și maxime pe coloane (beta)
    alpha = np.min(Q, axis=1)
    beta = np.max(Q, axis=0)
    v1 = np.max(alpha) # Maximin
    v2 = np.min(beta)  # Minimax
    
    # Verificare Punct Șa (v1 = v2)
    if abs(v1 - v2) < 1e-10:
        idx_linie = np.where(alpha == v1)[0][0]
        idx_col = np.where(beta == v2)[0][0]
        return True, v1, (idx_linie, idx_col), alpha, beta
    return False, (v1, v2), None, alpha, beta

# --- INTERFAȚĂ UTILIZATOR (STREAMLIT) ---

st.set_page_config(page_title="Teoria Jocurilor - Simplex App", layout="wide")
st.title("🛡️ Capitolul II: Teoria Jocurilor")
st.subheader("Jocuri de 2 persoane cu sumă nulă (Abordare Matriceală)")

# Sidebar pentru configurarea matricei
st.sidebar.header("⚙️ Configurare Joc")
n_linii = st.sidebar.number_input("Strategii Jucător A (Linii)", 2, 6, 3)
n_coloane = st.sidebar.number_input("Strategii Jucător B (Coloane)", 2, 6, 3)

st.write("### 1. Definirea Matricei de Câștig $Q$")
st.info("Introduceți valorile matricei. Jucătorul A câștigă, Jucătorul B pierde.")

# Tabel interactiv pentru matrice
input_data = np.zeros((n_linii, n_coloane))
df_edit = pd.DataFrame(input_data, 
                       columns=[f"b{j+1}" for j in range(n_coloane)], 
                       index=[f"a{i+1}" for i in range(n_linii)])
edited_df = st.data_editor(df_edit)
Q = edited_df.values

if st.button("🚀 Calculează Soluția Optimă"):
    st.divider()
    
    # PASUL 1: Analiza Minimax (Strategii Pure)
    st.write("### 2. Pasul 1: Analiza în strategii pure")
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
        # PASUL 2: Strategii Mixte (Simplex)
        st.warning("⚠️ $v_1 < v_2$: Jocul nu are punct șa. Trecem la Strategii Mixte prin Programare Liniară.")
        
        # Pozitivarea matricei (dacă există valori negative)
        k = 0
        if np.min(Q) <= 0:
            k = abs(np.min(Q)) + 1
            st.write(f"**Regulă:** Pozitivăm matricea adunând constanta $k = {k}$ la toate elementele.")
        
        Q_ajustat = Q + k
        
        # Construirea modelului de Programare Liniară pentru Jucătorul B (Dualitate)
        # Max g = y'1 + y'2 + ... sub restricția Q_ajustat * Y' <= 1
        st.write("### 3. Pasul 2: Rezolvarea prin Algoritmul Simplex")
        A_pl = Q_ajustat
        b_pl = [1] * n_linii
        c_pl = [1] * n_coloane
        semne = ['<='] * n_linii
        tip_x = ['>=0'] * n_coloane
        
        # Pregătire și Execuție Simplex
        TS_init, b_lucru, Cj_std, nume_v, baza_init, mapare = pregateste_forma_standard(A_pl, b_pl, c_pl, semne, tip_x, 'MAX', 1000)
        XB_f, Z_f, Dj_f, baza_f, TS_f = ruleaza_simplex_interactiv(TS_init.T, b_lucru, Cj_std, baza_init, nume_v, 'MAX', st)
        
        # PASUL 3: Recuperarea soluției finale
        st.write("### 4. Pasul 3: Rezultatele Finale (Interpretare)")
        
        v_joc = (1 / Z_f) - k # Valoarea reală a jocului
        
        # Extracție Y (Jucător B) din variabilele de decizie
        Y_opt = np.zeros(n_coloane)
        val_tab_final = {nume_v[i]: 0.0 for i in range(len(nume_v))}
        for i in range(len(XB_f)): val_tab_final[nume_v[baza_f[i]]] = XB_f[i]
        for j in range(n_coloane): Y_opt[j] = val_tab_final[f"x{j+1}"] * (1/Z_f)
        
        # Extracție X (Jucător A) din Delta_j variabilelor de ecart (Dualitate)
        X_opt = np.zeros(n_linii)
        for i in range(n_linii):
            idx_ecart = n_coloane + i # Variabilele y apar după x în Cj
            X_opt[i] = abs(Dj_f[idx_ecart]) * (1/Z_f)

        # Afișare metrici finale
        res1, res2, res3 = st.columns(3)
        res1.metric("Valoarea Jocului (v)", f(v_joc))
        res2.write("**Strategii Mixte A ($X_0$):**")
        res2.write([f(x) for x in X_opt])
        res3.write("**Strategii Mixte B ($Y_0$):**")
        res3.write([f(y) for y in Y_opt])
        
     

