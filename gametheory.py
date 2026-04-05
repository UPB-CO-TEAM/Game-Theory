import streamlit as st                                          # librarie pentru interfata web
import numpy as np                                             # librarie pentru calcule matematice
import pandas as pd                                            # librarie pentru tabele de date
from simplex import f, pregateste_forma_standard, ruleaza_iteratii_simplex, validare_solutie

                                                                # CONFIGURARE PAGINA SI DESIGN 
st.set_page_config(page_title="Teoria Jocurilor", layout="wide") # setari de baza pagina

st.markdown("""
    <style>
    .title-box { background-color: #E1BEE7; border-radius: 10px; padding: 25px; text-align: center; margin-bottom: 20px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5); }
    .title-text { color: #4A148C; font-size: 60px; font-weight: 900; margin: 0; font-family: 'Segoe UI', sans-serif; }
    .authors-box { color: #CE93D8; text-align: right; font-family: 'Segoe UI', sans-serif; margin-bottom: 40px; }
    .authors-title { color: #CE93D8; font-weight: bold; font-style: italic; font-size: 20px; margin-bottom: 8px; }
    .authors-names { color: #CE93D8; line-height: 1.6; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)                                    # design css pentru cutia cu titlu

st.markdown('''
    <div class="title-box">
        <p class="title-text">💻🎲 Joc de 2 persoane cu sumă nulă 🎲💻 </p>
    </div>
''', unsafe_allow_html=True)                                    # afisare titlu principal 
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
''', unsafe_allow_html=True)                                    # detalii despre echipa 

st.divider()                                                    # linie de separare

                                                                # LOGICA TEORIA JOCURILOR 

def analiza_strategii_pure(Q):                                  # cautam punctul sa
    alpha = np.min(Q, axis=1)                                   # minimul pe fiecare linie
    beta = np.max(Q, axis=0)                                    # maximul pe fiecare coloana
    v1 = np.max(alpha)                                          # maximul minimelor (maximin)
    v2 = np.min(beta)                                           # minimul maximelor (minimax)
    
    if abs(v1 - v2) < 1e-10:                                    # daca maximin = minimax avem pct sa
        idx_linie = np.where(alpha == v1)[0][0]                 # gasim pozitia liniei
        idx_col = np.where(beta == v2)[0][0]                    # gasim pozitia coloanei
        return True, v1, (idx_linie, idx_col), alpha, beta      # returnam rezultatul gasit
    return False, (v1, v2), None, alpha, beta                   # returnam valorile fara punct sa

                                                                # INTERFATA UTILIZATOR 

st.sidebar.header("⚙️ Configurare Joc")                          # meniu lateral setari
n_linii = st.sidebar.number_input("Strategii Jucător A", 2, 6, 3)   # alegem cate linii 
n_coloane = st.sidebar.number_input("Strategii Jucător B", 2, 6, 3) # alegem cate coloane 

st.markdown("<h3 style='color: #CE93D8;'>1. Definirea Matricei de Câștig Q</h3>", unsafe_allow_html=True)

matrice_default = [[1, 1, 2], [3, 2, 1], [2, 4, 5]]             # date initiale pentru test
input_data = np.zeros((n_linii, n_coloane))                     # initializam matricea cu zero
for i in range(min(n_linii, len(matrice_default))):             # incarcam datele default daca exista
    for j in range(min(n_coloane, len(matrice_default[0]))):    # parcurgem coloanele
        input_data[i, j] = matrice_default[i][j]                # adaugam valorile

df_edit = pd.DataFrame(input_data, columns=[f"b{j+1}" for j in range(n_coloane)], index=[f"a{i+1}" for i in range(n_linii)])
edited_df = st.data_editor(df_edit)                             # lasam utilizatorul sa editeze tabelul
Q = edited_df.values                                            # preluam valorile din tabel

if st.button("🚀 Calculează Soluția Optimă", type="primary", use_container_width=True):
    st.divider()                                                # buton declansare 
    
                                                                # PASUL 1: Strategii Pure 
    st.markdown("<h3 style='color: #CE93D8;'>2. Pasul 1: Calculăm: α, β, v1, v2</h3>", unsafe_allow_html=True)
    are_sa, val_sa, pos, alpha, beta = analiza_strategii_pure(Q) # apelam functia de verificare
    v1, v2 = np.max(alpha), np.min(beta)                        # salvam v1 si v2
    
    col1, col2 = st.columns(2)                                  # afisam pe 2 coloane
    with col1:
        st.write("**Vectorul minimelor pe linii (α):**", [f(x) for x in alpha]) 
        st.write(f"➡️ **Valoarea inferioară (Maximin) $v_1 = {f(v1)}$**")         
    with col2:
        st.write("**Vectorul maximelor pe coloane (β):**", [f(x) for x in beta]) 
        st.write(f"➡️ **Valoarea superioară (Minimax) $v_2 = {f(v2)}$**")        

    if are_sa:                                                  # cazul simplu: avem punct sa
        st.success(f"✅ PUNCT ȘA DETECTAT: a{pos[0]+1}, b{pos[1]+1}") 
        st.metric("Valoarea Jocului (v)", f(val_sa))            # afisare valoare joc
    else:
                                                                # PASUL 2: Strategii Mixte
        st.warning(f"⚠️ $v_1 < v_2$: Jocul nu are punct șa. Trecem la Strategii Mixte.")
        
        k = 0                                                   # variabila pentru pozitivare
        if np.min(Q) <= 0:                                      # verificam daca avem valori <= 0
            k = abs(np.min(Q)) + 1                              # calculam k pentru translatie
        
        Q_ajustat = Q + k                                       # noua matrice pentru simplex

                                                                # AFISARE MODELE DUALITATE 
        st.markdown("<h3 style='color: #CE93D8;'>3. Pasul 2: Modelele de Programare Liniară (Duale)</h3>", unsafe_allow_html=True)
        st.write("Atașăm problemele de Programare Liniară corespunzătoare celor doi jucători, conform regulii de dualitate:")

        col_pla, col_plb = st.columns(2)                        # punem modelele unul langa altul

        with col_pla:                                           # construim modelul de MIN (PLA)
            st.markdown("<h4 style='text-align: center; color: #4A148C;'>PLA (MIN)</h4>", unsafe_allow_html=True)
            obj_pla = " + ".join([f"x_{i+1}" for i in range(n_linii)]) 
            st.latex(rf"f_{{min}} = {obj_pla}")                # afisare functie obiectiv 
            restr_pla = []                                      # lista pentru restrictii
            for j in range(n_coloane):                          # parcurgem coloanele
                termeni = [f"{f(Q_ajustat[i][j])}x_{i+1}" for i in range(n_linii)]
                restr_pla.append(" + ".join(termeni) + r" \ge 1") 
            st.latex(r"\begin{cases} " + r" \\ ".join(restr_pla) + r" \end{cases}") 
            st.latex(rf"x_i \ge 0, \quad i=\overline{{1,{n_linii}}}") 

        with col_plb:                                           # construim modelul de MAX (PLB)
            st.markdown("<h4 style='text-align: center; color: #4A148C;'>PLB (MAX)</h4>", unsafe_allow_html=True)
            obj_plb = " + ".join([f"y_{j+1}" for j in range(n_coloane)]) 
            st.latex(rf"g_{{max}} = {obj_plb}")                # afisare functie g
            restr_plb = []                                      # lista restrictii PLB
            for i in range(n_linii):                            # parcurgem liniile
                termeni = [f"{f(Q_ajustat[i][j])}y_{j+1}" for j in range(n_coloane)] 
                restr_plb.append(" + ".join(termeni) + r" \le 1") 
            st.latex(r"\begin{cases} " + r" \\ ".join(restr_plb) + r" \end{cases}") 
            st.latex(rf"y_j \ge 0, \quad j=\overline{{1,{n_coloane}}}") 

        st.info(" **Convenție:** Se va rezolva problema **PLB (MAX)** folosind Algoritmul Simplex Primal (ASP).")
        st.markdown("---")

        st.markdown("<h3 style='color: #CE93D8;'> Rezolvarea prin Algoritmul Simplex Primal (pentru PLB)</h3>", unsafe_allow_html=True)
        
        A_pl, b_pl, c_pl = Q_ajustat, [1]*n_linii, [1]*n_coloane # datele pentru algoritm
        semne, tip_x = ['<=']*n_linii, ['>=0']*n_coloane       # setam constrangerile
        
        TS_init, b_lucru, Cj_std, nume_v, baza_init, mapare = pregateste_forma_standard(A_pl, b_pl, c_pl, semne, tip_x, 'MAX', 1000)
        A_prim_init, b_backup = TS_init.copy(), b_lucru.copy()  # copiem datele pentru validare
        
        XB_f, Z_f, Dj_f, baza_f, TS_f = ruleaza_iteratii_simplex(TS_init, b_lucru.copy(), Cj_std, baza_init, nume_v, 'MAX')
        validare_solutie(XB_f, Z_f, Dj_f, baza_f, TS_f, A_prim_init, b_backup, c_pl, mapare, nume_v, 'MAX')
        
                                                                # RECUPERARE DATE JOC 
        v_joc = (1 / Z_f) - k                                   # calcul v real din k
        inv_z = 1 / Z_f                                         # factor de normalizare
        
        val_tab_final = {nume_v[i]: 0.0 for i in range(len(nume_v))} # cautam valorile in baza finala
        for i in range(len(XB_f)): 
            val_tab_final[nume_v[baza_f[i]]] = XB_f[i]          # extragere din baza
            
        Y_B = [val_tab_final[f"x{j+1}"] for j in range(n_coloane)] # solutia bruta B
        X_A = [abs(Dj_f[n_coloane + i]) for i in range(n_linii)]   # solutia bruta A
        
        Y_opt = [y * inv_z for y in Y_B]                        # transformare in probabilitati
        X_opt = [x * inv_z for x in X_A]                        # transformare in probabilitati

                                                                # EXPLICATII FORMULE 
        st.markdown("---")
        st.markdown("<h3 style='color: #4A148C; background-color: #F3E5F5; padding: 12px; border-radius: 8px; border-left: 5px solid #CE93D8;'>🔄 Trecerea de la Soluția PL la Soluția Jocului</h3>", unsafe_allow_html=True)
        
        ex1, ex2 = st.columns(2)                                # afisare logica 
        with ex1:
            st.markdown("**1. Preluarea soluțiilor din tabelul optim:**")
            st.markdown("- **Pentru Jucătorul B ($Y_B$)**: Se citesc din coloana $X_B$.")
            st.latex(rf"Y_B = ({', '.join([f(y) for y in Y_B])})")
            st.markdown("- **Pentru Jucătorul A ($X_A$)**: Se citesc din linia $\Delta_j$.")
            st.latex(rf"X_A = ({', '.join([f(x) for x in X_A])})")
            
        with ex2:
            st.markdown(f"**2. Calculul valorii jocului ($v$):**")
            st.write(f"Valoarea funcției obiectiv: $g_{{max}} = {f(Z_f)}$.")
            if k > 0:                                           # daca am folosit k
                st.latex(rf"v = \frac{{1}}{{g_{{max}}}} - {k} = {f(v_joc)}")
            else:                                               # fara translatie k
                st.latex(rf"v = \frac{{1}}{{g_{{max}}}} = {f(v_joc)}")
                
        st.markdown("**3. Calculul probabilităților optime ($X_0, Y_0$):**")
        st.latex(rf"X_0 = \frac{{1}}{{g_{{max}}}} \cdot X_A \implies X_0 = ({', '.join([f(x) for x in X_opt])})")
        st.latex(rf"Y_0 = \frac{{1}}{{g_{{max}}}} \cdot Y_B \implies Y_0 = ({', '.join([f(y) for y in Y_opt])})")

                                                                # REZULTATE FINALE 
        st.markdown("---")
        st.markdown("<h3 style='color: #CE93D8;'>4. Pasul 3: Rezultate Finale</h3>", unsafe_allow_html=True)
        
        res1, res2, res3 = st.columns(3)                        # afisare rezultate 
        res1.metric("Valoarea Jocului (v)", f(v_joc))
        res2.write("**Strategii Mixte A ($X_0$):**"); res2.write([f(x) for x in X_opt])
        res3.write("**Strategii Mixte B ($Y_0$):**"); res3.write([f(y) for y in Y_opt])
        
                                                                # VALIDARE TEORETICA 
        st.markdown("---")
        st.markdown("<h3 style='color: #CE93D8; text-align: center;'> Verificări Specifice Teoriei Jocurilor</h3>", unsafe_allow_html=True)
        
        val_col1, val_col2 = st.columns(2)
        with val_col1:                                          # verificari normalizare
            st.markdown("**V1 & V2: Normalizare ($\sum=1$) și încadrare**")
            st.success(f"$\sum X = {f(sum(X_opt))}$ , $\sum Y = {f(sum(Y_opt))}$")
            st.success(f"$v_1$ ({f(v1)}) $\le v$ ({f(v_joc)}) $\le v_2$ ({f(v2)})")
        with val_col2:                                          # verificari relatie fundamentala
            st.markdown("**V3: Relația fundamentală $v = X_0 \cdot Q \cdot Y_0^T$**")
            val_calc = np.dot(np.dot(X_opt, Q), Y_opt)          # calcul matricial
            if abs(val_calc - v_joc) < 1e-5: 
                st.success(f"✅ Calcul direct în funcție: {f(val_calc)} == {f(v_joc)}")
            else: 
                st.error(f"❌ {f(val_calc)} != {f(v_joc)}")
