from model.xgb_model import Model
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import os 


st.set_page_config(page_title="Loan Approval Predictor", page_icon="💰", layout="wide")
model = Model()
#<----------------------------------------------------------Heading---------------------------------------------------------------------->
st.write("Project UTS - Derrick Vericho - 2702291305")
st.divider()
st.title("💰 Loan Approval Prediction App with XGBoost")

#<--------------------------------------------------------Setting piechart--------------------------------------------------------------->
st.divider()
url = "https://raw.githubusercontent.com/DerrickVericho/Model-Deployment/master/Dataset_A_loan.csv"
df = model.read_data(url)
labels = ['Rejected (0)', 'Approved (1)']
sizes = df['loan_status'].value_counts().sort_index()
colors = ['#ff6f69', '#88d8b0']
explode = (0.05, 0.05)

fig, ax = plt.subplots(figsize=(7, 7))
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    explode=explode,
    textprops={'color': "black", 'fontsize': 12}
)
ax.set_title('Loan Status Distribution', fontsize=16, weight='bold')
plt.setp(autotexts, size=13, weight="bold")

#<----------------------------------------------------------Layout piechart------------------------------------------------------------->
col1, col2= st.columns([1,4])  # Kolom kiri 2x lebih lebar dari kanan

with col2:
    loan_counts = df['loan_status'].value_counts(normalize=True).sort_index()
    approved_pct = loan_counts[1] * 100
    rejected_pct = loan_counts[0] * 100
    st.markdown(f"""
    ## 💡 Fakta Menarik dari Data Pengajuan Pinjaman

    📉 **Hanya 1 dari 5 orang yang berhasil mendapatkan pinjaman.**

    Meskipun banyak yang mengajukan, **sekitar 78% pengajuan ditolak** oleh sistem.

    Ini menunjukkan bahwa:
    - Banyak calon peminjam yang **tidak memenuhi kriteria kelayakan**.
    - Bisa jadi ada faktor-faktor tertentu seperti skor kredit, penghasilan, atau tujuan pinjaman yang berpengaruh besar dalam keputusan.

    Coba cek apakah kamu layak untuk mendapatkan pinjaman!
    """)

with col1:
    st.pyplot(fig)



#<----------------------------------------------------------Bagian Input--------------------------------------------------------------->
age = st.slider("Umur", 18, 70, 18)
gender = st.radio("Jenis Kelamin", ("male", "female"))
education = st.selectbox("Pendidikan Terakhir", ["High School", "Associate", "Bachelor", "Master", "Doctorate"])
home = st.selectbox("Status Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE","OTHER"])
intent = st.selectbox("Tujuan Pinjaman", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
income = st.number_input("Pendapatan Tahunan (USD)", 1000, 1000000, 50000)
income_missing = st.checkbox("Apakah pendapatan tidak tersedia?")
loan_amount = st.number_input("Jumlah Pinjaman (USD)", 500, 100000, 1000)
interest_rate = st.slider("Bunga Pinjaman (%)", 0.0, 30.0, 5.0)
percent_income = round(loan_amount / (income if income > 0 else 1), 2)  # Hitungan dari hutang/income, jika tidak ada income maka = hutang
cred_hist = st.slider("Lama Riwayat Kredit (tahun)", 1, 30, 10)
emp_exp = st.slider("Pengalaman Kerja (tahun)", 0, 40, 10)
credit_score = st.slider("Skor Kredit", 300, 850, 500)
previous_default = st.radio("Pernah Gagal Bayar?", ("No", "Yes"))

if st.button("Prediksi"):
    # Encode binary manually
    gender_encoded = 1 if gender == "male" else 0
    default_encoded = 1 if previous_default == "Yes" else 0
    income_missing_val = 1 if income_missing else 0

    input_data = {
        'person_age': age,
        'person_gender': gender_encoded,
        'person_education': education,
        'person_home_ownership': home,
        'loan_intent': intent,
        'person_income': income,
        'person_income_missing': income_missing_val,
        'loan_amnt': loan_amount,
        'loan_int_rate': interest_rate / 100,
        'loan_percent_income': percent_income,
        'cb_person_cred_hist_length': cred_hist,
        'person_emp_exp': emp_exp,
        'credit_score': credit_score,
        'previous_loan_defaults_on_file': default_encoded
    }

    # Prediksi & confidence
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)

    st.subheader("📊 Hasil Prediksi")
    if pred == 1:
        st.success(f"🎉 Pinjaman kemungkinan **DISETUJUI** ✅")
        st.write(f"Dengan persentase disetujui {max(prob) * 100:.2f}%")
    else:
        st.error(f"❌ Pinjaman kemungkinan **DITOLAK**")
        st.write(f"Dengan persentase ditolak {max(prob) * 100:.2f}%")

