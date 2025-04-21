# 💰 Loan Approval Prediction App

### The creator:
- Nama : Derrick Vericho
- NIM : 2702291305
- Class : LB-09

## About the Project: 
Tujuan dari proyek ini adalah untuk mengembangkan dan melakukan deployment model machine learning yang mampu memprediksi apakah pengajuan pinjaman seseorang akan disetujui atau ditolak, berdasarkan karakteristik data peminjam.




## About the dataset:
| No  | Kolom                          | Deskripsi                                                                 | Tipe Data        | Kategori         | Target/Feature |
|-----|--------------------------------|---------------------------------------------------------------------------|------------------|------------------|----------------|
| 1   | person_age                     | Usia dari orang tersebut                                                  | Numerik (float)    | Continuous       | Feature        |
| 2   | person_gender                  | Gender dari orang tersebut                                                | Kategorikal      | Binary           | Feature        |
| 3   | person_education               | Tingkat pendidikan tertinggi                                              | Kategorikal      | Multiclass       | Feature        |
| 4   | person_income                 | Pendapatan tahunan                                                        | Numerik (float)  | Continuous       | Feature        |
| 5   | person_emp_exp                 | Tahun pengalaman bekerja                                                  | Numerik (int)  | Continuous       | Feature        |
| 6   | person_home_ownership         | Status kepemilikan tempat huni                                            | Kategorikal      | Multiclass       | Feature        |
| 7   | loan_amnt                      | Jumlah pinjaman yang diminta                                              | Numerik (float)  | Continuous       | Feature        |
| 8   | loan_intent                    | Tujuan dari pinjaman                                                      | Kategorikal      | Multiclass       | Feature        |
| 9   | loan_int_rate                  | Suku bunga pinjaman                                                       | Numerik (float)  | Continuous       | Feature        |
| 10  | loan_percent_income           | Jumlah pinjaman sebagai persentase dari pendapatan tahunan                | Numerik (float)  | Continuous       | Feature        |
| 11  | cb_person_cred_hist_length    | Durasi riwayat kredit dalam tahun                                         | Numerik (float)    | Continuous       | Feature        |
| 12  | credit_score                   | Skor kredit dari orang tersebut                                           | Numerik (int)  | Continuous       | Feature        |
| 13  | previous_loan_defaults_on_file| Indikator tunggakan pinjaman sebelumnya (0: Tidak, 1: Ya)                | Kategorikal      | Binary           | Feature        |
| 14  | loan_status                    | Persetujuan pinjaman (1: diterima, 0: ditolak)                            | Kategorikal      | Binary           | **Target**     |


Visit the app : **https://model-deployment-hcvvkc4clubmfjsanlzcmv.streamlit.app/**
