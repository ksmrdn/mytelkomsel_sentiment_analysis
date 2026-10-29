# Panduan pelabelan ulang: ulasan campuran

File: `data/4_relabel_campuran.csv` (116 ulasan). Isi kolom **`label_baru`**, dan **`catatan`** kalau ragu.

Ulasan di file ini dipilih **hanya dari teksnya**: mengandung kata pujian (bagus, lancar, baik, mantap, …) dan juga
kata kontras (tapi, namun, walaupun, cuma, …) atau kata keluhan. Label lama, rating bintang, val/test, dan prediksi
model sengaja **tidak ditampilkan**. Sebagian kandidat sebenarnya keluhan murni (misalnya "suka error" = "sering
error"); labeli saja apa adanya.

## Pertanyaan kunci

> **Apakah penulis mengalami masalah nyata yang belum selesai, sehingga ulasan ini perlu ditanggapi CS?**

| Jawaban | `label_baru` |
|---|---|
| **Ya**: ada masalah konkret yang dialami (gagal, error, lemot, sinyal hilang, pulsa/kuota hilang, harga dirasa terlalu mahal, tagihan salah), walaupun ada pujian | `negative` |
| **Tidak**: isinya puas/memuji, dan bagian "tapi"-nya hanya harapan, saran ringan, pengandaian ("kalau nanti lemot lagi…"), atau masalah yang **sudah** beres | `positive` |
| Tidak memuji dan tidak mengeluh: faktual, pertanyaan, atau saran tanpa masalah yang dialami | `neutral` |
| Teks tidak bisa dinilai (kosong maknanya, bahasa lain, terpotong parah) | `skip` |

Dalam mode 2 kelas notebook, `neutral` tetap dihitung sebagai "bukan keluhan" (positive). Isi `neutral` kalau memang
netral, supaya informasinya tidak hilang.

## Contoh

| Ulasan | Label | Alasan |
|---|---|---|
Contoh di bawah adalah kalimat buatan, bukan ulasan dari gold.

| Ulasan (contoh buatan) | Label | Alasan |
|---|---|---|
| tampilan barunya keren, tapi menu riwayat transaksi sering kosong | `negative` | masalah yang dialami belum selesai |
| harga paketnya oke, cuma masa aktifnya kependekan buat saya | `negative` | masa aktif pendek = keluhan konkret |
| proses tukar poin gampang banget, sayang hadiahnya makin sedikit | `negative` | keluhan konkret, walaupun diawali pujian |
| kemarin sempat gagal bayar, sekarang sudah lancar lagi, makasih | `positive` | masalahnya sudah beres |
| fiturnya lengkap, semoga nanti ada mode gelap | `positive` | harapan, bukan masalah |
| wah hebat sekali, tiap malam sinyalnya hilang dengan tepat waktu | `negative` | sarkasme; isi `catatan`: sarkasme |

Perbedaan dengan panduan lama ("pilih sentimen yang dominan"): sekarang **satu keluhan konkret sudah cukup untuk
`negative`**. Aturan ini lebih mudah dijalankan secara konsisten, dan sesuai dengan tujuan model: menemukan ulasan
yang perlu ditanggapi.

Kalau kamu lebih memilih aturan "dominan", tulis di baris pertama kolom `catatan` "ATURAN: dominan", lalu labeli
dengan aturan itu secara konsisten.

## Setelah selesai

Simpan file (tetap CSV, nama sama), lalu kabari. Notebook akan:
1. mengganti label lama dengan `label_baru` untuk baris yang terisi (sumber menjadi `human_relabel`);
2. melaporkan berapa label yang berubah;
3. mengevaluasi ulang baseline dan IndoBERT di **gold_val**. Prediksi IndoBERT dari Colab tetap bisa dipakai, karena
   yang berubah hanya label gold, bukan data training.
