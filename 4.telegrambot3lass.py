from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from html2image import  Html2Image
import numpy as np
import os

label = {0: 'Banjir', 1: 'Tidak Banjir', 2: 'Tidak Terdeteksi'}
os.chdir('D:/@skripsi/cnntransferlearning - 3 class/')

#model = keras.models.load_model('modelklasifikasibanjirDATASET602020.h5')
model = keras.models.load_model('modelklasifikasibanjirDATASET801010.h5')

welcome_message = "Selamat datang di Deteksi Banjir Samarinda! tekan /menu untuk tampilan utama bot."
gif_url = 'https://media.giphy.com/media/8Dmjv4tkbIvAJGclcL/giphy.gif'
def start(update, context):
    # mengirim pesan selamat datang dan tutorial penggunaan dalam bentuk gif kepada pengguna
    context.bot.send_animation(chat_id=update.effective_chat.id, animation=gif_url, caption=welcome_message)

#/citra_satelit
def menu(updater, context):
    updater.message.reply_text("""
    Pilih perintah berikut: 
    /simpang_lembus
    /jalan_pramuka
    /simpang_airhitam
    /simpang_airputih
    /simpang_ayani
    /simpang_sempaja
    /simpang_pasundan
    /simpang_kebon_agung
    /simpang_gunungkapur
    /simpang_antasari_siradjsalman
    /saran
    """)

def citra_satelit(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://inderaja.bmkg.go.id/IMAGE/HIMA/H08_EH_Kaltim.png', save_as='satelit.jpg')
    updater.message.reply_document(document=open('evaluate/satelit.jpg', 'rb'), filename='satelit.jpg')
    
def preprocess_image(file):
    img_path = 'evaluate/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)
    
def simpang_lembus(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-lembuswana', save_as='realtimelembus.jpg')
    img = preprocess_image('realtimelembus.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang lembus', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text('simpang lembus')
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimelembus.jpg', 'rb'), filename='simpang lembus.png')
    else:
        updater.message.reply_text("Citra simpang lembus tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")
    
def jalan_pramuka(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-lembuswana-analytic', save_as='realtimepramuka.jpg')
    
    img = preprocess_image('realtimeanalytic.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('jalan pramuka',prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimepramuka.jpg', 'rb'), filename='jalan pramuka.png')
    else:
        updater.message.reply_text("Citra jalan pramuka Tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_airhitam(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-air-hitam', save_as='realtimeairhitam.jpg')
    img = preprocess_image('realtimeairhitam.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang air hitam', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimeairhitam.jpg', 'rb'), filename='simpang air hitam.png')
    else:
        updater.message.reply_text("Citra simpang air hitam tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_ayani(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-a-yani-gatsu', save_as='realtimeayani.jpg')

    img = preprocess_image('realtimeayani.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang ayani', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimeayani.jpg', 'rb'), filename='simpang ahmad yani.png')
    else:
        updater.message.reply_text("Citra simpang ahmad yani tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_airputih(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-air-putih', save_as='realtimeairputih.jpg')

    img = preprocess_image('realtimeairputih.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang air putih', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimeairputih.jpg', 'rb'), filename='simpang air putih.png')
    else:
        updater.message.reply_text("Citra simpang air putih tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_pasundan(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-pasundan', save_as='realtimepasundan.jpg')

    img = preprocess_image('realtimepasundan.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang pasundan', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimepasundan.jpg', 'rb'), filename='simpang pasundan.png')
    else:
        updater.message.reply_text("Citra simpang pasundan tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_kebon_agung(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-3-kebun-agung', save_as='realtimekebonagung.jpg')

    img = preprocess_image('realtimekebonagung.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang kebon agung', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimekebonagung.jpg', 'rb'), filename='simpang kebun agung.png')
    else:
        updater.message.reply_text("Citra simpang kebon agung tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_sempaja(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-sempaja', save_as='realtimesempaja.jpg')

    img = preprocess_image('realtimesempaja.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang sempaja', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimesempaja.jpg', 'rb'), filename='simpang sempaja.png')
    else:
        updater.message.reply_text("Citra simpang sempaja tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_gunungkapur(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-gn-kapur', save_as='realtimegn.jpg')

    img = preprocess_image('realtimegn.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang gn kapur', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimegn.jpg', 'rb'), filename='simpang gunung kapur.png')
    else:
        updater.message.reply_text("Citra simpang gunung kapur tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def simpang_antasari_siradjsalman(updater, context):
    global hti
    hti = Html2Image(output_path='evaluate')
    hti.screenshot(url='https://diskominfo.samarindakota.go.id/api/cctv/simpang-antasari-siradj-salman', save_as='realtimeantasira.jpg')

    img = preprocess_image('realtimeantasira.jpg')
    hasil = model.predict(img)
    print(hasil)
    prediksimax = np.argmax(hasil)
    prediksi = label[prediksimax]
    print('simpang antasari siradj salman', prediksi)
    if prediksimax == 0 or prediksimax == 1:
        updater.message.reply_text(prediksi)
        updater.message.reply_document(document=open('evaluate/realtimeantasira.jpg', 'rb'), filename='simpang antasari siradj salman.png')
    else:
        updater.message.reply_text("Citra simpang antasari - siradj salman tidak terdeteksi")
    updater.message.reply_text("Ingin pilih simpang yang lain?\n/menu")

def saran(updater, context):
    updater.message.reply_text('Terima kasih telah menggunakan Bot Deteksi Banjir Samarinda. Mohon kesediaan untuk mengisi https://forms.gle/AFSF5HTMjeyj3HW5A')

updater = Updater("5794009909:AAErEEOB4PFAybzq0a-Wl9mAjQSwG8NJFzA")
#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 OPENAI

dispatch = updater.dispatcher
dispatch.add_handler(CommandHandler("start", start))
dispatch.add_handler(CommandHandler("menu", menu))
dispatch.add_handler(CommandHandler("saran", saran))
dispatch.add_handler(CommandHandler("citra_satelit", citra_satelit))
dispatch.add_handler(CommandHandler("simpang_lembus", simpang_lembus))
dispatch.add_handler(CommandHandler("jalan_pramuka", jalan_pramuka))
dispatch.add_handler(CommandHandler("simpang_airhitam", simpang_airhitam))
dispatch.add_handler(CommandHandler("simpang_ayani", simpang_ayani))
dispatch.add_handler(CommandHandler("simpang_airputih", simpang_airputih))
dispatch.add_handler(CommandHandler("simpang_pasundan", simpang_pasundan))
dispatch.add_handler(CommandHandler("simpang_kebon_agung", simpang_kebon_agung))
dispatch.add_handler(CommandHandler("simpang_sempaja", simpang_sempaja))
dispatch.add_handler(CommandHandler("simpang_gunungkapur", simpang_gunungkapur))
dispatch.add_handler(CommandHandler("simpang_antasari_siradjsalman", simpang_antasari_siradjsalman))

#dispatch.add_handler(MessageHandler(filters.ALL, menu, pass_args=True))
dispatch.add_handler(MessageHandler(Filters.text, menu))
dispatch.add_handler(MessageHandler(Filters.photo, menu))

updater.start_polling()
updater.idle()