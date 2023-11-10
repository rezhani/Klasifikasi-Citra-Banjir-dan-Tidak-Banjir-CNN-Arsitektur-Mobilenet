from keras.applications import MobileNet
from keras.models import Sequential
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

# menggunakan library os untuk mengakses file di komputer
import os
os.chdir('D:/@skripsi/cnntransferlearning - 3 class/')

# prepare the data
labels = ['Banjir','Tidak Banjir','Tidak Terdeteksi']
datagen = ImageDataGenerator(rescale=1./255,
                             rotation_range=30,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2)

# Skenario Dataset
folder = 'DATASET602020'
#folder = 'DATASET801010'

# data augmentasi
train_generator = datagen.flow_from_directory(folder+'/train', target_size=(224, 224), batch_size=32)
val_generator = datagen.flow_from_directory(folder+'/valid', target_size=(224, 224), batch_size=32)
test_generator = datagen.flow_from_directory(folder+'/test', target_size=(224, 224), batch_size=32, shuffle=False)

# tanpa data augmentasi
#train_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(folder+'/train', target_size=(224, 224), batch_size=32)
#val_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(folder+'/valid', target_size=(224, 224), batch_size=32)
#test_generator = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(folder+'/test', target_size=(224, 224), batch_size=32, shuffle=False)

# menggunakan pretrained model dan menghapus head layernya untuk finetuning atau transfer learning model
base_model = MobileNet(weights='imagenet', include_top=False)
base_model.summary()

# menambah layer baru untuk head dengan output 3 kelas
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Flatten())
model.add(Dense(units=3, activation='softmax'))
model.summary()

# Menampilkan nama-nama layer dari model tambahan
print("\nLayer yang dibuat:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}")

#%%

# Freeze layer dari base model kecuali 5 layer terakhir (1,1 juta parameter)
#for layer in base_model.layers[:-5]:
#    layer.trainable = False
#model.summary()

# Freeze seluruh layer dari base model (131K Parameter)
for layer in base_model.layers:
    layer.trainable = False
model.summary()

#%%

# kompilasi model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# latih model
history=model.fit(train_generator,
                  steps_per_epoch=len(train_generator),
                  epochs=10,
                  validation_data=val_generator,
                  validation_steps=len(val_generator)
)

#%%

# evaluasi model yang telah dibuat dengan data test yang belum dilihat sebelumnya
loss, accuracy = model.evaluate(test_generator, steps=len(test_generator))
print('Test accuracy:', accuracy)
print('Test loss:', loss)

# simpan model
model.save("modelklasifikasibanjir"+folder+".h5")

#%%

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Loss')

plt.show()

#%%

import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

#memperoleh atribut nama file dari dataset test
filenames = test_generator.filenames
#memperoleh atribut label asli dari dataset test
y_true = test_generator.classes
#memperoleh prediksi dataset test dari model yang dibuat
y_pred = model.predict(test_generator)
#memperoleh prediksi tertinggi untuk confussion matrix
y_pred_classes = np.argmax(y_pred, axis=1)
#membuat confussion matrix
cm = confusion_matrix(y_true, y_pred_classes)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Confusion Matrix Plot
plot_confusion_matrix(cm=cm, classes=labels, title='Confusion Matrix')

#%%
predictions = model.predict(test_generator, steps=len(test_generator))

recall = metrics.recall_score(y_true=test_generator.classes, y_pred=predictions.argmax(axis=1), average='macro')
precision = metrics.precision_score(y_true=test_generator.classes, y_pred=predictions.argmax(axis=1), average='macro')
f1_score = metrics.f1_score(y_true=test_generator.classes, y_pred=predictions.argmax(axis=1), average='macro')
accuracy = metrics.accuracy_score(y_true=test_generator.classes, y_pred=predictions.argmax(axis=1))

print('Recall: ', recall)
print('Precision: ', precision)
print('F1 Score: ', f1_score)
print('Accuracy: ', accuracy)
#%%

# Mencetak daftar file yang terprediksi salah
misclassified_files = []
for i in range(len(y_true)):
    if y_true[i] != y_pred_classes[i]:
        misclassified_files.append(filenames[i])

print("Dataset Test yang Terprediksi Salah:")
for file in misclassified_files:
    print(file)